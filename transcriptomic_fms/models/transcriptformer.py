"""TranscriptFormer embedding model."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from transcriptomic_fms.models.base import BaseEmbeddingModel
from transcriptomic_fms.models.registry import register_model
from transcriptomic_fms.utils.gene_ids import looks_like_ensembl_id, normalize_ensembl_ids
from transcriptomic_fms.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from transcriptformer.model.inference import run_inference as _run_inference
except ImportError as e:  # pragma: no cover - import-time guard
    # Only treat a missing top-level `transcriptformer` module as "not installed".
    if getattr(e, "name", None) == "transcriptformer":
        _run_inference = None
    else:
        raise

VARIANTS = {
    "tf-sapiens": "tf_sapiens",
    "tf-exemplar": "tf_exemplar",
    "tf-metazoa": "tf_metazoa",
}
WEIGHTS_URL = "https://czi-transcriptformer.s3.amazonaws.com/weights/{name}.tar.gz"
DEFAULT_CHECKPOINT_DIR = "models/transcriptformer"

# Column names used in the minimal AnnData handed to TranscriptFormer. The checkpoint
# config sets ``aux_cols: assay``, so the assay column must be named exactly "assay".
_GENE_COL = "ensembl_id"
_ASSAY_COL = "assay"
_ROW_COL = "_tfms_row"
_UNKNOWN_ASSAY = "unknown"


def normalize_variant(variant: str) -> str:
    """Map a user-facing variant name (``tf-sapiens`` / ``tf_sapiens``) to its directory name."""
    key = str(variant).strip().lower().replace("_", "-")
    if key not in VARIANTS:
        raise ValueError(
            f"Unknown TranscriptFormer variant '{variant}'. "
            f"Choose one of: {', '.join(sorted(VARIANTS))}"
        )
    return VARIANTS[key]


@register_model("transcriptformer")
class TranscriptFormerModel(BaseEmbeddingModel):
    """TranscriptFormer generative cross-species model for single-cell embeddings."""

    def __init__(
        self,
        model_name: str,
        variant: str = "tf-sapiens",
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        auto_download: bool = True,
        pretrained_embedding: Optional[str] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        clip_counts: int = 30,
        normalize_to_scale: float = 0,
        assay: Optional[str] = None,
        assay_column: str = "assay",
        remove_duplicate_genes: bool = False,
        disable_compile_block_mask: bool = False,
        requires_gpu: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize TranscriptFormer model.

        Args:
            model_name: Model identifier (fixed to ``"transcriptformer"`` in registry).
            variant: Checkpoint to use: ``tf-sapiens`` (human-only), ``tf-exemplar``
                     (human + 4 model organisms) or ``tf-metazoa`` (12 species).
            checkpoint_dir: Directory holding extracted checkpoints; the variant is
                            expected in ``<checkpoint_dir>/<tf_variant>/``.
            checkpoint_path: Explicit path to an extracted checkpoint directory
                             (overrides ``variant`` / ``checkpoint_dir``).
            auto_download: Download the checkpoint from the public S3 bucket if missing.
            pretrained_embedding: Comma-separated ESM-2 gene embedding ``.h5`` file(s) for
                                  species outside the checkpoint's training set.
            device: ``"cuda"`` or ``"cpu"``. If ``None``, auto-detects.
            precision: Lightning precision (``"16-mixed"`` or ``"32"``). Defaults to
                       ``"16-mixed"`` on CUDA and ``"32"`` on CPU.
            clip_counts: Maximum count value per gene (TranscriptFormer default: 30).
            normalize_to_scale: Scale total counts per cell to this value (0 disables).
            assay: Assay label applied to every cell (overrides ``assay_column``).
            assay_column: ``obs`` column with CELLxGENE-style assay labels.
            remove_duplicate_genes: Keep the first occurrence of duplicated Ensembl IDs
                                    instead of raising.
            disable_compile_block_mask: Disable block-mask compilation (forced on CPU).
            requires_gpu: Whether this model requires a GPU (default: True).
            **kwargs: Additional, model-specific configuration (stored in ``self.config``).
        """
        super().__init__(model_name=model_name, requires_gpu=requires_gpu, **kwargs)

        if _run_inference is None:
            raise ImportError(
                "TranscriptFormer is not installed. Install with:\n"
                "  make install-model MODEL=transcriptformer\n"
                "  or: uv sync --extra transcriptformer\n"
                "  or: pip install transcriptformer==0.6.1\n"
                "Note: TranscriptFormer pins torch==2.5.1 and numpy==2.2.6; the HPC "
                "container is the recommended way to run it."
            )

        self.variant = normalize_variant(variant)
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.checkpoint_path = checkpoint_path
        self.auto_download = auto_download
        self.pretrained_embedding = pretrained_embedding
        self.clip_counts = clip_counts
        self.normalize_to_scale = normalize_to_scale
        self.assay = assay
        self.assay_column = assay_column
        self.remove_duplicate_genes = remove_duplicate_genes

        if device is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
                self.device = "cuda" if cuda_available else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = str(device)

        # Lightning rejects fp16 mixed precision on CPU, and FlexAttention block-mask
        # compilation needs a GPU toolchain.
        if precision is None:
            precision = "16-mixed" if self.device == "cuda" else "32"
        self.precision = str(precision)
        self.compile_block_mask = not disable_compile_block_mask and self.device != "cpu"

    def _resolve_checkpoint(self) -> Path:
        """Return the extracted checkpoint directory, downloading it if allowed."""
        if self.checkpoint_path:
            ckpt = Path(self.checkpoint_path).expanduser().resolve()
        else:
            ckpt = (Path(self.checkpoint_dir) / self.variant).expanduser().resolve()

        required = [ckpt / "config.json", ckpt / "model_weights.pt", ckpt / "vocabs"]
        if all(p.exists() for p in required):
            return ckpt

        if self.checkpoint_path or not self.auto_download:
            missing = ", ".join(str(p) for p in required if not p.exists())
            raise FileNotFoundError(
                f"TranscriptFormer checkpoint is incomplete at {ckpt} (missing: {missing}).\n"
                f"Download it with: make download-transcriptformer VARIANT="
                f"{self.variant.replace('_', '-')}"
            )

        self._download_checkpoint(ckpt.parent)
        return ckpt

    def _download_checkpoint(self, target_dir: Path) -> None:
        import tarfile
        import urllib.request

        url = WEIGHTS_URL.format(name=self.variant)
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading TranscriptFormer checkpoint {url} -> {target_dir} ...")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", dir=target_dir) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            with tarfile.open(tmp.name, mode="r:gz") as tar:
                tar.extractall(path=target_dir, filter="data")
        logger.info("Checkpoint download complete.")

    def _get_gene_ids(self, adata: sc.AnnData) -> list[str]:
        """
        Extract Ensembl gene IDs for TranscriptFormer.

        Checks in order:
        1. ``var['ensembl_id']`` column (TranscriptFormer's native column)
        2. ``var['gene_id']`` column
        3. ``var.index``
        """
        for column in ("ensembl_id", "gene_id"):
            if column in adata.var.columns:
                return normalize_ensembl_ids(adata.var[column])
        if adata.var_names is not None and len(adata.var_names) > 0:
            return normalize_ensembl_ids(adata.var_names)
        raise ValueError(
            "Cannot determine gene IDs for TranscriptFormer. Please ensure one of:\n"
            "  - var['ensembl_id'] column exists\n"
            "  - var['gene_id'] column exists\n"
            "  - var.index contains Ensembl gene IDs"
        )

    def _get_assay_labels(self, adata: sc.AnnData) -> list[str]:
        """Return per-cell assay labels, falling back to TranscriptFormer's unknown token."""
        if self.assay is not None:
            return [str(self.assay)] * adata.n_obs
        if self.assay_column in adata.obs.columns:
            labels = adata.obs[self.assay_column].astype(object)
            return [_UNKNOWN_ASSAY if pd.isna(v) or not str(v).strip() else str(v) for v in labels]
        logger.warning(
            f"obs['{self.assay_column}'] not found; embedding all cells with the "
            f"'{_UNKNOWN_ASSAY}' assay token. Pass --assay or --assay-column to set it."
        )
        return [_UNKNOWN_ASSAY] * adata.n_obs

    def preprocess(self, adata: sc.AnnData, output_path: Optional[Path] = None) -> sc.AnnData:
        """
        Preprocess data for TranscriptFormer.

        Requirements:
        - Raw (unnormalized) counts in ``adata.X``.
        - Ensembl gene IDs via ``var['ensembl_id']``, ``var['gene_id']`` or ``var.index``.

        Populates ``var['ensembl_id']`` with version-stripped Ensembl IDs. Count clipping
        and vocabulary filtering happen inside TranscriptFormer.
        """
        adata = adata.copy()
        gene_ids = self._get_gene_ids(adata)

        sample = gene_ids[: min(200, len(gene_ids))]
        if sample and not any(looks_like_ensembl_id(g) for g in sample):
            raise ValueError(
                "TranscriptFormer requires Ensembl gene IDs, but none of the first "
                f"{len(sample)} gene identifiers look like Ensembl IDs "
                f"(e.g. {sample[:3]}). Provide var['ensembl_id']."
            )

        adata.var[_GENE_COL] = gene_ids

        if output_path:
            adata.write(output_path)

        return adata

    def _build_input(self, adata: sc.AnnData) -> sc.AnnData:
        """Build the minimal AnnData TranscriptFormer consumes (counts, genes, assay, row id)."""
        x = adata.X
        # TranscriptFormer densifies via isinstance checks on scipy sparse *matrix* types.
        x = sp.csr_matrix(x, dtype=np.float32) if sp.issparse(x) else np.asarray(x, np.float32)
        obs = pd.DataFrame(
            {
                _ASSAY_COL: self._get_assay_labels(adata),
                _ROW_COL: np.arange(adata.n_obs, dtype=np.int64),
            },
            index=adata.obs_names.astype(str),
        )
        var = pd.DataFrame(
            {_GENE_COL: adata.var[_GENE_COL].astype(str).to_numpy()},
            index=adata.var_names.astype(str),
        )
        return sc.AnnData(X=x, obs=obs, var=var)

    def _build_config(self, ckpt: Path, batch_size: int) -> Any:
        """Build the inference config the same way ``transcriptformer inference`` does."""
        from omegaconf import OmegaConf
        import transcriptformer.cli as tf_cli

        base_cfg = OmegaConf.load(Path(tf_cli.__file__).parent / "conf" / "inference_config.yaml")
        with open(ckpt / "config.json") as f:
            cfg = OmegaConf.merge(OmegaConf.create(json.load(f)), base_cfg)

        cfg.model.checkpoint_path = str(ckpt)
        cfg.model.model_type = "transcriptformer"
        cfg.model.inference_config.batch_size = batch_size
        cfg.model.inference_config.precision = self.precision
        cfg.model.inference_config.emb_type = "cell"
        cfg.model.inference_config.output_keys = ["embeddings"]
        cfg.model.inference_config.obs_keys = [_ROW_COL]
        # Multi-GPU runs spawn DDP processes and write per-rank files, which breaks the
        # in-process return value this adapter relies on.
        cfg.model.inference_config.num_gpus = 1
        cfg.model.inference_config.device = self.device
        cfg.model.inference_config.use_oom_dataloader = False
        cfg.model.inference_config.load_checkpoint = str(ckpt / "model_weights.pt")
        cfg.model.inference_config.pretrained_embedding = (
            [p.strip() for p in str(self.pretrained_embedding).split(",") if p.strip()]
            if self.pretrained_embedding
            else None
        )
        cfg.model.data_config.gene_col_name = _GENE_COL
        cfg.model.data_config.use_raw = False
        cfg.model.data_config.clip_counts = self.clip_counts
        cfg.model.data_config.normalize_to_scale = self.normalize_to_scale
        cfg.model.data_config.filter_to_vocabs = True
        cfg.model.data_config.remove_duplicate_genes = self.remove_duplicate_genes
        cfg.model.data_config.n_data_workers = 0
        cfg.model.data_config.aux_vocab_path = str(ckpt / "vocabs")
        cfg.model.data_config.esm2_mappings_path = str(ckpt / "vocabs")
        cfg.model.model_config.compile_block_mask = self.compile_block_mask
        return cfg

    def embed(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> sc.AnnData:
        """
        Generate TranscriptFormer cell embeddings (mean-pooled over genes).

        Args:
            adata: Preprocessed AnnData (must contain ``var['ensembl_id']``).
            output_path: Path where embeddings will be saved (kept for API consistency).
            batch_size: Inference batch size (default: 8; use 1-4 on 16GB GPUs).
            **kwargs: Ignored (model options are set in the constructor).
        """
        if _GENE_COL not in adata.var.columns:
            raise ValueError(
                f"Required gene ID column '{_GENE_COL}' not found in adata.var. "
                "Run preprocess() first."
            )

        ckpt = self._resolve_checkpoint()
        cfg = self._build_config(ckpt, batch_size=batch_size or 8)
        tf_input = self._build_input(adata)

        logger.info(
            f"Extracting TranscriptFormer embeddings (variant={self.variant}, "
            f"device={self.device}, precision={self.precision})..."
        )

        # Lightning's CSVLogger writes ./logs relative to the working directory, which
        # may be read-only inside the container.
        prev_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="transcriptformer_") as tmp:
            os.chdir(tmp)
            try:
                result = _run_inference(cfg, data_files=[tf_input])
            finally:
                os.chdir(prev_cwd)

        if "embeddings" not in result.obsm:
            raise ValueError(
                "TranscriptFormer did not return obsm['embeddings']. "
                f"Available keys: {list(result.obsm.keys())}"
            )

        embeddings = np.asarray(result.obsm["embeddings"], dtype=np.float32)
        rows = np.asarray(result.obs[_ROW_COL]).astype(np.int64).ravel()
        if len(rows) != embeddings.shape[0]:
            raise ValueError(
                f"TranscriptFormer returned {embeddings.shape[0]} embeddings but "
                f"{len(rows)} row identifiers."
            )
        if len(rows) != adata.n_obs:
            logger.warning(f"TranscriptFormer returned {len(rows)} of {adata.n_obs} cells.")

        result_adata = sc.AnnData(X=embeddings, obs=adata.obs.iloc[rows].copy())
        self.validate_embeddings(result_adata)
        return result_adata

    def compute_sensitivity(
        self,
        adata: sc.AnnData,
        output_path: Path,
        batch_size: Optional[int] = None,
        n_cells: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Sensitivity analysis not yet implemented for TranscriptFormer."""
        raise NotImplementedError(
            "Sensitivity analysis for TranscriptFormer is not yet implemented."
        )

    def get_container_command(
        self,
        adata_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> list[str]:
        """Get command to run this model in a container."""
        cmd = [
            "python",
            "-m",
            "transcriptomic_fms.cli.main",
            "embed",
            "--model",
            self.model_name,
            "--input",
            str(adata_path),
            "--output",
            str(output_path),
            "--variant",
            self.variant.replace("_", "-"),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--device",
            self.device,
        ]
        if self.checkpoint_path:
            cmd.extend(["--checkpoint-path", str(self.checkpoint_path)])
        if self.pretrained_embedding:
            cmd.extend(["--pretrained-embedding", str(self.pretrained_embedding)])
        if self.assay is not None:
            cmd.extend(["--assay", str(self.assay)])

        for k, v in kwargs.items():
            if v is not None:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        return cmd

    def get_required_dependencies(self) -> list[str]:
        """Get required dependencies for TranscriptFormer."""
        return ["transcriptformer==0.6.1"]

    def get_optional_dependency_group(self) -> Optional[str]:
        """Get optional dependency group name for TranscriptFormer."""
        return "transcriptformer"

    def get_container_name(self) -> Optional[str]:
        """Get container name for TranscriptFormer."""
        return "transcriptformer"
