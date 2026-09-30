"""Tests for the TranscriptFormer adapter (no transcriptformer install or GPU required)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from transcriptomic_fms.models import transcriptformer as tf_module
from transcriptomic_fms.models.transcriptformer import TranscriptFormerModel, normalize_variant


def _make_model(**attrs: object) -> TranscriptFormerModel:
    model = object.__new__(TranscriptFormerModel)
    model.model_name = "transcriptformer"
    model.variant = "tf_sapiens"
    model.checkpoint_dir = "models/transcriptformer"
    model.checkpoint_path = None
    model.auto_download = False
    model.assay = None
    model.assay_column = "assay"
    model.device = "cpu"
    model.precision = "32"
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _make_adata(n_obs: int = 4) -> ad.AnnData:
    x = np.arange(n_obs * 3, dtype=np.float32).reshape(n_obs, 3)
    adata = ad.AnnData(
        X=sp.csr_matrix(x),
        obs={"cell_type": [f"t{i}" for i in range(n_obs)]},
        var={"ensembl_id": ["ENSG00000000003.15", "ENSG00000000005", "ENSG00000000419.2"]},
    )
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = ["TSPAN6", "TNMD", "DPM1"]
    return adata


class TestTranscriptFormerAdapter(unittest.TestCase):
    def test_normalize_variant_accepts_dash_and_underscore(self) -> None:
        self.assertEqual(normalize_variant("tf-sapiens"), "tf_sapiens")
        self.assertEqual(normalize_variant("TF_Metazoa"), "tf_metazoa")
        with self.assertRaises(ValueError):
            normalize_variant("tf-unknown")

    def test_preprocess_strips_versions_into_ensembl_id(self) -> None:
        out = _make_model().preprocess(_make_adata())
        self.assertEqual(
            list(out.var["ensembl_id"]),
            ["ENSG00000000003", "ENSG00000000005", "ENSG00000000419"],
        )

    def test_preprocess_falls_back_to_gene_id_then_index(self) -> None:
        adata = _make_adata()
        adata.var = adata.var.rename(columns={"ensembl_id": "gene_id"})
        out = _make_model().preprocess(adata)
        self.assertEqual(out.var["ensembl_id"].iloc[0], "ENSG00000000003")

        adata = _make_adata()
        adata.var_names = list(adata.var["ensembl_id"])
        del adata.var["ensembl_id"]
        out = _make_model().preprocess(adata)
        self.assertEqual(out.var["ensembl_id"].iloc[2], "ENSG00000000419")

    def test_preprocess_rejects_gene_symbols(self) -> None:
        adata = _make_adata()
        del adata.var["ensembl_id"]
        with self.assertRaisesRegex(ValueError, "Ensembl"):
            _make_model().preprocess(adata)

    def test_assay_labels_explicit_column_and_missing(self) -> None:
        adata = _make_adata(3)
        self.assertEqual(
            _make_model(assay="10x 3' v3")._get_assay_labels(adata), ["10x 3' v3"] * 3
        )

        adata.obs["assay"] = ["10x 3' v2", None, ""]
        self.assertEqual(
            _make_model()._get_assay_labels(adata), ["10x 3' v2", "unknown", "unknown"]
        )

        del adata.obs["assay"]
        with self.assertLogs(tf_module.logger, level="WARNING"):
            labels = _make_model()._get_assay_labels(adata)
        self.assertEqual(labels, ["unknown"] * 3)

    def test_build_input_is_minimal_and_sparse_matrix(self) -> None:
        adata = _make_model().preprocess(_make_adata())
        adata.X = sp.csr_array(adata.X)
        tf_input = _make_model(assay="Smart-seq2")._build_input(adata)
        self.assertIsInstance(tf_input.X, sp.csr_matrix)
        self.assertEqual(list(tf_input.obs.columns), ["assay", "_tfms_row"])
        self.assertEqual(list(tf_input.obs["_tfms_row"]), [0, 1, 2, 3])
        self.assertEqual(list(tf_input.var.columns), ["ensembl_id"])

    def test_resolve_checkpoint_missing_without_download_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_model(checkpoint_dir=tmp)
            with self.assertRaisesRegex(FileNotFoundError, "download-transcriptformer"):
                model._resolve_checkpoint()

            ckpt = Path(tmp) / "tf_sapiens"
            (ckpt / "vocabs").mkdir(parents=True)
            (ckpt / "config.json").write_text("{}")
            (ckpt / "model_weights.pt").write_bytes(b"")
            self.assertEqual(model._resolve_checkpoint(), ckpt.resolve())

    def test_embed_realigns_rows_and_restores_obs(self) -> None:
        adata = _make_model().preprocess(_make_adata())
        n_obs, dim = adata.n_obs, 5

        def fake_run_inference(cfg: object, data_files: list[ad.AnnData]) -> ad.AnnData:
            tf_input = data_files[0]
            order = np.array([2, 0, 3, 1])
            rows = tf_input.obs["_tfms_row"].to_numpy()[order]
            emb = np.repeat(rows[:, None].astype(np.float16), dim, axis=1)
            return ad.AnnData(
                obs=pd.DataFrame({"_tfms_row": rows}),
                obsm={"embeddings": emb},
            )

        model = _make_model()
        with (
            mock.patch.object(tf_module, "_run_inference", fake_run_inference),
            mock.patch.object(
                TranscriptFormerModel, "_resolve_checkpoint", return_value=Path(".")
            ),
            mock.patch.object(TranscriptFormerModel, "_build_config", return_value=object()),
        ):
            result = model.embed(adata, Path("unused.h5ad"))

        self.assertEqual(result.shape, (n_obs, dim))
        self.assertEqual(result.X.dtype, np.float32)
        self.assertEqual(list(result.obs_names), ["cell2", "cell0", "cell3", "cell1"])
        self.assertEqual(list(result.obs["cell_type"]), ["t2", "t0", "t3", "t1"])
        np.testing.assert_array_equal(result.X[:, 0], [2, 0, 3, 1])


if __name__ == "__main__":
    unittest.main()
