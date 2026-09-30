"""Validate AnnData files before embedding with bundled transcriptomic FMs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import pickle
import re
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from transcriptomic_fms.utils.gene_ids import looks_like_ensembl_id, normalize_ensembl_set

_ENSEMBL_COL_CANDIDATES = (
    "ensembl_id",
    "ensembl_gene_id",
    "ensembl ids",
    "ensembl_ids",
    "gene_id",
    "gene_ids",
    "feature_id",
    "id",
)
_SYMBOL_COL_CANDIDATES = (
    "feature_name",
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_names",
    "symbol",
    "name",
)
_EMBED_OUTPUT_SUFFIXES = ("_embeddings.h5ad", "_emb.h5ad")
_ENSEMBL_MODELS = frozenset({"geneformer", "scconcept", "transcriptformer"})
_SPECIAL_TOKEN_RE = re.compile(r"^<.*>$")
_DEFAULT_VOCABS_DIR = Path(__file__).resolve().parents[1] / "models" / "vocabs"


class Level(str, Enum):
    """Validation severity."""

    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class Finding:
    """Single validation finding."""

    level: Level
    message: str


@dataclass
class FileReport:
    """Validation findings for one file or directory-level check."""

    path: Path
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst_level(self) -> Level:
        if any(f.level == Level.ERROR for f in self.findings):
            return Level.ERROR
        if any(f.level == Level.WARN for f in self.findings):
            return Level.WARN
        return Level.OK


def _add(report: FileReport, level: Level, message: str) -> None:
    report.findings.append(Finding(level=level, message=message))


def _first_var_column(adata: ad.AnnData, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in adata.var.columns), None)


def _looks_like_ensembl_id(value: object) -> bool:
    return looks_like_ensembl_id(value)


def _sample_matrix_values(x: sp.spmatrix | np.ndarray, max_values: int = 200_000) -> np.ndarray:
    if sp.issparse(x):
        data = x.data
    else:
        data = np.asarray(x).ravel()
    if data.size == 0:
        return data.astype(np.float64)
    if data.size <= max_values:
        return data.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(data.size, size=max_values, replace=False)
    return data[idx].astype(np.float64)


def check_raw_counts_in_x(x: sp.spmatrix | np.ndarray) -> list[Finding]:
    """Heuristically validate that ``adata.X`` stores raw counts."""
    findings: list[Finding] = []
    values = _sample_matrix_values(x)
    if values.size == 0:
        findings.append(Finding(Level.ERROR, "X is empty"))
        return findings

    if np.any(values < 0):
        findings.append(Finding(Level.ERROR, "X contains negative values (expected raw counts)"))
        return findings

    rounded = np.round(values)
    integer_like = float(np.mean(np.isclose(values, rounded, rtol=0, atol=1e-5)))
    if integer_like < 0.9:
        findings.append(
            Finding(
                Level.ERROR,
                "X does not look like integer raw counts "
                f"(integer-like fraction={integer_like:.3f})",
            )
        )
    elif integer_like < 0.99:
        findings.append(
            Finding(
                Level.WARN,
                "X is mostly integer-like but not fully "
                f"(integer-like fraction={integer_like:.3f})",
            )
        )

    positive = values[values > 0]
    if positive.size > 0:
        frac_fractional = float(np.mean((positive % 1) > 1e-5))
        if frac_fractional > 0.05:
            findings.append(
                Finding(
                    Level.WARN,
                    f"Many non-zero X entries are non-integer (fraction={frac_fractional:.3f})",
                )
            )
        max_val = float(np.max(positive))
        if integer_like < 0.95 and max_val < 25 and float(np.mean(positive < 15)) > 0.8:
            findings.append(
                Finding(
                    Level.WARN,
                    "X value range resembles log-normalized data (max<25, most values<15)",
                )
            )

    return findings


def check_zero_count_cells(adata: ad.AnnData, *, max_sample_names: int = 5) -> list[Finding]:
    """Flag cells with zero total counts in ``adata.X`` (breaks count-based embedders)."""
    findings: list[Finding] = []
    x = adata.X
    if sp.issparse(x):
        totals = np.asarray(x.sum(axis=1)).ravel()
    else:
        totals = np.asarray(x).sum(axis=1).ravel()
    zero_mask = totals <= 0
    n_zero = int(zero_mask.sum())
    n_total = int(totals.shape[0])
    if n_zero == 0:
        return findings

    frac = n_zero / n_total if n_total else 0.0
    sample = adata.obs_names[zero_mask][:max_sample_names].astype(str).tolist()
    sample_txt = f"; examples: {', '.join(sample)}" if sample else ""
    findings.append(
        Finding(
            Level.ERROR,
            f"{n_zero}/{n_total} cells have zero total counts ({frac:.1%}){sample_txt}",
        )
    )
    return findings


def extract_gene_symbols(adata: ad.AnnData) -> list[str]:
    """Mirror the model adapters' gene-symbol lookup order."""
    column = _first_var_column(adata, _SYMBOL_COL_CANDIDATES)
    if column is not None:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def extract_ensembl_ids(adata: ad.AnnData, *, ensembl_col: str = "ensembl_id") -> list[str]:
    if ensembl_col in adata.var.columns:
        return adata.var[ensembl_col].astype(str).tolist()
    column = _first_var_column(adata, _ENSEMBL_COL_CANDIDATES)
    if column is not None:
        return adata.var[column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _normalize_gene_set(genes: Iterable[str], *, uppercase: bool = False) -> set[str]:
    out: set[str] = set()
    for gene in genes:
        value = str(gene).strip()
        if not value or value.lower() == "nan" or _SPECIAL_TOKEN_RE.match(value):
            continue
        out.add(value.upper() if uppercase else value)
    return out


def _normalize_ensembl_set(genes: Iterable[str]) -> set[str]:
    return normalize_ensembl_set(genes)


def _read_json_genes(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [str(key) for key in data]
    if isinstance(data, list):
        return [str(value) for value in data]
    raise ValueError(f"Unsupported JSON vocab format: {path}")


def _read_text_genes(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _read_pickle_keys(path: Path) -> list[str]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle dictionary in {path}, got {type(data).__name__}")
    return [str(key) for key in data if not _SPECIAL_TOKEN_RE.match(str(key))]


def load_bundled_model_gene_lists(
    *,
    models: Sequence[str] | None = None,
    vocabs_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Load bundled model vocabularies without importing optional model packages."""
    if vocabs_dir is None:
        vocabs_dir = _DEFAULT_VOCABS_DIR

    requested = {model.lower() for model in models} if models else None
    loaders = {
        "scgpt": lambda: _read_json_genes(vocabs_dir / "scgpt_vocab.json"),
        "scimilarity": lambda: _read_text_genes(vocabs_dir / "scimilarity_vocab.tsv"),
        "scfoundation": lambda: pd.read_csv(vocabs_dir / "scfoundation_vocab.tsv", sep="\t")[
            "gene_name"
        ]
        .dropna()
        .astype(str)
        .tolist(),
        "scconcept": lambda: pd.read_csv(vocabs_dir / "scconcept_vocab.csv")["gene_id"]
        .dropna()
        .astype(str)
        .tolist(),
        "geneformer": lambda: _read_pickle_keys(
            vocabs_dir / "geneformer_vocab" / "token_dictionary_gc30M.pkl"
        ),
        "transcriptformer": lambda: _read_text_genes(
            vocabs_dir / "transcriptformer_vocab" / "gene_vocab.txt"
        ),
    }

    unknown = sorted((requested or set()) - set(loaders))
    if unknown:
        available = ", ".join(sorted(loaders))
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Available: {available}")

    model_names = sorted(requested) if requested else sorted(loaders)
    lists: dict[str, list[str]] = {}
    for model in model_names:
        lists[model] = loaders[model]()
    return lists


def check_structure(adata: ad.AnnData) -> list[Finding]:
    findings: list[Finding] = []
    n_obs, n_vars = adata.shape
    if adata.n_obs != n_obs or adata.n_vars != n_vars:
        findings.append(Finding(Level.ERROR, "AnnData internal shape inconsistent"))
    if len(adata.obs_names) != n_obs:
        findings.append(Finding(Level.ERROR, "obs index length does not match n_obs"))
    if len(adata.var_names) != n_vars:
        findings.append(Finding(Level.ERROR, "var index length does not match n_vars"))

    if not adata.obs_names.is_unique:
        n_dup = int(adata.obs_names.duplicated().sum())
        findings.append(Finding(Level.ERROR, f"obs_names are not unique ({n_dup} duplicates)"))
    if not adata.var_names.is_unique:
        n_dup = int(adata.var_names.duplicated().sum())
        findings.append(Finding(Level.ERROR, f"var_names are not unique ({n_dup} duplicates)"))

    if adata.raw is not None:
        findings.append(
            Finding(
                Level.WARN,
                "adata.raw is set (embedding inputs are expected to keep counts in .X)",
            )
        )
    if adata.layers:
        findings.append(
            Finding(
                Level.WARN,
                f"adata.layers present ({', '.join(adata.layers.keys())}); "
                "embedding pipeline expects counts in .X",
            )
        )
    return findings


def check_gene_metadata(
    adata: ad.AnnData,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
) -> list[Finding]:
    findings: list[Finding] = []
    symbols = adata.var_names.astype(str)
    sample = symbols[: min(200, len(symbols))]
    ensembl_frac = (
        float(np.mean([_looks_like_ensembl_id(value) for value in sample])) if sample.size else 0.0
    )
    if ensembl_frac > 0.5:
        findings.append(
            Finding(
                Level.ERROR,
                "var_names look like Ensembl IDs; expected gene symbols in var_names",
            )
        )

    if gene_name_column not in adata.var.columns:
        findings.append(Finding(Level.ERROR, f"Missing adata.var['{gene_name_column}']"))
    else:
        gene_names = adata.var[gene_name_column].astype(str)
        empty = int(((gene_names.str.strip() == "") | adata.var[gene_name_column].isna()).sum())
        if empty:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"adata.var['{gene_name_column}'] has {empty} missing/empty entries",
                )
            )
        mismatches = int((gene_names.to_numpy() != symbols.to_numpy()).sum())
        if mismatches:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"var_names disagree with var['{gene_name_column}'] on {mismatches} genes",
                )
            )

    ensembl_column = ensembl_id_column if ensembl_id_column in adata.var.columns else None
    if ensembl_column is None:
        ensembl_column = _first_var_column(adata, _ENSEMBL_COL_CANDIDATES)
    if ensembl_column is None:
        findings.append(
            Finding(
                Level.ERROR,
                f"Missing Ensembl ID column (expected var['{ensembl_id_column}'])",
            )
        )
    else:
        ensembl = adata.var[ensembl_column].astype(str).map(lambda value: str(value).strip())
        empty_ensembl = int(((ensembl == "") | (ensembl.str.lower() == "nan")).sum())
        if empty_ensembl:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"var['{ensembl_column}'] has {empty_ensembl} missing Ensembl IDs",
                )
            )
        sample_ensembl = ensembl.head(min(200, len(ensembl)))
        valid = float(np.mean([_looks_like_ensembl_id(value) for value in sample_ensembl]))
        if valid < 0.8:
            findings.append(
                Finding(
                    Level.WARN,
                    f"var['{ensembl_column}'] entries often do not match Ensembl ID pattern "
                    f"(valid fraction in sample={valid:.3f})",
                )
            )

    return findings


def check_model_gene_overlap(
    adata: ad.AnnData,
    model_gene_lists: Mapping[str, Sequence[str]],
    *,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
) -> list[Finding]:
    """Report the fraction of dataset genes present in bundled model vocabularies."""
    findings: list[Finding] = []
    if not model_gene_lists:
        return findings

    file_symbols_upper = _normalize_gene_set(extract_gene_symbols(adata), uppercase=True)
    file_ensembl = _normalize_ensembl_set(extract_ensembl_ids(adata))

    for model, genes in sorted(model_gene_lists.items()):
        model_name = model.lower()
        if model_name in _ENSEMBL_MODELS:
            ref = _normalize_ensembl_set(genes)
            file_genes = file_ensembl
            overlap = len(file_genes & ref) / len(file_genes) if file_genes else 0.0
            metric = "Ensembl IDs"
            threshold = min_ensembl_overlap
        else:
            ref = _normalize_gene_set(genes, uppercase=True)
            file_genes = file_symbols_upper
            overlap = len(file_genes & ref) / len(file_genes) if file_genes else 0.0
            metric = "gene symbols"
            threshold = min_symbol_overlap

        if not ref:
            findings.append(Finding(Level.WARN, f"{model_name}: vocab is empty"))
            continue
        if not file_genes:
            findings.append(Finding(Level.ERROR, f"{model_name}: no dataset {metric} available"))
            continue

        n_matched = len(file_genes & ref)
        n_dataset = len(file_genes)
        n_vocab = len(ref)
        pct = 100.0 * overlap
        overlap_detail = (
            f"{n_matched}/{n_dataset} dataset {metric} found in model vocab "
            f"({n_vocab} vocab genes)"
        )
        if overlap < threshold:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{model_name}: only {pct:.1f}% overlap; {overlap_detail}",
                )
            )
        elif overlap < 0.5:
            findings.append(
                Finding(
                    Level.WARN,
                    f"{model_name}: {pct:.1f}% overlap; {overlap_detail}",
                )
            )
        else:
            findings.append(
                Finding(
                    Level.OK,
                    f"{model_name}: {pct:.1f}% overlap; {overlap_detail}",
                )
            )

    return findings


def load_transcriptformer_assay_vocab(vocabs_dir: Path | None = None) -> set[str]:
    """Load the assay labels TranscriptFormer checkpoints recognize."""
    vocabs_dir = vocabs_dir or _DEFAULT_VOCABS_DIR
    path = vocabs_dir / "transcriptformer_vocab" / "assay_vocab.json"
    return {label for label in json.loads(path.read_text()) if label != "unknown"}


def check_transcriptformer_assay(
    adata: ad.AnnData,
    *,
    assay_vocab: set[str] | None = None,
    assay_column: str = "assay",
    max_sample_labels: int = 5,
) -> list[Finding]:
    """Warn when cells would fall back to TranscriptFormer's 'unknown' assay token."""
    if assay_vocab is None:
        assay_vocab = load_transcriptformer_assay_vocab()
    if assay_column not in adata.obs.columns:
        return [
            Finding(
                Level.WARN,
                f"transcriptformer: obs['{assay_column}'] missing; all cells will use the "
                "'unknown' assay token (set it or pass --assay)",
            )
        ]

    labels = adata.obs[assay_column].astype(object)
    unmatched = sorted(
        {
            "<missing>" if pd.isna(v) else str(v)
            for v in labels
            if pd.isna(v) or str(v) not in assay_vocab
        }
    )
    if not unmatched:
        return []
    sample = ", ".join(repr(v) for v in unmatched[:max_sample_labels])
    return [
        Finding(
            Level.WARN,
            f"transcriptformer: {len(unmatched)} obs['{assay_column}'] label(s) not in the "
            f"assay vocab and will map to 'unknown': {sample}",
        )
    ]


def validate_h5ad_file(
    path: Path,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
    model_gene_lists: Mapping[str, Sequence[str]] | None = None,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
) -> FileReport:
    report = FileReport(path=path)
    try:
        adata = ad.read_h5ad(path)
    except Exception as exc:
        _add(report, Level.ERROR, f"Failed to read h5ad: {exc}")
        return report

    for finding in check_structure(adata):
        _add(report, finding.level, finding.message)
    for finding in check_raw_counts_in_x(adata.X):
        _add(report, finding.level, finding.message)
    for finding in check_zero_count_cells(adata):
        _add(report, finding.level, finding.message)
    for finding in check_gene_metadata(
        adata,
        gene_name_column=gene_name_column,
        ensembl_id_column=ensembl_id_column,
    ):
        _add(report, finding.level, finding.message)

    try:
        symbols = extract_gene_symbols(adata)
        if len(set(symbols)) < len(symbols):
            _add(
                report,
                Level.WARN,
                "Duplicate gene symbols in feature metadata (var_names should be unique)",
            )
    except Exception as exc:
        _add(report, Level.ERROR, f"Cannot extract gene symbols: {exc}")

    if model_gene_lists:
        for finding in check_model_gene_overlap(
            adata,
            model_gene_lists,
            min_symbol_overlap=min_symbol_overlap,
            min_ensembl_overlap=min_ensembl_overlap,
        ):
            _add(report, finding.level, finding.message)
        if "transcriptformer" in {model.lower() for model in model_gene_lists}:
            for finding in check_transcriptformer_assay(adata):
                _add(report, finding.level, finding.message)

    if not report.findings:
        _add(report, Level.OK, "All checks passed")
    return report


def iter_h5ad_paths(
    directory: Path,
    *,
    skip_embed_outputs: bool = True,
) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(directory.glob("*.h5ad"))
    if not skip_embed_outputs:
        return paths

    return [
        path
        for path in paths
        if not any(path.name.endswith(suffix) for suffix in _EMBED_OUTPUT_SUFFIXES)
    ]


def _reference_path(paths: Sequence[Path]) -> Path:
    for candidate in paths:
        if candidate.name == "reference.h5ad":
            return candidate
    return paths[0]


def check_cross_file_consistency(paths: Sequence[Path]) -> list[Finding]:
    """Ensure directory inputs share the same cell and gene indices as the reference file."""
    if len(paths) < 2:
        return []

    ref_path = _reference_path(paths)
    try:
        ref = ad.read_h5ad(ref_path, backed="r")
        ref_obs = list(ref.obs_names)
        ref_var = list(ref.var_names)
        ref.file.close()
    except Exception as exc:
        return [Finding(Level.ERROR, f"Cannot read reference {ref_path.name}: {exc}")]

    findings: list[Finding] = []
    ref_obs_set = set(ref_obs)
    ref_var_set = set(ref_var)

    for path in paths:
        if path == ref_path:
            continue
        try:
            adata = ad.read_h5ad(path, backed="r")
            obs = list(adata.obs_names)
            var = list(adata.var_names)
            adata.file.close()
        except Exception as exc:
            findings.append(
                Finding(Level.ERROR, f"{path.name}: cannot read for cross-check: {exc}")
            )
            continue

        if obs != ref_obs:
            missing_obs = len(ref_obs_set - set(obs))
            extra_obs = len(set(obs) - ref_obs_set)
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{path.name}: obs_names differ from {ref_path.name} "
                    f"(missing={missing_obs}, extra={extra_obs}, order_mismatch={obs != ref_obs})",
                )
            )
        if var != ref_var:
            missing_var = len(ref_var_set - set(var))
            extra_var = len(set(var) - ref_var_set)
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{path.name}: var_names differ from {ref_path.name} "
                    f"(missing={missing_var}, extra={extra_var}, order_mismatch={var != ref_var})",
                )
            )

    if not findings:
        findings.append(
            Finding(
                Level.OK,
                f"All {len(paths) - 1} files match {ref_path.name} obs/var names and order",
            )
        )
    return findings


def validate_path(
    input_path: Path,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
    model_gene_lists: Mapping[str, Sequence[str]] | None = None,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
    skip_embed_outputs: bool = True,
) -> tuple[list[FileReport], int]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix != ".h5ad":
            raise ValueError(f"Expected a .h5ad file or directory, got: {input_path}")
        reports = [
            validate_h5ad_file(
                input_path,
                gene_name_column=gene_name_column,
                ensembl_id_column=ensembl_id_column,
                model_gene_lists=model_gene_lists,
                min_symbol_overlap=min_symbol_overlap,
                min_ensembl_overlap=min_ensembl_overlap,
            )
        ]
    else:
        paths = iter_h5ad_paths(input_path, skip_embed_outputs=skip_embed_outputs)
        if not paths:
            raise FileNotFoundError(f"No .h5ad files found in {input_path}")
        reports = [
            validate_h5ad_file(
                path,
                gene_name_column=gene_name_column,
                ensembl_id_column=ensembl_id_column,
                model_gene_lists=model_gene_lists,
                min_symbol_overlap=min_symbol_overlap,
                min_ensembl_overlap=min_ensembl_overlap,
            )
            for path in paths
        ]
        cross = (
            check_cross_file_consistency(paths)
            if any(path.name == "reference.h5ad" for path in paths)
            else []
        )
        if cross:
            summary = FileReport(path=input_path / "[cross-file]")
            for finding in cross:
                _add(summary, finding.level, finding.message)
            reports.append(summary)

    exit_code = 1 if any(report.worst_level == Level.ERROR for report in reports) else 0
    return reports, exit_code


def format_report(reports: Sequence[FileReport], *, input_path: Path) -> str:
    lines = [
        f"=== pre-embedding check: {input_path} ({len(reports)} reports) ===",
        "",
    ]
    n_err = n_warn = n_ok = 0
    for report in reports:
        level = report.worst_level
        if level == Level.ERROR:
            n_err += 1
        elif level == Level.WARN:
            n_warn += 1
        else:
            n_ok += 1
        lines.append(f"{level.value:5}  {report.path.name}")
        for finding in report.findings:
            if finding.level == Level.OK and finding.message == "All checks passed":
                continue
            lines.append(f"       [{finding.level.value}] {finding.message}")
        lines.append("")

    lines.append(f"Summary: {n_ok} ok, {n_warn} warn, {n_err} error (of {len(reports)} reports)")
    return "\n".join(lines)
