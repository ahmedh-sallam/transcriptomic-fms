"""Tests for lightweight pre-embedding h5ad validation."""

from __future__ import annotations

from pathlib import Path
import pickle
import tempfile
import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from transcriptomic_fms.validation.embed_inputs import (
    Level,
    check_raw_counts_in_x,
    check_transcriptformer_assay,
    load_bundled_model_gene_lists,
    validate_h5ad_file,
    validate_path,
)


def _make_valid_adata(n_obs: int = 5, n_vars: int = 8) -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32)
    symbols = [f"GENE{i}" for i in range(n_vars)]
    ensembl = [f"ENSG0000000000{i:02d}" for i in range(n_vars)]
    adata = ad.AnnData(
        X=sp.csr_matrix(x),
        obs={"batch": ["a"] * n_obs},
        var={"gene_name": symbols, "ensembl_id": ensembl},
    )
    adata.var_names = symbols
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    return adata


class TestPreEmbeddingCheck(unittest.TestCase):
    def test_raw_counts_check_fails_log_like_floats(self) -> None:
        x = np.array([[0.1, 1.5], [2.2, 0.0]], dtype=np.float32)
        findings = check_raw_counts_in_x(x)
        self.assertTrue(any(f.level == Level.ERROR for f in findings))

    def test_validate_good_file_with_symbol_and_ensembl_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.h5ad"
            _make_valid_adata().write_h5ad(path)
            report = validate_h5ad_file(
                path,
                model_gene_lists={
                    "scgpt": [f"GENE{i}" for i in range(8)],
                    "geneformer": [f"ENSG0000000000{i:02d}" for i in range(8)],
                },
            )
            self.assertEqual(report.worst_level, Level.OK)

    def test_validate_detects_ensembl_var_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adata = _make_valid_adata()
            adata.var_names = adata.var["ensembl_id"].astype(str)
            path = Path(tmp) / "bad.h5ad"
            adata.write_h5ad(path)
            report = validate_h5ad_file(path)
            self.assertEqual(report.worst_level, Level.ERROR)

    def test_validate_detects_missing_gene_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adata = _make_valid_adata()
            del adata.var["gene_name"]
            del adata.var["ensembl_id"]
            path = Path(tmp) / "missing_metadata.h5ad"
            adata.write_h5ad(path)
            report = validate_h5ad_file(path)
            messages = "\n".join(f.message for f in report.findings)
            self.assertEqual(report.worst_level, Level.ERROR)
            self.assertIn("Missing adata.var['gene_name']", messages)
            self.assertIn("Missing Ensembl ID column", messages)

    def test_validate_path_detects_directory_cross_file_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_valid_adata().write_h5ad(root / "reference.h5ad")
            _make_valid_adata(n_vars=6).write_h5ad(root / "other.h5ad")
            reports, exit_code = validate_path(root)
            cross = [report for report in reports if report.path.name == "[cross-file]"]
            self.assertEqual(exit_code, 1)
            self.assertEqual(len(cross), 1)
            self.assertEqual(cross[0].worst_level, Level.ERROR)

    def test_geneformer_pickle_vocab_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vocabs = Path(tmp)
            geneformer_dir = vocabs / "geneformer_vocab"
            geneformer_dir.mkdir()
            token_path = geneformer_dir / "token_dictionary_gc30M.pkl"
            with token_path.open("wb") as handle:
                pickle.dump({"<pad>": 0, "<mask>": 1, "ENSG00000000003": 2}, handle)

            loaded = load_bundled_model_gene_lists(models=["geneformer"], vocabs_dir=vocabs)
            self.assertEqual(loaded, {"geneformer": ["ENSG00000000003"]})

    def test_vocab_overlap_error_uses_ensembl_for_geneformer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.h5ad"
            _make_valid_adata().write_h5ad(path)
            report = validate_h5ad_file(
                path,
                model_gene_lists={"geneformer": ["ENSG999999999999"]},
                min_ensembl_overlap=0.5,
            )
            self.assertEqual(report.worst_level, Level.ERROR)
            self.assertTrue(
                any("geneformer" in f.message and "dataset Ensembl IDs" in f.message for f in report.findings)
            )

    def test_zero_count_cells_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero_cell.h5ad"
            adata = _make_valid_adata()
            adata.X[0, :] = 0
            adata.write_h5ad(path)
            report = validate_h5ad_file(path)
            self.assertEqual(report.worst_level, Level.ERROR)
            self.assertTrue(any("zero total counts" in f.message for f in report.findings))

    def test_bundled_transcriptformer_vocab_is_ensembl(self) -> None:
        loaded = load_bundled_model_gene_lists(models=["transcriptformer"])
        genes = loaded["transcriptformer"]
        self.assertGreater(len(genes), 20_000)
        self.assertTrue(all(gene.startswith("ENSG") for gene in genes))

    def test_transcriptformer_assay_checks(self) -> None:
        adata = _make_valid_adata(n_obs=3)
        findings = check_transcriptformer_assay(adata, assay_vocab={"10x 3' v3"})
        self.assertEqual(len(findings), 1)
        self.assertIn("missing", findings[0].message)

        adata.obs["assay"] = ["10x 3' v3", "10x 3' v3", "10X_3prime"]
        findings = check_transcriptformer_assay(adata, assay_vocab={"10x 3' v3"})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].level, Level.WARN)
        self.assertIn("10X_3prime", findings[0].message)

        adata.obs["assay"] = ["10x 3' v3"] * 3
        self.assertEqual(check_transcriptformer_assay(adata, assay_vocab={"10x 3' v3"}), [])

    def test_transcriptformer_overlap_and_assay_warning_in_file_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.h5ad"
            _make_valid_adata().write_h5ad(path)
            report = validate_h5ad_file(
                path,
                model_gene_lists={"transcriptformer": [f"ENSG0000000000{i:02d}" for i in range(8)]},
            )
            messages = [f.message for f in report.findings]
            self.assertEqual(report.worst_level, Level.WARN)
            self.assertTrue(any("transcriptformer" in m and "Ensembl IDs" in m for m in messages))
            self.assertTrue(any("obs['assay'] missing" in m for m in messages))


if __name__ == "__main__":
    unittest.main()

