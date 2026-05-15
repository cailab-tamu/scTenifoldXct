"""
Stage 0 — Characterization / regression tests.

These run the full pipeline on committed example data with pre-built GRNs so no
network construction is needed (fast).  Any refactoring that accidentally changes
scientific results will break these tests.

Run manually (or in a dedicated CI job) before merging changes that touch core,
nn, stat, merge, or pcNet:

    pytest tests/test_regression.py -v
"""
import pathlib
import pytest
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
NETS_N = ROOT / "tutorials" / "Net_B2Fib_N"
NETS_T = ROOT / "tutorials" / "Net_B2Fib_T"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adata_merge():
    import anndata
    return anndata.read_h5ad(DATA / "adata_merge_example.h5ad")


@pytest.fixture(scope="module")
def xct_normal(adata_merge):
    from scTenifoldXct.core import scTenifoldXct
    ada = adata_merge[adata_merge.obs["NormalvsTumor"] == "N", :].copy()
    return scTenifoldXct(
        data=ada,
        source_celltype="B cells",
        target_celltype="Fibroblasts",
        obs_label="ident",
        rebuild_GRN=False,
        GRN_file_dir=str(NETS_N),
        verbose=False,
        n_cpus=1,
    )


@pytest.fixture(scope="module")
def xct_tumor(adata_merge):
    from scTenifoldXct.core import scTenifoldXct
    ada = adata_merge[adata_merge.obs["NormalvsTumor"] == "T", :].copy()
    return scTenifoldXct(
        data=ada,
        source_celltype="B cells",
        target_celltype="Fibroblasts",
        obs_label="ident",
        rebuild_GRN=False,
        GRN_file_dir=str(NETS_T),
        verbose=False,
        n_cpus=1,
    )


@pytest.fixture(scope="module")
def single_result(xct_normal):
    xct_normal.get_embeds(train=True, n_steps=500)
    return xct_normal.null_test()


@pytest.fixture(scope="module")
def merge_result(xct_normal, xct_tumor):
    from scTenifoldXct.merge import merge_scTenifoldXct
    merged = merge_scTenifoldXct(xct_normal, xct_tumor, verbose=False)
    emb = merged.get_embeds(train=True, n_steps=500, verbose=False)
    merged.nn_aligned_diff(emb)
    return merged.chi2_diff_test()


# ---------------------------------------------------------------------------
# Single-sample regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSingleSampleRegression:
    EXPECTED_COLS = {"ligand", "receptor", "dist", "correspondence", "p_val", "enriched_rank"}

    def test_returns_dataframe(self, single_result):
        assert isinstance(single_result, pd.DataFrame)

    def test_expected_columns(self, single_result):
        assert self.EXPECTED_COLS.issubset(single_result.columns)

    def test_nonempty(self, single_result):
        assert len(single_result) > 0, "null_test returned no enriched pairs"

    def test_distances_positive(self, single_result):
        assert (single_result["dist"] > 0).all()

    def test_pvals_within_threshold(self, single_result):
        assert (single_result["p_val"] >= 0).all()
        assert (single_result["p_val"] <= 0.05).all()

    def test_enriched_rank_sequential(self, single_result):
        assert list(single_result["enriched_rank"]) == list(range(1, len(single_result) + 1))


# ---------------------------------------------------------------------------
# Merge / differential regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMergeRegression:
    EXPECTED_COLS = {"ligand", "receptor", "FC", "p_val", "enriched_rank"}

    def test_returns_dataframe(self, merge_result):
        assert isinstance(merge_result, pd.DataFrame)

    def test_expected_columns(self, merge_result):
        assert self.EXPECTED_COLS.issubset(merge_result.columns)

    def test_nonempty(self, merge_result):
        assert len(merge_result) > 0, "chi2_diff_test returned no enriched pairs"

    def test_fc_positive(self, merge_result):
        assert (merge_result["FC"] >= 0).all()

    def test_pvals_within_threshold(self, merge_result):
        assert (merge_result["p_val"] >= 0).all()
        assert (merge_result["p_val"] <= 0.05).all()

    def test_enriched_rank_sequential(self, merge_result):
        assert list(merge_result["enriched_rank"]) == list(range(1, len(merge_result) + 1))
