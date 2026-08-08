"""End-to-end tests for the local FastAPI web UI (scTenifoldXct.webapp)."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

import anndata
from fastapi.testclient import TestClient

from scTenifoldXct.webapp.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(grn_dir=str(tmp_path / "GRNs"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def small_adata():
    """A tiny, already log-normalised AnnData with two cell types.

    Includes a real ligand/receptor pair (BDNF -> TRPC1, from database/LR.csv)
    so enrichment tests exercise a non-empty candidate pool end-to-end.
    """
    rng = np.random.default_rng(0)
    genes = ["BDNF", "TRPC1", "GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"]
    n_cells_a, n_cells_b = 10, 8
    n_genes = len(genes)
    X_a = rng.lognormal(size=(n_cells_a, n_genes)).astype(np.float32)
    X_b = rng.lognormal(size=(n_cells_b, n_genes)).astype(np.float32)
    X = np.vstack([X_a, X_b])
    obs = {"ident": ["cell_A"] * n_cells_a + ["cell_B"] * n_cells_b}
    adata = anndata.AnnData(X=X, obs=obs, var={"gene": genes})
    adata.var_names = genes
    return adata


@pytest.fixture
def small_h5ad_path(small_adata, tmp_path):
    path = tmp_path / "small.h5ad"
    small_adata.write_h5ad(path)
    return path


def _upload(client, path, already_normalized=True):
    with open(path, "rb") as fh:
        resp = client.post(
            "/api/datasets",
            files={"file": ("small.h5ad", fh, "application/octet-stream")},
            data={"already_normalized": "true" if already_normalized else "false"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _run_job_to_completion(client, dataset, **overrides):
    payload = {
        "dataset_id": dataset["dataset_id"],
        "source_celltype": "cell_A",
        "target_celltype": "cell_B",
        "obs_label": "ident",
        "n_steps": 2,
        "seed": 0,
        **overrides,
    }
    resp = client.post("/api/jobs", json=payload)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + 60
    status = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert status is not None and status["status"] == "done", status
    return job_id


def test_upload_dataset_returns_shape_and_labels(client, small_h5ad_path, small_adata):
    info = _upload(client, small_h5ad_path)
    assert info["n_genes"] == small_adata.n_vars
    assert info["n_cells"] == small_adata.n_obs
    assert info["obs_labels"]["ident"] == ["cell_A", "cell_B"]
    assert info["prebuilt_grn"] is False


def test_upload_rejects_non_h5ad(client, tmp_path):
    bogus = tmp_path / "not_h5ad.txt"
    bogus.write_text("nope")
    with open(bogus, "rb") as fh:
        resp = client.post("/api/datasets", files={"file": ("not_h5ad.txt", fh, "text/plain")})
    assert resp.status_code == 400


def test_run_job_end_to_end(client, small_h5ad_path):
    dataset = _upload(client, small_h5ad_path)

    # pval just under 1: accept every candidate pair regardless of where the
    # untrained, 2-step network happens to place BDNF/TRPC1 — the point is to
    # exercise the response schema end-to-end, not the statistics of a
    # barely-trained model.
    job_id = _run_job_to_completion(client, dataset, pval=0.999999)

    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["source_celltype"] == "cell_A"
    assert result["target_celltype"] == "cell_B"
    assert result["test_method"] == "null"
    assert len(result["rows"]) >= 1
    assert any(row["ligand"] == "BDNF" and row["receptor"] == "TRPC1" for row in result["rows"])

    csv_resp = client.get(f"/api/jobs/{job_id}/result.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert csv_resp.text.splitlines()[0].startswith("pair,")


def test_run_job_chi2(client, small_h5ad_path):
    dataset = _upload(client, small_h5ad_path)
    job_id = _run_job_to_completion(client, dataset, test_method="chi2")
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["test_method"] == "chi2"


def test_same_source_and_target_celltype_rejected(client, small_h5ad_path):
    dataset = _upload(client, small_h5ad_path)
    resp = client.post(
        "/api/jobs",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_celltype": "cell_A",
            "target_celltype": "cell_A",
        },
    )
    assert resp.status_code == 422


def test_invalid_celltype_rejected(client, small_h5ad_path):
    dataset = _upload(client, small_h5ad_path)
    resp = client.post(
        "/api/jobs",
        json={
            "dataset_id": dataset["dataset_id"],
            "source_celltype": "cell_A",
            "target_celltype": "NOT_A_REAL_CELLTYPE",
        },
    )
    assert resp.status_code == 400


def test_unknown_dataset_id_rejected(client):
    resp = client.post(
        "/api/jobs",
        json={"dataset_id": "does-not-exist", "source_celltype": "cell_A", "target_celltype": "cell_B"},
    )
    assert resp.status_code == 404


def test_unknown_job_id_returns_404(client):
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    resp = client.get("/api/jobs/does-not-exist/result")
    assert resp.status_code == 404


def test_example_dataset_when_available(client):
    """Only meaningful when data/adata_short_example.h5ad is present (git checkout)."""
    resp = client.get("/api/datasets/example")
    if resp.status_code == 404:
        pytest.skip("bundled example dataset not present")
    assert resp.status_code == 200
    info = resp.json()
    assert info["n_genes"] > 0
    assert info["n_cells"] > 0
