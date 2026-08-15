"""FastAPI app for the scTenifoldXct local web UI.

Route layout:
    GET  /api/datasets/example       load the bundled example dataset
    POST /api/datasets               upload a .h5ad dataset
    POST /api/jobs                   start a cell-cell interaction run
    GET  /api/jobs/{id}              poll job status/stage
    GET  /api/jobs/{id}/result       ranked ligand-receptor pairs as JSON
    GET  /api/jobs/{id}/result.csv   ranked ligand-receptor pairs as a CSV download
    GET  /                           static single-page app
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

import scanpy as sc
from anndata import AnnData
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from scTenifoldXct.dataLoader import build_adata

from .jobs import DatasetNotFoundError, JobManager, JobNotFoundError
from .schemas import (
    DatasetInfo,
    JobCreate,
    JobCreated,
    JobResult,
    JobStatus,
    PairRow,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_LABEL_VALUES = 50  # obs columns with more distinct values aren't useful cell-type labels


def _find_example_dataset() -> Path | None:
    """Locate the bundled example .h5ad, if available.

    Example/tutorial data is not shipped inside the pip wheel (only
    scTenifoldXct/database/*.csv is packaged), so this only resolves when
    running out of a git checkout — set SCTENIFOLDXCT_EXAMPLE_DATA to
    override, otherwise upload a dataset instead.
    """
    env_path = os.environ.get("SCTENIFOLDXCT_EXAMPLE_DATA")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    # repo checkout, e.g. `pip install -e .` or running from a clone
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "adata_short_example.h5ad")
    candidates.append(Path.cwd() / "data" / "adata_short_example.h5ad")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _find_example_grn_dir() -> Path | None:
    """Locate the precomputed GRN cache for the bundled example, if available."""
    env_path = os.environ.get("SCTENIFOLDXCT_EXAMPLE_GRN_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[2] / "tutorials" / "Net_example")
    candidates.append(Path.cwd() / "tutorials" / "Net_example")
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _dataset_info(dataset_id: str, name: str, adata: AnnData, prebuilt_grn: bool) -> DatasetInfo:
    obs_labels: dict[str, list[str]] = {}
    for col in adata.obs.columns:
        try:
            values = adata.obs[col].unique()
        except TypeError:
            continue
        if 1 < len(values) <= MAX_LABEL_VALUES:
            obs_labels[col] = sorted(str(v) for v in values)
    return DatasetInfo(
        dataset_id=dataset_id,
        name=name,
        n_genes=adata.n_vars,
        n_cells=adata.n_obs,
        obs_labels=obs_labels,
        prebuilt_grn=prebuilt_grn,
    )


def create_app(grn_dir: str = "GRNs") -> FastAPI:
    app = FastAPI(title="scTenifoldXct", description="Local UI for scTenifoldXct cell-cell interaction analysis")
    manager = JobManager(grn_dir=grn_dir)

    # -- datasets -----------------------------------------------------
    @app.get("/api/datasets/example", response_model=DatasetInfo)
    def load_example_dataset():
        path = _find_example_dataset()
        if path is None:
            raise HTTPException(
                404,
                "bundled example dataset not found (only available from a git "
                "checkout) — upload your own .h5ad instead",
            )
        adata = sc.read_h5ad(path)  # already log-normalised, so loaded as-is (no build_adata)
        grn_dir = _find_example_grn_dir()
        dataset_id = manager.add_dataset(adata, name=path.name, prebuilt_grn_dir=grn_dir)
        return _dataset_info(dataset_id, path.name, adata, prebuilt_grn=grn_dir is not None)

    @app.post("/api/datasets", response_model=DatasetInfo)
    async def upload_dataset(file: UploadFile, already_normalized: bool = Form(False)):
        if not file.filename.endswith(".h5ad"):
            raise HTTPException(400, "only .h5ad files are supported")

        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        try:
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large (limit 2 GB)")
                tmp.write(chunk)
            tmp.close()
            try:
                # already_normalized: load as-is (user vouches for log-normalised .X).
                # otherwise: run the same raw-counts -> normalize -> log1p pipeline as the CLI.
                adata = sc.read_h5ad(tmp.name) if already_normalized else build_adata(counts_path=tmp.name)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"could not read .h5ad file: {exc}") from exc
        finally:
            tmp.close()
            os.unlink(tmp.name)

        dataset_id = manager.add_dataset(adata, name=file.filename)
        return _dataset_info(dataset_id, file.filename, adata, prebuilt_grn=False)

    # -- jobs -----------------------------------------------------------
    @app.post("/api/jobs", response_model=JobCreated)
    def create_job(params: JobCreate):
        try:
            entry = manager.get_dataset(params.dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(404, f"unknown dataset_id {params.dataset_id!r}") from exc

        adata = entry.adata
        if params.obs_label not in adata.obs.columns:
            raise HTTPException(400, f"unknown obs_label column {params.obs_label!r}")
        labels = set(adata.obs[params.obs_label].astype(str))
        missing = [c for c in (params.source_celltype, params.target_celltype) if c not in labels]
        if missing:
            raise HTTPException(400, f"cell type(s) not found in {params.obs_label!r}: {missing}")

        job_id = manager.submit(params)
        return JobCreated(job_id=job_id)

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def get_job_status(job_id: str):
        try:
            job = manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
        return JobStatus(**job.to_status_dict())

    @app.get("/api/jobs/{job_id}/result", response_model=JobResult)
    def get_job_result(job_id: str):
        job = _require_finished_job(manager, job_id)
        df = job.result
        rows = [
            PairRow(
                pair=str(pair),
                ligand=str(row["ligand"]),
                receptor=str(row["receptor"]),
                dist=float(row["dist"]),
                correspondence=float(row["correspondence"]) if "correspondence" in row else None,
                p_val=float(row["p_val"]) if "p_val" in row else None,
                q_val=float(row["q_val"]) if "q_val" in row else None,
                FC=float(row["FC"]) if "FC" in row else None,
                enriched_rank=int(row["enriched_rank"]) if "enriched_rank" in row else None,
            )
            for pair, row in df.iterrows()
        ]
        return JobResult(
            job_id=job_id,
            source_celltype=job.params.source_celltype,
            target_celltype=job.params.target_celltype,
            test_method=job.params.test_method,
            rows=rows,
        )

    @app.get("/api/jobs/{job_id}/result.csv")
    def get_job_result_csv(job_id: str):
        job = _require_finished_job(manager, job_id)
        buf = io.StringIO()
        job.result.to_csv(buf, index_label="pair")
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="xct_{job_id}.csv"'}
        return StreamingResponse(buf, media_type="text/csv", headers=headers)

    # -- static frontend --------------------------------------------
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def _require_finished_job(manager: JobManager, job_id: str):
    try:
        job = manager.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"unknown job_id {job_id!r}") from exc
    if job.status == "error":
        raise HTTPException(500, f"job failed: {job.error}")
    if job.status != "done":
        raise HTTPException(409, f"job not finished yet (status={job.status}, stage={job.stage})")
    return job


app = create_app()
