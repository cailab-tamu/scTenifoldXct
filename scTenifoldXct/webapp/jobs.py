"""In-process job manager for the scTenifoldXct web UI.

Network reconstruction, manifold-alignment training, and the enrichment test
are synchronous and CPU/GPU-bound (see scTenifoldXct/core.py), so each run is
executed on a single-worker background thread pool and tracked by job id.
This is a local, single-user tool: jobs are intentionally serialized and
everything lives in memory — no persistence across process restarts.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from anndata import AnnData

from scTenifoldXct.core import scTenifoldXct

from .schemas import JobCreate

logger = logging.getLogger(__name__)

# Stages surfaced to the UI as a simple progress stepper.
STAGE_QUEUED = "queued"
STAGE_BUILDING_GRN = "building gene regulatory networks"
STAGE_TRAINING = "training manifold alignment"
STAGE_ENRICHMENT = "computing enrichment"
STAGE_DONE = "done"


@dataclass
class DatasetEntry:
    adata: AnnData
    name: str
    grn_dir: str
    prebuilt_grn: bool


@dataclass
class Job:
    id: str
    dataset_id: str
    params: JobCreate
    status: str = "queued"  # queued | running | done | error
    stage: str = STAGE_QUEUED
    error: str | None = None
    result: pd.DataFrame | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.stage = stage

    def to_status_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
            }


class DatasetNotFoundError(KeyError):
    pass


class JobNotFoundError(KeyError):
    pass


class JobManager:
    """Holds loaded datasets and runs scTenifoldXct jobs on a single worker thread."""

    def __init__(self, grn_dir: str = "GRNs"):
        self._datasets: dict[str, DatasetEntry] = {}
        self._jobs: dict[str, Job] = {}
        self._active_jobs: dict[str, int] = {}  # dataset_id -> in-flight (queued/running) job count
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xct-job")
        self._grn_root = grn_dir

    # -- datasets ---------------------------------------------------
    def add_dataset(self, adata: AnnData, name: str, prebuilt_grn_dir: str | Path | None = None) -> str:
        self._evict_idle_datasets()
        dataset_id = uuid.uuid4().hex[:12]
        if prebuilt_grn_dir is not None:
            grn_dir, prebuilt = str(prebuilt_grn_dir), True
        else:
            # per-dataset cache: a rebuild here is reused by later runs of the same
            # upload (e.g. re-running with a different sender/receiver pair).
            grn_dir, prebuilt = str(Path(self._grn_root) / dataset_id), False
        self._datasets[dataset_id] = DatasetEntry(adata=adata, name=name, grn_dir=grn_dir, prebuilt_grn=prebuilt)
        return dataset_id

    def get_dataset(self, dataset_id: str) -> DatasetEntry:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(dataset_id) from exc

    def _evict_idle_datasets(self) -> None:
        """Drop previously loaded datasets that have no in-flight job.

        The UI only ever shows one active dataset at a time — a fresh upload
        replaces it — so anything left over from before would otherwise leak
        forever: the AnnData stays in memory and its GRN cache stays on disk
        with nothing left able to reference it. A dataset with a still
        queued/running job is left alone (it stays referenced by that job
        regardless of this dict) and gets swept on the next upload once it's
        idle.
        """
        for dataset_id in list(self._datasets):
            if self._active_jobs.get(dataset_id, 0) > 0:
                continue
            entry = self._datasets.pop(dataset_id)
            self._active_jobs.pop(dataset_id, None)
            if not entry.prebuilt_grn:
                shutil.rmtree(entry.grn_dir, ignore_errors=True)

    # -- jobs ---------------------------------------------------------
    def submit(self, params: JobCreate) -> str:
        entry = self.get_dataset(params.dataset_id)  # fail fast if unknown dataset
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dataset_id=params.dataset_id, params=params)
        self._jobs[job_id] = job
        self._active_jobs[params.dataset_id] = self._active_jobs.get(params.dataset_id, 0) + 1
        self._executor.submit(self._run, job, entry)
        return job_id

    def get_job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _run(self, job: Job, entry: DatasetEntry) -> None:
        job.status = "running"
        params = job.params
        try:
            if params.seed is not None:
                from scTenifoldXct.nn import set_seed
                set_seed(params.seed)

            job.set_stage(STAGE_BUILDING_GRN)
            xct = scTenifoldXct(
                data=entry.adata,
                source_celltype=params.source_celltype,
                target_celltype=params.target_celltype,
                obs_label=params.obs_label,
                GRN_file_dir=entry.grn_dir,
                rebuild_GRN=params.rebuild_grn,
                query_DB=params.query_db,
                alpha=params.alpha,
                mu=params.mu,
                scale_w=params.scale_w,
                n_dim=params.n_dim,
                verbose=False,
                n_cpus=params.n_cpus,
            )

            job.set_stage(STAGE_TRAINING)
            xct.get_embeds(train=True, n_steps=params.n_steps, lr=params.lr, dist_metric=params.dist_metric)

            job.set_stage(STAGE_ENRICHMENT)
            if params.test_method == "chi2":
                df = xct.chi2_test(dof=params.dof, pval=params.pval, cal_FDR=params.fdr)
            else:
                df = xct.null_test(filter_zeros=params.filter_zeros, pval=params.pval)

            job.result = df
            job.set_stage(STAGE_DONE)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            logger.exception("job %s failed", job.id)
            job.error = str(exc)
            job.status = "error"
        finally:
            # marks entry.dataset_id idle again so a later upload can evict it
            self._active_jobs[job.dataset_id] -= 1
