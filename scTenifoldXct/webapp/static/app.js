"use strict";

const state = {
  dataset: null, // {dataset_id, name, n_genes, n_cells, obs_labels, prebuilt_grn}
  resultRows: null,
  testMethod: "null",
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      /* ignore non-JSON error bodies */
    }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

function showError(el, message) {
  el.textContent = message;
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

// -- dataset ------------------------------------------------------------

$("use-example").addEventListener("click", async () => {
  hide($("dataset-error"));
  $("use-example").disabled = true;
  try {
    const info = await api("/api/datasets/example");
    setDataset(info);
  } catch (err) {
    showError($("dataset-error"), err.message);
  } finally {
    $("use-example").disabled = false;
  }
});

$("file-input").addEventListener("change", async (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  hide($("dataset-error"));
  const form = new FormData();
  form.append("file", file);
  form.append("already_normalized", $("already-normalized").checked ? "true" : "false");
  try {
    const info = await api("/api/datasets", { method: "POST", body: form });
    setDataset(info);
  } catch (err) {
    showError($("dataset-error"), err.message);
  } finally {
    evt.target.value = "";
  }
});

function setDataset(info) {
  state.dataset = info;

  const obsCols = Object.keys(info.obs_labels || {});
  const obsSelect = $("obs-label-select");
  obsSelect.innerHTML = "";
  for (const col of obsCols) {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    obsSelect.appendChild(opt);
  }
  if (obsCols.includes("ident")) obsSelect.value = "ident";
  populateCelltypeOptions();

  $("rebuild-grn").checked = !info.prebuilt_grn;
  $("rebuild-hint").classList.toggle("hidden", !info.prebuilt_grn);

  const infoEl = $("dataset-info");
  infoEl.textContent =
    `Loaded "${info.name}" — ${info.n_genes.toLocaleString()} genes × ${info.n_cells.toLocaleString()} cells.`;
  infoEl.classList.remove("hidden");

  $("run-section").classList.remove("hidden");
  $("results-section").classList.add("hidden");
}

$("obs-label-select").addEventListener("change", populateCelltypeOptions);

function populateCelltypeOptions() {
  const col = $("obs-label-select").value;
  const values = (state.dataset.obs_labels || {})[col] || [];
  for (const id of ["source-celltype-select", "target-celltype-select"]) {
    const select = $(id);
    select.innerHTML = "";
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    }
  }
  if (values.length > 1) $("target-celltype-select").value = values[1];
}

// -- enrichment test toggle -------------------------------------------

$("test-method").addEventListener("change", (evt) => {
  state.testMethod = evt.target.value;
  const isChi2 = state.testMethod === "chi2";
  $("dof-field").hidden = !isChi2;
  $("fdr-row").hidden = !isChi2;
  $("filter-zeros-row").hidden = isChi2;
});

// -- run + poll -----------------------------------------------------

$("run-form").addEventListener("submit", async (evt) => {
  evt.preventDefault();
  hide($("dataset-error"));

  const payload = {
    dataset_id: state.dataset.dataset_id,
    source_celltype: $("source-celltype-select").value,
    target_celltype: $("target-celltype-select").value,
    obs_label: $("obs-label-select").value,
    query_db: $("query-db").value || null,
    alpha: Number($("alpha").value),
    mu: Number($("mu").value),
    scale_w: $("scale-w").checked,
    n_dim: Number($("n-dim").value),
    n_steps: Number($("n-steps").value),
    lr: Number($("lr").value),
    dist_metric: $("dist-metric").value,
    test_method: $("test-method").value,
    pval: Number($("pval").value),
    filter_zeros: $("filter-zeros").checked,
    dof: Number($("dof").value),
    fdr: $("fdr").checked,
    rebuild_grn: $("rebuild-grn").checked,
    n_cpus: Math.trunc(Number($("n-cpus").value)),
    seed: $("seed").value === "" ? null : Number($("seed").value),
  };

  if (payload.source_celltype === payload.target_celltype) {
    showError($("dataset-error"), "sender and receiver cell types must differ");
    return;
  }

  $("run-button").disabled = true;
  $("results-section").classList.remove("hidden");
  hide($("results-error"));
  $("results-table-wrap").innerHTML = "";
  hide($("download-csv"));
  $("status-bar").textContent = "submitting…";
  setProgress(0);

  try {
    const { job_id } = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await pollJob(job_id);
  } catch (err) {
    $("status-bar").textContent = "";
    setProgress(null, "error");
    showError($("results-error"), err.message);
  } finally {
    $("run-button").disabled = false;
  }
});

// Rough progress reflects the job's current stage (see
// scTenifoldXct/webapp/jobs.py STAGE_* constants) — training sub-progress
// isn't tracked per-step, so each stage jumps straight to its target percentage.
const STAGE_PROGRESS = {
  queued: 3,
  "building gene regulatory networks": 15,
  "training manifold alignment": 55,
  "computing enrichment": 85,
  done: 100,
};

function setProgress(percent, mode) {
  const wrap = $("progress-wrap");
  const fill = $("progress-fill");
  wrap.classList.remove("hidden");
  fill.classList.toggle("done", mode === "done");
  fill.classList.toggle("error", mode === "error");
  if (percent != null) fill.style.width = `${percent}%`;
}

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let status;
      try {
        status = await api(`/api/jobs/${jobId}`);
      } catch (err) {
        reject(err);
        return;
      }
      if (status.status === "error") {
        setProgress(null, "error");
        reject(new Error(status.error || "job failed"));
        return;
      }
      $("status-bar").textContent = `${status.stage}…`;
      setProgress(STAGE_PROGRESS[status.stage] ?? 5);
      if (status.status === "done") {
        try {
          await loadResult(jobId);
          resolve();
        } catch (err) {
          reject(err);
        }
        return;
      }
      setTimeout(tick, 1000);
    };
    tick();
  });
}

const TOP_N = 15;

async function loadResult(jobId) {
  const result = await api(`/api/jobs/${jobId}/result`);
  $("status-bar").textContent = `done — ${result.rows.length} pairs enriched`;
  setProgress(100, "done");
  state.resultRows = result.rows;
  state.testMethodUsed = result.test_method;
  renderResultsTable();

  const link = $("download-csv");
  link.href = `/api/jobs/${jobId}/result.csv`;
  link.textContent = `Download full CSV (${result.rows.length} pairs)`;
  link.classList.remove("hidden");
}

// Shows only the top TOP_N pairs by enriched rank; the full ranked list is
// always available via the CSV download.
function renderResultsTable() {
  const rows = [...state.resultRows]
    .sort((a, b) => (a.enriched_rank ?? Infinity) - (b.enriched_rank ?? Infinity))
    .slice(0, TOP_N);

  if (rows.length === 0) {
    const wrap = $("results-table-wrap");
    wrap.innerHTML = "";
    const p = document.createElement("p");
    p.className = "table-caption";
    p.textContent = "No pairs passed the enrichment cutoff.";
    wrap.appendChild(p);
    return;
  }

  const isChi2 = state.testMethodUsed === "chi2";
  const columns = [
    { key: "enriched_rank", label: "Rank" },
    { key: "ligand", label: "Ligand" },
    { key: "receptor", label: "Receptor" },
    { key: "dist", label: "Distance", fmt: (v) => v.toFixed(4) },
  ];
  if (isChi2) {
    columns.push({ key: "FC", label: "FC", fmt: (v) => (v == null ? "–" : v.toFixed(3)) });
    columns.push({ key: "q_val", label: "q-value", fmt: (v) => (v == null ? "–" : v.toExponential(2)) });
  } else {
    columns.push({ key: "p_val", label: "p-value", fmt: (v) => (v == null ? "–" : v.toExponential(2)) });
  }

  const caption = document.createElement("p");
  caption.className = "table-caption";
  caption.textContent = `Top ${rows.length} of ${state.resultRows.length} enriched pairs:`;

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col.label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const col of columns) {
      const td = document.createElement("td");
      const value = row[col.key];
      td.textContent = col.fmt ? col.fmt(value) : value ?? "–";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  const wrap = $("results-table-wrap");
  wrap.innerHTML = "";
  wrap.appendChild(caption);
  wrap.appendChild(table);
}
