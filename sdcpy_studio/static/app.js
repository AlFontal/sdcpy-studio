const statusText = document.getElementById("status_text");
const statusError = document.getElementById("status_error");
const statusProgressWrap = document.getElementById("status_progress_wrap");
const statusProgressBar = document.getElementById("status_progress_bar");
const statusProgressMeta = document.getElementById("status_progress_meta");

const summaryStats = document.getElementById("summary_stats");
const summaryNotes = document.getElementById("summary_notes");
const linksTableBody = document.querySelector("#links_table tbody");
const analysisSettingsDetails = document.getElementById("analysis_settings_details");

const explorerAlphaInput = document.getElementById("explorer_alpha");

const datasetFileInput = document.getElementById("dataset_file");
const datasetDateSelect = document.getElementById("dataset_date_col");
const datasetTs1Select = document.getElementById("dataset_ts1_col");
const datasetTs2Select = document.getElementById("dataset_ts2_col");
const submitDatasetButton = document.getElementById("submit_dataset");
const datasetMeta = document.getElementById("dataset_meta");
const datasetPreviewHead = document.querySelector("#dataset_preview_table thead");
const datasetPreviewBody = document.querySelector("#dataset_preview_table tbody");

const downloadXlsxButton = document.getElementById("download_xlsx");
const downloadPngButton = document.getElementById("download_png");
const downloadSvgButton = document.getElementById("download_svg");

let activePoll = null;
let latestResult = null;
let latestJobId = null;
let latestDatasetId = null;
let alphaRenderTimer = null;
let datasetInspectToken = 0;

function getConfig() {
  return {
    fragment_size: Number(document.getElementById("fragment_size").value),
    n_permutations: Number(document.getElementById("n_permutations").value),
    method: document.getElementById("method").value,
    alpha: Number(document.getElementById("alpha").value),
    min_lag: Number(document.getElementById("min_lag").value),
    max_lag: Number(document.getElementById("max_lag").value),
    two_tailed: document.getElementById("two_tailed").checked,
    permutations: document.getElementById("permutations").checked,
    max_memory_gb: 1.0,
  };
}

function getExplorerAlpha(defaultAlpha = 0.05) {
  const value = Number(explorerAlphaInput.value);
  if (Number.isFinite(value) && value > 0 && value < 1) {
    return value;
  }
  return defaultAlpha;
}

function parseSeries(rawText) {
  return rawText
    .split(/[\n,;\s]+/)
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v));
}

function setStatus(text, isError = false) {
  statusText.textContent = text;
  if (!isError) {
    statusError.textContent = "";
  }
}

function setDownloadButtons(enabled) {
  downloadXlsxButton.disabled = !enabled;
  downloadPngButton.disabled = !enabled;
  downloadSvgButton.disabled = !enabled;
}

function updateProgress(progress, status) {
  if (!progress || !Number.isFinite(progress.total) || progress.total <= 0) {
    statusProgressWrap.classList.add("hidden");
    statusProgressMeta.textContent = "";
    return;
  }

  const pct = Math.max(0, Math.min(100, Math.round((progress.current / progress.total) * 100)));
  statusProgressWrap.classList.remove("hidden");
  statusProgressBar.style.width = `${pct}%`;
  statusProgressMeta.textContent = `${progress.description} (${progress.current}/${progress.total})`;

  if (status === "succeeded" || status === "failed") {
    statusProgressBar.style.width = "100%";
  }
}

function renderSummary(summary, notes, runtimeSeconds) {
  const keys = [
    ["series_length", "Series length"],
    ["fragment_size", "Fragment size"],
    ["n_pairs", "Compared pairs"],
    ["n_significant", "Significant pairs"],
    ["significant_rate", "Significant rate"],
    ["r_min", "Min r"],
    ["r_max", "Max r"],
    ["lag_min", "Min lag"],
    ["lag_max", "Max lag"],
    ["method", "Method"],
  ];

  const items = keys
    .filter(([key]) => Object.prototype.hasOwnProperty.call(summary, key))
    .map(([key, label]) => {
      const value = summary[key];
      const pretty =
        typeof value === "number" ? value.toFixed(4).replace(/\.0000$/, "") : String(value);
      return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${pretty}</span></div>`;
    });

  items.push(
    `<div class="stat"><span class="stat-label">Runtime (s)</span><span class="stat-value">${runtimeSeconds.toFixed(2)}</span></div>`
  );

  summaryStats.innerHTML = items.join("");
  summaryNotes.innerHTML = (notes || []).map((note) => `<li>${note}</li>`).join("");
}

function renderLinks(rows) {
  linksTableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    ["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"].forEach((col) => {
      const td = document.createElement("td");
      const value = row[col];
      td.textContent =
        typeof value === "number" ? value.toFixed(4).replace(/\.0000$/, "") : String(value);
      tr.appendChild(td);
    });
    linksTableBody.appendChild(tr);
  });
}

function maskedSignificantMatrix(rMatrix, pMatrix, alpha) {
  return rMatrix.map((row, i) =>
    row.map((rv, j) => {
      const p = pMatrix?.[i]?.[j];
      if (rv === null || rv === undefined || p === null || p === undefined) {
        return null;
      }
      return p <= alpha ? rv : null;
    })
  );
}

function buildAxisTicks(labels, nTicks = 6) {
  if (!labels?.length) {
    return { tickvals: [], ticktext: [] };
  }
  if (labels.length <= nTicks) {
    const vals = labels.map((_, i) => i);
    return { tickvals: vals, ticktext: labels.map((v) => String(v)) };
  }

  const step = Math.max(1, Math.floor((labels.length - 1) / (nTicks - 1)));
  const vals = [];
  for (let i = 0; i < labels.length; i += step) {
    vals.push(i);
  }
  if (vals[vals.length - 1] !== labels.length - 1) {
    vals.push(labels.length - 1);
  }
  return {
    tickvals: vals,
    ticktext: vals.map((idx) => {
      const value = String(labels[idx]);
      return value.length > 12 ? value.slice(0, 12) : value;
    }),
  };
}

function normalizePosition(value, maxIndex) {
  const rounded = Math.round(Number(value));
  if (!Number.isFinite(rounded)) {
    return 0;
  }
  return Math.max(0, Math.min(maxIndex, rounded));
}

function renderTwoWayExplorer(result) {
  const containerId = "two_way_explorer";
  const container = document.getElementById(containerId);
  if (!window.Plotly) {
    if (container) {
      container.textContent = "Plotly library unavailable in this environment.";
    }
    return;
  }

  const series = result.series;
  const matrixR = result.matrix_r;
  const matrixP = result.matrix_p;
  const fragmentSize = Number(result.summary.fragment_size || 1);
  const alpha = getExplorerAlpha(result.summary.alpha || 0.05);

  if (!series?.index?.length || !matrixR?.x?.length || !matrixR?.y?.length || !matrixR?.z?.length) {
    Plotly.newPlot(
      containerId,
      [],
      {
        title: "2-way explorer unavailable",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  const labels = series.index.map((v) => String(v));
  const positions = labels.map((_, idx) => idx);
  const tickConfig = buildAxisTicks(labels, 6);
  const zSig = maskedSignificantMatrix(matrixR.z, matrixP.z, alpha);

  const xMin = Math.min(...matrixR.x);
  const xMax = Math.max(...matrixR.x);
  const yMin = Math.min(...matrixR.y);
  const yMax = Math.max(...matrixR.y);

  const xFragEnd = Math.min(xMax, xMin + fragmentSize - 1);
  const yFragEnd = Math.min(yMax, yMin + fragmentSize - 1);

  const bracketShapes = [
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0: xMin,
      y0: yMin,
      x1: xFragEnd,
      y1: yMin,
      line: { color: "#111827", width: 5 },
    },
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0: xMin,
      y0: yMin,
      x1: xMin,
      y1: yFragEnd,
      line: { color: "#111827", width: 5 },
    },
  ];

  const traces = [
    {
      type: "heatmap",
      x: matrixR.x,
      y: matrixR.y,
      z: zSig,
      colorscale: "RdBu",
      zmid: 0,
      xaxis: "x",
      yaxis: "y",
      hovertemplate:
        "start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<extra>Significant only</extra>",
      name: "significant",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: positions,
      y: series.ts1,
      line: { color: "#111827", width: 1.6 },
      xaxis: "x2",
      yaxis: "y2",
      customdata: labels,
      hovertemplate: "t=%{customdata}<br>TS1=%{y:.3f}<extra></extra>",
      name: "TS1",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: series.ts2,
      y: positions,
      line: { color: "#111827", width: 1.6 },
      xaxis: "x3",
      yaxis: "y3",
      customdata: labels,
      hovertemplate: "TS2=%{x:.3f}<br>t=%{customdata}<extra></extra>",
      name: "TS2",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "#dc2626", width: 3 },
      xaxis: "x2",
      yaxis: "y2",
      hoverinfo: "skip",
      showlegend: false,
      name: "TS1 segment",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "#2563eb", width: 3 },
      xaxis: "x3",
      yaxis: "y3",
      hoverinfo: "skip",
      showlegend: false,
      name: "TS2 segment",
    },
  ];

  const layout = {
    title: `2-way explorer (significant only, alpha=${alpha.toFixed(3)})`,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 60, r: 74, b: 52, l: 56 },
    showlegend: false,
    xaxis: {
      domain: [0.23, 0.82],
      title: "TS1 fragment start",
      range: [xMin - 0.5, xMax + 0.5],
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    yaxis: {
      domain: [0.12, 0.72],
      title: "TS2 fragment start",
      range: [yMin - 0.5, yMax + 0.5],
      showgrid: true,
      gridcolor: "#e5e7eb",
      scaleanchor: "x",
      scaleratio: 1,
      zeroline: false,
    },
    xaxis2: {
      domain: [0.23, 0.82],
      anchor: "y2",
      range: [-0.5, positions.length - 0.5],
      title: "TS1",
      tickmode: "array",
      tickvals: tickConfig.tickvals,
      ticktext: tickConfig.ticktext,
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    yaxis2: {
      domain: [0.76, 0.97],
      anchor: "x2",
      title: "",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    xaxis3: {
      domain: [0.02, 0.20],
      anchor: "y3",
      title: "",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    yaxis3: {
      domain: [0.12, 0.72],
      anchor: "x3",
      range: [-0.5, positions.length - 0.5],
      tickmode: "array",
      tickvals: tickConfig.tickvals,
      ticktext: tickConfig.ticktext,
      title: "TS2",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    annotations: [
      {
        xref: "x",
        yref: "y",
        x: xMin + Math.max(2, fragmentSize * 0.45),
        y: yMin + Math.max(1, fragmentSize * 0.05),
        text: `s = ${fragmentSize} periods`,
        showarrow: false,
        font: { family: "JetBrains Mono, monospace", size: 12, color: "#111827" },
      },
    ],
    shapes: bracketShapes,
    height: 760,
  };
  traces[0].colorbar = {
    title: "Pearson r",
    x: 0.86,
    y: 0.42,
    len: 0.60,
  };

  const config = { responsive: true, displaylogo: false };

  Plotly.newPlot(containerId, traces, layout, config).then((gd) => {
    if (typeof gd.removeAllListeners === "function") {
      gd.removeAllListeners("plotly_hover");
      gd.removeAllListeners("plotly_unhover");
    }

    const clearHighlight = () => {
      Plotly.restyle(gd, { x: [[]], y: [[]] }, [3]);
      Plotly.restyle(gd, { x: [[]], y: [[]] }, [4]);
      Plotly.relayout(gd, { shapes: bracketShapes });
    };

    gd.on("plotly_hover", (event) => {
      const point = event?.points?.[0];
      if (!point || point.curveNumber !== 0 || point.z === null || point.z === undefined) {
        return;
      }

      const start1 = Number(point.x);
      const start2 = Number(point.y);
      if (!Number.isFinite(start1) || !Number.isFinite(start2)) {
        return;
      }

      const nSeries = positions.length;
      const start1Idx = normalizePosition(start1, nSeries - 1);
      const start2Idx = normalizePosition(start2, nSeries - 1);
      const stop1 = Math.min(nSeries - 1, start1Idx + fragmentSize - 1);
      const stop2 = Math.min(nSeries - 1, start2Idx + fragmentSize - 1);

      const ts1SegX = positions.slice(start1Idx, stop1 + 1);
      const ts1SegY = series.ts1.slice(start1Idx, stop1 + 1);
      const ts2SegX = series.ts2.slice(start2Idx, stop2 + 1);
      const ts2SegY = positions.slice(start2Idx, stop2 + 1);

      Plotly.restyle(gd, { x: [ts1SegX], y: [ts1SegY] }, [3]);
      Plotly.restyle(gd, { x: [ts2SegX], y: [ts2SegY] }, [4]);

      const markerRect = {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: start1Idx - 0.5,
        x1: start1Idx + 0.5,
        y0: start2Idx - 0.5,
        y1: start2Idx + 0.5,
        line: { color: "#f97316", width: 1.5 },
        fillcolor: "rgba(249, 115, 22, 0.15)",
      };
      Plotly.relayout(gd, { shapes: [...bracketShapes, markerRect] });
    });

    gd.on("plotly_unhover", clearHighlight);
  });
}

function populateSelect(selectEl, values, includeBlank = false, blankLabel = "(none)") {
  selectEl.innerHTML = "";
  if (includeBlank) {
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = blankLabel;
    selectEl.appendChild(blank);
  }

  values.forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    selectEl.appendChild(opt);
  });
}

function renderDatasetPreview(columns, rows) {
  if (!columns?.length) {
    datasetPreviewHead.innerHTML = "";
    datasetPreviewBody.innerHTML = "";
    return;
  }

  datasetPreviewHead.innerHTML = `<tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr>`;
  datasetPreviewBody.innerHTML = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((col) => `<td>${row[col] ?? ""}</td>`)
          .join("")}</tr>`
    )
    .join("");
}

function updateDatasetRunAvailability() {
  const ready =
    !!latestDatasetId &&
    !!datasetTs1Select.value &&
    !!datasetTs2Select.value &&
    datasetTs1Select.value !== datasetTs2Select.value;
  submitDatasetButton.disabled = !ready;
}

async function inspectDataset() {
  const file = datasetFileInput.files[0];
  if (!file) {
    throw new Error("Please select a dataset CSV file first.");
  }
  const token = ++datasetInspectToken;
  latestDatasetId = null;
  updateDatasetRunAvailability();
  datasetMeta.textContent = `Inspecting ${file.name}...`;

  const formData = new FormData();
  formData.append("dataset_file", file);

  const response = await fetch("/api/v1/datasets/inspect", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    datasetMeta.textContent = "";
    throw new Error(data.detail || "Dataset inspection failed.");
  }
  if (token !== datasetInspectToken) {
    return;
  }

  latestDatasetId = data.dataset_id;
  datasetMeta.textContent = `${data.filename}: ${data.n_rows} rows x ${data.n_columns} columns. ` +
    `Numeric: ${data.numeric_columns.join(", ") || "none"}. Date candidates: ${data.datetime_columns.join(", ") || "none"}.`;

  const tsChoices = data.numeric_columns.length ? data.numeric_columns : data.columns;
  populateSelect(datasetDateSelect, data.datetime_columns, true, "(no date column)");
  populateSelect(datasetTs1Select, tsChoices);
  populateSelect(datasetTs2Select, tsChoices);

  if (data.suggested_date_column) {
    datasetDateSelect.value = data.suggested_date_column;
  }
  if (tsChoices.length >= 2) {
    datasetTs1Select.value = tsChoices[0];
    datasetTs2Select.value = tsChoices[1];
  }

  renderDatasetPreview(data.columns, data.preview_rows);
  updateDatasetRunAvailability();
  if (analysisSettingsDetails) {
    analysisSettingsDetails.open = true;
  }
}

async function submitFromDataset() {
  if (!latestDatasetId) {
    throw new Error("Please inspect a dataset before submitting from selected columns.");
  }

  const payload = {
    dataset_id: latestDatasetId,
    ts1_column: datasetTs1Select.value,
    ts2_column: datasetTs2Select.value,
    date_column: datasetDateSelect.value || null,
    ...getConfig(),
  };

  const response = await fetch("/api/v1/jobs/sdc/dataset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Unable to submit dataset-based analysis.");
  }
  await pollJob(data.job_id);
}

async function fetchResult(jobId) {
  const response = await fetch(`/api/v1/jobs/${jobId}/result`);
  if (!response.ok) {
    throw new Error(`Could not fetch result: ${response.status}`);
  }

  const payload = await response.json();
  const result = payload.result;
  latestResult = result;
  latestJobId = jobId;

  if (!Number.isFinite(Number(explorerAlphaInput.value))) {
    explorerAlphaInput.value = String(result.summary.alpha ?? 0.05);
  }

  renderSummary(result.summary, result.notes, result.runtime_seconds);
  renderTwoWayExplorer(result);
  renderLinks(result.strongest_links || []);
  setDownloadButtons(true);
}

async function pollJob(jobId) {
  if (activePoll) {
    clearInterval(activePoll);
  }

  setDownloadButtons(false);
  latestJobId = jobId;
  setStatus(`Job ${jobId} submitted. Waiting for completion...`);

  activePoll = setInterval(async () => {
    try {
      const response = await fetch(`/api/v1/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Status request failed (${response.status})`);
      }

      const status = await response.json();
      updateProgress(status.progress, status.status);
      setStatus(`Job ${jobId}: ${status.status}`);

      if (status.status === "succeeded") {
        clearInterval(activePoll);
        await fetchResult(jobId);
        setStatus(`Job ${jobId} succeeded.`);
      } else if (status.status === "failed") {
        clearInterval(activePoll);
        statusError.textContent = status.error || "Unknown job failure.";
      }
    } catch (error) {
      clearInterval(activePoll);
      statusError.textContent = String(error);
    }
  }, 1200);
}

async function submitFromText() {
  const ts1 = parseSeries(document.getElementById("ts1_text").value);
  const ts2 = parseSeries(document.getElementById("ts2_text").value);

  const payload = {
    ...getConfig(),
    ts1,
    ts2,
  };

  const response = await fetch("/api/v1/jobs/sdc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Unable to submit analysis.");
  }

  await pollJob(data.job_id);
}

async function loadExample() {
  const response = await fetch("/api/v1/examples/synthetic");
  const data = await response.json();
  document.getElementById("ts1_text").value = data.ts1.join("\n");
  document.getElementById("ts2_text").value = data.ts2.join("\n");
  setStatus("Loaded synthetic example. You can run it directly.");
}

function triggerDownload(fmt) {
  if (!latestJobId) {
    return;
  }
  window.open(`/api/v1/jobs/${latestJobId}/download/${fmt}`, "_blank");
}

function attachHandlers() {
  document.getElementById("load_example").addEventListener("click", async () => {
    try {
      await loadExample();
    } catch (error) {
      statusError.textContent = String(error);
    }
  });

  document.getElementById("submit_text").addEventListener("click", async () => {
    try {
      await submitFromText();
    } catch (error) {
      statusError.textContent = String(error);
    }
  });

  document.getElementById("submit_dataset").addEventListener("click", async () => {
    try {
      await submitFromDataset();
    } catch (error) {
      statusError.textContent = String(error);
    }
  });

  datasetFileInput.addEventListener("change", async () => {
    if (!datasetFileInput.files?.length) {
      latestDatasetId = null;
      datasetMeta.textContent = "";
      renderDatasetPreview([], []);
      updateDatasetRunAvailability();
      if (analysisSettingsDetails) {
        analysisSettingsDetails.open = false;
      }
      return;
    }
    try {
      await inspectDataset();
      setStatus("Dataset inspected. Configure columns and run.");
    } catch (error) {
      statusError.textContent = String(error);
    }
  });

  datasetDateSelect.addEventListener("change", updateDatasetRunAvailability);
  datasetTs1Select.addEventListener("change", updateDatasetRunAvailability);
  datasetTs2Select.addEventListener("change", updateDatasetRunAvailability);

  explorerAlphaInput.addEventListener("input", () => {
    if (!latestResult) {
      return;
    }
    if (alphaRenderTimer) {
      clearTimeout(alphaRenderTimer);
    }
    alphaRenderTimer = setTimeout(() => renderTwoWayExplorer(latestResult), 180);
  });

  downloadXlsxButton.addEventListener("click", () => triggerDownload("xlsx"));
  downloadPngButton.addEventListener("click", () => triggerDownload("png"));
  downloadSvgButton.addEventListener("click", () => triggerDownload("svg"));
}

setDownloadButtons(false);
if (analysisSettingsDetails) {
  analysisSettingsDetails.open = false;
}
attachHandlers();
