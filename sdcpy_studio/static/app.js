const statusText = document.getElementById("status_text");
const statusError = document.getElementById("status_error");
const summaryStats = document.getElementById("summary_stats");
const summaryNotes = document.getElementById("summary_notes");
const linksTableBody = document.querySelector("#links_table tbody");
const explorerAlphaInput = document.getElementById("explorer_alpha");

let activePoll = null;
let latestResult = null;
let alphaRenderTimer = null;

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

function renderTwoWayExplorer(result) {
  if (!window.Plotly) {
    return;
  }

  const containerId = "two_way_explorer";
  const series = result.series;
  const matrixR = result.matrix_r;
  const matrixP = result.matrix_p;
  const ranges = result.ranges_panel;
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

  const zSig = maskedSignificantMatrix(matrixR.z, matrixP.z, alpha);

  const xMin = Math.min(...matrixR.x);
  const xMax = Math.max(...matrixR.x);
  const yMin = Math.min(...matrixR.y);
  const yMax = Math.max(...matrixR.y);

  const xFragEnd = Math.min(xMax, xMin + fragmentSize - 1);
  const yFragEnd = Math.max(yMin, yMax - fragmentSize + 1);

  const bracketShapes = [
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0: xMin,
      y0: yMax,
      x1: xFragEnd,
      y1: yMax,
      line: { color: "#111827", width: 5 },
    },
    {
      type: "line",
      xref: "x",
      yref: "y",
      x0: xMin,
      y0: yMax,
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
      colorbar: {
        title: "Pearson r",
        x: 1.02,
        y: 0.44,
        len: 0.62,
      },
      xaxis: "x",
      yaxis: "y",
      hovertemplate:
        "start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<extra>Significant only</extra>",
      name: "significant",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: series.index,
      y: series.ts1,
      line: { color: "#111827", width: 1.6 },
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate: "t=%{x}<br>TS1=%{y:.3f}<extra></extra>",
      name: "TS1",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: series.ts2,
      y: series.index,
      line: { color: "#111827", width: 1.6 },
      xaxis: "x3",
      yaxis: "y3",
      hovertemplate: "TS2=%{x:.3f}<br>t=%{y}<extra></extra>",
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
    {
      type: "scatter",
      mode: "lines",
      x: ranges?.positive_freq || [],
      y: ranges?.bin_center || [],
      xaxis: "x4",
      yaxis: "y4",
      line: { color: "#dc2626", width: 2.4 },
      name: "Positive freq",
      hovertemplate: "freq=%{x:.3f}<br>bin=%{y:.3f}<extra>Positive</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      x: ranges?.negative_freq || [],
      y: ranges?.bin_center || [],
      xaxis: "x4",
      yaxis: "y4",
      line: { color: "#2563eb", width: 2.4 },
      name: "Negative freq",
      hovertemplate: "freq=%{x:.3f}<br>bin=%{y:.3f}<extra>Negative</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      x: ranges?.ns_freq || [],
      y: ranges?.bin_center || [],
      xaxis: "x4",
      yaxis: "y4",
      line: { color: "#6b7280", width: 1.8, dash: "dot" },
      name: "NS freq",
      hovertemplate: "freq=%{x:.3f}<br>bin=%{y:.3f}<extra>NS</extra>",
    },
  ];

  const layout = {
    title: `2-way explorer (significant only, alpha=${alpha.toFixed(3)})`,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 48, r: 38, b: 46, l: 60 },
    showlegend: false,
    xaxis: {
      domain: [0.24, 0.76],
      title: "TS1 fragment start",
      range: [xMin - 0.5, xMax + 0.5],
      showgrid: true,
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      domain: [0.16, 0.68],
      title: "TS2 fragment start",
      range: [yMin - 0.5, yMax + 0.5],
      showgrid: true,
      gridcolor: "#e5e7eb",
      scaleanchor: "x",
      scaleratio: 1,
    },
    xaxis2: {
      domain: [0.24, 0.76],
      anchor: "y2",
      matches: "x",
      title: "TS1",
      showgrid: true,
      gridcolor: "#e5e7eb",
      tickangle: 0,
    },
    yaxis2: {
      domain: [0.72, 0.98],
      anchor: "x2",
      title: "",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    xaxis3: {
      domain: [0.02, 0.22],
      anchor: "y3",
      title: "",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    yaxis3: {
      domain: [0.16, 0.68],
      anchor: "x3",
      matches: "y",
      title: "TS2",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    xaxis4: {
      domain: [0.78, 0.98],
      anchor: "y4",
      title: "freq",
      range: [0, 1],
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    yaxis4: {
      domain: [0.16, 0.68],
      anchor: "x4",
      title: "TS1 value bins",
      showgrid: true,
      gridcolor: "#e5e7eb",
      zeroline: false,
    },
    annotations: [
      {
        xref: "x",
        yref: "y",
        x: xMin + Math.max(2, fragmentSize * 0.45),
        y: yMax - Math.max(1, fragmentSize * 0.06),
        text: `s = ${fragmentSize} periods`,
        showarrow: false,
        font: { family: "JetBrains Mono, monospace", size: 12, color: "#111827" },
      },
      {
        xref: "paper",
        yref: "paper",
        x: 0.88,
        y: 0.71,
        text: "get_ranges_df",
        showarrow: false,
        font: { size: 11, color: "#4b5563" },
      },
    ],
    shapes: bracketShapes,
    height: 780,
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

      const stop1 = Math.min(series.index.length - 1, start1 + fragmentSize - 1);
      const stop2 = Math.min(series.index.length - 1, start2 + fragmentSize - 1);

      const ts1SegX = series.index.slice(start1, stop1 + 1);
      const ts1SegY = series.ts1.slice(start1, stop1 + 1);
      const ts2SegX = series.ts2.slice(start2, stop2 + 1);
      const ts2SegY = series.index.slice(start2, stop2 + 1);

      Plotly.restyle(gd, { x: [ts1SegX], y: [ts1SegY] }, [3]);
      Plotly.restyle(gd, { x: [ts2SegX], y: [ts2SegY] }, [4]);

      const markerRect = {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: start1 - 0.5,
        x1: start1 + 0.5,
        y0: start2 - 0.5,
        y1: start2 + 0.5,
        line: { color: "#f97316", width: 1.5 },
        fillcolor: "rgba(249, 115, 22, 0.15)",
      };
      Plotly.relayout(gd, { shapes: [...bracketShapes, markerRect] });
    });

    gd.on("plotly_unhover", clearHighlight);
  });
}

async function fetchResult(jobId) {
  const response = await fetch(`/api/v1/jobs/${jobId}/result`);
  if (!response.ok) {
    throw new Error(`Could not fetch result: ${response.status}`);
  }

  const payload = await response.json();
  const result = payload.result;
  latestResult = result;

  if (!Number.isFinite(Number(explorerAlphaInput.value))) {
    explorerAlphaInput.value = String(result.summary.alpha ?? 0.05);
  }

  renderSummary(result.summary, result.notes, result.runtime_seconds);
  renderTwoWayExplorer(result);
  renderLinks(result.strongest_links || []);
}

async function pollJob(jobId) {
  if (activePoll) {
    clearInterval(activePoll);
  }

  setStatus(`Job ${jobId} submitted. Waiting for completion...`);

  activePoll = setInterval(async () => {
    try {
      const response = await fetch(`/api/v1/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Status request failed (${response.status})`);
      }

      const status = await response.json();
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
  }, 1500);
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

async function submitFromCsv() {
  const ts1File = document.getElementById("ts1_file").files[0];
  const ts2File = document.getElementById("ts2_file").files[0];
  if (!ts1File || !ts2File) {
    throw new Error("Please select both CSV files.");
  }

  const cfg = getConfig();
  const formData = new FormData();
  formData.append("ts1_file", ts1File);
  formData.append("ts2_file", ts2File);

  Object.entries(cfg).forEach(([key, value]) => {
    formData.append(key, String(value));
  });

  const response = await fetch("/api/v1/jobs/sdc/csv", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Unable to submit CSV analysis.");
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

  document.getElementById("submit_csv").addEventListener("click", async () => {
    try {
      await submitFromCsv();
    } catch (error) {
      statusError.textContent = String(error);
    }
  });

  explorerAlphaInput.addEventListener("input", () => {
    if (!latestResult) {
      return;
    }
    if (alphaRenderTimer) {
      clearTimeout(alphaRenderTimer);
    }
    alphaRenderTimer = setTimeout(() => renderTwoWayExplorer(latestResult), 180);
  });
}

attachHandlers();
