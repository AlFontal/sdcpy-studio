const statusText = document.getElementById("status_text");
const statusError = document.getElementById("status_error");
const summaryStats = document.getElementById("summary_stats");
const summaryNotes = document.getElementById("summary_notes");
const linksTableBody = document.querySelector("#links_table tbody");

let activePoll = null;

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
      const pretty = typeof value === "number" ? value.toFixed(4).replace(/\.0000$/, "") : String(value);
      return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${pretty}</span></div>`;
    });

  items.push(
    `<div class="stat"><span class="stat-label">Runtime (s)</span><span class="stat-value">${runtimeSeconds.toFixed(2)}</span></div>`
  );

  summaryStats.innerHTML = items.join("");
  summaryNotes.innerHTML = (notes || []).map((note) => `<li>${note}</li>`).join("");
}

function renderHeatmap(containerId, matrix, title) {
  if (!window.Plotly) {
    return;
  }

  const data = [
    {
      z: matrix.z,
      x: matrix.x,
      y: matrix.y,
      type: "heatmap",
      colorscale: "RdBu",
      zmid: 0,
      colorbar: { title: "r" },
      hovertemplate: "start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<extra></extra>",
    },
  ];

  const layout = {
    title,
    margin: { t: 36, r: 20, b: 48, l: 56 },
    xaxis: { title: "start_1" },
    yaxis: { title: "start_2" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };

  Plotly.newPlot(containerId, data, layout, { responsive: true, displaylogo: false });
}

function renderTwoWayExplorer(result) {
  if (!window.Plotly) {
    return;
  }

  const containerId = "two_way_explorer";
  const matrix = result.heatmap_all;
  const series = result.series;
  const fragmentSize = Number(result.summary.fragment_size || 1);
  const maxIndex = series.index.length - 1;

  if (!matrix?.x?.length || !matrix?.y?.length || !series?.index?.length) {
    const emptyLayout = {
      title: "2-way explorer unavailable (no matrix data)",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.newPlot(containerId, [], emptyLayout, { responsive: true, displaylogo: false });
    return;
  }

  const traces = [
    {
      type: "heatmap",
      x: matrix.x,
      y: matrix.y,
      z: matrix.z,
      colorscale: "RdBu",
      zmid: 0,
      colorbar: { title: "r", x: 1.02, len: 0.74, y: 0.37 },
      xaxis: "x",
      yaxis: "y",
      hovertemplate: "start_1=%{x}<br>start_2=%{y}<br>r=%{z:.3f}<extra></extra>",
      name: "SDC",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: series.index,
      y: series.ts1,
      line: { color: "#0a7f5a", width: 1.6 },
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate: "t=%{x}<br>ts1=%{y:.3f}<extra>TS1</extra>",
      name: "TS1",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: series.ts2,
      y: series.index,
      line: { color: "#0077b6", width: 1.6 },
      xaxis: "x3",
      yaxis: "y3",
      hovertemplate: "ts2=%{x:.3f}<br>t=%{y}<extra>TS2</extra>",
      name: "TS2",
    },
    {
      type: "scattergl",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "#f97316", width: 4 },
      xaxis: "x2",
      yaxis: "y2",
      hoverinfo: "skip",
      name: "TS1 segment",
      showlegend: false,
    },
    {
      type: "scattergl",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "#f97316", width: 4 },
      xaxis: "x3",
      yaxis: "y3",
      hoverinfo: "skip",
      name: "TS2 segment",
      showlegend: false,
    },
  ];

  const layout = {
    title: "Interactive 2-way SDC explorer",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 44, r: 44, b: 46, l: 64 },
    showlegend: false,
    xaxis: {
      domain: [0.26, 1.0],
      title: "start_1",
      zeroline: false,
    },
    yaxis: {
      domain: [0.0, 0.74],
      title: "start_2",
      zeroline: false,
    },
    xaxis2: {
      domain: [0.26, 1.0],
      anchor: "y2",
      matches: "x",
      title: "time index (TS1)",
      zeroline: false,
      showgrid: true,
      gridcolor: "#eef1f5",
    },
    yaxis2: {
      domain: [0.79, 1.0],
      anchor: "x2",
      title: "TS1 value",
      zeroline: false,
      showgrid: true,
      gridcolor: "#eef1f5",
    },
    xaxis3: {
      domain: [0.0, 0.2],
      anchor: "y3",
      title: "TS2 value",
      zeroline: false,
      showgrid: true,
      gridcolor: "#eef1f5",
    },
    yaxis3: {
      domain: [0.0, 0.74],
      anchor: "x3",
      matches: "y",
      title: "time index (TS2)",
      zeroline: false,
      showgrid: true,
      gridcolor: "#eef1f5",
    },
    shapes: [],
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
      Plotly.relayout(gd, { shapes: [] });
    };

    gd.on("plotly_hover", (event) => {
      const point = event?.points?.[0];
      if (!point || point.curveNumber !== 0) {
        return;
      }

      const start1 = Number(point.x);
      const start2 = Number(point.y);
      if (!Number.isFinite(start1) || !Number.isFinite(start2)) {
        return;
      }

      const stop1 = Math.min(maxIndex, start1 + fragmentSize - 1);
      const stop2 = Math.min(maxIndex, start2 + fragmentSize - 1);

      const topX = series.index.slice(start1, stop1 + 1);
      const topY = series.ts1.slice(start1, stop1 + 1);
      const leftX = series.ts2.slice(start2, stop2 + 1);
      const leftY = series.index.slice(start2, stop2 + 1);

      Plotly.restyle(gd, { x: [topX], y: [topY] }, [3]);
      Plotly.restyle(gd, { x: [leftX], y: [leftY] }, [4]);

      const markerShape = {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: start1 - 0.5,
        x1: start1 + 0.5,
        y0: start2 - 0.5,
        y1: start2 + 0.5,
        line: { color: "#f97316", width: 2 },
        fillcolor: "rgba(249, 115, 22, 0.15)",
      };
      Plotly.relayout(gd, { shapes: [markerShape] });
    });

    gd.on("plotly_unhover", clearHighlight);
  });
}

function renderLinks(rows) {
  linksTableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    ["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"].forEach((col) => {
      const td = document.createElement("td");
      const value = row[col];
      td.textContent = typeof value === "number" ? value.toFixed(4).replace(/\.0000$/, "") : String(value);
      tr.appendChild(td);
    });
    linksTableBody.appendChild(tr);
  });
}

async function fetchResult(jobId) {
  const response = await fetch(`/api/v1/jobs/${jobId}/result`);
  if (!response.ok) {
    throw new Error(`Could not fetch result: ${response.status}`);
  }

  const payload = await response.json();
  const result = payload.result;

  renderSummary(result.summary, result.notes, result.runtime_seconds);
  renderTwoWayExplorer(result);
  renderHeatmap("heatmap_all", result.heatmap_all, "All correlations");
  renderHeatmap("heatmap_sig", result.heatmap_significant, "Significant correlations only");
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
}

attachHandlers();
