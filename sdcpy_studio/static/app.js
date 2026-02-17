const statusText = document.getElementById("status_text");
const statusError = document.getElementById("status_error");
const statusCard = document.getElementById("status_card");
const statusProgressLabel = document.getElementById("status_progress_label");
const statusProgressValue = document.getElementById("status_progress_value");
const statusProgressTrack = document.getElementById("status_progress_track");
const statusProgressFill = document.getElementById("status_progress_fill");

const summaryStats = document.getElementById("summary_stats");
const summaryNotes = document.getElementById("summary_notes");
const analysisSettingsDetails = document.getElementById("analysis_settings_details");
const explorerDetails = document.getElementById("explorer_details");

const fragmentSizeInput = document.getElementById("fragment_size");
const heatmapStepInput = document.getElementById("heatmap_step");
const heatmapStepNotice = document.getElementById("heatmap_step_notice");
const plotFragmentInput = document.getElementById("plot_fragment_size");
const plotAlphaInput = document.getElementById("plot_alpha");
const plotAlphaValue = document.getElementById("plot_alpha_value");
const plotHeatmapStepInput = document.getElementById("plot_heatmap_step");
const plotApplyButton = document.getElementById("plot_apply");
const plotLagMinInput = document.getElementById("plot_lag_min");
const plotLagMaxInput = document.getElementById("plot_lag_max");
const plotLagFocusSlider = document.getElementById("plot_lag_focus");
const plotLagFocusValue = document.getElementById("plot_lag_focus_value");
const plotLagFocusNumberInput = document.getElementById("plot_lag_focus_number");
const sdcLagControls = document.getElementById("sdc_lag_controls");
const sdcExplorerMatrixTabButton = document.getElementById("sdc_explorer_tab_matrix");
const sdcExplorerLagTabButton = document.getElementById("sdc_explorer_tab_lag");
const twoWayMatrixExplorerContainer = document.getElementById("two_way_explorer");
const twoWayLagExplorerContainer = document.getElementById("two_way_lag_explorer");
const workflowTabSdcButton = document.getElementById("workflow_tab_sdc");
const workflowTabMapButton = document.getElementById("workflow_tab_map");
const workflowSdcCards = Array.from(document.querySelectorAll(".workflow-sdc"));
const workflowMapCards = Array.from(document.querySelectorAll(".workflow-map"));

const datasetFileInput = document.getElementById("dataset_file");
const datasetDateSelect = document.getElementById("dataset_date_col");
const datasetTs1Select = document.getElementById("dataset_ts1_col");
const datasetTs2Select = document.getElementById("dataset_ts2_col");
const submitDatasetButton = document.getElementById("submit_dataset");
const datasetMeta = document.getElementById("dataset_meta");
const datasetPreviewHead = document.querySelector("#dataset_preview_table thead");
const datasetPreviewBody = document.querySelector("#dataset_preview_table tbody");
const datasetPreviewDetails = document.getElementById("dataset_preview_details");
const datasetPreviewSummary = document.getElementById("dataset_preview_summary");
const loadOniExampleButton = document.getElementById("load_oni_example");
const modeDatasetButton = document.getElementById("mode_dataset");
const modePasteButton = document.getElementById("mode_paste");
const datasetModePanel = document.getElementById("dataset_mode_panel");
const pasteModePanel = document.getElementById("paste_mode_panel");
const ts1TextInput = document.getElementById("ts1_text");
const ts2TextInput = document.getElementById("ts2_text");
const ts1NameInput = document.getElementById("ts1_name");
const ts2NameInput = document.getElementById("ts2_name");
const pasteValidationText = document.getElementById("paste_validation");

const downloadXlsxButton = document.getElementById("download_xlsx");
const downloadPngButton = document.getElementById("download_png");
const downloadSvgButton = document.getElementById("download_svg");
const mapDriverDatasetInput = document.getElementById("map_driver_dataset");
const mapFieldDatasetInput = document.getElementById("map_field_dataset");
const mapFragmentSizeInput = document.getElementById("map_fragment_size");
const mapAlphaInput = document.getElementById("map_alpha");
const mapTopFractionInput = document.getElementById("map_top_fraction");
const mapPermutationsInput = document.getElementById("map_n_permutations");
const mapMinLagInput = document.getElementById("map_min_lag");
const mapMaxLagInput = document.getElementById("map_max_lag");
const mapTimeStartInput = document.getElementById("map_time_start");
const mapTimeEndInput = document.getElementById("map_time_end");
const mapPeakDateInput = document.getElementById("map_peak_date");
const mapLatMinInput = document.getElementById("map_lat_min");
const mapLatMaxInput = document.getElementById("map_lat_max");
const mapLonMinInput = document.getElementById("map_lon_min");
const mapLonMaxInput = document.getElementById("map_lon_max");
const mapClearBoundsButton = document.getElementById("map_clear_bounds");
const mapLoadButton = document.getElementById("map_load");
const mapRunButton = document.getElementById("map_run");
const mapStatusText = document.getElementById("map_status");
const mapBoundsNotice = document.getElementById("map_bounds_notice");
const mapProgressLabel = document.getElementById("map_progress_label");
const mapProgressValue = document.getElementById("map_progress_value");
const mapProgressTrack = document.getElementById("map_progress_track");
const mapProgressFill = document.getElementById("map_progress_fill");
const mapProgressEta = document.getElementById("map_progress_eta");
const mapPhaseBadge = document.getElementById("map_phase_badge");
const mapExploreControls = document.getElementById("map_explore_controls");
const mapTimeSlider = document.getElementById("map_time_slider");
const mapTimePlayButton = document.getElementById("map_time_play");
const mapTimeSliderLabel = document.getElementById("map_time_slider_label");
const mapResultTabs = document.getElementById("map_result_tabs");
const mapPlot = document.getElementById("sdc_map_plot");
const mapSelectedCellText = document.getElementById("map_selected_cell");
const mapSummary = document.getElementById("sdc_map_summary");
const mapDownloadPngButton = document.getElementById("map_download_png");
const mapDownloadNcButton = document.getElementById("map_download_nc");
const mapDatasetDocsContent = document.getElementById("map_dataset_docs_content");
const mapDriverDatasetMeta = document.getElementById("map_driver_dataset_meta");
const mapFieldDatasetMeta = document.getElementById("map_field_dataset_meta");
const mapSaturationInput = document.getElementById("map_saturation");
const mapSaturationMeta = document.getElementById("map_saturation_meta");

let activePoll = null;
let latestResult = null;
let latestJobId = null;
let activeWorkflowTab = "sdc";
let latestDatasetId = null;
let alphaRenderTimer = null;
let datasetInspectToken = 0;
let analysisSettingsUnlocked = false;
let activeInputMode = "dataset";
let statusHideTimer = null;
let analysisRunStartedAt = 0;
let analysisProgressPeak = 0;
let hasExpandedExplorerAfterFirstRun = false;
let explorerResizeTimer = null;
let mapResizeTimer = null;
let pasteValidationTimer = null;
let mapActivePoll = null;
let latestMapResult = null;
let latestMapExplore = null;
let latestMapJobId = null;
let mapRunStartedAt = 0;
let mapRunBusy = false;
let mapLoadBusy = false;
let mapPhase = "idle";
let mapSelectedTimeIndex = 0;
let mapSelectedCell = null;
let mapActiveTab = 0;
let mapCellSeriesCache = new Map();
let mapDownloadsEnabled = false;
let mapBoundsShapeSync = false;
let mapTimePlayTimer = null;
let mapTimePlaying = false;
let heatmapStepManuallyOverridden = false;
let latestValidatedSeriesLength = 0;
let mapDatasetCatalog = null;
let activeSdcExplorerTab = "matrix";
let lagExplorerState = {
  availableLags: [],
  minLag: null,
  maxLag: null,
  selectedLag: null,
};

const MAP_DRIVER_LABEL_OVERRIDES = {
  pdo: "Pacific Decadal Oscillation (PDO)",
  nao: "North Atlantic Oscillation (NAO)",
  nino34: "Nino 3.4 anomaly",
};

const MAP_FIELD_LABEL_OVERRIDES = {
  ncep_air: "Near-surface air temperature anomaly",
  ersstv5_sst: "Sea-surface temperature anomaly (ERSSTv5)",
  oisst_v2_sst: "Sea-surface temperature anomaly (OISSTv2)",
};

const RD_BU_WHITE_CENTER = [
  [0.0, "#053061"],
  [0.125, "#2166ac"],
  [0.25, "#4393c3"],
  [0.375, "#92c5de"],
  [0.5, "#ffffff"],
  [0.625, "#f4a582"],
  [0.75, "#d6604d"],
  [0.875, "#b2182b"],
  [1.0, "#67001f"],
];
const TS1_PLOT_COLOR = "#f06f6c";
const TS2_PLOT_COLOR = "#18b8bd";

const ADAPTIVE_HEATMAP_BASE_LENGTH = 2000;
const MIN_ANALYSIS_SERIES_LENGTH = 20;

function getConfig() {
  return {
    fragment_size: Number(fragmentSizeInput.value),
    heatmap_step: Math.max(1, Math.round(Number(heatmapStepInput.value) || 1)),
    n_permutations: Number(document.getElementById("n_permutations").value),
    method: document.getElementById("method").value,
    alpha: getExplorerAlpha(0.05),
    min_lag: Number(document.getElementById("min_lag").value),
    max_lag: Number(document.getElementById("max_lag").value),
    two_tailed: document.getElementById("two_tailed").checked,
    permutations: document.getElementById("permutations").checked,
    max_memory_gb: 1.0,
  };
}

function getExplorerAlpha(defaultAlpha = 0.05) {
  const value = Number(plotAlphaInput?.value);
  if (Number.isFinite(value) && value >= 0 && value <= 1) {
    return value;
  }
  return defaultAlpha;
}

function methodCorrelationLabel(method) {
  const normalized = String(method || "").toLowerCase();
  if (normalized === "spearman") {
    return "Spearman rho";
  }
  if (normalized === "kendall") {
    return "Kendall tau";
  }
  return "Pearson r";
}

function formatAlphaValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "0.050";
  }
  return numeric.toFixed(3);
}

function recommendedFragmentSize(seriesLength) {
  const n = Math.max(0, Math.floor(Number(seriesLength) || 0));
  if (n <= 3) {
    return 2;
  }
  return Math.max(2, Math.min(n - 1, Math.round(n * 0.1)));
}

function applyRecommendedFragmentSize(seriesLength) {
  const next = recommendedFragmentSize(seriesLength);
  fragmentSizeInput.value = String(next);
  if (plotFragmentInput) {
    plotFragmentInput.value = String(next);
  }
}

function recommendedHeatmapStep(seriesLength) {
  const n = Math.max(0, Math.floor(Number(seriesLength) || 0));
  if (n <= ADAPTIVE_HEATMAP_BASE_LENGTH) {
    return 1;
  }

  let threshold = ADAPTIVE_HEATMAP_BASE_LENGTH;
  let nextStep = 1;
  while (n > threshold) {
    threshold *= 2;
    nextStep *= 2;
  }
  return Math.max(1, nextStep);
}

function formatCompactCount(value) {
  const n = Math.max(0, Math.round(Number(value) || 0));
  if (n >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(2).replace(/\.00$/, "")}B`;
  }
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(2).replace(/\.00$/, "")}M`;
  }
  if (n >= 1_000) {
    return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  }
  return String(n);
}

function updateHeatmapStepNotice({
  seriesLength = latestValidatedSeriesLength,
  autoApplied = false,
} = {}) {
  if (!heatmapStepNotice) {
    return;
  }
  const n = Math.max(0, Math.floor(Number(seriesLength) || 0));
  if (n <= ADAPTIVE_HEATMAP_BASE_LENGTH || n < MIN_ANALYSIS_SERIES_LENGTH) {
    heatmapStepNotice.hidden = true;
    return;
  }

  heatmapStepNotice.hidden = false;
  const recommendedStep = recommendedHeatmapStep(n);
  if (autoApplied) {
    heatmapStepNotice.textContent = `Large series detected (${formatCompactCount(n)} points). Heatmap step was auto-set to ${recommendedStep}. Reduce heatmap step at your own risk.`;
    return;
  }
  heatmapStepNotice.textContent = `Large series detected (${formatCompactCount(n)} points). Recommended heatmap step: ${recommendedStep}. Reduce heatmap step at your own risk, browser might crash.`;
}

function applyAdaptiveHeatmapStep(seriesLength, { resetManualOverride = false } = {}) {
  const n = Math.max(0, Math.floor(Number(seriesLength) || 0));
  latestValidatedSeriesLength = n;
  if (resetManualOverride) {
    heatmapStepManuallyOverridden = false;
  }

  const recommendedStep = recommendedHeatmapStep(n);
  const currentStep = Math.max(1, Math.round(Number(heatmapStepInput?.value) || 1));

  let autoApplied = false;
  if (!heatmapStepManuallyOverridden && currentStep !== recommendedStep) {
    heatmapStepInput.value = String(recommendedStep);
    if (plotHeatmapStepInput) {
      plotHeatmapStepInput.value = String(recommendedStep);
    }
    autoApplied = true;
  }

  updateHeatmapStepNotice({ seriesLength: n, autoApplied });
}

function applyAdaptiveDefaultsForSeries(seriesLength, { resetManualOverride = false } = {}) {
  applyRecommendedFragmentSize(seriesLength);
  applyAdaptiveHeatmapStep(seriesLength, { resetManualOverride });
}

function parseSeries(rawText) {
  return rawText
    .split(/[\n,;\s]+/)
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v));
}

function normalizeUserSeriesName(rawName, fallback) {
  const cleaned = String(rawName ?? "").trim();
  return cleaned || fallback;
}

function updatePasteValidationMessage(message, { isError = false, isValid = false } = {}) {
  if (!pasteValidationText) {
    return;
  }
  pasteValidationText.textContent = message;
  pasteValidationText.classList.toggle("error", isError);
  pasteValidationText.classList.toggle("valid", isValid);
}

function getPasteSeriesValidation({ updateMessage = false } = {}) {
  const ts1 = parseSeries(ts1TextInput?.value || "");
  const ts2 = parseSeries(ts2TextInput?.value || "");
  const ts1Label = normalizeUserSeriesName(ts1NameInput?.value, "TS1");
  const ts2Label = normalizeUserSeriesName(ts2NameInput?.value, "TS2");

  if (!ts1.length && !ts2.length) {
    if (updateMessage) {
      updatePasteValidationMessage("Paste values to validate.");
    }
    return { valid: false, ts1, ts2, ts1Label, ts2Label };
  }

  if (!ts1.length || !ts2.length) {
    if (updateMessage) {
      updatePasteValidationMessage("Both series need at least one numeric value.", { isError: true });
    }
    return { valid: false, ts1, ts2, ts1Label, ts2Label };
  }

  if (ts1.length !== ts2.length) {
    if (updateMessage) {
      updatePasteValidationMessage(
        `Length mismatch: Series 1 has ${ts1.length}, Series 2 has ${ts2.length}.`,
        { isError: true }
      );
    }
    return { valid: false, ts1, ts2, ts1Label, ts2Label };
  }

  if (ts1.length < 20) {
    if (updateMessage) {
      updatePasteValidationMessage(
        `Need at least 20 points per series (currently ${ts1.length}).`,
        { isError: true }
      );
    }
    return { valid: false, ts1, ts2, ts1Label, ts2Label };
  }

  if (updateMessage) {
    updatePasteValidationMessage(`Validation passed: ${ts1.length} paired points ready.`, {
      isValid: true,
    });
  }
  return { valid: true, ts1, ts2, ts1Label, ts2Label };
}

function setStatus(text, isError = false) {
  if (!statusText || !statusCard) {
    return;
  }
  statusText.textContent = text;
  statusCard.classList.remove("hidden");

  if (statusHideTimer) {
    clearTimeout(statusHideTimer);
    statusHideTimer = null;
  }

  if (!isError && statusError) {
    statusError.textContent = "";
  }

  if (!isError && /(succeeded|loaded|updated)/i.test(text)) {
    statusHideTimer = setTimeout(() => {
      statusCard.classList.add("hidden");
      if (statusError) {
        statusError.textContent = "";
      }
    }, 2600);
  }
}

function setStatusError(message) {
  if (statusError) {
    statusError.textContent = String(message);
  }
  setStatusProgress({ percent: analysisProgressPeak, label: "Failed" });
  setStatus("Analysis failed.", true);
}

function setStatusProgress({ percent = 0, label = "Idle" } = {}) {
  const clampedPercent = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  if (statusProgressLabel) {
    statusProgressLabel.textContent = label;
  }
  if (statusProgressValue) {
    statusProgressValue.textContent = `${clampedPercent}%`;
  }
  if (statusProgressFill) {
    statusProgressFill.style.width = `${clampedPercent}%`;
  }
  if (statusProgressTrack) {
    statusProgressTrack.setAttribute("aria-valuenow", String(clampedPercent));
  }
}

function setDownloadButtons(enabled) {
  downloadXlsxButton.disabled = !enabled;
  downloadPngButton.disabled = !enabled;
  downloadSvgButton.disabled = !enabled;
}

function setMapDownloadButtons(enabled) {
  mapDownloadsEnabled = !!enabled;
  const isReady = mapDownloadsEnabled && !!latestMapJobId;
  if (mapDownloadNcButton) {
    mapDownloadNcButton.disabled = !isReady;
  }
  if (mapDownloadPngButton) {
    mapDownloadPngButton.disabled = !isReady;
  }
}

function setMapStatus(message, isError = false) {
  if (!mapStatusText) {
    return;
  }
  mapStatusText.textContent = message;
  mapStatusText.classList.toggle("error", !!isError);
}

function setMapPhase(nextPhase) {
  mapPhase = nextPhase || "idle";
  if (mapPhase !== "explore") {
    stopMapTimePlayback();
  }
  if (mapPhaseBadge) {
    mapPhaseBadge.textContent = `Phase: ${mapPhase}`;
  }
  if (mapExploreControls) {
    mapExploreControls.hidden = mapPhase !== "explore";
  }
  if (mapResultTabs) {
    mapResultTabs.hidden = mapPhase !== "results";
  }
}

function refreshMapRunButtonState() {
  if (!mapRunButton) {
    return;
  }
  const canRun = !!latestMapExplore && !mapLoadBusy && !mapRunBusy;
  mapRunButton.disabled = !canRun;
}

function setMapLoadBusy(isBusy) {
  mapLoadBusy = !!isBusy;
  if (mapLoadButton) {
    mapLoadButton.disabled = mapLoadBusy || mapRunBusy;
    mapLoadButton.textContent = mapLoadBusy ? "Loading datasets..." : "Load driver + dataset";
  }
  refreshMapRunButtonState();
}

function setMapRunBusy(isBusy) {
  mapRunBusy = !!isBusy;
  if (!mapRunButton) {
    return;
  }
  mapRunButton.textContent = mapRunBusy ? "Running SDC map..." : "Run SDC map";
  if (mapLoadButton) {
    mapLoadButton.disabled = mapLoadBusy || mapRunBusy;
  }
  refreshMapRunButtonState();
}

function setWorkflowTab(mode) {
  const nextTab = mode === "map" ? "map" : "sdc";
  activeWorkflowTab = nextTab;

  workflowSdcCards.forEach((card) => {
    card.hidden = nextTab !== "sdc";
  });
  workflowMapCards.forEach((card) => {
    card.hidden = nextTab !== "map";
  });

  if (workflowTabSdcButton) {
    workflowTabSdcButton.classList.toggle("is-active", nextTab === "sdc");
    workflowTabSdcButton.setAttribute("aria-selected", nextTab === "sdc" ? "true" : "false");
  }
  if (workflowTabMapButton) {
    workflowTabMapButton.classList.toggle("is-active", nextTab === "map");
    workflowTabMapButton.setAttribute("aria-selected", nextTab === "map" ? "true" : "false");
  }
}

function parseOptionalNumberInput(inputEl) {
  if (!inputEl) {
    return null;
  }
  const raw = String(inputEl.value ?? "").trim();
  if (!raw) {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function getMapBoundsSelection() {
  const latMin = parseOptionalNumberInput(mapLatMinInput);
  const latMax = parseOptionalNumberInput(mapLatMaxInput);
  const lonMin = parseOptionalNumberInput(mapLonMinInput);
  const lonMax = parseOptionalNumberInput(mapLonMaxInput);
  const values = [latMin, latMax, lonMin, lonMax];
  const definedCount = values.filter((value) => value !== null).length;

  if (definedCount === 0) {
    return { hasBounds: false, lat_min: null, lat_max: null, lon_min: null, lon_max: null };
  }
  if (definedCount !== 4) {
    throw new Error("Provide all four bounds (lat/lon min+max), or clear them all to run full map.");
  }
  if (latMin >= latMax) {
    throw new Error("Latitude bounds must satisfy lat min < lat max.");
  }
  if (lonMin >= lonMax) {
    throw new Error("Longitude bounds must satisfy lon min < lon max.");
  }
  return { hasBounds: true, lat_min: latMin, lat_max: latMax, lon_min: lonMin, lon_max: lonMax };
}

function updateMapBoundsNotice() {
  if (!mapBoundsNotice) {
    return;
  }
  try {
    const bounds = getMapBoundsSelection();
    if (!bounds.hasBounds) {
      mapBoundsNotice.textContent =
        "No bounds selected: full map will be computed. Modebar: Draw rectangle sets bounds, Pan selects grid cells for the comparison series.";
      return;
    }
    mapBoundsNotice.textContent =
      `Selected bounds: lat [${bounds.lat_min.toFixed(2)}, ${bounds.lat_max.toFixed(2)}], ` +
      `lon [${bounds.lon_min.toFixed(2)}, ${bounds.lon_max.toFixed(2)}]. Modebar: Draw rectangle edits bounds, Pan selects grid cells.`;
  } catch (error) {
    mapBoundsNotice.textContent = String(error);
  }
}

function setMapBoundsInputs(bounds) {
  if (!mapLatMinInput || !mapLatMaxInput || !mapLonMinInput || !mapLonMaxInput) {
    return;
  }
  mapLatMinInput.value = bounds?.lat_min != null ? String(bounds.lat_min) : "";
  mapLatMaxInput.value = bounds?.lat_max != null ? String(bounds.lat_max) : "";
  mapLonMinInput.value = bounds?.lon_min != null ? String(bounds.lon_min) : "";
  mapLonMaxInput.value = bounds?.lon_max != null ? String(bounds.lon_max) : "";
  updateMapBoundsNotice();
}

function createMapBoundsShape(bounds) {
  if (!bounds) {
    return null;
  }
  return {
    type: "rect",
    xref: "x",
    yref: "y",
    editable: true,
    x0: bounds.lon_min,
    x1: bounds.lon_max,
    y0: bounds.lat_min,
    y1: bounds.lat_max,
    line: { color: "#f97316", width: 2, dash: "dash" },
    fillcolor: "rgba(249,115,22,0.06)",
  };
}

function normalizeBoundsObject(candidate) {
  if (!candidate) {
    return null;
  }
  const x0 = Number(candidate.x0);
  const x1 = Number(candidate.x1);
  const y0 = Number(candidate.y0);
  const y1 = Number(candidate.y1);
  if (![x0, x1, y0, y1].every((value) => Number.isFinite(value))) {
    return null;
  }
  const lonMin = Math.min(x0, x1);
  const lonMax = Math.max(x0, x1);
  const latMin = Math.min(y0, y1);
  const latMax = Math.max(y0, y1);
  if (!(lonMin < lonMax) || !(latMin < latMax)) {
    return null;
  }
  return {
    hasBounds: true,
    lat_min: latMin,
    lat_max: latMax,
    lon_min: lonMin,
    lon_max: lonMax,
  };
}

function parseBoundsFromRelayout(eventData, currentLayout = null) {
  if (!eventData) {
    return undefined;
  }
  if (Array.isArray(eventData.shapes)) {
    if (!eventData.shapes.length) {
      return null;
    }
    const lastShape = eventData.shapes[eventData.shapes.length - 1];
    return normalizeBoundsObject(lastShape);
  }

  const shapeUpdates = new Map();
  const pattern = /^shapes\[(\d+)\]\.(x0|x1|y0|y1)$/;
  Object.entries(eventData).forEach(([key, value]) => {
    const match = key.match(pattern);
    if (!match) {
      return;
    }
    const idx = Number(match[1]);
    if (!shapeUpdates.has(idx)) {
      shapeUpdates.set(idx, {});
    }
    shapeUpdates.get(idx)[match[2]] = value;
  });
  if (shapeUpdates.size) {
    const indexes = [...shapeUpdates.keys()].sort((a, b) => a - b);
    const idx = indexes[indexes.length - 1];
    const existing = currentLayout?.shapes?.[idx] || {};
    return normalizeBoundsObject({ ...existing, ...shapeUpdates.get(idx) });
  }
  return undefined;
}

async function syncExploreBoundsShape(bounds) {
  if (!window.Plotly || !mapPlot || mapPhase !== "explore") {
    return;
  }
  const shape = createMapBoundsShape(bounds);
  mapBoundsShapeSync = true;
  try {
    await Plotly.relayout(mapPlot, { shapes: shape ? [shape] : [] });
  } finally {
    mapBoundsShapeSync = false;
  }
}

function parseOptionalDateInput(inputEl) {
  const raw = String(inputEl?.value ?? "").trim();
  return raw || null;
}

function getMapConfig() {
  const bounds = getMapBoundsSelection();
  return {
    driver_dataset: mapDriverDatasetInput?.value || "pdo",
    field_dataset: mapFieldDatasetInput?.value || "ncep_air",
    fragment_size: Math.max(2, Math.round(Number(mapFragmentSizeInput?.value) || 12)),
    alpha: Number(mapAlphaInput?.value || 0.05),
    top_fraction: Number(mapTopFractionInput?.value || 0.25),
    n_permutations: Math.max(1, Math.round(Number(mapPermutationsInput?.value) || 49)),
    min_lag: Math.round(Number(mapMinLagInput?.value) || -6),
    max_lag: Math.round(Number(mapMaxLagInput?.value) || 6),
    time_start: parseOptionalDateInput(mapTimeStartInput),
    time_end: parseOptionalDateInput(mapTimeEndInput),
    peak_date: parseOptionalDateInput(mapPeakDateInput),
    two_tailed: false,
    lat_min: bounds.lat_min,
    lat_max: bounds.lat_max,
    lon_min: bounds.lon_min,
    lon_max: bounds.lon_max,
    lat_stride: 1,
    lon_stride: 1,
  };
}

function parseMapSaturationValue() {
  if (!mapSaturationInput) {
    return null;
  }
  const raw = String(mapSaturationInput.value || "").trim();
  if (!raw) {
    return null;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value;
}

function toSentenceCase(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function mapAxisDriverLabel(driverKey, driverMeta) {
  const key = String(driverKey || "").trim();
  if (MAP_DRIVER_LABEL_OVERRIDES[key]) {
    return MAP_DRIVER_LABEL_OVERRIDES[key];
  }
  const desc = String(driverMeta?.description || "").replace(/\.$/, "").trim();
  if (desc) {
    return toSentenceCase(desc.replace(/^NOAA PSL\s+/i, ""));
  }
  return key || "Driver";
}

function mapAxisFieldLabel(fieldKey, fieldMeta) {
  const key = String(fieldKey || "").trim();
  if (MAP_FIELD_LABEL_OVERRIDES[key]) {
    return MAP_FIELD_LABEL_OVERRIDES[key];
  }
  const desc = String(fieldMeta?.description || "").replace(/\.$/, "").trim();
  if (desc) {
    return toSentenceCase(desc.replace(/^NOAA\s+/i, ""));
  }
  const variable = String(fieldMeta?.variable || "").trim();
  if (variable) {
    return `${variable.toUpperCase()} anomaly`;
  }
  return key || "Selected cell";
}

function getMapAxisLabels(explore) {
  const summaryDriver = String(explore?.summary?.driver_dataset || mapDriverDatasetInput?.value || "");
  const summaryField = String(explore?.summary?.field_dataset || mapFieldDatasetInput?.value || "");
  const { driver, field } = getSelectedMapDatasets();
  return {
    driverLabel: mapAxisDriverLabel(summaryDriver, driver),
    fieldLabel: mapAxisFieldLabel(summaryField, field),
  };
}

function stopMapTimePlayback() {
  if (mapTimePlayTimer) {
    clearInterval(mapTimePlayTimer);
    mapTimePlayTimer = null;
  }
  mapTimePlaying = false;
  if (mapTimePlayButton) {
    mapTimePlayButton.textContent = "▶";
    mapTimePlayButton.title = "Play";
    mapTimePlayButton.setAttribute("aria-pressed", "false");
  }
}

function startMapTimePlayback() {
  if (!latestMapExplore || mapPhase !== "explore") {
    return;
  }
  stopMapTimePlayback();
  mapTimePlaying = true;
  if (mapTimePlayButton) {
    mapTimePlayButton.textContent = "⏸";
    mapTimePlayButton.title = "Pause";
    mapTimePlayButton.setAttribute("aria-pressed", "true");
  }
  mapTimePlayTimer = setInterval(() => {
    const maxIndex = Math.max(0, (latestMapExplore.time_index?.length || 1) - 1);
    if (maxIndex <= 0) {
      stopMapTimePlayback();
      return;
    }
    if (mapSelectedTimeIndex >= maxIndex) {
      stopMapTimePlayback();
      return;
    }
    updateExploreFrame(mapSelectedTimeIndex + 1);
  }, 380);
}

function toggleMapTimePlayback() {
  if (mapTimePlaying) {
    stopMapTimePlayback();
    return;
  }
  startMapTimePlayback();
}

function formatMetadataDateRange(start, end) {
  const s = String(start || "").trim();
  const e = String(end || "").trim();
  if (!s || !e) {
    return "";
  }
  return `${s} to ${e}`;
}

function formatShortIsoDate(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  return text.includes("T") ? text.split("T")[0] : text;
}

function formatLoadedWindow(start, end) {
  const s = formatShortIsoDate(start);
  const e = formatShortIsoDate(end);
  if (!s || !e) {
    return "";
  }
  return `${s} to ${e}`;
}

function formatBoundsSnippet(item) {
  const latMin = Number(item?.lat_min);
  const latMax = Number(item?.lat_max);
  const lonMin = Number(item?.lon_min);
  const lonMax = Number(item?.lon_max);
  if (![latMin, latMax, lonMin, lonMax].every((value) => Number.isFinite(value))) {
    return "";
  }
  return `Domain: lat [${latMin.toFixed(1)}, ${latMax.toFixed(1)}], lon [${lonMin.toFixed(
    1
  )}, ${lonMax.toFixed(1)}].`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMetaCard(container, { title, description, chips }) {
  if (!container) {
    return;
  }
  const chipHtml = (chips || [])
    .filter((chip) => chip && String(chip).trim())
    .map((chip) => `<span class="map-meta-chip">${escapeHtml(chip)}</span>`)
    .join("");
  container.innerHTML =
    `<p class="map-meta-title">${escapeHtml(title || "Metadata")}</p>` +
    `<p>${escapeHtml(description || "")}</p>` +
    `<div class="map-meta-chips">${chipHtml}</div>`;
}

function getSelectedMapDatasets() {
  if (!mapDatasetCatalog) {
    return { driver: null, field: null };
  }
  const driverKey = mapDriverDatasetInput?.value || "";
  const fieldKey = mapFieldDatasetInput?.value || "";
  const driver = (mapDatasetCatalog.drivers || []).find((item) => item.key === driverKey) || null;
  const field = (mapDatasetCatalog.fields || []).find((item) => item.key === fieldKey) || null;
  return { driver, field };
}

function renderMapSelectorOptions() {
  if (!mapDatasetCatalog) {
    return;
  }
  if (mapDriverDatasetInput) {
    const byKey = new Map((mapDatasetCatalog.drivers || []).map((item) => [item.key, item]));
    Array.from(mapDriverDatasetInput.options).forEach((option) => {
      const entry = byKey.get(option.value);
      if (!entry) {
        return;
      }
      const coverage = formatMetadataDateRange(entry.time_start, entry.time_end);
      option.textContent = coverage ? `${entry.key} (${coverage})` : entry.key;
      option.title = entry.description || entry.key;
    });
  }
  if (mapFieldDatasetInput) {
    const byKey = new Map((mapDatasetCatalog.fields || []).map((item) => [item.key, item]));
    Array.from(mapFieldDatasetInput.options).forEach((option) => {
      const entry = byKey.get(option.value);
      if (!entry) {
        return;
      }
      const variableSuffix = entry.variable ? ` · ${entry.variable}` : "";
      const coverage = formatMetadataDateRange(entry.time_start, entry.time_end);
      option.textContent = coverage
        ? `${entry.key}${variableSuffix} (${coverage})`
        : `${entry.key}${variableSuffix}`;
      option.title = entry.description || entry.key;
    });
  }
}

function renderMapSelectorMetadata() {
  if (!mapDatasetCatalog) {
    renderMetaCard(mapDriverDatasetMeta, {
      title: "Driver metadata",
      description: "Metadata unavailable right now.",
      chips: [],
    });
    renderMetaCard(mapFieldDatasetMeta, {
      title: "Field metadata",
      description: "Metadata unavailable right now.",
      chips: [],
    });
    return;
  }
  const { driver, field } = getSelectedMapDatasets();
  if (!driver) {
    renderMetaCard(mapDriverDatasetMeta, {
      title: "Driver metadata",
      description: "Select a driver dataset.",
      chips: [],
    });
  } else {
    const coverage = formatMetadataDateRange(driver.time_start, driver.time_end);
    const loadedWindow = formatLoadedWindow(
      driver.loaded_time_start ?? latestMapExplore?.summary?.time_start,
      driver.loaded_time_end ?? latestMapExplore?.summary?.time_end
    );
    const peakDate = driver.peak_date ? `Peak: ${driver.peak_date}` : "";
    const points = Number.isFinite(Number(driver.n_points))
      ? `${Number(driver.n_points).toLocaleString()} points`
      : "";
    const chips = [coverage ? `Coverage: ${coverage}` : loadedWindow ? `Loaded window: ${loadedWindow}` : "", points, peakDate];
    renderMetaCard(mapDriverDatasetMeta, {
      title: "Driver series",
      description: driver.description || "No description available.",
      chips,
    });
  }

  if (!field) {
    renderMetaCard(mapFieldDatasetMeta, {
      title: "Field metadata",
      description: "Select a field dataset.",
      chips: [],
    });
  } else {
    const coverage = formatMetadataDateRange(field.time_start, field.time_end);
    const loadedWindow = formatLoadedWindow(
      field.loaded_time_start ?? latestMapExplore?.summary?.time_start,
      field.loaded_time_end ?? latestMapExplore?.summary?.time_end
    );
    const variableChip = field.variable ? `Variable: ${field.variable}` : "";
    const gridShape =
      Number.isFinite(Number(field.n_lat)) && Number.isFinite(Number(field.n_lon))
        ? `Grid: ${field.n_lat}×${field.n_lon}`
        : "";
    const domain = formatBoundsSnippet(field).replace(/^Domain:\s*/i, "");
    const chips = [
      coverage ? `Coverage: ${coverage}` : loadedWindow ? `Loaded window: ${loadedWindow}` : "",
      variableChip,
      gridShape,
      domain,
    ];
    renderMetaCard(mapFieldDatasetMeta, {
      title: "Field grid",
      description: field.description || "No description available.",
      chips,
    });
  }
}

function renderMapDatasetDocs() {
  if (!mapDatasetDocsContent) {
    return;
  }
  if (!mapDatasetCatalog) {
    mapDatasetDocsContent.textContent = "Dataset descriptions unavailable in this environment.";
    return;
  }
  const { driver, field } = getSelectedMapDatasets();
  const driverKey = mapDriverDatasetInput?.value || "";
  const fieldKey = mapFieldDatasetInput?.value || "";
  const driverText = driver?.description || "No driver description available.";
  const fieldText = field?.description || "No field description available.";
  const variableSuffix = field?.variable ? ` (variable: ${field.variable})` : "";
  const driverCoverage =
    formatMetadataDateRange(driver?.time_start, driver?.time_end) ||
    formatLoadedWindow(driver?.loaded_time_start, driver?.loaded_time_end) ||
    "available after load";
  const fieldCoverage =
    formatMetadataDateRange(field?.time_start, field?.time_end) ||
    formatLoadedWindow(field?.loaded_time_start, field?.loaded_time_end) ||
    "available after load";
  const domainText = formatBoundsSnippet(field);
  mapDatasetDocsContent.innerHTML =
    `<p><strong>Driver (${driverKey})</strong>: ${driverText}<br><span class="hint">Coverage: ${driverCoverage}</span></p>` +
    `<p><strong>Field (${fieldKey})</strong>${variableSuffix}: ${fieldText}<br><span class="hint">Coverage: ${fieldCoverage}${domainText ? ` · ${domainText}` : ""}</span></p>`;
}

async function fetchMapCatalog() {
  const timeoutMs = 12000;
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  let timeoutId = null;
  if (controller) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }
  try {
    const response = await fetch("/api/v1/sdc-map/catalog", controller ? { signal: controller.signal } : {});
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to fetch map dataset catalog.");
    }
    mapDatasetCatalog = data;
  } catch (_error) {
    mapDatasetCatalog = null;
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
  renderMapSelectorOptions();
  renderMapSelectorMetadata();
  renderMapDatasetDocs();
}

async function applyMapDriverDefaults(driverKey) {
  const key = driverKey || mapDriverDatasetInput?.value || "pdo";
  const response = await fetch(`/api/v1/sdc-map/defaults?driver_dataset=${encodeURIComponent(key)}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Unable to compute driver defaults.");
  }
  if (mapPeakDateInput) {
    mapPeakDateInput.value = data.peak_date || "";
  }
  if (mapTimeStartInput) {
    mapTimeStartInput.value = data.time_start || "";
  }
  if (mapTimeEndInput) {
    mapTimeEndInput.value = data.time_end || "";
  }
  if (mapDatasetCatalog?.drivers?.length) {
    const entry = mapDatasetCatalog.drivers.find((item) => item.key === key);
    if (entry) {
      entry.time_start = data.driver_min_date || data.time_start || entry.time_start;
      entry.time_end = data.driver_max_date || data.time_end || entry.time_end;
      entry.peak_date = data.peak_date || entry.peak_date;
      if (Number.isFinite(Number(data.n_points))) {
        entry.n_points = Number(data.n_points);
      }
      renderMapSelectorOptions();
      renderMapSelectorMetadata();
      renderMapDatasetDocs();
    }
  }
}

function updateProgress(progress, status) {
  const normalizedStatus = String(status || "").toLowerCase();
  const rawCurrent = Math.max(0, Number(progress?.current) || 0);
  const total = Math.max(1, Number(progress?.total) || 1);
  const current = normalizedStatus === "succeeded" ? total : Math.min(rawCurrent, total);
  const description = progress?.description ? String(progress.description) : "Running";

  let percent = 0;
  if (normalizedStatus === "succeeded") {
    percent = 100;
  } else if (normalizedStatus === "failed") {
    percent = (current / total) * 100;
  } else if (normalizedStatus === "running") {
    percent = (current / total) * 100;
  }

  if (normalizedStatus === "queued") {
    analysisProgressPeak = 0;
  } else if (normalizedStatus === "running") {
    analysisProgressPeak = Math.max(analysisProgressPeak, percent);
    percent = analysisProgressPeak;
  }

  const stepLabel = total > 1 ? `Step ${Math.min(current, total)}/${total}` : "Step 1/1";
  let label = description;
  if (normalizedStatus === "queued") {
    label = "Queued";
  } else if (normalizedStatus === "running") {
    label = `${stepLabel}: ${description}`;
  } else if (normalizedStatus === "succeeded") {
    const elapsedSeconds = analysisRunStartedAt
      ? Math.max(0, (Date.now() - analysisRunStartedAt) / 1000)
      : 0;
    label = elapsedSeconds ? `Completed in ${formatDurationSeconds(elapsedSeconds)}` : "Completed";
  } else if (normalizedStatus === "failed") {
    label = description && description.toLowerCase() !== "failed" ? `Failed: ${description}` : "Failed";
  }

  setStatusProgress({ percent, label });
}

function formatDurationSeconds(totalSeconds) {
  const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) {
    return secs ? `${mins}m ${secs}s` : `${mins}m`;
  }
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return remMins ? `${hours}h ${remMins}m` : `${hours}h`;
}

function setMapProgress({
  percent = 0,
  label = "Idle",
  etaText = "Expected runtime: up to a few minutes.",
} = {}) {
  const clampedPercent = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));

  if (mapProgressLabel) {
    mapProgressLabel.textContent = label;
  }
  if (mapProgressValue) {
    mapProgressValue.textContent = `${clampedPercent}%`;
  }
  if (mapProgressFill) {
    mapProgressFill.style.width = `${clampedPercent}%`;
  }
  if (mapProgressTrack) {
    mapProgressTrack.setAttribute("aria-valuenow", String(clampedPercent));
  }
  if (mapProgressEta) {
    mapProgressEta.textContent = etaText;
  }
}

function updateMapProgress(progress, status) {
  const normalizedStatus = String(status || "").toLowerCase();
  const rawCurrent = Math.max(0, Number(progress?.current) || 0);
  const total = Math.max(1, Number(progress?.total) || 1);
  const current = normalizedStatus === "succeeded" ? total : Math.min(rawCurrent, total);
  const stageLabel = progress?.description ? String(progress.description) : "Running";

  let percent = 0;
  if (normalizedStatus === "succeeded") {
    percent = 100;
  } else if (normalizedStatus === "failed") {
    percent = (current / total) * 100;
  } else if (normalizedStatus === "running") {
    percent = (current / total) * 100;
  }

  const stepLabel = total > 1 ? `Step ${Math.min(current, total)}/${total}` : "Step 1/1";
  let label = stageLabel;
  if (normalizedStatus === "queued") {
    label = "Queued";
  } else if (normalizedStatus === "running") {
    label = `${stepLabel}: ${stageLabel}`;
  } else if (normalizedStatus === "succeeded") {
    label = "Completed";
  } else if (normalizedStatus === "failed") {
    label = stageLabel && stageLabel.toLowerCase() !== "failed" ? `Failed: ${stageLabel}` : "Failed";
  }

  const elapsedSeconds =
    mapRunStartedAt > 0 ? Math.max(0, (Date.now() - mapRunStartedAt) / 1000) : 0;
  let etaText = "Expected runtime: up to a few minutes.";
  if (normalizedStatus === "queued") {
    etaText = "Queued. Waiting for worker to pick up the job.";
  } else if (normalizedStatus === "succeeded") {
    etaText = `Completed in ${formatDurationSeconds(elapsedSeconds)}.`;
  } else if (normalizedStatus === "failed") {
    etaText = elapsedSeconds
      ? `Stopped due to an error after ${formatDurationSeconds(elapsedSeconds)}.`
      : "Stopped due to an error.";
  } else if (current > 0 && current < total && elapsedSeconds > 0) {
    const secondsPerStep = elapsedSeconds / Math.max(1, current);
    const remainingSeconds = Math.max(1, (total - current) * secondsPerStep);
    etaText = `Elapsed: ${formatDurationSeconds(elapsedSeconds)}. Remaining: ~${formatDurationSeconds(remainingSeconds)}.`;
  } else if (normalizedStatus === "running") {
    etaText = elapsedSeconds
      ? `Elapsed: ${formatDurationSeconds(elapsedSeconds)}. Estimating remaining...`
      : "Estimating remaining time...";
  }

  setMapProgress({
    percent,
    label,
    etaText,
  });
}

function syncPlotControlsFromSettings() {
  if (plotFragmentInput && fragmentSizeInput) {
    plotFragmentInput.value = fragmentSizeInput.value;
  }
  if (plotAlphaInput && plotAlphaValue) {
    plotAlphaValue.textContent = formatAlphaValue(plotAlphaInput.value);
  }
  if (plotHeatmapStepInput && heatmapStepInput) {
    plotHeatmapStepInput.value = heatmapStepInput.value;
  }
}

function expandExplorerAfterFirstSuccessfulRun() {
  if (hasExpandedExplorerAfterFirstRun || !explorerDetails) {
    return;
  }
  explorerDetails.open = true;
  hasExpandedExplorerAfterFirstRun = true;
}

async function rerunActiveWorkflow() {
  if (activeInputMode === "dataset") {
    await submitFromDataset();
    return;
  }
  await submitFromText();
}

async function applyPlotControls() {
  const nextAlpha = Number(plotAlphaInput?.value);
  const nextFragment = Math.round(Number(plotFragmentInput?.value));
  const nextHeatmapStep = Math.max(1, Math.round(Number(plotHeatmapStepInput?.value) || 1));

  if (Number.isFinite(nextAlpha) && nextAlpha >= 0 && nextAlpha <= 1) {
    if (plotAlphaInput) {
      plotAlphaInput.value = String(nextAlpha);
    }
    if (plotAlphaValue) {
      plotAlphaValue.textContent = formatAlphaValue(nextAlpha);
    }
    if (latestResult) {
      renderTwoWayExplorer(latestResult);
    }
  }

  if (!Number.isFinite(nextFragment) || nextFragment < 2) {
    return;
  }

  const currentFragment = Number(fragmentSizeInput.value);
  const currentHeatmapStep = Math.max(1, Math.round(Number(heatmapStepInput.value) || 1));
  if (nextFragment === currentFragment && nextHeatmapStep === currentHeatmapStep) {
    setStatus("Updated plot alpha.");
    return;
  }

  fragmentSizeInput.value = String(nextFragment);
  heatmapStepInput.value = String(nextHeatmapStep);
  if (nextHeatmapStep !== currentHeatmapStep) {
    heatmapStepManuallyOverridden = true;
  }
  syncPlotControlsFromSettings();
  updateHeatmapStepNotice();
  if (!latestResult) {
    setStatus("Plot settings updated. Run analysis to apply.");
    return;
  }

  setStatus(
    `Re-running with fragment size ${nextFragment} and heatmap step ${nextHeatmapStep}...`
  );
  await rerunActiveWorkflow();
}

function applyLagRangeFromInputs() {
  const lagValues = lagExplorerState.availableLags || [];
  if (!lagValues.length) {
    return;
  }
  const minAvailable = lagValues[0];
  const maxAvailable = lagValues[lagValues.length - 1];
  const nextMinRaw = Math.round(Number(plotLagMinInput?.value));
  const nextMaxRaw = Math.round(Number(plotLagMaxInput?.value));
  const nextMin = Number.isFinite(nextMinRaw) ? Math.max(minAvailable, nextMinRaw) : minAvailable;
  const nextMax = Number.isFinite(nextMaxRaw) ? Math.min(maxAvailable, nextMaxRaw) : maxAvailable;

  lagExplorerState.minLag = Math.min(nextMin, nextMax);
  lagExplorerState.maxLag = Math.max(nextMin, nextMax);
  lagExplorerState.selectedLag = clampLagToFilteredRange(
    lagExplorerState.selectedLag,
    lagValues,
    lagExplorerState.minLag,
    lagExplorerState.maxLag
  );
  syncLagControlsFromState();
  if (latestResult && activeSdcExplorerTab === "lag") {
    renderLagFocusExplorer(latestResult);
  }
}

function applyFocusedLagFromInput(rawValue) {
  const lagValues = lagExplorerState.availableLags || [];
  if (!lagValues.length) {
    return;
  }
  const minLag = Number(lagExplorerState.minLag);
  const maxLag = Number(lagExplorerState.maxLag);
  const parsed = Math.round(Number(rawValue));
  if (!Number.isFinite(parsed)) {
    syncLagControlsFromState();
    return;
  }
  const clamped = clampLagToFilteredRange(parsed, lagValues, minLag, maxLag);
  if (clamped === null) {
    syncLagControlsFromState();
    return;
  }
  lagExplorerState.selectedLag = clamped;
  syncLagControlsFromState();
  if (latestResult && activeSdcExplorerTab === "lag") {
    renderLagFocusExplorer(latestResult);
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
    ["method", "Method"],
    ["full_series_corr_lag0", "Full-series corr (lag 0)"],
  ];

  const items = keys
    .filter(([key]) => Object.prototype.hasOwnProperty.call(summary, key))
    .map(([key, label]) => {
      const value = summary[key];
      const pretty =
        typeof value === "number"
          ? value.toFixed(4).replace(/\.0000$/, "")
          : value === null || value === undefined
            ? "NA"
            : String(value);
      return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${pretty}</span></div>`;
    });

  items.push(
    `<div class="stat"><span class="stat-label">Runtime (s)</span><span class="stat-value">${runtimeSeconds.toFixed(2)}</span></div>`
  );

  summaryStats.innerHTML = items.join("");
  summaryNotes.innerHTML = (notes || []).map((note) => `<li>${note}</li>`).join("");
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

function formatTooltipDate(value) {
  const text = String(value ?? "").trim();
  const dateOnly = text.match(/^(\d{4}-\d{2}-\d{2})(?:[ T].*)$/);
  if (dateOnly) {
    return dateOnly[1];
  }
  return text;
}

function formatAxisTickLabel(value) {
  const text = String(value ?? "").trim();
  const monthLike = text.match(/^(\d{4}-\d{2})(?:-\d{2})?(?:[ T].*)?$/);
  if (monthLike) {
    return monthLike[1];
  }
  return text.length > 10 ? text.slice(0, 10) : text;
}

function normalizeSeriesLabel(value, fallback) {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function compactSeriesLabel(value, maxLen = 36) {
  const text = String(value ?? "").trim();
  if (text.length <= maxLen) {
    return text;
  }
  return `${text.slice(0, Math.max(1, maxLen - 1))}\u2026`;
}

function buildAxisTicks(labels, nTicks = 6, axisValues = null) {
  if (!labels?.length) {
    return { tickvals: [], ticktext: [] };
  }
  if (labels.length <= nTicks) {
    const vals = labels.map((_, i) => i);
    const tickvals = axisValues?.length === labels.length ? vals.map((i) => axisValues[i]) : vals;
    return { tickvals, ticktext: labels.map((v) => formatAxisTickLabel(v)) };
  }

  const step = Math.max(1, Math.floor((labels.length - 1) / (nTicks - 1)));
  const vals = [];
  for (let i = 0; i < labels.length; i += step) {
    vals.push(i);
  }
  const lastIndex = labels.length - 1;
  if (vals[vals.length - 1] !== lastIndex) {
    const prev = vals[vals.length - 1];
    const minSpacing = Math.max(1, Math.round(step * 0.65));
    if (Number.isFinite(prev) && lastIndex - prev < minSpacing) {
      vals[vals.length - 1] = lastIndex;
    } else {
      vals.push(lastIndex);
    }
  }
  const tickvals = axisValues?.length === labels.length ? vals.map((i) => axisValues[i]) : vals;
  return {
    tickvals,
    ticktext: vals.map((idx) => formatAxisTickLabel(labels[idx])),
  };
}

function normalizePosition(value, maxIndex) {
  const rounded = Math.round(Number(value));
  if (!Number.isFinite(rounded)) {
    return 0;
  }
  return Math.max(0, Math.min(maxIndex, rounded));
}

function resetLagExplorerState() {
  lagExplorerState = {
    availableLags: [],
    minLag: null,
    maxLag: null,
    selectedLag: null,
  };
}

function getLagExplorerMatrices(result) {
  const lagMatrixR = result?.lag_matrix_r;
  const lagMatrixP = result?.lag_matrix_p;
  if (!lagMatrixR || !lagMatrixP) {
    return null;
  }
  if (!Array.isArray(lagMatrixR.x) || !Array.isArray(lagMatrixR.y) || !Array.isArray(lagMatrixR.z)) {
    return null;
  }
  if (!Array.isArray(lagMatrixP.x) || !Array.isArray(lagMatrixP.y) || !Array.isArray(lagMatrixP.z)) {
    return null;
  }
  if (!lagMatrixR.x.length || !lagMatrixR.y.length || !lagMatrixR.z.length) {
    return null;
  }
  return { lagMatrixR, lagMatrixP };
}

function hasLagExplorerData(result) {
  return !!getLagExplorerMatrices(result);
}

function nearestLagValue(target, lagValues) {
  if (!Array.isArray(lagValues) || !lagValues.length) {
    return null;
  }
  let best = Number(lagValues[0]);
  let bestDist = Math.abs(best - Number(target));
  for (let i = 1; i < lagValues.length; i += 1) {
    const candidate = Number(lagValues[i]);
    const dist = Math.abs(candidate - Number(target));
    if (dist < bestDist) {
      best = candidate;
      bestDist = dist;
    }
  }
  return Number.isFinite(best) ? best : null;
}

function clampLagToFilteredRange(target, lagValues, minLag, maxLag) {
  const filtered = (lagValues || []).filter((lag) => lag >= minLag && lag <= maxLag);
  return nearestLagValue(target, filtered);
}

function syncLagControlVisibility() {
  const show = activeSdcExplorerTab === "lag" && hasLagExplorerData(latestResult);
  if (sdcLagControls) {
    sdcLagControls.hidden = !show;
  }
  if (twoWayMatrixExplorerContainer) {
    twoWayMatrixExplorerContainer.hidden = activeSdcExplorerTab !== "matrix";
  }
  if (twoWayLagExplorerContainer) {
    twoWayLagExplorerContainer.hidden = activeSdcExplorerTab !== "lag";
  }
}

function setSdcExplorerTab(nextTab, { rerender = true } = {}) {
  const normalized = nextTab === "lag" ? "lag" : "matrix";
  activeSdcExplorerTab = normalized;
  if (sdcExplorerMatrixTabButton) {
    const isActive = normalized === "matrix";
    sdcExplorerMatrixTabButton.classList.toggle("is-active", isActive);
    sdcExplorerMatrixTabButton.setAttribute("aria-selected", isActive ? "true" : "false");
  }
  if (sdcExplorerLagTabButton) {
    const isActive = normalized === "lag";
    sdcExplorerLagTabButton.classList.toggle("is-active", isActive);
    sdcExplorerLagTabButton.setAttribute("aria-selected", isActive ? "true" : "false");
  }
  syncLagControlVisibility();
  if (rerender && latestResult) {
    renderTwoWayExplorer(latestResult);
  }
}

function setSdcExplorerTabAvailability(result) {
  const hasLagData = hasLagExplorerData(result);
  if (sdcExplorerLagTabButton) {
    sdcExplorerLagTabButton.disabled = !hasLagData;
  }
  if (!hasLagData && activeSdcExplorerTab === "lag") {
    setSdcExplorerTab("matrix", { rerender: false });
  }
  syncLagControlVisibility();
}

function initializeLagExplorerState(result, { reset = false } = {}) {
  const matrices = getLagExplorerMatrices(result);
  if (!matrices) {
    resetLagExplorerState();
    return;
  }
  if (!reset && lagExplorerState.availableLags.length) {
    return;
  }

  const availableLags = matrices.lagMatrixR.y
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);

  if (!availableLags.length) {
    resetLagExplorerState();
    return;
  }

  const minLag = availableLags[0];
  const maxLag = availableLags[availableLags.length - 1];
  const preferredLagRaw = Number(result?.lag_default);
  const preferredLag = Number.isFinite(preferredLagRaw) ? preferredLagRaw : 0;
  const selectedLag = nearestLagValue(preferredLag, availableLags);

  lagExplorerState = {
    availableLags,
    minLag,
    maxLag,
    selectedLag,
  };
}

function syncLagControlsFromState() {
  const lagValues = lagExplorerState.availableLags || [];
  if (!lagValues.length) {
    return;
  }
  const minAvailable = lagValues[0];
  const maxAvailable = lagValues[lagValues.length - 1];
  const rawMin = Number(lagExplorerState.minLag);
  const rawMax = Number(lagExplorerState.maxLag);
  const minLagCandidate = Number.isFinite(rawMin) ? rawMin : minAvailable;
  const maxLagCandidate = Number.isFinite(rawMax) ? rawMax : maxAvailable;
  const minLag = Math.max(minAvailable, minLagCandidate);
  const maxLag = Math.min(maxAvailable, maxLagCandidate);
  const normalizedMin = minLag;
  const normalizedMax = Math.max(normalizedMin, maxLag);
  const selectedLagRaw = Number(lagExplorerState.selectedLag);
  const selectedLag =
    clampLagToFilteredRange(
      Number.isFinite(selectedLagRaw) ? selectedLagRaw : 0,
      lagValues,
      normalizedMin,
      normalizedMax
    ) ?? normalizedMin;

  lagExplorerState.minLag = normalizedMin;
  lagExplorerState.maxLag = normalizedMax;
  lagExplorerState.selectedLag = selectedLag;

  if (plotLagMinInput) {
    plotLagMinInput.value = String(normalizedMin);
    plotLagMinInput.min = String(minAvailable);
    plotLagMinInput.max = String(maxAvailable);
  }
  if (plotLagMaxInput) {
    plotLagMaxInput.value = String(normalizedMax);
    plotLagMaxInput.min = String(minAvailable);
    plotLagMaxInput.max = String(maxAvailable);
  }
  if (plotLagFocusSlider) {
    plotLagFocusSlider.min = String(normalizedMin);
    plotLagFocusSlider.max = String(normalizedMax);
    plotLagFocusSlider.step = "1";
    plotLagFocusSlider.value = String(selectedLag);
  }
  if (plotLagFocusValue) {
    plotLagFocusValue.textContent = String(selectedLag);
  }
  if (plotLagFocusNumberInput) {
    plotLagFocusNumberInput.value = String(selectedLag);
    plotLagFocusNumberInput.min = String(normalizedMin);
    plotLagFocusNumberInput.max = String(normalizedMax);
    plotLagFocusNumberInput.step = "1";
  }
}

function setAnalysisSettingsUnlocked(unlocked) {
  analysisSettingsUnlocked = !!unlocked;
  if (!analysisSettingsDetails) {
    return;
  }
  analysisSettingsDetails.dataset.locked = unlocked ? "false" : "true";
  if (!unlocked) {
    analysisSettingsDetails.open = false;
  }
}

function setInputMode(mode) {
  const usePasteMode = mode === "paste";
  activeInputMode = usePasteMode ? "paste" : "dataset";

  if (datasetModePanel) {
    datasetModePanel.hidden = usePasteMode;
  }
  if (pasteModePanel) {
    pasteModePanel.hidden = !usePasteMode;
  }
  if (modeDatasetButton) {
    modeDatasetButton.classList.toggle("is-active", !usePasteMode);
    modeDatasetButton.setAttribute("aria-selected", usePasteMode ? "false" : "true");
  }
  if (modePasteButton) {
    modePasteButton.classList.toggle("is-active", usePasteMode);
    modePasteButton.setAttribute("aria-selected", usePasteMode ? "true" : "false");
  }
  updateDatasetRunAvailability();
  if (usePasteMode) {
    const validation = getPasteSeriesValidation({ updateMessage: true });
    if (validation.valid) {
      applyAdaptiveHeatmapStep(validation.ts1.length);
    }
  }
  updateHeatmapStepNotice();
}

function renderTwoWayMatrixExplorer(result) {
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
  const seriesLength = Math.min(series.ts1.length, series.ts2.length, labels.length);
  if (seriesLength < fragmentSize) {
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

  const ts1Series = series.ts1.slice(0, seriesLength);
  const ts2Series = series.ts2.slice(0, seriesLength);
  const ts1LabelRaw = normalizeSeriesLabel(result?.summary?.ts1_label, "TS1");
  const ts2LabelRaw = normalizeSeriesLabel(result?.summary?.ts2_label, "TS2");
  const ts1Label = compactSeriesLabel(ts1LabelRaw);
  const ts2Label = compactSeriesLabel(ts2LabelRaw);
  const labelSeries = labels.slice(0, seriesLength).map((label) => formatTooltipDate(label));
  const seriesPositions = Array.from({ length: seriesLength }, (_, idx) => idx);
  const fragmentCenterOffset = (fragmentSize - 1) / 2;
  const maxStartIndex = Math.max(0, seriesLength - fragmentSize);
  const containerWidth = container?.clientWidth || 1200;
  const compactLayout = containerWidth < 1120;
  const tinyLayout = containerWidth < 860;
  const heatXStart = matrixR.x.map((v) => Number(v));
  const heatYStart = matrixR.y.map((v) => Number(v));
  const heatX = heatXStart.map((v) => v + fragmentCenterOffset);
  const heatY = heatYStart.map((v) => v + fragmentCenterOffset);
  const startIndexToLabel = (startIdx) => {
    const idx = Math.max(0, Math.min(seriesLength - 1, Math.round(Number(startIdx) || 0)));
    return labelSeries[idx] ?? String(idx);
  };
  const heatmapHoverData = heatYStart.map((start2, rowIdx) => {
    const start2Label = startIndexToLabel(start2);
    return heatXStart.map((start1, colIdx) => {
      const rawR = matrixR?.z?.[rowIdx]?.[colIdx];
      const rawP = matrixP?.z?.[rowIdx]?.[colIdx];
      const hasValues =
        typeof rawR === "number" &&
        Number.isFinite(rawR) &&
        typeof rawP === "number" &&
        Number.isFinite(rawP);
      let statText = "Not tested";
      if (hasValues) {
        const rValue = Number(rawR);
        const pValue = Number(rawP);
        statText = pValue <= alpha ? `r=${rValue.toFixed(3)}` : `NS (r=${rValue.toFixed(3)})`;
      }
      return [startIndexToLabel(start1), start2Label, statText];
    });
  });
  const centerToStartIndex = (centerPosition) =>
    Math.max(0, Math.min(maxStartIndex, Math.round(centerPosition - fragmentCenterOffset)));
  const topTickConfig = buildAxisTicks(
    labelSeries,
    tinyLayout ? 3 : compactLayout ? 4 : 5,
    seriesPositions
  );
  const sideTickConfig = buildAxisTicks(
    labelSeries,
    tinyLayout ? 4 : compactLayout ? 5 : 6,
    seriesPositions
  );
  const fragTickConfig = buildAxisTicks(
    labelSeries,
    tinyLayout ? 4 : compactLayout ? 5 : 6,
    seriesPositions
  );
  const zSig = maskedSignificantMatrix(matrixR.z, matrixP.z, alpha);
  const zSigNumeric = zSig.map((row) =>
    row.map((value) =>
      typeof value === "number" && Number.isFinite(value) ? value : Number.NaN
    )
  );

  const axisMin = -0.5;
  const axisMax = seriesLength - 0.5;

  let zAbsMax = 0;
  zSigNumeric.forEach((row) => {
    row.forEach((value) => {
      if (Number.isFinite(value)) {
        zAbsMax = Math.max(zAbsMax, Math.abs(value));
      }
    });
  });
  if (zAbsMax === 0) {
    zAbsMax = 1;
  }

  const colCount = matrixR.x.length;
  const rowCount = matrixR.y.length;

  const maxPositiveByCol = new Array(colCount).fill(null);
  const minNegativeByCol = new Array(colCount).fill(null);
  const maxPositiveByRow = new Array(rowCount).fill(null);
  const minNegativeByRow = new Array(rowCount).fill(null);

  for (let col = 0; col < colCount; col += 1) {
    let maxPos = -Infinity;
    let minNeg = Infinity;
    for (let row = 0; row < rowCount; row += 1) {
      const value = zSigNumeric?.[row]?.[col];
      if (!Number.isFinite(value)) {
        continue;
      }
      if (value > 0) {
        maxPos = Math.max(maxPos, value);
      }
      if (value < 0) {
        minNeg = Math.min(minNeg, value);
      }
    }
    maxPositiveByCol[col] = Number.isFinite(maxPos) ? maxPos : null;
    minNegativeByCol[col] = Number.isFinite(minNeg) ? minNeg : null;
  }

  for (let row = 0; row < rowCount; row += 1) {
    let maxPos = -Infinity;
    let minNeg = Infinity;
    for (let col = 0; col < colCount; col += 1) {
      const value = zSigNumeric?.[row]?.[col];
      if (!Number.isFinite(value)) {
        continue;
      }
      if (value > 0) {
        maxPos = Math.max(maxPos, value);
      }
      if (value < 0) {
        minNeg = Math.min(minNeg, value);
      }
    }
    maxPositiveByRow[row] = Number.isFinite(maxPos) ? maxPos : null;
    minNegativeByRow[row] = Number.isFinite(minNeg) ? minNeg : null;
  }

  const minNegativeAbsByCol = minNegativeByCol.map((value) =>
    Number.isFinite(value) ? Math.abs(value) : null
  );
  const minNegativeAbsByRow = minNegativeByRow.map((value) =>
    Number.isFinite(value) ? Math.abs(value) : null
  );

  const finiteTs2 = ts2Series.filter(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
  const finiteTs1 = ts1Series.filter(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
  const ts1Min = finiteTs1.length ? Math.min(...finiteTs1) : -1;
  const ts1Max = finiteTs1.length ? Math.max(...finiteTs1) : 1;
  const ts1Pad = Math.max(0.05, (ts1Max - ts1Min) * 0.04);
  const ts2Min = finiteTs2.length ? Math.min(...finiteTs2) : -1;
  const ts2Max = finiteTs2.length ? Math.max(...finiteTs2) : 1;
  const ts2Pad = Math.max(0.05, (ts2Max - ts2Min) * 0.08);
  const bottomRange = [0, 1];

  const layoutMargins = {
    t: tinyLayout ? 26 : 32,
    r: tinyLayout ? 42 : 50,
    b: tinyLayout ? 18 : 24,
    l: tinyLayout ? 34 : 42,
  };

  let domains;
  if (tinyLayout) {
    domains = {
      leftX: [0.0, 0.134],
      centerX: [0.147, 0.769],
      rightX: [0.782, 0.905],
      topY: [0.85, 0.97],
      mainY: [0.22, 0.84],
      bottomY: [0.08, 0.205],
    };
  } else if (compactLayout) {
    domains = {
      leftX: [0.0, 0.13],
      centerX: [0.142, 0.778],
      rightX: [0.79, 0.91],
      topY: [0.855, 0.975],
      mainY: [0.22, 0.845],
      bottomY: [0.08, 0.205],
    };
  } else {
    domains = {
      leftX: [0.0, 0.126],
      centerX: [0.138, 0.783],
      rightX: [0.795, 0.915],
      topY: [0.86, 0.98],
      mainY: [0.22, 0.85],
      bottomY: [0.08, 0.205],
    };
  }
  const plotHeight = Math.round(
    0.88 *
      Math.max(
        tinyLayout ? 620 : 700,
        Math.min(980, Math.round(containerWidth * (tinyLayout ? 0.95 : 0.72)))
      )
  );
  const colorbarX = tinyLayout ? 0.93 : compactLayout ? 0.942 : 0.95;
  const panelOutlineColor = "#000000";
  const panelOutlineWidth = 1.4;
  const heatmapDomainWidth = domains.centerX[1] - domains.centerX[0];
  const heatmapDomainHeight = domains.mainY[1] - domains.mainY[0];
  const indicatorInsetX = heatmapDomainWidth * (tinyLayout ? 0.02 : 0.016);
  const indicatorInsetY = heatmapDomainHeight * (tinyLayout ? 0.03 : 0.022);
  const indicatorCornerX = domains.centerX[0] + indicatorInsetX;
  const indicatorCornerY = domains.mainY[1] - indicatorInsetY;
  const indicatorSpanRatio = Math.max(0.06, Math.min(0.24, fragmentSize / Math.max(1, seriesLength)));
  const indicatorWidth = heatmapDomainWidth * indicatorSpanRatio;
  const indicatorHeight = heatmapDomainHeight * indicatorSpanRatio;
  const indicatorLabelX = indicatorCornerX + indicatorWidth + heatmapDomainWidth * 0.012;
  const indicatorLabelY = indicatorCornerY - indicatorHeight * 0.02;
  const corrLabel = methodCorrelationLabel(result?.summary?.method);
  const panelFrameShapes = [
    {
      type: "rect",
      xref: "paper",
      yref: "paper",
      x0: domains.centerX[0],
      x1: domains.centerX[1],
      y0: domains.topY[0],
      y1: domains.topY[1],
      line: { color: panelOutlineColor, width: panelOutlineWidth },
      fillcolor: "rgba(0,0,0,0)",
    },
    {
      type: "rect",
      xref: "paper",
      yref: "paper",
      x0: domains.leftX[0],
      x1: domains.leftX[1],
      y0: domains.mainY[0],
      y1: domains.mainY[1],
      line: { color: panelOutlineColor, width: panelOutlineWidth },
      fillcolor: "rgba(0,0,0,0)",
    },
    {
      type: "rect",
      xref: "paper",
      yref: "paper",
      x0: domains.rightX[0],
      x1: domains.rightX[1],
      y0: domains.mainY[0],
      y1: domains.mainY[1],
      line: { color: panelOutlineColor, width: panelOutlineWidth },
      fillcolor: "rgba(0,0,0,0)",
    },
    {
      type: "rect",
      xref: "paper",
      yref: "paper",
      x0: domains.centerX[0],
      x1: domains.centerX[1],
      y0: domains.bottomY[0],
      y1: domains.bottomY[1],
      line: { color: panelOutlineColor, width: panelOutlineWidth },
      fillcolor: "rgba(0,0,0,0)",
    },
  ];

  const bracketShapes = [
    {
      type: "line",
      xref: "paper",
      yref: "paper",
      x0: indicatorCornerX,
      y0: indicatorCornerY,
      x1: indicatorCornerX + indicatorWidth,
      y1: indicatorCornerY,
      line: { color: "#111827", width: 4 },
    },
    {
      type: "line",
      xref: "paper",
      yref: "paper",
      x0: indicatorCornerX,
      y0: indicatorCornerY,
      x1: indicatorCornerX,
      y1: indicatorCornerY - indicatorHeight,
      line: { color: "#111827", width: 4 },
    },
  ];

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: seriesPositions,
      y: ts1Series,
      line: { color: "#111827", width: 2.2, simplify: false },
      xaxis: "x2",
      yaxis: "y2",
      customdata: labelSeries,
      hovertemplate: `t=%{customdata}<br>${ts1LabelRaw}=%{y:.3f}<extra></extra>`,
      name: "TS1",
    },
    {
      type: "scatter",
      mode: "lines",
      x: ts2Series,
      y: seriesPositions,
      line: { color: "#111827", width: 2.2, simplify: false },
      xaxis: "x3",
      yaxis: "y3",
      customdata: labelSeries,
      hovertemplate: `${ts2LabelRaw}=%{x:.3f}<br>t=%{customdata}<extra></extra>`,
      name: "TS2",
    },
    {
      type: "heatmap",
      x: heatX,
      y: heatY,
      z: zSig,
      customdata: heatmapHoverData,
      colorscale: RD_BU_WHITE_CENTER,
      zmin: -zAbsMax,
      zmax: zAbsMax,
      zmid: 0,
      zsmooth: false,
      hoverongaps: false,
      xaxis: "x",
      yaxis: "y",
      hovertemplate: "start_1=%{customdata[0]}<br>start_2=%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
      name: "significant",
      colorbar: {
        title: { text: corrLabel },
        x: colorbarX,
        y: (domains.mainY[0] + domains.mainY[1]) / 2,
        len: domains.mainY[1] - domains.mainY[0],
        thickness: tinyLayout ? 12 : 14,
        tickfont: { size: tinyLayout ? 10 : 12 },
      },
    },
    {
      type: "scatter",
      mode: "lines",
      x: heatX,
      y: maxPositiveByCol,
      line: { color: "#b4232d", width: 3, simplify: false },
      xaxis: "x4",
      yaxis: "y4",
      hovertemplate: "center_1=%{x}<br>max positive r=%{y:.3f}<extra></extra>",
      name: "max positive (TS1)",
    },
    {
      type: "scatter",
      mode: "lines",
      x: heatX,
      y: minNegativeAbsByCol,
      line: { color: "#1f67a5", width: 3, simplify: false },
      xaxis: "x4",
      yaxis: "y4",
      hovertemplate: "center_1=%{x}<br>max negative |r|=%{y:.3f}<extra></extra>",
      name: "min negative (TS1)",
    },
    {
      type: "scatter",
      mode: "lines",
      x: maxPositiveByRow,
      y: heatY,
      line: { color: "#b4232d", width: 3, simplify: false },
      xaxis: "x5",
      yaxis: "y5",
      hovertemplate: "max positive r=%{x:.3f}<br>center_2=%{y}<extra></extra>",
      name: "max positive (TS2)",
    },
    {
      type: "scatter",
      mode: "lines",
      x: minNegativeAbsByRow,
      y: heatY,
      line: { color: "#1f67a5", width: 3, simplify: false },
      xaxis: "x5",
      yaxis: "y5",
      hovertemplate: "max negative |r|=%{x:.3f}<br>center_2=%{y}<extra></extra>",
      name: "min negative (TS2)",
    },
    {
      type: "scatter",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "rgba(17, 24, 39, 0.24)", width: 10.5, simplify: false },
      xaxis: "x2",
      yaxis: "y2",
      hoverinfo: "skip",
      showlegend: false,
      name: "TS1 segment shadow",
    },
    {
      type: "scatter",
      mode: "lines",
      x: [],
      y: [],
      line: { color: "rgba(17, 24, 39, 0.24)", width: 10.5, simplify: false },
      xaxis: "x3",
      yaxis: "y3",
      hoverinfo: "skip",
      showlegend: false,
      name: "TS2 segment shadow",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: layoutMargins,
    showlegend: false,
    xaxis: {
      domain: domains.centerX,
      anchor: "y",
      range: [axisMin, axisMax],
      showticklabels: false,
      ticks: "",
      automargin: true,
      showgrid: false,
      gridcolor: "#d1d5db",
      zeroline: false,
      constrain: "domain",
      mirror: false,
      showline: false,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    yaxis: {
      domain: domains.mainY,
      anchor: "x",
      range: [axisMax, axisMin],
      showticklabels: false,
      ticks: "",
      automargin: true,
      showgrid: false,
      gridcolor: "#d1d5db",
      zeroline: false,
      constrain: "domain",
      mirror: false,
      showline: false,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    xaxis2: {
      domain: domains.centerX,
      anchor: "y2",
      matches: "x",
      side: "top",
      tickmode: "array",
      tickvals: topTickConfig.tickvals,
      ticktext: topTickConfig.ticktext,
      tickangle: 0,
      title: "",
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      ticks: "outside",
      ticklen: 3,
      tickfont: { size: tinyLayout ? 10 : 11 },
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    yaxis2: {
      domain: domains.topY,
      anchor: "x2",
      range: [ts1Min - ts1Pad, ts1Max + ts1Pad],
      title: "",
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      ticks: "outside",
      ticklen: 3,
      nticks: 4,
      tickfont: { size: tinyLayout ? 10 : 11 },
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    xaxis3: {
      domain: domains.leftX,
      anchor: "y3",
      range: [ts2Max + ts2Pad, ts2Min - ts2Pad],
      side: "top",
      title: "",
      tickfont: { size: tinyLayout ? 10 : 11 },
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      nticks: 4,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    yaxis3: {
      domain: domains.mainY,
      anchor: "x3",
      matches: "y",
      tickmode: "array",
      tickvals: sideTickConfig.tickvals,
      ticktext: sideTickConfig.ticktext,
      title: ts2Label,
      titlefont: { size: tinyLayout ? 11 : 13 },
      tickfont: { size: tinyLayout ? 10 : 11 },
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      ticks: "outside",
      ticklen: 3,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    xaxis4: {
      domain: domains.centerX,
      anchor: "y4",
      matches: "x",
      tickmode: "array",
      tickvals: fragTickConfig.tickvals,
      ticktext: fragTickConfig.ticktext,
      tickangle: 0,
      title: "",
      titlefont: { size: tinyLayout ? 13 : 14 },
      tickfont: { size: tinyLayout ? 10 : 11 },
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    yaxis4: {
      domain: domains.bottomY,
      anchor: "x4",
      range: bottomRange,
      side: "right",
      title: "Max |corr|",
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      tickfont: { size: tinyLayout ? 10 : 11 },
      nticks: 4,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    xaxis5: {
      domain: domains.rightX,
      anchor: "y5",
      range: [0, 1],
      side: "top",
      title: "Max |corr|",
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      tickfont: { size: tinyLayout ? 10 : 11 },
      nticks: 4,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    yaxis5: {
      domain: domains.mainY,
      anchor: "x5",
      matches: "y",
      showticklabels: false,
      ticks: "",
      title: "",
      automargin: true,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: panelOutlineColor,
      linewidth: 1.2,
    },
    annotations: [
      {
        xref: "paper",
        yref: "paper",
        x: (domains.centerX[0] + domains.centerX[1]) / 2,
        y: Math.min(0.995, domains.topY[1] + (tinyLayout ? 0.012 : 0.014)),
        text: ts1Label,
        showarrow: false,
        font: {
          family: "Space Grotesk, sans-serif",
          size: tinyLayout ? 11 : 13,
          color: "#111827",
        },
        xanchor: "center",
        yanchor: "bottom",
        align: "center",
      },
      {
        xref: "paper",
        yref: "paper",
        x: indicatorLabelX,
        y: indicatorLabelY,
        text: `s = ${fragmentSize} periods`,
        showarrow: false,
        font: {
          family: "JetBrains Mono, monospace",
          size: tinyLayout ? 11 : 12,
          color: "#111827",
        },
        xanchor: "left",
        yanchor: "middle",
        align: "left",
      },
    ],
    shapes: [...panelFrameShapes, ...bracketShapes],
    height: plotHeight,
  };

  const config = { responsive: true, displaylogo: false };

  Plotly.newPlot(containerId, traces, layout, config).then((gd) => {
    if (typeof gd.removeAllListeners === "function") {
      gd.removeAllListeners("plotly_hover");
      gd.removeAllListeners("plotly_unhover");
    }

    const clearHighlight = () => {
      Plotly.restyle(gd, { x: [[]], y: [[]] }, [7, 8]);
      Plotly.relayout(gd, { shapes: [...panelFrameShapes, ...bracketShapes] });
    };

    gd.on("plotly_hover", (event) => {
      const point = event?.points?.[0];
      if (!point || point.curveNumber !== 2) {
        return;
      }

      const center1 = Number(point.x);
      const center2 = Number(point.y);
      if (!Number.isFinite(center1) || !Number.isFinite(center2)) {
        return;
      }

      const start1Idx = centerToStartIndex(center1);
      const start2Idx = centerToStartIndex(center2);
      const stop1 = Math.min(seriesLength - 1, start1Idx + fragmentSize - 1);
      const stop2 = Math.min(seriesLength - 1, start2Idx + fragmentSize - 1);

      const ts1SegX = [];
      const ts1SegY = [];
      for (let idx = start1Idx; idx <= stop1; idx += 1) {
        ts1SegX.push(idx);
        ts1SegY.push(ts1Series[normalizePosition(idx, seriesLength - 1)]);
      }

      const ts2SegX = [];
      const ts2SegY = [];
      for (let idx = start2Idx; idx <= stop2; idx += 1) {
        ts2SegX.push(ts2Series[normalizePosition(idx, seriesLength - 1)]);
        ts2SegY.push(idx);
      }

      Plotly.restyle(gd, { x: [ts1SegX], y: [ts1SegY] }, [7]);
      Plotly.restyle(gd, { x: [ts2SegX], y: [ts2SegY] }, [8]);

      const ts1ShadowRect = {
        type: "rect",
        xref: "x2",
        yref: "y2",
        x0: start1Idx - 0.5,
        x1: stop1 + 0.5,
        y0: ts1Min - ts1Pad,
        y1: ts1Max + ts1Pad,
        line: { width: 0 },
        fillcolor: "rgba(17, 24, 39, 0.14)",
      };
      const ts2ShadowRect = {
        type: "rect",
        xref: "x3",
        yref: "y3",
        x0: ts2Min - ts2Pad,
        x1: ts2Max + ts2Pad,
        y0: start2Idx - 0.5,
        y1: stop2 + 0.5,
        line: { width: 0 },
        fillcolor: "rgba(17, 24, 39, 0.14)",
      };
      const markerRect = {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: center1 - 0.5,
        x1: center1 + 0.5,
        y0: center2 - 0.5,
        y1: center2 + 0.5,
        line: { color: "#f97316", width: 1.5 },
        fillcolor: "rgba(249, 115, 22, 0.15)",
      };
      Plotly.relayout(gd, {
        shapes: [...panelFrameShapes, ...bracketShapes, ts1ShadowRect, ts2ShadowRect, markerRect],
      });
    });

    gd.on("plotly_unhover", clearHighlight);
  });
}

function renderLagFocusExplorer(result) {
  const containerId = "two_way_lag_explorer";
  const container = twoWayLagExplorerContainer || document.getElementById(containerId);
  if (!window.Plotly) {
    if (container) {
      container.textContent = "Plotly library unavailable in this environment.";
    }
    return;
  }

  const matrices = getLagExplorerMatrices(result);
  const series = result?.series;
  const fragmentSize = Number(result?.summary?.fragment_size || 1);
  const alpha = getExplorerAlpha(result?.summary?.alpha || 0.05);
  const corrLabel = methodCorrelationLabel(result?.summary?.method);

  if (!matrices || !series?.index?.length) {
    Plotly.newPlot(
      containerId,
      [],
      {
        title: "Lag focus view unavailable",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  initializeLagExplorerState(result);
  syncLagControlsFromState();

  const lagMatrixR = matrices.lagMatrixR;
  const lagMatrixP = matrices.lagMatrixP;
  const labels = series.index.map((v) => String(v));
  const seriesLength = Math.min(series.ts1.length, series.ts2.length, labels.length);
  if (seriesLength < fragmentSize) {
    Plotly.newPlot(
      containerId,
      [],
      {
        title: "Lag focus view unavailable",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  const ts1Series = series.ts1.slice(0, seriesLength);
  const ts2Series = series.ts2.slice(0, seriesLength);
  const labelSeries = labels.slice(0, seriesLength).map((label) => formatTooltipDate(label));
  const seriesPositions = Array.from({ length: seriesLength }, (_, idx) => idx);
  const fragmentCenterOffset = (fragmentSize - 1) / 2;
  const heatXStart = lagMatrixR.x.map((v) => Number(v));
  const heatX = heatXStart.map((v) => v + fragmentCenterOffset);
  const availableLags = lagExplorerState.availableLags || [];
  if (!availableLags.length) {
    Plotly.newPlot(
      containerId,
      [],
      {
        title: "Lag focus view unavailable",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  const minLag = Number(lagExplorerState.minLag);
  const maxLag = Number(lagExplorerState.maxLag);
  const filteredLagIndices = [];
  availableLags.forEach((lagValue, idx) => {
    if (lagValue >= minLag && lagValue <= maxLag) {
      filteredLagIndices.push(idx);
    }
  });
  if (!filteredLagIndices.length) {
    Plotly.newPlot(
      containerId,
      [],
      {
        title: "Lag focus view unavailable for selected lag range",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  const filteredLags = filteredLagIndices.map((idx) => availableLags[idx]);
  const clampedSelectedLag =
    clampLagToFilteredRange(lagExplorerState.selectedLag, availableLags, minLag, maxLag) ??
    filteredLags[0];
  lagExplorerState.selectedLag = clampedSelectedLag;
  syncLagControlsFromState();

  const zMasked = filteredLagIndices.map((rowIdx) => {
    const rowR = lagMatrixR.z?.[rowIdx] || [];
    const rowP = lagMatrixP.z?.[rowIdx] || [];
    return heatX.map((_, colIdx) => {
      const rawR = rowR?.[colIdx];
      const rawP = rowP?.[colIdx];
      if (
        typeof rawR !== "number" ||
        !Number.isFinite(rawR) ||
        typeof rawP !== "number" ||
        !Number.isFinite(rawP)
      ) {
        return null;
      }
      return rawP <= alpha ? rawR : null;
    });
  });

  let zAbsMax = 0;
  zMasked.forEach((row) => {
    row.forEach((value) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        zAbsMax = Math.max(zAbsMax, Math.abs(value));
      }
    });
  });
  if (zAbsMax <= 0) {
    zAbsMax = 1;
  }

  const selectedLagGlobalIndex = availableLags.findIndex((lag) => lag === clampedSelectedLag);
  const selectedLine =
    selectedLagGlobalIndex >= 0
      ? heatX.map((_, colIdx) => {
          const rawValue = lagMatrixR?.z?.[selectedLagGlobalIndex]?.[colIdx];
          return typeof rawValue === "number" && Number.isFinite(rawValue) ? rawValue : null;
        })
      : new Array(heatX.length).fill(null);
  const axisMin = -0.5;
  const axisMax = seriesLength - 0.5;
  const maxStartIndex = Math.max(0, seriesLength - fragmentSize);
  const containerWidth = container?.clientWidth || 1200;
  const compactLayout = containerWidth < 1000;
  const tinyLayout = containerWidth < 760;
  const tsTicks = buildAxisTicks(
    labelSeries,
    tinyLayout ? 4 : compactLayout ? 5 : 7,
    seriesPositions
  );

  const finiteTs1 = ts1Series.filter(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
  const finiteTs2 = ts2Series.filter(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
  const ts1Min = finiteTs1.length ? Math.min(...finiteTs1) : -1;
  const ts1Max = finiteTs1.length ? Math.max(...finiteTs1) : 1;
  const ts1Pad = Math.max(0.05, (ts1Max - ts1Min) * 0.04);
  const ts2Min = finiteTs2.length ? Math.min(...finiteTs2) : -1;
  const ts2Max = finiteTs2.length ? Math.max(...finiteTs2) : 1;
  const ts2Pad = Math.max(0.05, (ts2Max - ts2Min) * 0.04);
  const ts1Label = normalizeSeriesLabel(result?.summary?.ts1_label, "TS1");
  const ts2Label = normalizeSeriesLabel(result?.summary?.ts2_label, "TS2");

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: seriesPositions,
      y: ts1Series,
      xaxis: "x",
      yaxis: "y",
      name: ts1Label,
      line: { color: TS1_PLOT_COLOR, width: 2.1, simplify: false },
      customdata: labelSeries,
      hovertemplate: `t=%{customdata}<br>${ts1Label}=%{y:.3f}<extra></extra>`,
    },
    {
      type: "scatter",
      mode: "lines",
      x: seriesPositions,
      y: ts2Series,
      xaxis: "x",
      yaxis: "y2",
      name: ts2Label,
      line: { color: TS2_PLOT_COLOR, width: 2.1, simplify: false },
      customdata: labelSeries,
      hovertemplate: `t=%{customdata}<br>${ts2Label}=%{y:.3f}<extra></extra>`,
    },
    {
      type: "heatmap",
      x: heatX,
      y: filteredLags,
      z: zMasked,
      xaxis: "x2",
      yaxis: "y3",
      colorscale: RD_BU_WHITE_CENTER,
      zmin: -zAbsMax,
      zmax: zAbsMax,
      zmid: 0,
      zsmooth: false,
      hoverongaps: false,
      hovertemplate: "t=%{x:.1f}<br>lag=%{y}<br>r=%{z:.3f}<extra></extra>",
      colorbar: {
        title: { text: corrLabel },
        len: 0.33,
        thickness: tinyLayout ? 12 : 14,
        y: 0.52,
        x: 0.965,
      },
      name: "Lag heatmap",
    },
    {
      type: "scatter",
      mode: "lines",
      x: heatX,
      y: selectedLine,
      xaxis: "x3",
      yaxis: "y4",
      connectgaps: false,
      line: { color: "#111827", width: 2.1, simplify: false },
      customdata: heatX.map((value) => {
        const idx = normalizePosition(Math.round(value), seriesLength - 1);
        return labelSeries[idx] || String(idx);
      }),
      hovertemplate: `t=%{customdata}<br>lag=${clampedSelectedLag}<br>${corrLabel}=%{y:.3f}<extra></extra>`,
      name: `Focused lag r (lag=${clampedSelectedLag})`,
    },
  ];

  const zeroInRange = minLag <= 0 && maxLag >= 0;
  const shapes = [];
  if (zeroInRange) {
    shapes.push({
      type: "line",
      xref: "x2",
      yref: "y3",
      x0: axisMin,
      x1: axisMax,
      y0: 0,
      y1: 0,
      line: { color: "#111827", width: 1.2, dash: "dash" },
    });
  }
  shapes.push({
    type: "line",
    xref: "x2",
    yref: "y3",
    x0: axisMin,
    x1: axisMax,
    y0: clampedSelectedLag,
    y1: clampedSelectedLag,
    line: { color: "#f97316", width: 1.4, dash: "dot" },
  });
  shapes.push({
    type: "line",
    xref: "x3",
    yref: "y4",
    x0: axisMin,
    x1: axisMax,
    y0: 0,
    y1: 0,
    line: { color: "#111827", width: 1.2, dash: "dash" },
  });
  const baseShapes = [...shapes];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: {
      t: tinyLayout ? 30 : 34,
      r: tinyLayout ? 44 : 52,
      b: tinyLayout ? 36 : 40,
      l: tinyLayout ? 48 : 54,
    },
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0,
      y: 1.02,
      bgcolor: "rgba(255,255,255,0.65)",
    },
    height: Math.round(
      Math.max(
        tinyLayout ? 620 : 700,
        Math.min(980, Math.round(containerWidth * (tinyLayout ? 1.0 : 0.78)))
      )
    ),
    xaxis: {
      domain: [0.0, 0.94],
      anchor: "y",
      range: [axisMin, axisMax],
      showticklabels: false,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    yaxis: {
      domain: [0.74, 0.98],
      anchor: "x",
      title: ts1Label,
      range: [ts1Min - ts1Pad, ts1Max + ts1Pad],
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    yaxis2: {
      domain: [0.74, 0.98],
      overlaying: "y",
      side: "right",
      title: ts2Label,
      range: [ts2Min - ts2Pad, ts2Max + ts2Pad],
      showgrid: false,
      zeroline: false,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    xaxis2: {
      domain: [0.0, 0.94],
      anchor: "y3",
      range: [axisMin, axisMax],
      matches: "x",
      showticklabels: false,
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    yaxis3: {
      domain: [0.34, 0.68],
      anchor: "x2",
      title: "Lag",
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    xaxis3: {
      domain: [0.0, 0.94],
      anchor: "y4",
      range: [axisMin, axisMax],
      matches: "x",
      tickmode: "array",
      tickvals: tsTicks.tickvals,
      ticktext: tsTicks.ticktext,
      tickangle: 0,
      title: "Series 1 time",
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    yaxis4: {
      domain: [0.08, 0.28],
      anchor: "x3",
      title: corrLabel,
      range: [-1, 1],
      showgrid: true,
      gridcolor: "#d1d5db",
      zeroline: false,
      mirror: true,
      showline: true,
      linecolor: "#000",
      linewidth: 1.1,
    },
    annotations: [
      {
        xref: "paper",
        yref: "paper",
        x: 0.0,
        y: 1.05,
        xanchor: "left",
        yanchor: "bottom",
        text: `alpha <= ${alpha.toFixed(3)}`,
        showarrow: false,
        font: {
          family: "JetBrains Mono, monospace",
          size: tinyLayout ? 10 : 11,
          color: "#334155",
        },
      },
      {
        xref: "paper",
        yref: "paper",
        x: 0.0,
        y: 0.305,
        xanchor: "left",
        yanchor: "bottom",
        text: `Focused lag = ${clampedSelectedLag}`,
        showarrow: false,
        font: {
          family: "JetBrains Mono, monospace",
          size: tinyLayout ? 10 : 11,
          color: "#334155",
        },
      },
    ],
    shapes: baseShapes,
  };

  Plotly.newPlot(containerId, traces, layout, { responsive: true, displaylogo: false }).then((gd) => {
    if (typeof gd.removeAllListeners === "function") {
      gd.removeAllListeners("plotly_hover");
      gd.removeAllListeners("plotly_unhover");
    }

    const clearHighlight = () => {
      Plotly.relayout(gd, { shapes: baseShapes });
    };

    gd.on("plotly_hover", (event) => {
      const point = event?.points?.[0];
      if (!point || point.curveNumber !== 2) {
        return;
      }
      const center1 = Number(point.x);
      const lagValue = Number(point.y);
      if (!Number.isFinite(center1) || !Number.isFinite(lagValue)) {
        return;
      }

      const start1 = Math.max(
        0,
        Math.min(maxStartIndex, Math.round(center1 - fragmentCenterOffset))
      );
      const stop1 = Math.min(seriesLength - 1, start1 + fragmentSize - 1);
      const start2 = Math.max(0, Math.min(maxStartIndex, start1 - Math.round(lagValue)));
      const stop2 = Math.min(seriesLength - 1, start2 + fragmentSize - 1);
      const topPanelY0 = 0.74;
      const topPanelY1 = 0.98;

      const ts1ShadowRect = {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: start1 - 0.5,
        x1: stop1 + 0.5,
        y0: topPanelY0,
        y1: topPanelY1,
        line: { width: 0 },
        fillcolor: "rgba(240, 111, 108, 0.18)",
      };
      const ts2ShadowRect = {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: start2 - 0.5,
        x1: stop2 + 0.5,
        y0: topPanelY0,
        y1: topPanelY1,
        line: { width: 0 },
        fillcolor: "rgba(24, 184, 189, 0.18)",
      };
      const heatMarkerRect = {
        type: "rect",
        xref: "x2",
        yref: "y3",
        x0: center1 - 0.5,
        x1: center1 + 0.5,
        y0: lagValue - 0.5,
        y1: lagValue + 0.5,
        line: { color: "#f97316", width: 1.4 },
        fillcolor: "rgba(249, 115, 22, 0.12)",
      };

      Plotly.relayout(gd, {
        shapes: [...baseShapes, ts1ShadowRect, ts2ShadowRect, heatMarkerRect],
      });
    });

    gd.on("plotly_unhover", clearHighlight);
  });
}

function renderTwoWayExplorer(result) {
  setSdcExplorerTabAvailability(result);
  initializeLagExplorerState(result);
  syncLagControlsFromState();
  if (activeSdcExplorerTab === "lag" && hasLagExplorerData(result)) {
    renderLagFocusExplorer(result);
    return;
  }
  renderTwoWayMatrixExplorer(result);
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

function renderDatasetPreview(columns, rows, { collapse = false } = {}) {
  if (!columns?.length) {
    datasetPreviewHead.innerHTML = "";
    datasetPreviewBody.innerHTML = "";
    if (datasetPreviewSummary) {
      datasetPreviewSummary.textContent = "Dataset preview";
    }
    if (datasetPreviewDetails) {
      datasetPreviewDetails.hidden = true;
      datasetPreviewDetails.open = false;
    }
    return;
  }

  if (datasetPreviewDetails) {
    datasetPreviewDetails.hidden = false;
    datasetPreviewDetails.open = !collapse;
  }
  if (datasetPreviewSummary) {
    datasetPreviewSummary.textContent = `Dataset preview: first ${rows.length} rows`;
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
  if (!submitDatasetButton) {
    return;
  }

  const datasetReady =
    !!latestDatasetId &&
    !!datasetTs1Select.value &&
    !!datasetTs2Select.value &&
    datasetTs1Select.value !== datasetTs2Select.value;
  const pasteReady = getPasteSeriesValidation({ updateMessage: false }).valid;
  const readyForActiveMode = activeInputMode === "paste" ? pasteReady : datasetReady;

  setAnalysisSettingsUnlocked(readyForActiveMode);

  if (activeInputMode === "paste") {
    submitDatasetButton.disabled = !pasteReady;
    if (pasteReady && analysisSettingsDetails) {
      analysisSettingsDetails.open = true;
    }
    return;
  }

  submitDatasetButton.disabled = !datasetReady;
}

function applyDatasetInspection(
  data,
  {
    collapsePreview = true,
    preferredDateColumn = null,
    preferredTs1Column = null,
    preferredTs2Column = null,
  } = {}
) {
  latestDatasetId = data.dataset_id;
  datasetMeta.textContent = `${data.n_rows} rows x ${data.n_columns} columns. ` +
    `Numeric: ${data.numeric_columns.join(", ") || "none"}. Date candidates: ${data.datetime_columns.join(", ") || "none"}.`;

  const tsChoices = data.numeric_columns.length ? data.numeric_columns : data.columns;
  populateSelect(datasetDateSelect, data.datetime_columns, true, "(no date column)");
  populateSelect(datasetTs1Select, tsChoices);
  populateSelect(datasetTs2Select, tsChoices);

  if (tsChoices.length >= 1) {
    datasetTs1Select.value = tsChoices[0];
  }
  if (tsChoices.length >= 2) {
    datasetTs2Select.value = tsChoices[1];
  }
  if (tsChoices.length === 1) {
    datasetTs2Select.value = tsChoices[0];
  }

  const preferredDate = preferredDateColumn || data.suggested_date_column;
  if (preferredDate && data.datetime_columns.includes(preferredDate)) {
    datasetDateSelect.value = preferredDate;
  }
  if (preferredTs1Column && tsChoices.includes(preferredTs1Column)) {
    datasetTs1Select.value = preferredTs1Column;
  }
  if (
    preferredTs2Column &&
    tsChoices.includes(preferredTs2Column) &&
    preferredTs2Column !== datasetTs1Select.value
  ) {
    datasetTs2Select.value = preferredTs2Column;
  }

  applyAdaptiveDefaultsForSeries(data.n_rows, { resetManualOverride: true });
  renderDatasetPreview(data.columns, data.preview_rows, { collapse: collapsePreview });
  updateDatasetRunAvailability();
  if (analysisSettingsDetails && analysisSettingsUnlocked) {
    analysisSettingsDetails.open = true;
  }
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

  applyDatasetInspection(data, { collapsePreview: true });
}

async function loadOniExampleDataset() {
  const token = ++datasetInspectToken;
  latestDatasetId = null;
  updateDatasetRunAvailability();
  datasetMeta.textContent = "Loading ONI sample dataset...";
  setInputMode("dataset");
  datasetFileInput.value = "";

  const response = await fetch("/api/v1/examples/oni-dataset");
  const data = await response.json();
  if (!response.ok) {
    datasetMeta.textContent = "";
    throw new Error(data.detail || "Unable to load ONI example dataset.");
  }
  if (token !== datasetInspectToken) {
    return;
  }

  applyDatasetInspection(data, {
    collapsePreview: true,
    preferredDateColumn: "date",
    preferredTs1Column: "oni_anomaly",
    preferredTs2Column: "temp_anomaly_sa",
  });
  await syncOniFilenameInFileInput();
}

async function syncOniFilenameInFileInput() {
  if (!datasetFileInput || typeof DataTransfer === "undefined" || typeof File === "undefined") {
    return;
  }
  try {
    const response = await fetch("/api/v1/examples/oni-dataset.csv");
    if (!response.ok) {
      return;
    }
    const blob = await response.blob();
    const file = new File([blob], "oni_temp_sa.csv", { type: "text/csv" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    datasetFileInput.files = transfer.files;
  } catch (_error) {
    // Best effort only. The ONI workflow is already loaded server-side.
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
  fragmentSizeInput.value = String(result.summary.fragment_size ?? fragmentSizeInput.value);
  if (Number.isFinite(Number(result?.summary?.series_length))) {
    latestValidatedSeriesLength = Math.max(0, Math.floor(Number(result.summary.series_length)));
  }
  resetLagExplorerState();
  initializeLagExplorerState(result, { reset: true });
  setSdcExplorerTabAvailability(result);
  syncLagControlsFromState();
  expandExplorerAfterFirstSuccessfulRun();
  syncPlotControlsFromSettings();
  updateHeatmapStepNotice();

  renderSummary(result.summary, result.notes, result.runtime_seconds);
  renderTwoWayExplorer(result);
  setDownloadButtons(true);
}

async function pollJob(jobId) {
  if (activePoll) {
    clearInterval(activePoll);
  }

  setDownloadButtons(false);
  latestJobId = jobId;
  analysisRunStartedAt = Date.now();
  analysisProgressPeak = 0;
  setStatus("Running analysis...");
  setStatusProgress({ percent: 0, label: "Queued" });

  activePoll = setInterval(async () => {
    try {
      const response = await fetch(`/api/v1/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Status request failed (${response.status})`);
      }

      const status = await response.json();
      updateProgress(status.progress, status.status);
      if (status.status === "queued" || status.status === "running") {
        setStatus("Running analysis...");
      }

      if (status.status === "succeeded") {
        clearInterval(activePoll);
        await fetchResult(jobId);
        setStatus("Analysis succeeded.");
      } else if (status.status === "failed") {
        clearInterval(activePoll);
        setStatusError(status.error || "Unknown job failure.");
      }
    } catch (error) {
      clearInterval(activePoll);
      setStatusError(error);
    }
  }, 1200);
}

async function submitFromText() {
  const validation = getPasteSeriesValidation({ updateMessage: true });
  if (!validation.valid) {
    throw new Error("Please fix pasted series before running analysis.");
  }
  applyAdaptiveHeatmapStep(validation.ts1.length);

  const payload = {
    ...getConfig(),
    ts1: validation.ts1,
    ts2: validation.ts2,
    ts1_label: validation.ts1Label,
    ts2_label: validation.ts2Label,
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
  if (ts1TextInput) {
    ts1TextInput.value = data.ts1.join("\n");
  }
  if (ts2TextInput) {
    ts2TextInput.value = data.ts2.join("\n");
  }
  if (ts1NameInput && !ts1NameInput.value.trim()) {
    ts1NameInput.value = "TS1";
  }
  if (ts2NameInput && !ts2NameInput.value.trim()) {
    ts2NameInput.value = "TS2";
  }
  applyAdaptiveDefaultsForSeries(data.ts1.length, { resetManualOverride: true });
  getPasteSeriesValidation({ updateMessage: true });
  updateDatasetRunAvailability();
  setStatus("Loaded synthetic example. You can run it directly.");
}

function triggerDownload(fmt) {
  if (!latestJobId) {
    return;
  }
  window.open(`/api/v1/jobs/${latestJobId}/download/${fmt}`, "_blank");
}

function triggerMapDownload(fmt) {
  if (latestMapJobId && (fmt === "png" || fmt === "nc")) {
    window.open(`/api/v1/jobs/sdc-map/${latestMapJobId}/download/${fmt}`, "_blank");
  }
}

function nearestIndex(values, target) {
  if (!values?.length) {
    return 0;
  }
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let idx = 0; idx < values.length; idx += 1) {
    const dist = Math.abs(Number(values[idx]) - Number(target));
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

function getCellSeries(explore, latIndex, lonIndex) {
  const key = `${latIndex}:${lonIndex}`;
  if (mapCellSeriesCache.has(key)) {
    return mapCellSeriesCache.get(key);
  }
  const series = (explore.field_frames || []).map((frame) => {
    const value = frame?.[latIndex]?.[lonIndex];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  });
  mapCellSeriesCache.set(key, series);
  return series;
}

function pickInitialCell(explore) {
  const latCount = explore?.lat?.length || 0;
  const lonCount = explore?.lon?.length || 0;
  if (!latCount || !lonCount) {
    return { latIndex: 0, lonIndex: 0 };
  }
  const preferred = explore?.summary?.first_valid_index;
  if (
    Array.isArray(preferred) &&
    preferred.length >= 3 &&
    Number.isFinite(preferred[1]) &&
    Number.isFinite(preferred[2])
  ) {
    return {
      latIndex: Math.max(0, Math.min(latCount - 1, Math.round(Number(preferred[1])))),
      lonIndex: Math.max(0, Math.min(lonCount - 1, Math.round(Number(preferred[2])))),
    };
  }
  return {
    latIndex: Math.floor(latCount / 2),
    lonIndex: Math.floor(lonCount / 2),
  };
}

function renderMapSummary(summary, runtimeSeconds = null) {
  if (!mapSummary) {
    return;
  }
  const keys = [
    ["driver_dataset", "Driver"],
    ["field_dataset", "Field"],
    ["peak_date", "Peak date"],
    ["time_start", "Time start"],
    ["time_end", "Time end"],
    ["fragment_size", "Fragment"],
    ["alpha", "Alpha"],
    ["top_fraction", "Top fraction"],
    ["n_time", "Time points"],
    ["n_lat", "Lat points"],
    ["n_lon", "Lon points"],
    ["valid_cells", "Valid cells"],
    ["valid_values", "Valid values"],
    ["full_bounds_selected", "Full bounds"],
    ["mean_abs_corr", "Mean |corr|"],
  ];
  const items = keys
    .filter(([key]) => Object.prototype.hasOwnProperty.call(summary, key))
    .map(([key, label]) => {
      const value = summary[key];
      let pretty;
      if (typeof value === "boolean") {
        pretty = value ? "yes" : "no";
      } else if (typeof value === "number") {
        pretty = value.toFixed(4).replace(/\.0000$/, "");
      } else {
        pretty = value === null || value === undefined ? "NA" : String(value);
      }
      return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${pretty}</span></div>`;
    });
  if (typeof runtimeSeconds === "number" && Number.isFinite(runtimeSeconds)) {
    items.push(
      `<div class="stat"><span class="stat-label">Runtime (s)</span><span class="stat-value">${runtimeSeconds.toFixed(2)}</span></div>`
    );
  }
  mapSummary.innerHTML = items.join("");
}

function setSelectedCellSummary(explore, latIndex, lonIndex) {
  if (!mapSelectedCellText) {
    return;
  }
  const latValue = explore?.lat?.[latIndex];
  const lonValue = explore?.lon?.[lonIndex];
  const date = explore?.time_index?.[mapSelectedTimeIndex] || "NA";
  mapSelectedCellText.textContent = `Selected grid cell: lat=${Number(latValue).toFixed(2)}, lon=${Number(lonValue).toFixed(2)} at ${date}. Comparison plot uses dual y-axes (driver left, field right).`;
}

function updateExploreSliderLabel(explore) {
  if (!mapTimeSliderLabel) {
    return;
  }
  const date = explore?.time_index?.[mapSelectedTimeIndex] || "NA";
  mapTimeSliderLabel.textContent = date;
}

function pickExploreColorSettings(minValue, maxValue, saturationAbs = null) {
  const min = Number(minValue);
  const max = Number(maxValue);
  const sat = Number(saturationAbs);
  const hasSat = Number.isFinite(sat) && sat > 0;

  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    const defaultMax = Number.isFinite(max) ? Math.abs(max) : 1;
    const usedSat = hasSat ? sat : Math.max(defaultMax, 1);
    return {
      colorscale: "cividis",
      zmin: Number.isFinite(min) ? Math.max(min, -usedSat) : 0,
      zmax: usedSat,
      zmid: undefined,
      colorbarTitle: "Field value",
    };
  }

  const hasPositive = max > 0;
  const hasNegative = min < 0;
  const span = Math.max(1e-9, max - min);
  const centerSkew = Math.abs((max + min) / span);
  const useDiverging = hasPositive && hasNegative && centerSkew <= 0.35;

  if (useDiverging) {
    const absBound = hasSat ? sat : Math.max(Math.abs(min), Math.abs(max), 1e-6);
    return {
      colorscale: RD_BU_WHITE_CENTER,
      zmin: -absBound,
      zmax: absBound,
      zmid: 0,
      colorbarTitle: "Field anomaly",
    };
  }

  const clippedMin = hasSat ? Math.max(min, -sat) : min;
  const clippedMax = hasSat ? Math.min(max, sat) : max;
  const safeMin = clippedMin < clippedMax ? clippedMin : min;
  const safeMax = clippedMin < clippedMax ? clippedMax : max;

  return {
    colorscale: "cividis",
    zmin: safeMin,
    zmax: safeMax,
    zmid: undefined,
    colorbarTitle: "Field value",
  };
}

async function renderMapExploration(explore) {
  if (!window.Plotly || !mapPlot) {
    return;
  }
  const timeIndex = explore.time_index || [];
  const driverValues = (explore.driver_values || []).map((value) =>
    typeof value === "number" && Number.isFinite(value) ? value : null
  );
  const frames = explore.field_frames || [];
  const lat = explore.lat || [];
  const lon = explore.lon || [];
  const coastlineLon = explore.coastline?.lon || [];
  const coastlineLat = explore.coastline?.lat || [];

  mapSelectedTimeIndex = Math.max(0, Math.min(timeIndex.length - 1, mapSelectedTimeIndex));
  if (!mapSelectedCell || mapSelectedCell.latIndex >= lat.length || mapSelectedCell.lonIndex >= lon.length) {
    mapSelectedCell = pickInitialCell(explore);
  }

  const selectedSeriesRaw = getCellSeries(explore, mapSelectedCell.latIndex, mapSelectedCell.lonIndex);
  const selectedSeries = selectedSeriesRaw.map((value) =>
    typeof value === "number" && Number.isFinite(value) ? value : null
  );
  const { driverLabel, fieldLabel } = getMapAxisLabels(explore);
  const currentFrame = frames[mapSelectedTimeIndex] || [];
  const currentDate = timeIndex[mapSelectedTimeIndex] || "NA";

  let fixedMin = Number(explore?.summary?.field_value_min);
  let fixedMax = Number(explore?.summary?.field_value_max);
  if (!Number.isFinite(fixedMin) || !Number.isFinite(fixedMax)) {
    let localMin = Infinity;
    let localMax = -Infinity;
    for (const frame of frames) {
      for (const row of frame || []) {
        for (const value of row || []) {
          if (typeof value === "number" && Number.isFinite(value)) {
            localMin = Math.min(localMin, value);
            localMax = Math.max(localMax, value);
          }
        }
      }
    }
    fixedMin = Number.isFinite(localMin) ? localMin : -1;
    fixedMax = Number.isFinite(localMax) ? localMax : 1;
  }
  const autoAbsSaturation = Math.max(Math.abs(fixedMin), Math.abs(fixedMax), 1e-6);
  const customSaturation = parseMapSaturationValue();
  const usedSaturation = customSaturation ?? autoAbsSaturation;
  if (mapSaturationMeta) {
    mapSaturationMeta.textContent = customSaturation
      ? "Manual saturation override."
      : "Auto from field range.";
  }
  const colorSettings = pickExploreColorSettings(fixedMin, fixedMax, usedSaturation);
  const fieldLatMin = Number(explore?.summary?.field_lat_min);
  const fieldLatMax = Number(explore?.summary?.field_lat_max);
  const fieldLonMin = Number(explore?.summary?.field_lon_min);
  const fieldLonMax = Number(explore?.summary?.field_lon_max);
  const latPadding = 2;
  const lonPadding = 3;
  const latFallbackMin = lat.length ? Math.min(...lat) : -90;
  const latFallbackMax = lat.length ? Math.max(...lat) : 90;
  const lonFallbackMin = lon.length ? Math.min(...lon) : -180;
  const lonFallbackMax = lon.length ? Math.max(...lon) : 180;
  const mapLatRange =
    Number.isFinite(fieldLatMin) && Number.isFinite(fieldLatMax)
      ? [fieldLatMin - latPadding, fieldLatMax + latPadding]
      : [latFallbackMin, latFallbackMax];
  const mapLonRange =
    Number.isFinite(fieldLonMin) && Number.isFinite(fieldLonMax)
      ? [fieldLonMin - lonPadding, fieldLonMax + lonPadding]
      : [lonFallbackMin, lonFallbackMax];
  let selectedBounds = null;
  try {
    const bounds = getMapBoundsSelection();
    if (bounds.hasBounds) {
      selectedBounds = bounds;
    }
  } catch (_error) {
    selectedBounds = null;
  }
  const mapDomainX = [0, 1];
  const seriesDomainX = [0, 1];
  const mapDomainY = [0.56, 1];
  const seriesDomainY = [0, 0.4];

  const traces = [
    {
      type: "heatmap",
      x: lon,
      y: lat,
      z: currentFrame,
      xaxis: "x",
      yaxis: "y",
      colorscale: colorSettings.colorscale,
      zmin: colorSettings.zmin,
      zmax: colorSettings.zmax,
      zmid: colorSettings.zmid,
      hoverongaps: true,
      colorbar: {
        title: { text: colorSettings.colorbarTitle },
        thickness: 12,
        len: 0.34,
        y: 0.76,
      },
      hovertemplate: "Lon=%{x:.2f}<br>Lat=%{y:.2f}<br>Value=%{z:.3f}<extra></extra>",
      name: "Field frame",
    },
    {
      type: "scatter",
      mode: "lines",
      x: coastlineLon,
      y: coastlineLat,
      xaxis: "x",
      yaxis: "y",
      line: { color: "#111827", width: 1.1, simplify: false },
      hoverinfo: "skip",
      showlegend: false,
      name: "Coastline",
    },
    {
      type: "scatter",
      mode: "lines",
      x: timeIndex,
      y: driverValues,
      xaxis: "x2",
      yaxis: "y2",
      line: { color: "#111827", width: 2.1, simplify: false },
      name: driverLabel,
      hovertemplate: `Date=%{x}<br>${driverLabel}=%{y:.3f}<extra></extra>`,
    },
    {
      type: "scatter",
      mode: "lines",
      x: timeIndex,
      y: selectedSeries,
      xaxis: "x2",
      yaxis: "y3",
      line: { color: "#0d9488", width: 2.1, simplify: false },
      name: `${fieldLabel} (selected cell)`,
      hovertemplate: `Date=%{x}<br>${fieldLabel}=%{y:.3f}<extra></extra>`,
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 30, r: 42, b: 36, l: 54 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.03 },
    height: 860,
    xaxis: {
      domain: mapDomainX,
      anchor: "y",
      title: "Longitude",
      showgrid: false,
      tickfont: { size: 10 },
      range: mapLonRange,
    },
    yaxis: {
      domain: mapDomainY,
      anchor: "x",
      title: "Latitude",
      showgrid: false,
      tickfont: { size: 10 },
      range: mapLatRange,
      scaleanchor: "x",
      scaleratio: 1,
    },
    xaxis2: {
      domain: seriesDomainX,
      anchor: "y2",
      title: "Date",
      showgrid: true,
      gridcolor: "#d1d5db",
      tickfont: { size: 10 },
    },
    yaxis2: {
      domain: seriesDomainY,
      anchor: "x2",
      title: driverLabel,
      showgrid: true,
      gridcolor: "#d1d5db",
      tickfont: { size: 10 },
    },
    yaxis3: {
      domain: seriesDomainY,
      anchor: "x2",
      overlaying: "y2",
      side: "right",
      title: fieldLabel,
      showgrid: false,
      tickfont: { size: 10, color: "#0d9488" },
    },
    annotations: [
      {
        xref: "paper",
        yref: "paper",
        x: (mapDomainX[0] + mapDomainX[1]) / 2,
        y: Math.min(1.04, mapDomainY[1] + 0.03),
        showarrow: false,
        text: `Field map @ ${currentDate}`,
        font: { size: 12, color: "#111827" },
      },
      {
        xref: "paper",
        yref: "paper",
        x: (seriesDomainX[0] + seriesDomainX[1]) / 2,
        y: Math.min(1.04, seriesDomainY[1] + 0.03),
        showarrow: false,
        text: "Selected grid cell vs driver (dual axis)",
        font: { size: 12, color: "#111827" },
      },
    ],
    dragmode: "drawrect",
    newshape: {
      line: { color: "#f97316", width: 2, dash: "dash" },
      fillcolor: "rgba(249,115,22,0.06)",
      layer: "above",
    },
    shapes: selectedBounds ? [createMapBoundsShape(selectedBounds)] : [],
  };

  await Plotly.newPlot(mapPlot, traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToAdd: ["drawrect", "eraseshape"],
  });
  mapPlot.dataset.resultMapInitialized = "0";
  if (typeof mapPlot.removeAllListeners === "function") {
    mapPlot.removeAllListeners("plotly_click");
    mapPlot.removeAllListeners("plotly_relayout");
  }
  mapPlot.on("plotly_click", (event) => {
    const point = event?.points?.[0];
    if (!point || point.curveNumber !== 0) {
      return;
    }
    const nextLon = Number(point.x);
    const nextLat = Number(point.y);
    if (!Number.isFinite(nextLon) || !Number.isFinite(nextLat)) {
      return;
    }
    mapSelectedCell = {
      latIndex: nearestIndex(lat, nextLat),
      lonIndex: nearestIndex(lon, nextLon),
    };
    const nextSeries = getCellSeries(explore, mapSelectedCell.latIndex, mapSelectedCell.lonIndex);
    Plotly.restyle(mapPlot, { y: [nextSeries] }, [3]);
    setSelectedCellSummary(explore, mapSelectedCell.latIndex, mapSelectedCell.lonIndex);
  });
  mapPlot.on("plotly_relayout", async (eventData) => {
    if (mapBoundsShapeSync || mapPhase !== "explore") {
      return;
    }
    const nextBounds = parseBoundsFromRelayout(eventData, mapPlot.layout);
    if (nextBounds === undefined) {
      return;
    }
    if (nextBounds === null) {
      setMapBoundsInputs(null);
      invalidateMapPreparation();
      return;
    }
    setMapBoundsInputs(nextBounds);
    invalidateMapPreparation();
    await syncExploreBoundsShape(nextBounds);
    // After drawing bounds, switch to pan so users can click heatmap cells to inspect series.
    mapBoundsShapeSync = true;
    try {
      await Plotly.relayout(mapPlot, { dragmode: "pan" });
    } finally {
      mapBoundsShapeSync = false;
    }
    setMapStatus("Bounds updated. Click heatmap cells to inspect local series.", false);
  });

  if (mapTimeSlider) {
    mapTimeSlider.min = "0";
    mapTimeSlider.max = String(Math.max(0, timeIndex.length - 1));
    mapTimeSlider.step = "1";
    mapTimeSlider.value = String(mapSelectedTimeIndex);
  }
  if (mapTimePlayButton) {
    mapTimePlayButton.disabled = timeIndex.length <= 1;
  }
  updateExploreSliderLabel(explore);
  setSelectedCellSummary(explore, mapSelectedCell.latIndex, mapSelectedCell.lonIndex);
}

function updateExploreFrame(index) {
  if (mapPhase !== "explore" || !latestMapExplore || !window.Plotly || !mapPlot?.data?.length) {
    return;
  }
  const maxIndex = Math.max(0, (latestMapExplore.time_index?.length || 1) - 1);
  mapSelectedTimeIndex = Math.max(0, Math.min(maxIndex, Math.round(Number(index) || 0)));
  const frame = latestMapExplore.field_frames?.[mapSelectedTimeIndex] || [];
  const dateLabel = latestMapExplore.time_index?.[mapSelectedTimeIndex] || "NA";
  Plotly.restyle(mapPlot, { z: [frame] }, [0]);
  Plotly.relayout(mapPlot, { "annotations[0].text": `Field map @ ${dateLabel}` });
  if (mapTimeSlider) {
    mapTimeSlider.value = String(mapSelectedTimeIndex);
  }
  if (mapTimePlaying && mapSelectedTimeIndex >= maxIndex) {
    stopMapTimePlayback();
  }
  updateExploreSliderLabel(latestMapExplore);
  if (mapSelectedCell) {
    setSelectedCellSummary(latestMapExplore, mapSelectedCell.latIndex, mapSelectedCell.lonIndex);
  }
}

function getVisibleMapResultLayers(layerMaps) {
  const layers = layerMaps?.layers || [];
  return layers.filter((layer) => layer && layer.key !== "dominant_sign");
}

function resetMapPlotCanvas() {
  if (!window.Plotly || !mapPlot) {
    return;
  }
  try {
    window.Plotly.purge(mapPlot);
  } catch (_error) {
    // Best effort; `newPlot`/`react` below will still rebuild.
  }
  mapPlot.innerHTML = "";
}

function computeMapResultLayoutMetrics(mapLonRange, mapLatRange) {
  const lonSpan = Math.max(
    1e-6,
    Math.abs(Number(mapLonRange?.[1] ?? 1) - Number(mapLonRange?.[0] ?? 0))
  );
  const latSpan = Math.max(
    1e-6,
    Math.abs(Number(mapLatRange?.[1] ?? 1) - Number(mapLatRange?.[0] ?? 0))
  );
  const panelWidth = Math.max(560, Math.round(mapPlot?.clientWidth || 920));
  const margins = { t: 78, r: 24, b: 44, l: 56 };
  const innerWidth = Math.max(360, panelWidth - margins.l - margins.r);
  const mapHeight = innerWidth * (latSpan / lonSpan);
  const height = Math.max(440, Math.min(760, Math.round(mapHeight + margins.t + margins.b + 12)));
  return { margins, height };
}

function renderMapResultTabs(layerMaps) {
  if (!mapResultTabs) {
    return;
  }
  const layers = getVisibleMapResultLayers(layerMaps);
  mapResultTabs.innerHTML = "";
  layers.forEach((layer, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `map-tab-btn${index === mapActiveTab ? " is-active" : ""}`;
    btn.textContent = layer.label || layer.key || `Layer ${index + 1}`;
    btn.dataset.index = String(index);
    btn.addEventListener("click", () => {
      mapActiveTab = index;
      renderActiveMapResultLayer();
    });
    mapResultTabs.appendChild(btn);
  });
}

async function renderActiveMapResultLayer() {
  if (!latestMapResult?.layer_maps || !window.Plotly || !mapPlot) {
    return;
  }
  const layerMaps = latestMapResult.layer_maps;
  const layers = getVisibleMapResultLayers(layerMaps);
  if (!layers.length) {
    return;
  }
  mapActiveTab = Math.max(0, Math.min(layers.length - 1, mapActiveTab));
  if (mapResultTabs) {
    Array.from(mapResultTabs.querySelectorAll(".map-tab-btn")).forEach((btn, idx) => {
      btn.classList.toggle("is-active", idx === mapActiveTab);
      btn.setAttribute("aria-selected", idx === mapActiveTab ? "true" : "false");
    });
  }
  const layer = layers[mapActiveTab];
  const summary = latestMapResult?.summary || {};
  const fieldLatMin = Number(summary.field_lat_min);
  const fieldLatMax = Number(summary.field_lat_max);
  const fieldLonMin = Number(summary.field_lon_min);
  const fieldLonMax = Number(summary.field_lon_max);
  const latValues = Array.isArray(layerMaps.lat) ? layerMaps.lat : [];
  const lonValues = Array.isArray(layerMaps.lon) ? layerMaps.lon : [];
  const latFallbackMin = latValues.length ? Math.min(...latValues) : -90;
  const latFallbackMax = latValues.length ? Math.max(...latValues) : 90;
  const lonFallbackMin = lonValues.length ? Math.min(...lonValues) : -180;
  const lonFallbackMax = lonValues.length ? Math.max(...lonValues) : 180;
  const latPadding = 2;
  const lonPadding = 3;
  const mapLatRange =
    Number.isFinite(fieldLatMin) && Number.isFinite(fieldLatMax)
      ? [fieldLatMin - latPadding, fieldLatMax + latPadding]
      : [latFallbackMin, latFallbackMax];
  const mapLonRange =
    Number.isFinite(fieldLonMin) && Number.isFinite(fieldLonMax)
      ? [fieldLonMin - lonPadding, fieldLonMax + lonPadding]
      : [lonFallbackMin, lonFallbackMax];
  const layoutMetrics = computeMapResultLayoutMetrics(mapLonRange, mapLatRange);
  const traces = [
    {
      type: "heatmap",
      x: layerMaps.lon,
      y: layerMaps.lat,
      z: layer.values,
      xaxis: "x",
      yaxis: "y",
      colorscale: layer.colorscale,
      zmin: layer.zmin,
      zmax: layer.zmax,
      zmid: layer.zmin === null || layer.zmax === null ? undefined : 0,
      hoverongaps: true,
      colorbar: {
        orientation: "h",
        title: { text: layer.label || layer.key, side: "top" },
        thickness: 11,
        len: 0.66,
        x: 0.5,
        xanchor: "center",
        y: 1.03,
        yanchor: "bottom",
        tickformat: ".2f",
      },
      hovertemplate: "Lon=%{x:.2f}<br>Lat=%{y:.2f}<br>Value=%{z:.3f}<extra></extra>",
      name: layer.label,
    },
    {
      type: "scatter",
      mode: "lines",
      x: layerMaps.coastline?.lon || [],
      y: layerMaps.coastline?.lat || [],
      xaxis: "x",
      yaxis: "y",
      line: { color: "#111827", width: 1.1, simplify: false },
      hoverinfo: "skip",
      showlegend: false,
      name: "Coastline",
    },
  ];
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: layoutMetrics.margins,
    height: layoutMetrics.height,
    showlegend: false,
    xaxis: {
      title: "Longitude",
      showgrid: false,
      tickfont: { size: 10 },
      range: mapLonRange,
      domain: [0, 1],
      constrain: "domain",
    },
    yaxis: {
      title: "Latitude",
      showgrid: false,
      tickfont: { size: 10 },
      range: mapLatRange,
      scaleanchor: "x",
      scaleratio: 1,
      domain: [0, 1],
      constrain: "domain",
    },
  };
  await Plotly.newPlot(mapPlot, traces, layout, { responsive: true, displaylogo: false });
  mapPlot.dataset.resultMapInitialized = "1";
  if (typeof mapPlot.removeAllListeners === "function") {
    mapPlot.removeAllListeners("plotly_click");
  }
}

async function fetchMapResult(jobId) {
  const response = await fetch(`/api/v1/jobs/sdc-map/${jobId}/result`);
  if (!response.ok) {
    throw new Error(`Could not fetch map result: ${response.status}`);
  }
  const payload = await response.json();
  const result = payload.result;
  latestMapResult = result;
  latestMapJobId = jobId;
  mapActiveTab = 0;
  setMapPhase("results");
  resetMapPlotCanvas();
  renderMapSummary(result.summary, result.runtime_seconds);
  renderMapResultTabs(result.layer_maps);
  await renderActiveMapResultLayer();
  mapDownloadsEnabled = true;
  setMapDownloadButtons(true);
  setMapStatus(`SDC map ready in ${formatDurationSeconds(result.runtime_seconds)}.`);
}

async function submitMapExplore() {
  if (mapLoadBusy || mapRunBusy) {
    return;
  }
  stopMapTimePlayback();
  setWorkflowTab("map");
  setMapLoadBusy(true);
  mapDownloadsEnabled = false;
  latestMapJobId = null;
  latestMapResult = null;
  resetMapPlotCanvas();
  setMapDownloadButtons(false);
  setMapStatus("Loading driver and gridded dataset...");
  setMapProgress({
    percent: 15,
    label: "Loading datasets",
    etaText: "Preparing exploration view...",
  });
  try {
    const config = getMapConfig();
    const response = await fetch("/api/v1/sdc-map/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load map exploration.");
    }
    latestMapExplore = data.result;
    const summary = latestMapExplore?.summary || {};
    if (mapDatasetCatalog?.drivers?.length) {
      const driverKey = String(summary.driver_dataset || mapDriverDatasetInput?.value || "");
      const driverEntry = mapDatasetCatalog.drivers.find((item) => item.key === driverKey);
      if (driverEntry) {
        driverEntry.loaded_time_start = summary.time_start || driverEntry.loaded_time_start;
        driverEntry.loaded_time_end = summary.time_end || driverEntry.loaded_time_end;
        if (summary.peak_date) {
          driverEntry.peak_date = summary.peak_date;
        }
      }
    }
    if (mapDatasetCatalog?.fields?.length) {
      const fieldKey = String(summary.field_dataset || mapFieldDatasetInput?.value || "");
      const fieldEntry = mapDatasetCatalog.fields.find((item) => item.key === fieldKey);
      if (fieldEntry) {
        fieldEntry.loaded_time_start = summary.time_start || fieldEntry.loaded_time_start;
        fieldEntry.loaded_time_end = summary.time_end || fieldEntry.loaded_time_end;
        if (Number.isFinite(Number(summary.n_lat))) {
          fieldEntry.n_lat = Number(summary.n_lat);
        }
        if (Number.isFinite(Number(summary.n_lon))) {
          fieldEntry.n_lon = Number(summary.n_lon);
        }
        if (Number.isFinite(Number(summary.field_lat_min))) {
          fieldEntry.lat_min = Number(summary.field_lat_min);
        }
        if (Number.isFinite(Number(summary.field_lat_max))) {
          fieldEntry.lat_max = Number(summary.field_lat_max);
        }
        if (Number.isFinite(Number(summary.field_lon_min))) {
          fieldEntry.lon_min = Number(summary.field_lon_min);
        }
        if (Number.isFinite(Number(summary.field_lon_max))) {
          fieldEntry.lon_max = Number(summary.field_lon_max);
        }
      }
    }
    renderMapSelectorMetadata();
    renderMapDatasetDocs();
    if (mapPeakDateInput) {
      mapPeakDateInput.value = latestMapExplore?.summary?.peak_date || mapPeakDateInput.value;
    }
    if (mapTimeStartInput) {
      mapTimeStartInput.value = latestMapExplore?.summary?.time_start || mapTimeStartInput.value;
    }
    if (mapTimeEndInput) {
      mapTimeEndInput.value = latestMapExplore?.summary?.time_end || mapTimeEndInput.value;
    }
    if (latestMapExplore?.summary?.full_bounds_selected) {
      setMapBoundsInputs(null);
    }
    mapCellSeriesCache = new Map();
    mapSelectedCell = pickInitialCell(latestMapExplore);
    mapSelectedTimeIndex = Math.floor((latestMapExplore.time_index?.length || 1) / 2);
    setMapPhase("explore");
    await renderMapExploration(latestMapExplore);
    renderMapSummary(latestMapExplore.summary, null);
    const fullBounds = Boolean(latestMapExplore?.summary?.full_bounds_selected);
    setMapStatus(
      fullBounds
        ? "Exploration ready. No bounds selected: full map run will be expensive. Draw bounds on the map to constrain it."
        : "Exploration ready. Move the date slider, click grid cells, and run SDC map."
    );
    setMapProgress({
      percent: 100,
      label: "Exploration ready",
      etaText: "Datasets are loaded in memory.",
    });
  } finally {
    setMapLoadBusy(false);
  }
}

async function pollMapJob(jobId) {
  if (mapActivePoll) {
    clearInterval(mapActivePoll);
  }
  mapDownloadsEnabled = false;
  setMapDownloadButtons(false);
  latestMapJobId = jobId;
  mapRunStartedAt = Date.now();
  setMapRunBusy(true);
  updateMapProgress({ current: 0, total: 1, description: "Queued" }, "queued");
  setMapStatus("Submitting map job...");

  mapActivePoll = setInterval(async () => {
    try {
      const response = await fetch(`/api/v1/jobs/sdc-map/${jobId}`);
      if (!response.ok) {
        throw new Error(`Map status request failed (${response.status})`);
      }

      const status = await response.json();
      updateMapProgress(status.progress || {}, status.status);
      if (status.status === "queued" || status.status === "running") {
        const progress = status.progress || {};
        const description = progress.description ? String(progress.description) : "Running";
        setMapStatus(`Running: ${description}`);
      }

      if (status.status === "succeeded") {
        clearInterval(mapActivePoll);
        mapActivePoll = null;
        await fetchMapResult(jobId);
        updateMapProgress(
          status.progress || { current: 1, total: 1, description: "Completed" },
          "succeeded"
        );
        setMapRunBusy(false);
      } else if (status.status === "failed") {
        clearInterval(mapActivePoll);
        mapActivePoll = null;
        updateMapProgress(
          status.progress || { current: 0, total: 1, description: "Failed" },
          "failed"
        );
        setMapStatus(status.error || "Map job failed.", true);
        setMapRunBusy(false);
      }
    } catch (error) {
      clearInterval(mapActivePoll);
      mapActivePoll = null;
      updateMapProgress({ current: 0, total: 1, description: "Failed" }, "failed");
      setMapStatus(String(error), true);
      setMapRunBusy(false);
    }
  }, 700);
}

async function submitMapJob() {
  if (mapRunBusy || mapLoadBusy) {
    return;
  }
  if (!latestMapExplore) {
    throw new Error("Load and explore driver + dataset first.");
  }
  stopMapTimePlayback();
  setWorkflowTab("map");
  const config = getMapConfig();
  const response = await fetch("/api/v1/jobs/sdc-map", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  const data = await response.json();
  if (!response.ok) {
    setMapRunBusy(false);
    throw new Error(data.detail || "Unable to submit SDC map job.");
  }
  await pollMapJob(data.job_id);
}

function invalidateMapPreparation({ hard = false } = {}) {
  if (hard) {
    latestMapExplore = null;
  }
  refreshMapRunButtonState();
  if (mapPhase === "explore") {
    setMapStatus(
      hard
        ? "Map settings changed. Reload driver + dataset to refresh exploration."
        : "Map settings updated. You can run SDC map with current settings."
    );
  } else if (mapPhase === "results") {
    setMapStatus(
      hard
        ? "Map settings changed. Reload driver + dataset before running again."
        : "Map settings updated. Run SDC map again to refresh results."
    );
  }
}

function attachHandlers() {
  if (modeDatasetButton) {
    modeDatasetButton.addEventListener("click", () => setInputMode("dataset"));
  }
  if (modePasteButton) {
    modePasteButton.addEventListener("click", () => setInputMode("paste"));
  }
  if (workflowTabSdcButton) {
    workflowTabSdcButton.addEventListener("click", () => setWorkflowTab("sdc"));
  }
  if (workflowTabMapButton) {
    workflowTabMapButton.addEventListener("click", () => setWorkflowTab("map"));
  }
  if (sdcExplorerMatrixTabButton) {
    sdcExplorerMatrixTabButton.addEventListener("click", () => {
      setSdcExplorerTab("matrix");
    });
  }
  if (sdcExplorerLagTabButton) {
    sdcExplorerLagTabButton.addEventListener("click", () => {
      if (sdcExplorerLagTabButton.disabled) {
        return;
      }
      setSdcExplorerTab("lag");
    });
  }

  document.getElementById("load_example").addEventListener("click", async () => {
    try {
      setInputMode("paste");
      await loadExample();
    } catch (error) {
      setStatusError(error);
    }
  });

  document.getElementById("submit_dataset").addEventListener("click", async () => {
    try {
      await rerunActiveWorkflow();
    } catch (error) {
      setStatusError(error);
    }
  });

  if (loadOniExampleButton) {
    loadOniExampleButton.addEventListener("click", async () => {
      try {
        await loadOniExampleDataset();
        setStatus("ONI example loaded. Configure settings and run.");
      } catch (error) {
        setStatusError(error);
      }
    });
  }

  datasetFileInput.addEventListener("change", async () => {
    if (!datasetFileInput.files?.length) {
      latestDatasetId = null;
      datasetMeta.textContent = "";
      renderDatasetPreview([], []);
      updateDatasetRunAvailability();
      return;
    }
    try {
      await inspectDataset();
      setStatus("Dataset inspected. Configure columns and run.");
    } catch (error) {
      setStatusError(error);
    }
  });

  if (analysisSettingsDetails) {
    analysisSettingsDetails.addEventListener("toggle", () => {
      if (!analysisSettingsUnlocked && analysisSettingsDetails.open) {
        analysisSettingsDetails.open = false;
        setStatus("Provide a valid dataset or pasted series to unlock analysis settings.");
      }
    });
  }

  datasetDateSelect.addEventListener("change", updateDatasetRunAvailability);
  datasetTs1Select.addEventListener("change", updateDatasetRunAvailability);
  datasetTs2Select.addEventListener("change", updateDatasetRunAvailability);

  const schedulePasteValidation = () => {
    if (pasteValidationTimer) {
      clearTimeout(pasteValidationTimer);
    }
    pasteValidationTimer = setTimeout(() => {
      const validation = getPasteSeriesValidation({ updateMessage: activeInputMode === "paste" });
      if (validation.valid) {
        applyAdaptiveHeatmapStep(validation.ts1.length);
      } else {
        updateHeatmapStepNotice();
      }
      updateDatasetRunAvailability();
    }, 140);
  };

  if (ts1TextInput) {
    ts1TextInput.addEventListener("input", schedulePasteValidation);
  }
  if (ts2TextInput) {
    ts2TextInput.addEventListener("input", schedulePasteValidation);
  }
  if (ts1NameInput) {
    ts1NameInput.addEventListener("input", schedulePasteValidation);
  }
  if (ts2NameInput) {
    ts2NameInput.addEventListener("input", schedulePasteValidation);
  }

  fragmentSizeInput.addEventListener("input", () => {
    syncPlotControlsFromSettings();
    updateHeatmapStepNotice();
  });
  heatmapStepInput.addEventListener("input", () => {
    heatmapStepManuallyOverridden = true;
    syncPlotControlsFromSettings();
    updateHeatmapStepNotice();
  });

  if (plotAlphaInput) {
    plotAlphaInput.addEventListener("input", () => {
      if (plotAlphaValue) {
        plotAlphaValue.textContent = formatAlphaValue(plotAlphaInput.value);
      }
      if (!latestResult) {
        return;
      }
      if (alphaRenderTimer) {
        clearTimeout(alphaRenderTimer);
      }
      alphaRenderTimer = setTimeout(() => renderTwoWayExplorer(latestResult), 180);
    });
  }

  if (plotApplyButton) {
    plotApplyButton.addEventListener("click", async () => {
      try {
        await applyPlotControls();
      } catch (error) {
        setStatusError(error);
      }
    });
  }

  if (plotFragmentInput) {
    plotFragmentInput.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      try {
        await applyPlotControls();
      } catch (error) {
        setStatusError(error);
      }
    });
  }

  if (plotHeatmapStepInput) {
    plotHeatmapStepInput.addEventListener("input", () => {
      heatmapStepManuallyOverridden = true;
      updateHeatmapStepNotice();
    });
    plotHeatmapStepInput.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      try {
        await applyPlotControls();
      } catch (error) {
        setStatusError(error);
      }
    });
  }

  if (plotLagMinInput) {
    plotLagMinInput.addEventListener("change", () => {
      applyLagRangeFromInputs();
    });
  }
  if (plotLagMaxInput) {
    plotLagMaxInput.addEventListener("change", () => {
      applyLagRangeFromInputs();
    });
  }
  if (plotLagFocusSlider) {
    plotLagFocusSlider.addEventListener("input", () => {
      applyFocusedLagFromInput(plotLagFocusSlider.value);
    });
  }
  if (plotLagFocusNumberInput) {
    plotLagFocusNumberInput.addEventListener("change", () => {
      applyFocusedLagFromInput(plotLagFocusNumberInput.value);
    });
    plotLagFocusNumberInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      event.preventDefault();
      applyFocusedLagFromInput(plotLagFocusNumberInput.value);
    });
  }

  downloadXlsxButton.addEventListener("click", () => triggerDownload("xlsx"));
  downloadPngButton.addEventListener("click", () => triggerDownload("png"));
  downloadSvgButton.addEventListener("click", () => triggerDownload("svg"));
  if (mapDownloadPngButton) {
    mapDownloadPngButton.addEventListener("click", () => triggerMapDownload("png"));
  }
  if (mapDownloadNcButton) {
    mapDownloadNcButton.addEventListener("click", () => triggerMapDownload("nc"));
  }
  if (mapLoadButton) {
    mapLoadButton.addEventListener("click", async () => {
      try {
        await submitMapExplore();
      } catch (error) {
        setMapStatus(String(error), true);
      }
    });
  }
  if (mapRunButton) {
    mapRunButton.addEventListener("click", async () => {
      try {
        await submitMapJob();
      } catch (error) {
        setMapStatus(String(error), true);
      }
    });
  }
  if (mapTimeSlider) {
    mapTimeSlider.addEventListener("input", () => {
      stopMapTimePlayback();
      updateExploreFrame(Number(mapTimeSlider.value));
    });
  }
  if (mapTimePlayButton) {
    mapTimePlayButton.addEventListener("click", () => {
      toggleMapTimePlayback();
    });
  }

  if (mapClearBoundsButton) {
    mapClearBoundsButton.addEventListener("click", async () => {
      setMapBoundsInputs(null);
      invalidateMapPreparation();
      await syncExploreBoundsShape(null);
    });
  }
  if (mapDriverDatasetInput) {
    mapDriverDatasetInput.addEventListener("change", async () => {
      try {
        await applyMapDriverDefaults(mapDriverDatasetInput.value);
      } catch (error) {
        setMapStatus(String(error), true);
      }
      renderMapSelectorMetadata();
      renderMapDatasetDocs();
      invalidateMapPreparation({ hard: true });
    });
  }
  if (mapFieldDatasetInput) {
    mapFieldDatasetInput.addEventListener("change", () => {
      renderMapSelectorMetadata();
      renderMapDatasetDocs();
      invalidateMapPreparation({ hard: true });
    });
  }

  const mapConfigInputs = [
    mapFragmentSizeInput,
    mapAlphaInput,
    mapTopFractionInput,
    mapPermutationsInput,
    mapMinLagInput,
    mapMaxLagInput,
    mapTimeStartInput,
    mapTimeEndInput,
    mapPeakDateInput,
    mapLatMinInput,
    mapLatMaxInput,
    mapLonMinInput,
    mapLonMaxInput,
  ].filter(Boolean);
  mapConfigInputs.forEach((input) => {
    input.addEventListener("change", async () => {
      updateMapBoundsNotice();
      invalidateMapPreparation();
      try {
        const bounds = getMapBoundsSelection();
        await syncExploreBoundsShape(bounds.hasBounds ? bounds : null);
      } catch (_error) {
        // Keep invalid intermediate states local to form validation.
      }
    });
  });

  if (mapSaturationInput) {
    mapSaturationInput.addEventListener("input", () => {
      if (mapPhase === "explore" && latestMapExplore) {
        void renderMapExploration(latestMapExplore);
      } else if (mapSaturationMeta) {
        mapSaturationMeta.textContent = mapSaturationInput.value.trim()
          ? "Manual saturation override."
          : "Auto from field range.";
      }
    });
  }

  window.addEventListener("resize", () => {
    if (activeWorkflowTab === "map" && mapPhase === "results" && latestMapResult) {
      if (mapResizeTimer) {
        clearTimeout(mapResizeTimer);
      }
      mapResizeTimer = setTimeout(() => {
        void renderActiveMapResultLayer();
      }, 180);
      return;
    }
    if (activeWorkflowTab !== "sdc" || !latestResult) {
      return;
    }
    if (explorerResizeTimer) {
      clearTimeout(explorerResizeTimer);
    }
    explorerResizeTimer = setTimeout(() => renderTwoWayExplorer(latestResult), 170);
  });
}

setDownloadButtons(false);
setStatusProgress();
setMapDownloadButtons(false);
setAnalysisSettingsUnlocked(false);
setInputMode(activeInputMode);
setWorkflowTab(activeWorkflowTab);
setSdcExplorerTab("matrix", { rerender: false });
setSdcExplorerTabAvailability(null);
syncPlotControlsFromSettings();
updateHeatmapStepNotice();
setMapProgress();
setMapPhase("idle");
refreshMapRunButtonState();
setMapStatus("Load datasets to start exploration, then run SDC map.");
if (mapSelectedCellText) {
  mapSelectedCellText.textContent =
    "Selected grid cell: none. Click the field heatmap to inspect a cell time series.";
}
updateMapBoundsNotice();
attachHandlers();

void (async () => {
  await fetchMapCatalog();
  try {
    await applyMapDriverDefaults(mapDriverDatasetInput?.value || "pdo");
  } catch (error) {
    setMapStatus(String(error), true);
  }
  updateMapBoundsNotice();
})();
