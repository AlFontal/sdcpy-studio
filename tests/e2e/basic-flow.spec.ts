import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import path from "node:path";

function buildDatasetCsv(n: number): string {
  const rows = ["date,driver,mapped"];
  for (let i = 0; i < n; i += 1) {
    const year = 2000 + Math.floor(i / 12);
    const month = (i % 12) + 1;
    const date = `${year}-${String(month).padStart(2, "0")}-01`;
    const driver = Math.sin(i / 6) + 0.2 * Math.cos(i / 2);
    const mapped = Math.sin((i + 2) / 6) + 0.25 * Math.cos(i / 2 + 0.3);
    rows.push(`${date},${driver.toFixed(6)},${mapped.toFixed(6)}`);
  }
  return rows.join("\n");
}

test("dataset workflow runs analysis and renders outputs", async ({ page }, testInfo) => {
  const csvPath = testInfo.outputPath("dataset.csv");
  writeFileSync(csvPath, buildDatasetCsv(72), "utf8");

  await page.goto("/");
  await expect(page.getByTestId("analysis-settings-details")).toHaveJSProperty("open", false);
  await expect(page.getByTestId("explorer-details")).toHaveJSProperty("open", false);

  await page.getByTestId("dataset-file-input").setInputFiles(csvPath);
  await expect(page.getByTestId("dataset-meta")).toContainText("rows", { timeout: 20_000 });
  await expect(page.getByTestId("analysis-settings-details")).toHaveJSProperty("open", true);
  await expect(page.getByTestId("dataset-preview-details")).toHaveJSProperty("open", false);
  await expect(page.getByTestId("analysis-settings-card").getByTestId("dataset-run-button")).toBeVisible();

  await page.getByTestId("n-permutations-input").fill("9");
  await page.getByTestId("fragment-size-input").fill("12");
  await page.getByTestId("min-lag-input").fill("-12");
  await page.getByTestId("max-lag-input").fill("12");

  const runButton = page.getByTestId("dataset-run-button");
  await expect(runButton).toBeEnabled();
  await runButton.click();

  await expect(page.getByTestId("status-text")).toContainText("succeeded", { timeout: 180_000 });
  await expect(page.getByTestId("summary-stats")).toContainText("Series length");
  await expect(page.getByTestId("explorer-details")).toHaveJSProperty("open", true);

  const explorer = page.getByTestId("two-way-explorer");
  await expect(explorer).toBeVisible();
  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const container = document.getElementById("two_way_explorer");
        if (!container) {
          return false;
        }
        const renderedPlot = container.matches(".js-plotly-plot") || !!container.querySelector(".js-plotly-plot");
        const fallbackText = (container.textContent ?? "").includes("Plotly library unavailable");
        return renderedPlot || fallbackText;
      });
    })
    .toBe(true);

  const plotChecks = await page.evaluate(() => {
    const gd = document.querySelector(
      "#two_way_explorer.js-plotly-plot, #two_way_explorer .js-plotly-plot"
    ) as
      | ({ data?: Array<{ name?: string; zmid?: number; zmin?: number; zmax?: number; type?: string }> } & Element)
      | null;
    if (!gd || !Array.isArray(gd.data)) {
      return {
        hasPlot: false,
      };
    }
    const names = gd.data.map((trace) => String(trace.name ?? ""));
    const hasRangesTrace = gd.data.some((trace) => {
      const name = String(trace.name ?? "").toLowerCase();
      return name.includes("range") || name.includes("freq");
    });
    const heatmap = gd.data.find((trace) => trace.type === "heatmap");
    const topTs = gd.data.find((trace) => trace.name === "TS1");
    const xaxisRange = (gd as { layout?: { xaxis?: { range?: [number, number] } } }).layout?.xaxis?.range;
    const symmetricScale =
      typeof heatmap?.zmin === "number" &&
      typeof heatmap?.zmax === "number" &&
      Math.abs(Math.abs(heatmap.zmin) - Math.abs(heatmap.zmax)) < 1e-6;
    return {
      hasPlot: true,
      hasRangesTrace,
      names,
      zmid: heatmap?.zmid ?? null,
      symmetricScale,
      heatXFirst: Array.isArray(heatmap?.x) ? Number(heatmap.x[0]) : null,
      topTracePoints: Array.isArray(topTs?.x) ? topTs.x.length : null,
      xAxisRange: xaxisRange ?? null,
    };
  });
  if (plotChecks.hasPlot) {
    expect(plotChecks.hasRangesTrace).toBeFalsy();
    expect(plotChecks.names).toContain("max positive (TS1)");
    expect(plotChecks.names).toContain("min negative (TS1)");
    expect(plotChecks.names).toContain("max positive (TS2)");
    expect(plotChecks.names).toContain("min negative (TS2)");
    expect(plotChecks.zmid).toBe(0);
    expect(plotChecks.symmetricScale).toBeTruthy();
    expect(plotChecks.heatXFirst).toBeCloseTo(5.5, 6);
    expect(plotChecks.topTracePoints).toBe(72);
    expect(plotChecks.xAxisRange?.[0]).toBeCloseTo(-0.5, 6);
    expect(plotChecks.xAxisRange?.[1]).toBeCloseTo(71.5, 6);
  }

  const matrixTab = page.getByTestId("sdc-explorer-tab-matrix");
  const lagTab = page.getByTestId("sdc-explorer-tab-lag");
  await expect(matrixTab).toHaveAttribute("aria-selected", "true");
  await expect(lagTab).toBeEnabled();
  await lagTab.click();
  await expect(lagTab).toHaveAttribute("aria-selected", "true");
  await expect(matrixTab).toHaveAttribute("aria-selected", "false");

  const lagExplorer = page.getByTestId("two-way-lag-explorer");
  await expect(lagExplorer).toBeVisible();
  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const container = document.getElementById("two_way_lag_explorer");
        if (!container) {
          return false;
        }
        const renderedPlot = container.matches(".js-plotly-plot") || !!container.querySelector(".js-plotly-plot");
        const fallbackText = (container.textContent ?? "").includes("Plotly library unavailable");
        return renderedPlot || fallbackText;
      });
    })
    .toBe(true);

  await expect(page.getByTestId("plot-lag-focus-slider")).toHaveValue("0");
  await expect(page.getByTestId("plot-lag-focus-number-input")).toHaveValue("0");

  const initialLagTraceName = await page.evaluate(() => {
    const gd = document.querySelector(
      "#two_way_lag_explorer.js-plotly-plot, #two_way_lag_explorer .js-plotly-plot"
    ) as ({ data?: Array<{ name?: string }> } & Element) | null;
    if (!gd || !Array.isArray(gd.data)) {
      return "";
    }
    const trace = gd.data.find((item) => String(item?.name ?? "").includes("Focused lag r"));
    return String(trace?.name ?? "");
  });
  expect(initialLagTraceName).toContain("lag=0");

  await page.getByTestId("plot-lag-focus-slider").evaluate((element) => {
    const input = element as any;
    input.value = "3";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.getByTestId("plot-lag-focus-number-input")).toHaveValue("3");

  const updatedLagTraceName = await page.evaluate(() => {
    const gd = document.querySelector(
      "#two_way_lag_explorer.js-plotly-plot, #two_way_lag_explorer .js-plotly-plot"
    ) as ({ data?: Array<{ name?: string }> } & Element) | null;
    if (!gd || !Array.isArray(gd.data)) {
      return "";
    }
    const trace = gd.data.find((item) => String(item?.name ?? "").includes("Focused lag r"));
    return String(trace?.name ?? "");
  });
  expect(updatedLagTraceName).toContain("lag=3");

  await page.getByTestId("plot-alpha-input").evaluate((element) => {
    const input = element as any;
    input.value = "0.001";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.getByTestId("plot-alpha-input")).toHaveValue("0.001");
  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const gd = document.querySelector(
          "#two_way_lag_explorer.js-plotly-plot, #two_way_lag_explorer .js-plotly-plot"
        ) as ({ layout?: { annotations?: Array<{ text?: string }> } } & Element) | null;
        const annotationText = gd?.layout?.annotations?.[0]?.text;
        return String(annotationText ?? "");
      });
    })
    .toContain("alpha <= 0.001");
});

test("oni sample can be loaded without uploading a file", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("load-oni-example-button").click();
  await expect(page.getByTestId("dataset-meta")).toContainText("300 rows x 3 columns", {
    timeout: 20_000,
  });
  await expect(page.getByTestId("dataset-meta")).not.toContainText("oni_temp_sa.csv");
  await expect
    .poll(async () => page.locator('input[type="file"][data-testid="dataset-file-input"]').inputValue())
    .toContain("oni_temp_sa.csv");
  await expect(page.getByTestId("dataset-date-select")).toHaveValue("date");
  await expect(page.getByTestId("dataset-ts1-select")).toHaveValue("oni_anomaly");
  await expect(page.getByTestId("dataset-ts2-select")).toHaveValue("temp_anomaly_sa");
  await expect(page.getByTestId("dataset-run-button")).toBeEnabled();
  await expect(page.getByTestId("dataset-preview-details")).toHaveJSProperty("open", false);
});

test("sdc map accepts custom driver CSV and custom field NetCDF uploads", async ({ page }) => {
  const driverCsvPath = path.resolve("tests/fixtures/map_custom_driver.csv");
  const fieldNcPath = path.resolve("tests/fixtures/map_custom_field.nc");

  let exploreRequest: Record<string, unknown> | null = null;
  let mapSubmitRequest: Record<string, unknown> | null = null;
  let previewRequest: Record<string, unknown> | null = null;
  let statusPollCount = 0;

  await page.route("**/api/v1/sdc-map/driver/preview", async (route) => {
    previewRequest = route.request().postDataJSON() as Record<string, unknown>;
    const nPositive = Number(previewRequest?.n_positive_peaks ?? 3);
    const nNegative = Number(previewRequest?.n_negative_peaks ?? 3);
    const makeEvents = (sign: "positive" | "negative", count: number) =>
      Array.from({ length: count }, (_, index) => ({
        index,
        date: `2000-${String(index + 1).padStart(2, "0")}-01`,
        value: sign === "positive" ? 1.2 - index * 0.1 : -1.1 + index * 0.05,
        sign,
      }));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        result: {
          summary: {
            driver_dataset: "map_custom_driver.csv",
            time_start: "1998-01-01",
            time_end: "2001-12-01",
            correlation_width: 12,
            n_positive_peaks: nPositive,
            n_negative_peaks: nNegative,
            base_state_beta: 0.5,
            n_points: 48,
          },
          event_catalog: {
            selected_positive: makeEvents("positive", nPositive),
            selected_negative: makeEvents("negative", nNegative),
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.55,
            base_state_count: 18,
            warnings: [],
          },
          time_index: ["1998-01-01", "1998-02-01", "1998-03-01"],
          driver_values: [0.1, 0.2, -0.1],
        },
      }),
    });
  });

  await page.route("**/api/v1/sdc-map/explore", async (route) => {
    exploreRequest = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        result: {
          summary: {
            driver_dataset: "map_custom_driver.csv",
            field_dataset: "map_custom_field.nc",
            driver_source_type: "upload",
            field_source_type: "upload",
            field_variable: "sst_anom_custom",
            time_start: "1998-01-01",
            time_end: "2001-12-01",
            correlation_width: 12,
            n_positive_peaks: 3,
            n_negative_peaks: 3,
            base_state_beta: 0.5,
            n_time: 48,
            n_lat: 5,
            n_lon: 6,
            valid_values: 1440,
            valid_rate: 1.0,
            first_valid_index: [0, 0, 0],
            field_lat_min: -20,
            field_lat_max: 20,
            field_lon_min: -160,
            field_lon_max: 180,
            field_value_min: -2,
            field_value_max: 2,
            used_lat_min: -20,
            used_lat_max: 20,
            used_lon_min: -160,
            used_lon_max: 180,
            full_bounds_selected: true,
            selected_positive_events: 2,
            selected_negative_events: 2,
            base_state_count: 18,
            base_state_threshold: 0.55,
          },
          event_catalog: {
            selected_positive: [
              { index: 0, date: "1999-03-01", value: 1.14, sign: "positive" },
              { index: 1, date: "2000-08-01", value: 1.22, sign: "positive" },
            ],
            selected_negative: [
              { index: 0, date: "1998-11-01", value: -1.01, sign: "negative" },
              { index: 1, date: "2001-02-01", value: -0.97, sign: "negative" },
            ],
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.55,
            base_state_count: 18,
            warnings: [],
          },
          time_index: ["1998-01-01", "1998-02-01"],
          driver_values: [0.1, 0.2],
          lat: [-20, 0, 20],
          lon: [-160, -140, -120],
          field_frames: [
            [
              [0.1, 0.2, 0.3],
              [0.2, 0.3, 0.4],
              [0.3, 0.4, 0.5],
            ],
            [
              [0.2, 0.3, 0.4],
              [0.3, 0.4, 0.5],
              [0.4, 0.5, 0.6],
            ],
          ],
          coastline: { lat: [null], lon: [null] },
        },
      }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    mapSubmitRequest = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "map-job-1",
        status: "queued",
        message: "ok",
      }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map/map-job-1", async (route) => {
    statusPollCount += 1;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "map-job-1",
        status: "succeeded",
        created_at: "2026-02-23T00:00:00Z",
        started_at: "2026-02-23T00:00:00Z",
        completed_at: "2026-02-23T00:00:01Z",
        progress: { current: 1, total: 1, description: "Completed" },
      }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map/map-job-1/result", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "map-job-1",
        status: "succeeded",
        result: {
          summary: {
            driver_dataset: "map_custom_driver.csv",
            field_dataset: "map_custom_field.nc",
            driver_source_type: "upload",
            field_source_type: "upload",
            field_variable: "sst_anom_custom",
            time_start: "1998-01-01",
            time_end: "2001-12-01",
            correlation_width: 12,
            n_positive_peaks: 3,
            n_negative_peaks: 3,
            base_state_beta: 0.5,
            n_permutations: 49,
            alpha: 0.05,
            min_lag: -6,
            max_lag: 6,
            lat_min: -20,
            lat_max: 20,
            lon_min: -160,
            lon_max: 180,
            lat_stride: 1,
            lon_stride: 1,
            n_time: 48,
            n_lat: 5,
            n_lon: 6,
            total_cells: 30,
            valid_cells: 30,
            valid_cell_rate: 1.0,
            field_lat_min: -20,
            field_lat_max: 20,
            field_lon_min: -160,
            field_lon_max: 180,
            mean_abs_corr: 0.42,
            full_bounds_selected: true,
            selected_positive_events: 2,
            selected_negative_events: 2,
            base_state_count: 18,
            base_state_threshold: 0.55,
          },
          event_catalog: {
            selected_positive: [
              { index: 8, date: "1999-03-01", value: 1.14, sign: "positive" },
              { index: 25, date: "2000-08-01", value: 1.22, sign: "positive" },
            ],
            selected_negative: [
              { index: 4, date: "1998-11-01", value: -1.01, sign: "negative" },
              { index: 31, date: "2001-02-01", value: -0.97, sign: "negative" },
            ],
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.55,
            base_state_count: 18,
            warnings: [],
          },
          class_results: {
            positive: {
              summary: { selected_event_count: 2, valid_cells: 30, lag_valid_cells: [24, 30, 27] },
              events: [
                { date: "1999-03-01", value: 1.14, sign: "positive" },
                { date: "2000-08-01", value: 1.22, sign: "positive" },
              ],
              empty_reason: null,
              layer_maps: {
                lat: [-20, 0, 20],
                lon: [-160, -140, -120],
                coastline: { lat: [null], lon: [null] },
                layers: [
                  {
                    key: "corr_mean",
                    label: "A. Correlation",
                    colorscale: "RdBu",
                    zmin: -1,
                    zmax: 1,
                    values: [
                      [0.1, 0.2, 0.3],
                      [0.0, -0.1, -0.2],
                      [0.2, 0.1, 0.0],
                    ],
                  },
                  {
                    key: "driver_rel_time_mean",
                    label: "B. Position",
                    colorscale: "PuOr",
                    zmin: -6,
                    zmax: 6,
                    values: [
                      [-4, -4, -4],
                      [-4, -4, -4],
                      [-4, -4, -4],
                    ],
                  },
                  {
                    key: "lag_mean",
                    label: "C. Lag",
                    colorscale: "RdYlBu",
                    zmin: -6,
                    zmax: 6,
                    values: [
                      [1, 2, 3],
                      [0, -1, -2],
                      [2, 1, 0],
                    ],
                  },
                  {
                    key: "timing_combo",
                    label: "D. Timing",
                    colorscale: "BrBG",
                    zmin: -8,
                    zmax: 8,
                    values: [
                      [-3, -2, -1],
                      [-4, -5, -6],
                      [-2, -3, -4],
                    ],
                  },
                ],
              },
              lag_maps: {
                lat: [-20, 0, 20],
                lon: [-160, -140, -120],
                lags: [-1, 0, 1],
                coastline: { lat: [null], lon: [null] },
                corr_by_lag: [
                  [
                    [0.0, 0.1, 0.2],
                    [0.0, -0.1, -0.1],
                    [0.1, 0.0, 0.0],
                  ],
                  [
                    [0.1, 0.2, 0.3],
                    [0.0, -0.1, -0.2],
                    [0.2, 0.1, 0.0],
                  ],
                  [
                    [0.2, 0.1, 0.1],
                    [0.1, 0.0, -0.1],
                    [0.3, 0.2, 0.1],
                  ],
                ],
                event_count_by_lag: [
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                ],
              },
            },
            negative: {
              summary: { selected_event_count: 2, valid_cells: 28, lag_valid_cells: [22, 28, 24] },
              events: [
                { date: "1998-11-01", value: -1.01, sign: "negative" },
                { date: "2001-02-01", value: -0.97, sign: "negative" },
              ],
              empty_reason: null,
              layer_maps: {
                lat: [-20, 0, 20],
                lon: [-160, -140, -120],
                coastline: { lat: [null], lon: [null] },
                layers: [
                  {
                    key: "corr_mean",
                    label: "A. Correlation",
                    colorscale: "RdBu",
                    zmin: -1,
                    zmax: 1,
                    values: [
                      [-0.2, -0.1, 0.0],
                      [-0.1, -0.2, -0.3],
                      [0.0, -0.1, -0.2],
                    ],
                  },
                  {
                    key: "driver_rel_time_mean",
                    label: "B. Position",
                    colorscale: "PuOr",
                    zmin: -6,
                    zmax: 6,
                    values: [
                      [-4, -4, -4],
                      [-4, -4, -4],
                      [-4, -4, -4],
                    ],
                  },
                  {
                    key: "lag_mean",
                    label: "C. Lag",
                    colorscale: "RdYlBu",
                    zmin: -6,
                    zmax: 6,
                    values: [
                      [-1, 0, 1],
                      [0, 1, 2],
                      [1, 2, 3],
                    ],
                  },
                  {
                    key: "timing_combo",
                    label: "D. Timing",
                    colorscale: "BrBG",
                    zmin: -8,
                    zmax: 8,
                    values: [
                      [-5, -4, -3],
                      [-4, -3, -2],
                      [-3, -2, -1],
                    ],
                  },
                ],
              },
              lag_maps: {
                lat: [-20, 0, 20],
                lon: [-160, -140, -120],
                lags: [-1, 0, 1],
                coastline: { lat: [null], lon: [null] },
                corr_by_lag: [
                  [
                    [-0.1, -0.1, 0.0],
                    [-0.1, -0.2, -0.2],
                    [0.0, -0.1, -0.1],
                  ],
                  [
                    [-0.2, -0.1, 0.0],
                    [-0.1, -0.2, -0.3],
                    [0.0, -0.1, -0.2],
                  ],
                  [
                    [-0.1, 0.0, 0.1],
                    [0.0, -0.1, -0.2],
                    [0.1, 0.0, -0.1],
                  ],
                ],
                event_count_by_lag: [
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                  [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                  ],
                ],
              },
            },
          },
          notes: [],
          runtime_seconds: 1.2,
          figure_png_base64:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGMwMDD4DwAD1QG6hQm8WQAAAABJRU5ErkJggg==",
          download_formats: ["png", "nc"],
          layer_maps: {
            lat: [-20, 0, 20],
            lon: [-160, -140, -120],
            coastline: { lat: [null], lon: [null] },
            layers: [
              {
                key: "corr_mean",
                label: "Mean extreme correlation",
                colorscale: "RdBu",
                zmin: -1,
                zmax: 1,
                values: [
                  [0.1, 0.2, 0.3],
                  [0.0, -0.1, -0.2],
                  [0.2, 0.1, 0.0],
                ],
              },
              {
                key: "lag_mean",
                label: "C. Lag",
                colorscale: "RdYlBu",
                zmin: -6,
                zmax: 6,
                values: [
                  [1, 2, 3],
                  [0, -1, -2],
                  [2, 1, 0],
                ],
              },
              {
                key: "driver_rel_time_mean",
                label: "B. Position",
                colorscale: "BrBG",
                zmin: -6,
                zmax: 6,
                values: [
                  [1, 1, 1],
                  [0, 0, 0],
                  [-1, -1, -1],
                ],
              },
            ],
          },
        },
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("tab", { name: /SDC Map/i }).click();

  await page.getByTestId("map-driver-file-input").setInputFiles(driverCsvPath);
  await expect(page.getByTestId("map-driver-upload-status")).toContainText("Ready:", { timeout: 20_000 });
  await expect(page.getByTestId("map-driver-date-select")).toHaveValue("date");
  await expect(page.getByTestId("map-driver-value-select")).toHaveValue("driver_index");

  await page.getByTestId("map-field-file-input").setInputFiles(fieldNcPath);
  await expect(page.getByTestId("map-field-upload-status")).toContainText("Ready:", { timeout: 20_000 });
  await expect(page.getByTestId("map-field-variable-select")).toHaveValue("sst_anom_custom");

  await expect(page.locator("#map_time_start")).toHaveValue(/\d{4}-\d{2}-\d{2}/);
  await expect(page.locator("#map_event_preview")).toContainText("Positive events (N+)");
  await expect(page.getByTestId("map-event-preview-chart")).toBeVisible();
  const previewChartMeta = await page.evaluate(() => {
    const gd = document.querySelector(
      "#map_event_preview_chart.js-plotly-plot, #map_event_preview_chart .js-plotly-plot"
    ) as
      | ({
          data?: Array<{ name?: string; hovertemplate?: string; hoverinfo?: string; showlegend?: boolean }>;
          layout?: { shapes?: Array<unknown> };
        } & Element)
      | null;
    if (!gd || !Array.isArray(gd.data)) {
      return { hasPlot: false, shapeCount: 0, traceNames: [], markerHoverInfo: "", thresholdLegendCount: 0 };
    }
    const selectedPositiveTrace = gd.data.find((trace) => trace.name === "Selected positive");
    const traceNames = gd.data.map((trace) => String(trace.name ?? ""));
    const thresholdLegendCount = gd.data.filter(
      (trace) => trace.name === "Base-state threshold" && trace.showlegend !== false
    ).length;
    return {
      hasPlot: true,
      shapeCount: Array.isArray(gd.layout?.shapes) ? gd.layout.shapes.length : 0,
      traceNames,
      markerHoverInfo: String(selectedPositiveTrace?.hoverinfo ?? ""),
      thresholdLegendCount,
    };
  });
  expect(previewChartMeta.hasPlot).toBeTruthy();
  expect(previewChartMeta.shapeCount).toBeGreaterThan(0);
  expect(previewChartMeta.traceNames).not.toContain("Ignored positive");
  expect(previewChartMeta.traceNames).not.toContain("Ignored negative");
  expect(previewChartMeta.thresholdLegendCount).toBe(1);
  expect(previewChartMeta.markerHoverInfo).toBe("skip");
  await page.locator("#map_n_positive_peaks").fill("2");
  await expect.poll(() => previewRequest?.n_positive_peaks).toBe(2);
  await expect(page.locator("#map_event_preview")).toContainText("2000-02-01");

  await page.locator("#map_load").click();
  await expect.poll(() => exploreRequest).not.toBeNull();
  expect(exploreRequest?.driver_source_type).toBe("upload");
  expect(exploreRequest?.field_source_type).toBe("upload");
  expect(typeof exploreRequest?.driver_upload_id).toBe("string");
  expect(typeof exploreRequest?.field_upload_id).toBe("string");
  expect(exploreRequest?.driver_date_column).toBe("date");
  expect(exploreRequest?.driver_value_column).toBe("driver_index");
  expect(exploreRequest?.field_variable).toBe("sst_anom_custom");
  expect(exploreRequest?.correlation_width).toBe(12);
  expect(exploreRequest?.n_positive_peaks).toBe(2);
  expect(exploreRequest?.n_negative_peaks).toBe(3);

  await expect(page.locator("#map_status")).toContainText("Exploration ready", { timeout: 15_000 });
  await expect(page.locator("#map_run")).toBeEnabled();
  await expect(page.locator(".map-stage")).not.toContainText("dual axis");
  await expect(page.locator(".map-stage")).not.toContainText("Selected grid cell");
  await expect(page.locator("#map_explore_controls")).toBeVisible();
  await expect(page.locator("#map_result_lag_controls")).toBeHidden();
  await expect(page.locator("#map_class_tabs")).toBeHidden();
  await expect(page.locator("#map_static_summary_section")).toBeHidden();

  await page.locator("#map_run").click();
  await expect.poll(() => mapSubmitRequest).not.toBeNull();
  expect(mapSubmitRequest?.driver_source_type).toBe("upload");
  expect(mapSubmitRequest?.field_source_type).toBe("upload");
  await expect(page.locator("#map_status")).toContainText("SDC map ready", { timeout: 15_000 });
  expect(statusPollCount).toBeGreaterThan(0);
  await expect(page.locator("#map_explore_controls")).toBeHidden();
  await expect(page.locator("#map_class_tabs")).toBeVisible();
  await expect(page.locator("#map_result_lag_controls")).toBeVisible();
  await expect(page.locator("#map_static_summary_section")).toBeVisible();
  await expect(page.locator("#sdc_map_summary")).toContainText("map_custom_driver.csv");
  await expect(page.locator("#map_class_tabs")).toContainText("Positive");
  await expect(page.getByTestId("map-result-lag-slider")).toHaveValue("0");
  await expect(page.locator("#sdc_map_summary")).toContainText("Active lag");
  await expect(page.locator("#map_static_summary_section")).toContainText("A. Correlation");
  await expect(page.locator("#map_static_summary_section")).toContainText("D. Timing");
  await expect(page.locator(".map-static-help")).toHaveCount(4);
  await expect(page.locator(".map-static-help").first()).toHaveAttribute("data-tooltip", /Strongest significant/);
  const sectionOrder = await page.evaluate(() => {
    const staticSection = document.getElementById("map_static_summary_section");
    const lagControls = document.getElementById("map_result_lag_controls");
    const plot = document.getElementById("sdc_map_plot");
    const staticTop = staticSection?.getBoundingClientRect().top ?? null;
    const lagTop = lagControls?.getBoundingClientRect().top ?? null;
    const plotTop = plot?.getBoundingClientRect().top ?? null;
    return { staticTop, lagTop, plotTop };
  });
  expect(sectionOrder.staticTop).not.toBeNull();
  expect(sectionOrder.lagTop).not.toBeNull();
  expect(sectionOrder.plotTop).not.toBeNull();
  expect(Number(sectionOrder.staticTop)).toBeLessThan(Number(sectionOrder.lagTop));
  expect(Number(sectionOrder.lagTop)).toBeLessThan(Number(sectionOrder.plotTop));
  const initialLagMap = await page.evaluate(() => {
    const gd = document.querySelector("#sdc_map_plot.js-plotly-plot") as
      | ({ data?: Array<{ name?: string; z?: Array<Array<number | null>> }> } & Element)
      | null;
    const trace = gd?.data?.[0];
    return {
      name: String(trace?.name ?? ""),
      z00: trace?.z?.[0]?.[0] ?? null,
    };
  });
  expect(initialLagMap.name).toContain("+0");
  expect(initialLagMap.z00).toBe(0.1);
  await page.getByTestId("map-result-lag-number").fill("1");
  await page.getByTestId("map-result-lag-number").press("Enter");
  await expect(page.getByTestId("map-result-lag-slider")).toHaveValue("1");
  await expect(page.locator("#sdc_map_summary")).toContainText("+1");
  const shiftedLagMap = await page.evaluate(() => {
    const gd = document.querySelector("#sdc_map_plot.js-plotly-plot") as
      | ({ data?: Array<{ name?: string; z?: Array<Array<number | null>> }> } & Element)
      | null;
    const trace = gd?.data?.[0];
    return {
      name: String(trace?.name ?? ""),
      z00: trace?.z?.[0]?.[0] ?? null,
    };
  });
  expect(shiftedLagMap.name).toContain("+1");
  expect(shiftedLagMap.z00).toBe(0.2);
});

test("custom map upload errors are cleared after a valid re-upload", async ({ page }, testInfo) => {
  const invalidDriverPath = testInfo.outputPath("driver-semicolon.csv");
  writeFileSync(
    invalidDriverPath,
    ["date;driver_index;alt_series", "2000-01-01;0.1;1.0", "2000-02-01;0.2;0.9"].join("\n"),
    "utf8"
  );
  const invalidFieldPath = testInfo.outputPath("invalid-field.nc");
  writeFileSync(invalidFieldPath, "not a netcdf file", "utf8");

  const validDriverPath = path.resolve("tests/fixtures/map_custom_driver.csv");
  const validFieldPath = path.resolve("tests/fixtures/map_custom_field.nc");

  await page.goto("/");
  await page.getByRole("tab", { name: /SDC Map/i }).click();

  await page.getByTestId("map-driver-file-input").setInputFiles(invalidDriverPath);
  await expect(page.getByTestId("map-driver-upload-status")).toContainText("parseable date column", {
    timeout: 20_000,
  });
  await expect(page.locator("#map_status")).toContainText("parseable date column");

  await page.getByTestId("map-driver-file-input").setInputFiles(validDriverPath);
  await expect(page.getByTestId("map-driver-upload-status")).toContainText("Ready:", { timeout: 20_000 });
  await expect(page.locator("#map_status")).not.toContainText("parseable date column");

  await page.getByTestId("map-field-file-input").setInputFiles(invalidFieldPath);
  await expect(page.getByTestId("map-field-upload-status")).toContainText("Could not open NetCDF file", {
    timeout: 20_000,
  });
  await expect(page.locator("#map_status")).toContainText("Could not open NetCDF file");

  await page.getByTestId("map-field-file-input").setInputFiles(validFieldPath);
  await expect(page.getByTestId("map-field-upload-status")).toContainText("Ready:", { timeout: 20_000 });
  await expect(page.locator("#map_status")).not.toContainText("Could not open NetCDF file");
  await expect(page.locator("#map_status")).toContainText("Load datasets to start exploration");
});

test("custom NetCDF uploads expose extra-dimension selectors and submit the chosen value", async ({ page }) => {
  const driverCsvPath = path.resolve("tests/fixtures/map_custom_driver.csv");
  const fieldNcPath = path.resolve("tests/fixtures/map_custom_field.nc");
  let exploreRequest: Record<string, unknown> | null = null;

  await page.route("**/api/v1/sdc-map/field/inspect", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        upload_id: "field-upload-levels",
        filename: "field_levels.nc",
        variables: ["air"],
        compatible_variables: ["air"],
        variable_options: [
          {
            name: "air",
            selectors: [
              {
                dimension: "level",
                label: "level (hPa)",
                suggested_value: "850",
                options: [
                  { value: "850", label: "850 hPa" },
                  { value: "500", label: "500 hPa" },
                ],
              },
            ],
            normalization: {
              original_dims: ["time", "level", "lat", "lon"],
              squeezed_dims: [],
              selected_dimensions: { level: "850" },
            },
            warnings: [
              {
                code: "dimension_selection_available",
                message: "'air' requires selecting extra dimensions before analysis. Suggested: level=850.",
                columns: ["air"],
              },
            ],
          },
        ],
        incompatible_variables: [],
        suggested_variable: "air",
        dims: { time: 12, lat: 3, lon: 4 },
        normalization: {
          original_dims: ["time", "level", "lat", "lon"],
          squeezed_dims: [],
          selected_dimensions: { level: "850" },
        },
        warnings: [
          {
            code: "dimension_selection_available",
            message: "'air' requires selecting extra dimensions before analysis. Suggested: level=850.",
            columns: ["air"],
          },
        ],
        time_start: "2000-01-01",
        time_end: "2000-12-01",
        lat_min: -10,
        lat_max: 10,
        lon_min: 120,
        lon_max: 180,
      }),
    });
  });

  await page.route("**/api/v1/sdc-map/explore", async (route) => {
    exploreRequest = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        result: {
          summary: {
            driver_dataset: "map_custom_driver.csv",
            field_dataset: "field_levels.nc",
            time_start: "2000-01-01",
            time_end: "2000-12-01",
            correlation_width: 12,
            n_positive_peaks: 3,
            n_negative_peaks: 3,
            base_state_beta: 0.5,
            n_time: 12,
            n_lat: 3,
            n_lon: 4,
            valid_values: 144,
            field_lat_min: -10,
            field_lat_max: 10,
            field_lon_min: 120,
            field_lon_max: 180,
            field_variable: "air",
            field_dimension_selections: { level: "500" },
            full_bounds_selected: true,
          },
          event_catalog: {
            selected_positive: [{ index: 0, date: "2000-08-01", value: 1.1, sign: "positive" }],
            selected_negative: [{ index: 0, date: "2000-03-01", value: -0.9, sign: "negative" }],
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.45,
            base_state_count: 4,
            warnings: [],
          },
          time_index: ["2000-01-01"],
          driver_values: [0.1],
          lat: [-10],
          lon: [120],
          field_frames: [[[0.2]]],
          coastline: { lat: [null], lon: [null] },
        },
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("tab", { name: /SDC Map/i }).click();
  await page.getByTestId("map-driver-file-input").setInputFiles(driverCsvPath);
  await expect(page.getByTestId("map-driver-upload-status")).toContainText("Ready:", { timeout: 20_000 });
  await page.getByTestId("map-field-file-input").setInputFiles(fieldNcPath);
  await expect(page.getByTestId("map-field-upload-status")).toContainText("Selection: level=850", {
    timeout: 20_000,
  });

  const levelSelect = page.getByTestId("map-field-dimension-level");
  await expect(levelSelect).toBeVisible();
  await expect(levelSelect).toHaveValue("850");
  await levelSelect.selectOption("500");

  await page.locator("#map_load").click();
  await expect.poll(() => exploreRequest).not.toBeNull();
  expect(exploreRequest?.field_dimension_selections).toEqual({ level: "500" });
});

test("map results warn when no valid cells pass filtering", async ({ page }) => {
  await page.route("**/api/v1/sdc-map/catalog", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        drivers: [{ key: "pdo", label: "PDO" }],
        fields: [{ key: "ncep_air", label: "NCEP Air" }],
      }),
    });
  });

  await page.route("**/api/v1/sdc-map/defaults?driver_dataset=pdo", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        driver_dataset: "pdo",
        time_start: "1998-01-01",
        time_end: "2001-12-01",
        driver_min_date: "1998-01-01",
        driver_max_date: "2001-12-01",
        n_points: 48,
        event_catalog: {
          selected_positive: [{ index: 31, date: "2000-08-01", value: 1.0, sign: "positive" }],
          selected_negative: [{ index: 13, date: "1999-02-01", value: -0.9, sign: "negative" }],
          ignored_positive: [],
          ignored_negative: [],
          base_state_threshold: 0.5,
          base_state_count: 6,
          warnings: [],
        },
      }),
    });
  });

  await page.route("**/api/v1/sdc-map/explore", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        result: {
          summary: {
            driver_dataset: "pdo",
            field_dataset: "ncep_air",
            time_start: "1998-01-01",
            time_end: "2001-12-01",
            correlation_width: 12,
            n_positive_peaks: 3,
            n_negative_peaks: 3,
            base_state_beta: 0.5,
            n_time: 12,
            n_lat: 2,
            n_lon: 2,
            valid_values: 48,
            field_lat_min: -10,
            field_lat_max: 10,
            field_lon_min: -150,
            field_lon_max: -130,
            full_bounds_selected: false,
          },
          event_catalog: {
            selected_positive: [{ index: 31, date: "2000-08-01", value: 1.0, sign: "positive" }],
            selected_negative: [{ index: 13, date: "1999-02-01", value: -0.9, sign: "negative" }],
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.5,
            base_state_count: 6,
            warnings: [],
          },
          time_index: ["2000-01-01"],
          driver_values: [0.1],
          lat: [-10, 10],
          lon: [-150, -130],
          field_frames: [[[0.2, 0.1], [0.0, -0.1]]],
          coastline: { lat: [null], lon: [null] },
        },
      }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ job_id: "map-job-zero", status: "queued", message: "ok" }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map/map-job-zero", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "map-job-zero",
        status: "succeeded",
        created_at: "2026-02-23T00:00:00Z",
        started_at: "2026-02-23T00:00:00Z",
        completed_at: "2026-02-23T00:00:01Z",
        progress: { current: 1, total: 1, description: "Completed" },
      }),
    });
  });

  await page.route("**/api/v1/jobs/sdc-map/map-job-zero/result", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job_id: "map-job-zero",
        status: "succeeded",
        result: {
          summary: {
            driver_dataset: "pdo",
            field_dataset: "ncep_air",
            field_variable: "air",
            time_start: "1998-01-01",
            time_end: "2001-12-01",
            correlation_width: 12,
            n_positive_peaks: 3,
            n_negative_peaks: 3,
            base_state_beta: 0.5,
            n_permutations: 49,
            alpha: 0.05,
            min_lag: -6,
            max_lag: 6,
            lat_min: -10,
            lat_max: 10,
            lon_min: -150,
            lon_max: -130,
            lat_stride: 1,
            lon_stride: 1,
            n_time: 12,
            n_lat: 2,
            n_lon: 2,
            total_cells: 4,
            valid_cells: 0,
            valid_cell_rate: 0,
            field_lat_min: -10,
            field_lat_max: 10,
            field_lon_min: -150,
            field_lon_max: -130,
            mean_abs_corr: null,
            full_bounds_selected: false,
            selected_positive_events: 1,
            selected_negative_events: 1,
            base_state_count: 6,
            base_state_threshold: 0.5,
          },
          event_catalog: {
            selected_positive: [{ index: 31, date: "2000-08-01", value: 1.0, sign: "positive" }],
            selected_negative: [{ index: 13, date: "1999-02-01", value: -0.9, sign: "negative" }],
            ignored_positive: [],
            ignored_negative: [],
            base_state_threshold: 0.5,
            base_state_count: 6,
            warnings: [],
          },
          class_results: {
            positive: {
              summary: { selected_event_count: 1, valid_cells: 0, lag_valid_cells: [0] },
              events: [{ date: "2000-08-01", value: 1.0, sign: "positive" }],
              empty_reason: "No valid positive map cells passed filtering with the current parameters.",
              layer_maps: {
                lat: [-10, 10],
                lon: [-150, -130],
                coastline: { lat: [null], lon: [null] },
                layers: [
                  {
                    key: "corr_mean",
                    label: "Mean extreme correlation",
                    colorscale: "RdBu",
                    zmin: -1,
                    zmax: 1,
                    values: [
                      [null, null],
                      [null, null],
                    ],
                  },
                ],
              },
              lag_maps: {
                lat: [-10, 10],
                lon: [-150, -130],
                lags: [0],
                coastline: { lat: [null], lon: [null] },
                corr_by_lag: [
                  [
                    [null, null],
                    [null, null],
                  ],
                ],
                event_count_by_lag: [
                  [
                    [0, 0],
                    [0, 0],
                  ],
                ],
              },
            },
            negative: {
              summary: { selected_event_count: 1, valid_cells: 0, lag_valid_cells: [0] },
              events: [{ date: "1999-02-01", value: -0.9, sign: "negative" }],
              empty_reason: "No valid negative map cells passed filtering with the current parameters.",
              layer_maps: {
                lat: [-10, 10],
                lon: [-150, -130],
                coastline: { lat: [null], lon: [null] },
                layers: [
                  {
                    key: "corr_mean",
                    label: "Mean extreme correlation",
                    colorscale: "RdBu",
                    zmin: -1,
                    zmax: 1,
                    values: [
                      [null, null],
                      [null, null],
                    ],
                  },
                ],
              },
              lag_maps: {
                lat: [-10, 10],
                lon: [-150, -130],
                lags: [0],
                coastline: { lat: [null], lon: [null] },
                corr_by_lag: [
                  [
                    [null, null],
                    [null, null],
                  ],
                ],
                event_count_by_lag: [
                  [
                    [0, 0],
                    [0, 0],
                  ],
                ],
              },
            },
          },
          notes: ["No valid grid cells passed filtering with the current parameters."],
          runtime_seconds: 1.1,
          figure_png_base64:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGMwMDD4DwAD1QG6hQm8WQAAAABJRU5ErkJggg==",
          download_formats: ["png", "nc"],
          layer_maps: {
            lat: [-10, 10],
            lon: [-150, -130],
            coastline: { lat: [null], lon: [null] },
            layers: [
              {
                key: "corr_mean",
                label: "Mean extreme correlation",
                colorscale: "RdBu",
                zmin: -1,
                zmax: 1,
                values: [
                  [null, null],
                  [null, null],
                ],
              },
            ],
          },
        },
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("tab", { name: /SDC Map/i }).click();
  await page.locator("#map_load").click();
  await expect(page.locator("#map_status")).toContainText("Exploration ready", { timeout: 15_000 });
  await page.locator("#map_run").click();
  await expect(page.locator("#map_status")).toContainText("no valid grid cells passed filtering", {
    timeout: 15_000,
  });
  await expect(page.locator("#sdc_map_summary")).toContainText("No valid grid cells passed filtering");
});
