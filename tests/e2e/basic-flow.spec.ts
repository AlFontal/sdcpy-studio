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
  let statusPollCount = 0;

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
            peak_date: "2000-08-01",
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
            peak_date: "2000-08-01",
            fragment_size: 12,
            n_permutations: 49,
            alpha: 0.05,
            top_fraction: 0.25,
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
            positive_dominant_cells: 15,
            negative_dominant_cells: 15,
            mean_abs_corr: 0.42,
            full_bounds_selected: true,
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
                label: "Mean lag (months)",
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
                label: "Mean driver-relative time (months)",
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
  await expect(page.locator("#map_peak_date")).toHaveValue(/\d{4}-\d{2}-\d{2}/);

  await page.locator("#map_load").click();
  await expect.poll(() => exploreRequest).not.toBeNull();
  expect(exploreRequest?.driver_source_type).toBe("upload");
  expect(exploreRequest?.field_source_type).toBe("upload");
  expect(typeof exploreRequest?.driver_upload_id).toBe("string");
  expect(typeof exploreRequest?.field_upload_id).toBe("string");
  expect(exploreRequest?.driver_date_column).toBe("date");
  expect(exploreRequest?.driver_value_column).toBe("driver_index");
  expect(exploreRequest?.field_variable).toBe("sst_anom_custom");

  await expect(page.locator("#map_status")).toContainText("Exploration ready", { timeout: 15_000 });
  await expect(page.locator("#map_run")).toBeEnabled();

  await page.locator("#map_run").click();
  await expect.poll(() => mapSubmitRequest).not.toBeNull();
  expect(mapSubmitRequest?.driver_source_type).toBe("upload");
  expect(mapSubmitRequest?.field_source_type).toBe("upload");
  await expect(page.locator("#map_status")).toContainText("SDC map ready", { timeout: 15_000 });
  expect(statusPollCount).toBeGreaterThan(0);
  await expect(page.locator("#sdc_map_summary")).toContainText("map_custom_driver.csv");
});
