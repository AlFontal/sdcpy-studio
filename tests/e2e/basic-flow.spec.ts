import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";

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
