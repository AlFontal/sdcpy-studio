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

  await page.getByTestId("dataset-file-input").setInputFiles(csvPath);
  await expect(page.getByTestId("dataset-meta")).toContainText("rows", { timeout: 20_000 });
  await expect(page.getByTestId("analysis-settings-details")).toHaveJSProperty("open", true);

  await page.getByTestId("n-permutations-input").fill("9");
  await page.getByTestId("fragment-size-input").fill("12");
  await page.getByTestId("min-lag-input").fill("-12");
  await page.getByTestId("max-lag-input").fill("12");

  const runButton = page.getByTestId("dataset-run-button");
  await expect(runButton).toBeEnabled();
  await runButton.click();

  await expect(page.getByTestId("status-text")).toContainText("succeeded", { timeout: 180_000 });
  await expect(page.getByTestId("summary-stats")).toContainText("Series length");

  const explorer = page.getByTestId("two-way-explorer");
  await expect(explorer).toBeVisible();
  await expect
    .poll(async () => {
      const text = (await explorer.textContent()) ?? "";
      return text.includes("2-way explorer") || text.includes("Plotly library unavailable");
    })
    .toBe(true);
});
