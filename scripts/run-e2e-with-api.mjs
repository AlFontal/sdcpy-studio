import { spawn } from "node:child_process";
import process from "node:process";

const host = "127.0.0.1";
const port = "8000";
const baseUrl = `http://${host}:${port}`;
const healthUrl = `${baseUrl}/health`;
const timeoutMs = 90_000;

function spawnProcess(command, args, extraEnv = {}) {
  return spawn(command, args, {
    stdio: "inherit",
    env: { ...process.env, ...extraEnv },
  });
}

function commandForNpx() {
  return process.platform === "win32" ? "npx.cmd" : "npx";
}

async function waitForHealth(url, timeout) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // Keep retrying until timeout.
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
  throw new Error(`Timed out waiting for health check: ${url}`);
}

async function main() {
  const api = spawnProcess("uv", [
    "run",
    "uvicorn",
    "sdcpy_studio.main:create_app",
    "--factory",
    "--host",
    host,
    "--port",
    port,
  ]);

  const stopApi = () => {
    if (!api.killed) {
      api.kill("SIGTERM");
    }
  };

  process.on("SIGINT", stopApi);
  process.on("SIGTERM", stopApi);

  try {
    await waitForHealth(healthUrl, timeoutMs);
    const test = spawnProcess(commandForNpx(), ["playwright", "test"], {
      PLAYWRIGHT_TEST_BASE_URL: baseUrl,
    });

    const exitCode = await new Promise((resolve) => {
      test.on("exit", (code) => resolve(code ?? 1));
    });

    stopApi();
    process.exit(exitCode);
  } catch (error) {
    stopApi();
    console.error(String(error));
    process.exit(1);
  }
}

main();
