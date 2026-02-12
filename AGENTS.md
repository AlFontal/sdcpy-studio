# Agent Runbook

## Run the App

```bash
npm run dev:api
```

The API/UI is served at `http://127.0.0.1:8000`.

## Run E2E Tests

```bash
npm run test:e2e:with-api
```

This command starts the FastAPI app, waits for `/health`, runs Playwright tests, and stops the API process.

If the app is already running:

```bash
npm run test:e2e
```

## UI Bug Reproduction with Playwright MCP

Before changing UI behavior:

1. Reproduce the issue with Playwright MCP (navigate, interact, screenshot).
2. Save a screenshot and note the exact failing user path.
3. Apply the minimal UI/API change.
4. Re-run Playwright E2E and include results.

Example smoke command:

```bash
codex exec -C . "Use MCP Playwright to open http://127.0.0.1:8000 and take a screenshot."
```

## Troubleshooting

- MCP not detected in VS Code:
  - Confirm server exists with `codex mcp list`.
  - Use CLI first (`codex exec ...`) to verify MCP works, then reload VS Code.
- Playwright browser missing:
  - Run `npx playwright install`.
- Linux headless environments:
  - Use `headless: true` (already configured).
  - If needed for non-headless debugging, set display env (for example `DISPLAY=:99`) and run with Xvfb.
