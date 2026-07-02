# Harness Tap Rename Design

## Context

The project started with a client-specific name for a local reverse proxy and
trace viewer. The implemented tool is broader than that original client: any
harness or agent runtime that can configure an OpenAI-compatible base URL can
send traffic through the proxy and view the captured trace.

The product name should reflect that broader scope.

## Decision

Rename the product-facing surface to `Harness Tap`.

Use a full migration:

- Present the app, README, CLI output, and viewer as `Harness Tap`.
- Replace the console script with `harness-tap`.
- Change the default trace directory to `.harness-tap/traces`.
- Rename the Python module to `harness_tap`.
- Remove previous-name references from tracked source, tests, and current docs.

## Product Language

The new language should describe the tool as a local OpenAI-compatible trace
proxy and viewer for configurable harnesses. It should avoid implying that any
one client is the only supported client.

Acceptable client examples include local agents, test harnesses, coding tools,
and other OpenAI-compatible clients.

## Migration Scope

Update:

- `README.md` title, description, examples, and trace directory references.
- `pyproject.toml` project metadata and console scripts.
- CLI parser program name, default trace paths, and startup output.
- Viewer title/header text.
- Capture metadata client label to a generic harness value.
- Tests that assert product name, command metadata, or default trace paths.
- Python package directory and imports to `harness_tap`.
- Existing design and implementation docs so tracked documentation uses the
  `Harness Tap` name consistently.

Do not update in this pass:

- Existing git history.
- Existing local trace payload contents. Captured prompts may still contain
  client-generated strings from old sessions.
- The workspace directory name or git branch name, unless the user explicitly
  asks for that filesystem or branch migration.

## Compatibility

Users should run:

```bash
.venv/bin/harness-tap serve --trace-dir .harness-tap/traces
```

If local legacy trace data exists, migrate it to `.harness-tap/traces` without
deleting captured records. Do not rewrite captured JSONL payloads just to remove
historical client text.

## Testing

Verification should include:

- Full Python test suite.
- Python compile check for `harness_tap` and `tests`.
- A browser reload of `/viewer` confirming the UI title/header reads
  `Harness Tap`.
- A quick CLI help check confirming `harness-tap` is the command.
- A tracked-file search confirming no previous-name references remain in source,
  tests, README, or docs.
