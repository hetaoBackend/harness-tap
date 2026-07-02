# Harness Tap Full Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete the rename so tracked project files use `Harness Tap`,
`harness-tap`, and `harness_tap` consistently.

**Architecture:** Keep the proxy, store, SSE, and viewer behavior unchanged.
Rename the package surface, command surface, default trace directory, docs, and
tests in one focused migration.

**Tech Stack:** Python 3.11, aiohttp, pytest, editable setuptools install, local
browser verification.

---

### Task 1: Rename Package And CLI Surface

**Files:**
- Move: legacy package directory to `harness_tap/`
- Modify: `pyproject.toml`
- Modify: `harness_tap/cli.py`
- Modify: `harness_tap/__main__.py`
- Modify: all tests importing the package

- [x] **Step 1: Move the package directory**

Run:

```bash
git mv "$(printf '\172\143\157\144\145_tap')" harness_tap
```

- [x] **Step 2: Update package metadata**

Set the project name to `harness-tap`, description to generic harness language,
and console script to:

```toml
[project.scripts]
harness-tap = "harness_tap.cli:main"
```

- [x] **Step 3: Update imports and CLI defaults**

Replace package imports with `harness_tap`, parser prog with `harness-tap`,
default trace directory with `.harness-tap/traces`, and startup copy with
generic OpenAI-compatible harness wording.

- [x] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py tests/test_proxy.py tests/test_viewer.py -q
```

Expected: all selected tests pass.

### Task 2: Update Product Copy And Documentation

**Files:**
- Modify: `README.md`
- Modify: `harness_tap/viewer.py`
- Rename and modify: existing design and implementation docs under
  `docs/superpowers/`

- [x] **Step 1: Update README**

Use `Harness Tap`, `harness-tap`, `.harness-tap/traces`, and generic
OpenAI-compatible harness setup language.

- [x] **Step 2: Update viewer copy**

Change the browser title and header to `Harness Tap Trace Viewer`.

- [x] **Step 3: Rename current docs**

Use `git mv` so existing current docs are named around `harness-tap`, then edit
their content to use generic harness language.

- [x] **Step 4: Run product-copy tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_viewer.py -q
```

Expected: all viewer tests pass.

### Task 3: Verify Clean Migration

**Files:**
- Local trace directory: `.harness-tap/traces`
- Local virtual environment entrypoints

- [x] **Step 1: Reinstall editable package**

Run:

```bash
.venv/bin/python -m pip uninstall -y harness-tap
.venv/bin/python -m pip install -e '.[dev]'
```

Expected: `.venv/bin/harness-tap` exists.

- [x] **Step 2: Migrate local trace data**

If legacy local trace data exists, move or copy it into `.harness-tap/traces`
without editing captured JSONL payloads.

- [x] **Step 3: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall harness_tap tests
.venv/bin/harness-tap --help
git grep -ni "$(printf '\172\143\157\144\145')" -- ':!*.jsonl'
```

Expected: tests pass, compile succeeds, help renders, and tracked-file search
returns no matches.

- [x] **Step 4: Browser verification**

Restart the local server with `harness-tap`, reload `/viewer`, and confirm the
viewer title/header reads `Harness Tap Trace Viewer`.
