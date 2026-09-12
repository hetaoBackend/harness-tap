"""Browser regressions. Install with pip install -e '.[browser]' and playwright install chromium."""
from __future__ import annotations

import asyncio
import copy

import pytest
from aiohttp import web

from harness_tap.proxy import create_app
from harness_tap.store import JsonlTraceStore

playwright = pytest.importorskip("playwright.async_api")

MESSAGES = [
    {"role": "system", "content": "System context"},
    {"role": "developer", "content": "Developer context"},
    {"role": "user", "content": "Inspect the files"},
    {"role": "assistant", "content": "Checking now", "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a b.txt"}'}}
    ]},
    {"role": "tool", "tool_call_id": "call-1", "content": "Result " * 2000},
    {"role": "user", "content": "Continue"},
    {"role": "assistant", "content": "Done"},
]


@pytest.fixture
async def viewer(tmp_path):
    store = JsonlTraceStore(tmp_path, session_id="session-many")
    record = {
        "timestamp": "2026-09-12T09:00:00+00:00", "duration_ms": 1200,
        "request": {"path": "/v1/chat/completions", "body": {"model": "model-" + "x" * 80, "messages": MESSAGES}},
        "response": {"status": 200, "body": {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}},
    }
    for turn in range(1, 28):
        store.append({**record, "turn": turn})
    store.append({**record, "turn": 1, "session_id": "session-other", "timestamp": "2026-09-11T09:00:00+00:00"})
    runner = web.AppRunner(create_app(upstream_base_url="https://api.example.test/v1", trace_store=store))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with playwright.async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(f"http://127.0.0.1:{port}/viewer")
        await page.locator(".turn-card").last.wait_for()
        try:
            yield page, store, record
        finally:
            await browser.close()
            await runner.cleanup()


async def test_many_turns_remain_readable_and_last_turn_reachable(viewer):
    page, _, _ = viewer
    heights = await page.locator(".turn-card").evaluate_all("els => els.map(e => e.getBoundingClientRect().height)")
    assert len(heights) == 27
    assert min(heights) >= 48
    await page.locator('[data-record="26"]').click()
    assert await page.locator("#inspect-title").inner_text() == "Turn 27"


async def test_request_preserves_order_roles_and_tool_calls(viewer):
    page, _, _ = viewer
    await page.locator('[data-tab="request"]').click()
    cards = page.locator(".prompt-card")
    roles = await cards.evaluate_all('els => els.map(e => e.className.split(" ")[1])')
    assert roles == [message["role"] for message in MESSAGES]
    await cards.nth(3).locator("summary").click()
    await playwright.expect(cards.nth(3).locator("pre")).to_contain_text("read_file")
    assert "Checking now" in await cards.nth(3).inner_text()
    assert "read_file" in await cards.nth(3).inner_text()
    assert "call-1" in await cards.nth(3).inner_text()
    await cards.nth(4).locator("summary").click()
    await playwright.expect(cards.nth(4).locator("pre")).to_contain_text("call-1")
    assert MESSAGES[4]["content"] in await cards.nth(4).locator("pre").text_content()


@pytest.mark.parametrize("width", [390, 1024, 1280, 1440])
async def test_layout_fits_viewport_and_inspector_is_reachable(viewer, width):
    page, _, _ = viewer
    await page.set_viewport_size({"width": width, "height": 900})
    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    if width > 980:
        assert await page.locator(".inspect-body").evaluate("e => e.getBoundingClientRect().bottom <= innerHeight + 1")


async def test_delta_detects_replaced_context(viewer):
    page, store, record = viewer
    changed = copy.deepcopy(record)
    changed["request"]["body"]["messages"][2]["content"] = "Different instruction"
    store.append({**changed, "turn": 28})
    await page.evaluate("loadSessions()")
    await page.locator('[data-record="27"]').click()
    await page.locator('[data-tab="delta"]').click()
    assert "replaced" in (await page.locator("#payload").inner_text()).lower()


async def test_late_session_response_does_not_override_new_selection(viewer):
    page, _, _ = viewer
    async def delay(route):
        await asyncio.sleep(0.3)
        await route.continue_()
    await page.route("**/api/sessions/session-other", delay)
    await page.locator('[data-session="session-other"]').click()
    await page.locator('[data-session="session-many"]').click()
    await page.wait_for_timeout(500)
    assert await page.locator("#session-title").inner_text() == "session-many"


async def test_refresh_preserves_open_message_and_scroll_at_200_turns(viewer):
    page, store, record = viewer
    for turn in range(28, 201):
        store.append({**record, "turn": turn})
    await page.evaluate("loadSessions()")
    await page.locator("#jump-turn").select_option("199")
    card = page.locator(".prompt-card").nth(4)
    await card.locator("summary").click()
    await card.locator("pre").wait_for()
    await page.locator("#payload").evaluate("e => e.scrollTop = 600")
    before = await page.evaluate("({turns: $('turns').scrollTop, payload: $('payload').scrollTop})")
    store.append({**record, "turn": 201})
    await page.evaluate("loadSessions()")
    await playwright.expect(card.locator("pre")).to_be_attached()
    after = await page.evaluate("({turns: $('turns').scrollTop, payload: $('payload').scrollTop})")
    assert after == before
    assert await page.locator("#inspect-title").inner_text() == "Turn 200"
    assert await page.locator(".turn-card").count() == 201
    assert await page.locator(".turn-card").evaluate_all("els => els.every(e => e.getBoundingClientRect().height >= 48)")


async def test_selecting_and_expanding_are_independent_and_filter_navigation_works(viewer):
    page, store, record = viewer
    store.append({**record, "turn": 28, "response": {"status": 500, "body": {"error": "upstream failed"}}})
    await page.evaluate("loadSessions()")
    await page.locator('[data-expand-record="1"]').click()
    assert await page.locator("#inspect-title").inner_text() == "Turn 1"
    await page.locator('[data-record="1"]').click()
    assert await page.locator('[data-expand-record="1"]').get_attribute("aria-expanded") == "true"
    await page.locator("#errors-only").click()
    assert await page.locator(".turn-card").count() == 1
    await page.locator("#next-turn").click()
    assert await page.locator("#inspect-title").inner_text() == "Turn 28"
    await page.locator("#search").fill("no-such-text")
    assert await page.locator(".turn-card").count() == 0
    assert await page.locator("#jump-turn").is_disabled()
    await page.locator('[data-session="session-other"]').click()
    await playwright.expect(page.locator("#session-title")).to_have_text("session-other")
    assert await page.locator("#search").input_value() == ""
    assert await page.locator(".turn-card").count() == 1


async def test_failed_refresh_and_copy_report_errors_without_losing_payload(viewer):
    page, _, _ = viewer
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.route("**/api/sessions", lambda route: route.fulfill(status=503, body="unavailable"))
    await page.evaluate("loadSessions()")
    assert "failed" in await page.locator("#refresh-label").inner_text()
    assert await page.locator(".prompt-card").count() == len(MESSAGES)
    await page.unroute("**/api/sessions")
    await page.evaluate("loadSessions()")
    assert "auto refresh" in await page.locator("#refresh-label").inner_text()
    await page.evaluate('Object.defineProperty(navigator, "clipboard", {value: {writeText: async () => {throw new Error("denied")}}})')
    await page.locator("#copy").click()
    await playwright.expect(page.locator("#payload-label")).to_contain_text("Copy failed")
    assert not errors


async def test_raw_scroll_and_keyboard_focus_survive_refresh(viewer):
    page, store, record = viewer
    await page.locator('[data-tab="raw"]').click()
    await page.locator("#payload").evaluate("e => e.scrollTop = 500")
    assert await page.locator("#payload").evaluate("e => e.scrollTop") == 500
    store.append({**record, "turn": 28})
    await page.evaluate("loadSessions()")
    assert await page.locator("#payload").evaluate("e => e.scrollTop") == 500
    assert await page.evaluate('document.activeElement.dataset.tab') == "raw"
    await page.locator('[data-session="session-many"]').focus()
    await page.evaluate("loadSessions()")
    assert await page.evaluate('document.activeElement.dataset.session') == "session-many"
