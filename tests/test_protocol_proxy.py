import asyncio
import gzip
import json

import pytest
from aiohttp import ClientPayloadError, ClientSession, web
from multidict import CIMultiDict
from yarl import URL

from harness_tap.headers import filter_headers
from harness_tap.proxy import create_app
from harness_tap.store import JsonlTraceStore
from test_protocol_streams import frame


@pytest.fixture
async def serve():
    runners = []
    async def start(app):
        runner = web.AppRunner(app)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        return f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    yield start
    for runner in reversed(runners):
        await runner.cleanup()


@pytest.mark.parametrize('path,protocol', [('/responses', 'openai-responses'), ('/messages', 'anthropic-messages'), ('/chat/completions', 'openai-chat-completions')])
async def test_native_json_route_query_auth_and_wire_body(serve, tmp_path, path, protocol):
    seen = {}
    body = b'{ "model": "test", "input": "hello", "stream": false, "vendor_field": 2 }'
    result = b'{ "id": "native", "content": [], "output": [] }'
    async def upstream(request):
        seen.update(path=request.path, query=request.rel_url.raw_query_string,
                    body=await request.read(), headers=dict(request.headers))
        return web.Response(status=201, body=result, headers={'Content-Type': 'application/json', 'request-id': 'r1'})
    app = web.Application()
    app.router.add_post('/gateway/v1' + path, upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/gateway/v1/', trace_store=store))
    query = 'beta=true&x=a%2Fb&x=a+b&x=%2B'
    async with ClientSession() as client:
        async with client.post(URL(proxy + '/v1' + path + '?' + query, encoded=True), data=body, headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer trace-only', 'x-api-key': 'trace-only',
            'anthropic-version': '2023-06-01', 'anthropic-beta': 'test-beta', 'Connection': 'X-Hop', 'X-Hop': 'drop',
        }) as response:
            assert await response.read() == result
            assert response.status == 201 and response.headers['request-id'] == 'r1'
    assert seen['path'] == '/gateway/v1' + path
    assert seen['query'] == query and seen['body'] == body
    assert seen['headers']['Authorization'] == 'Bearer trace-only'
    assert seen['headers']['x-api-key'] == 'trace-only'
    assert seen['headers']['anthropic-version'] == '2023-06-01'
    assert seen['headers']['anthropic-beta'] == 'test-beta'
    assert 'X-Hop' not in seen['headers']
    records = store.load_session(store.session_id)
    assert len(records) == 1
    assert records[0]['capture']['protocol'] == protocol
    assert records[0]['schema_version'] == 2
    assert records[0]['request']['headers']['Authorization'] == '***'
    assert records[0]['request']['headers']['x-api-key'] == '***'
    assert records[0]['request']['query_string'] == query
    assert records[0]['response']['body'] == json.loads(result)


async def test_per_protocol_overrides_and_cookie_isolation(serve, tmp_path):
    calls = []
    async def upstream(request):
        calls.append((request.path, request.headers.get('Cookie')))
        return web.json_response({'ok': True}, headers={'Set-Cookie': 'leak=secret'})
    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/default/v1', responses_upstream=up + '/openai/v1', messages_upstream=up + '/anthropic/v1', trace_store=store))
    for suffix in ['/chat/completions', '/responses', '/messages']:
        async with ClientSession() as client:
            async with client.post(proxy + '/v1' + suffix, json={}) as response:
                await response.read()
    assert calls == [('/default/v1/chat/completions', None), ('/openai/v1/responses', None), ('/anthropic/v1/messages', None)]


@pytest.mark.parametrize('path,first,last,output', [
    ('/responses', frame('response.created', response={'id': 'r1', 'output': []}), frame('response.completed', response={'id': 'r1', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hi'}]}], 'status': 'completed'}), 'output'),
    ('/messages', frame('message_start', message={'id': 'm1', 'content': [], 'usage': {'input_tokens': 1}}), frame('message_stop'), 'content'),
])
async def test_stream_first_frame_arrives_before_upstream_completion(serve, tmp_path, path, first, last, output):
    release = asyncio.Event()
    async def upstream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(first)
        await asyncio.wait_for(release.wait(), 3)
        await response.write(last)
        return response
    app = web.Application()
    app.router.add_post('/v1' + path, upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1' + path, json={'stream': True}) as response:
            try:
                received = await asyncio.wait_for(response.content.readexactly(len(first)), 2)
                assert received == first
            finally:
                release.set()
            assert received + await response.read() == first + last
    record = store.load_session(store.session_id)[0]
    assert record['capture']['model_status'] == 'completed'
    assert not record['capture'].get('partial')
    assert output in record['response']['body']


@pytest.mark.parametrize('path', ['/responses', '/messages'])
@pytest.mark.parametrize('status', [400, 401, 429, 500])
async def test_stream_request_json_error_remains_json(serve, tmp_path, path, status):
    async def upstream(request):
        return web.json_response({'error': {'message': 'Rejected'}}, status=status)
    app = web.Application()
    app.router.add_post('/v1' + path, upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1' + path, json={'stream': True}) as response:
            assert response.status == status
            assert await response.json() == {'error': {'message': 'Rejected'}}
    record = store.load_session(store.session_id)[0]
    assert record['transport'] == 'http' and 'sse_events' not in record['response']


@pytest.mark.parametrize('sse', [False, True])
async def test_gzip_bytes_and_headers_survive_with_decoded_capture(serve, tmp_path, sse):
    raw = frame('message_start', message={'content': []}) + frame('message_stop') if sse else b'{"content": [{"type":"text","text":"Hi"}]}'
    compressed = gzip.compress(raw)
    seen = []
    async def upstream(request):
        seen.append(await request.json())
        return web.Response(body=compressed, headers={'Content-Encoding': 'gzip', 'Content-Type': 'text/event-stream' if sse else 'application/json'})
    app = web.Application()
    app.router.add_post('/v1/messages', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession(auto_decompress=False) as client:
        async with client.post(proxy + '/v1/messages', data=gzip.compress(b'{"model":"test"}'), headers={'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}) as response:
            assert response.headers['Content-Encoding'] == 'gzip'
            assert await response.read() == compressed
    assert seen == [{'model': 'test'}]
    record = store.load_session(store.session_id)[0]
    assert record['request']['body'] == {'model': 'test'}
    assert 'content' in record['response']['body']
    assert not record['capture'].get('parse_errors')


async def test_stream_capture_budget_does_not_truncate_wire(serve, tmp_path):
    raw = frame('response.created', response={'output': []}) + frame('response.output_text.delta', output_index=0, content_index=0, delta='x' * 10000)
    async def upstream(request):
        return web.Response(body=gzip.compress(raw), headers={'Content-Type': 'text/event-stream', 'Content-Encoding': 'gzip'})
    app = web.Application()
    app.router.add_post('/v1/responses', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store, max_capture_bytes=200))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1/responses', json={}) as response:
            assert await response.read() == raw
    record = store.load_session(store.session_id)[0]
    assert record['capture']['response_truncated'] and record['capture']['partial']
    assert len(json.dumps(record)) < 3000


@pytest.mark.parametrize('abrupt', [False, True])
async def test_missing_terminal_and_broken_transport_record_once(serve, tmp_path, abrupt):
    first = frame('message_start', message={'content': []})
    async def upstream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(first)
        if abrupt:
            request.transport.close()
        return response
    app = web.Application()
    app.router.add_post('/v1/messages', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1/messages', json={}) as response:
            assert response.status == 200
            if abrupt:
                with pytest.raises(ClientPayloadError):
                    await response.read()
            else:
                assert await response.read() == first
    records = store.load_session(store.session_id)
    assert len(records) == 1
    assert records[0]['response']['status'] == 200
    assert records[0]['capture']['partial']
    assert bool(records[0]['capture'].get('transport_error')) == abrupt


async def test_malformed_events_are_forwarded_and_captured(serve, tmp_path):
    raw = frame('message_start', message={'content': []}) + b'event: bad\ndata: junk\n\n' + frame('error', error={'message': 'Busy'})
    async def upstream(request):
        return web.Response(body=raw, content_type='text/event-stream')
    app = web.Application()
    app.router.add_post('/v1/messages', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1/messages', json={}) as response:
            assert await response.read() == raw
    record = store.load_session(store.session_id)[0]
    assert record['capture']['parse_errors']
    assert record['capture']['stream_error'] == {'message': 'Busy'}
    assert record['response']['sse_events'][1]['raw_data'] == 'junk'


def test_connection_header_tokens_are_removed_case_insensitively():
    headers = CIMultiDict([('Connection', 'keep-alive, X-Private'), ('X-Private', 'private'), ('anthropic-beta', 'keep')])
    assert 'X-Private' not in filter_headers(headers)
    assert filter_headers(headers)['anthropic-beta'] == 'keep'


async def test_duplicate_response_headers_preserved_and_redirect_not_followed(serve, tmp_path):
    followups = []
    async def upstream(request):
        if request.path.endswith('/target'):
            followups.append(True)
        return web.Response(status=307, headers=CIMultiDict([
            ('Location', '/target'), ('Set-Cookie', 'a=1'), ('Set-Cookie', 'b=2'),
        ]))
    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1/responses', json={}, allow_redirects=False) as response:
            await response.read()
            assert response.status == 307
            assert response.headers.getall('Set-Cookie') == ['a=1', 'b=2']
    assert not followups
    assert store.load_session(store.session_id)[0]['response']['headers']['Set-Cookie'] == '***'


@pytest.mark.parametrize('send_headers', [False, True])
async def test_read_timeout_before_or_after_headers_is_recorded_once(serve, tmp_path, send_headers):
    release = asyncio.Event()
    async def upstream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        if send_headers:
            await response.prepare(request)
            await response.write(frame('message_start', message={'content': []}))
        await release.wait()
        return response
    app = web.Application()
    app.router.add_post('/v1/messages', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store, read_timeout=.05))
    try:
        async with ClientSession() as client:
            async with client.post(proxy + '/v1/messages', json={}) as response:
                if send_headers:
                    assert response.status == 200
                    with pytest.raises(ClientPayloadError):
                        await response.read()
                else:
                    assert response.status == 504
                    assert (await response.json())['error']
    finally:
        release.set()
    records = store.load_session(store.session_id)
    assert len(records) == 1
    assert records[0]['response']['status'] == (200 if send_headers else 504)
    assert records[0]['capture']['transport_error']


async def test_client_disconnect_records_one_partial_trace(serve, tmp_path):
    saved = asyncio.Event()
    release = asyncio.Event()
    class NotifyingStore(JsonlTraceStore):
        def append(self, record):
            result = super().append(record)
            saved.set()
            return result
    async def upstream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(frame('message_start', message={'content': []}))
        await release.wait()
        try:
            for _ in range(20):
                await response.write(b': ping\n\n')
                await asyncio.sleep(.005)
        except ConnectionResetError:
            pass
        return response
    app = web.Application()
    app.router.add_post('/v1/messages', upstream)
    up = await serve(app)
    store = NotifyingStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store))
    async with ClientSession() as client:
        response = await client.post(proxy + '/v1/messages', json={})
        assert await response.content.readany()
        response.close()
        release.set()
        await asyncio.wait_for(saved.wait(), 2)
    records = store.load_session(store.session_id)
    assert len(records) == 1 and records[0]['capture']['partial']


@pytest.mark.parametrize('path', ['/responses', '/messages'])
async def test_new_routes_reject_method_and_oversize_request(serve, tmp_path, path):
    calls = []
    async def upstream(request):
        calls.append(True)
        return web.json_response({})
    app = web.Application()
    app.router.add_post('/v1' + path, upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store, max_request_bytes=32))
    async with ClientSession() as client:
        async with client.get(proxy + '/v1' + path) as response:
            assert response.status == 405
        async with client.post(proxy + '/v1' + path, data=b'x' * 100) as response:
            assert response.status == 413
    assert not calls
    records = store.load_session(store.session_id)
    assert len(records) == 1 and records[0]['capture']['rejected']


async def test_request_capture_limit_preserves_full_forwarded_payload(serve, tmp_path):
    raw = json.dumps({'input': 'x' * 1000}).encode()
    async def upstream(request):
        assert await request.read() == raw
        return web.json_response({'id': 'r1', 'output': []})
    app = web.Application()
    app.router.add_post('/v1/responses', upstream)
    up = await serve(app)
    store = JsonlTraceStore(tmp_path)
    proxy = await serve(create_app(upstream_base_url=up + '/v1', trace_store=store, max_capture_bytes=64))
    async with ClientSession() as client:
        async with client.post(proxy + '/v1/responses', data=raw) as response:
            assert response.status == 200
            await response.read()
    record = store.load_session(store.session_id)[0]
    assert record['capture']['request_truncated']
    assert len(record['request']['body']) == 64
