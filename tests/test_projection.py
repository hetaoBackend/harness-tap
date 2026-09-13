import copy

from aiohttp import ClientSession

from harness_tap.cli import main
from harness_tap.projection import project_record
from harness_tap.proxy import create_app
from harness_tap.store import JsonlTraceStore
from test_protocol_proxy import serve


def responses_record():
    return {
        'turn': 1, 'timestamp': '2026-09-12T09:00:00+00:00', 'duration_ms': 12,
        'request': {'path': '/v1/responses', 'body': {
            'model': 'test', 'instructions': 'Use tools carefully', 'previous_response_id': 'resp_old',
            'input': [
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Look up'}]},
                {'type': 'function_call', 'call_id': 'c1', 'name': 'lookup', 'arguments': '{}'},
                {'type': 'function_call_output', 'call_id': 'c1', 'output': 'Found'},
                {'type': 'item_reference', 'id': 'opaque'},
            ], 'tools': [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}],
        }},
        'response': {'status': 200, 'body': {'id': 'r1', 'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
            'output': [{'type': 'reasoning', 'encrypted_content': 'opaque-value'},
                       {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Answer'}]},
                       {'type': 'function_call', 'name': 'other', 'call_id': 'c2', 'arguments': '{}'}],
            'usage': {'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14, 'output_tokens_details': {'reasoning_tokens': 2}},
        }},
    }


def messages_record():
    return {
        'turn': 2, 'timestamp': '2026-09-12T09:00:01+00:00',
        'request': {'path': '/v1/messages', 'body': {
            'model': 'test', 'system': [{'type': 'text', 'text': 'Be careful', 'cache_control': {'type': 'ephemeral'}}],
            'messages': [
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'lookup', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'Found'}]},
            ], 'tools': [{'name': 'lookup', 'input_schema': {'type': 'object'}}],
        }},
        'response': {'status': 200, 'body': {'id': 'm1', 'stop_reason': 'tool_use',
            'content': [{'type': 'thinking', 'thinking': 'Think', 'signature': 'opaque-signature'},
                        {'type': 'text', 'text': 'Here it is'},
                        {'type': 'tool_use', 'id': 't2', 'name': 'other', 'input': {'q': 'hi'}},
                        {'type': 'future_block', 'payload': {'keep': True}}],
            'usage': {'input_tokens': 3, 'output_tokens': 4, 'cache_read_input_tokens': 8, 'cache_creation_input_tokens': 5,
                      'cache_creation': {'ephemeral_5m_input_tokens': 5}},
        }},
    }


def test_responses_projection_preserves_items_sources_and_server_history():
    record = responses_record()
    before = copy.deepcopy(record)
    view = project_record(record)
    assert record == before
    assert view['protocol'] == 'openai-responses'
    assert [i['source_path'] for i in view['input_items']] == ['instructions', 'input[0]', 'input[1]', 'input[2]', 'input[3]']
    assert view['input_items'][3]['role'] == 'tool'
    assert view['input_items'][3]['value']['call_id'] == 'c1'
    assert view['context_references'] == {'previous_response_id': 'resp_old'}
    assert 'server-side history' in view['context_note']
    assert len(view['output_items']) == 3
    assert view['output_items'][0]['value']['encrypted_content'] == 'opaque-value'
    assert view['outcome']['is_error'] and view['outcome']['status'] == 'incomplete'
    assert view['usage']['total_tokens'] == 14


def test_messages_projection_preserves_blocks_tool_ids_and_cache_usage():
    record = messages_record()
    view = project_record(record)
    assert view['protocol'] == 'anthropic-messages'
    assert view['input_items'][0]['source_path'] == 'system'
    assert view['input_items'][2]['value']['content'][0]['tool_use_id'] == 't1'
    assert view['output_items'][0]['value']['signature'] == 'opaque-signature'
    assert view['output_items'][3]['value']['payload'] == {'keep': True}
    assert view['tools'][0]['input_schema'] == {'type': 'object'}
    assert view['usage']['total_tokens'] == 20
    assert not view['outcome']['is_error']


def test_legacy_chat_trace_and_string_input():
    legacy = project_record({'request': {'body': {'messages': [{'role': 'user', 'content': 'Hi'}]}},
                             'response': {'body': {'choices': [{'message': {'content': 'OK'}}]}}})
    assert legacy['protocol'] == 'openai-chat-completions'
    assert legacy['output_items'][0]['text'] == 'OK'
    view = project_record({'request': {'path': '/v1/responses', 'body': {'input': 'Hello'}}})
    assert view['input_items'][0]['value'] == 'Hello'
    assert view['usage']['total_tokens'] is None


async def test_viewer_api_returns_projection_without_changing_raw_record(serve, tmp_path):
    store = JsonlTraceStore(tmp_path)
    store.append(responses_record())
    store.append(messages_record())
    stored = store.load_session(store.session_id)
    proxy = await serve(create_app(upstream_base_url='https://example.test/v1', trace_store=store))
    async with ClientSession() as client:
        async with client.get(proxy + '/api/sessions/' + store.session_id) as response:
            result = await response.json()
    assert result['records'] == stored
    assert result['session']['error_count'] == 1
    assert result['session']['total_tokens'] == 34
    assert result['projections'][0]['context_note']
    assert result['turns'][0]['message_count'] == 5
    assert result['turns'][1]['protocol'] == 'anthropic-messages'


def test_cli_prints_native_protocol_inputs_tools_outputs_and_status(tmp_path, capsys):
    store = JsonlTraceStore(tmp_path)
    store.append(responses_record())
    store.append(messages_record())
    assert main(['inspect', '--trace-dir', str(tmp_path), store.session_id]) == 0
    text = capsys.readouterr().out
    for needle in ['openai-responses', 'anthropic-messages', 'Use tools carefully', 'Look up', 'Found',
                   'lookup', 'Answer', 'Think', 'Here it is', 'server-side history', 'incomplete', 'cache_read_input_tokens']:
        assert needle in text


def test_http_200_stream_error_and_capture_errors_are_visible():
    record = messages_record()
    record['capture'] = {'stream_error': {'message': 'Busy'}, 'partial': True}
    view = project_record(record)
    assert view['outcome']['is_error']
    assert view['outcome']['error'] == {'message': 'Busy'}


def test_malformed_native_fields_are_still_inspectable():
    record = {'capture': None, 'request': {'path': '/v1/responses', 'body': {
        'input': [{'type': {'invalid': True}, 'role': ['invalid']}, None],
        'tools': [None, 'invalid', {'function': 'invalid'}],
    }}, 'response': {'status': 400, 'body': {'status': {'invalid': True}, 'error': 'Invalid input'}}}
    view = project_record(record)
    assert view['outcome']['is_error']
    assert view['input_items'][0]['value'] == record['request']['body']['input'][0]


def test_native_empty_output_is_not_synthesized_as_an_assistant_item():
    record = messages_record()
    record['response']['body']['content'] = []
    assert project_record(record)['output_items'] == []
