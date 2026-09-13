import json

import pytest

from harness_tap.sse import MessagesSSEReassembler, ResponsesSSEReassembler, SSEDecoder


def frame(kind, **data):
    return f"event: {kind}\r\ndata: {json.dumps({'type': kind, **data}, ensure_ascii=False)}\r\n\r\n".encode()


def feed_split(parser, chunks):
    for byte in b"".join(chunks):
        parser.feed_bytes(bytes([byte]))
    parser.finish()
    return parser.final_response()


def test_sse_framing_preserves_metadata_comments_multiline_and_utf8():
    events = []
    parser = SSEDecoder(events.append)
    data = '\ufeff: heartbeat\r\nevent: custom\rid: 2\nretry: 50\ndata: {"text":\ndata: "你好"}\n\n'.encode()
    for byte in data:
        parser.feed_bytes(bytes([byte]))
    assert parser.finish() is False
    assert events == [{"comments": [" heartbeat"], "event": "custom", "id": "2", "retry": "50", "raw_data": '{"text":\n"你好"}'}]


@pytest.mark.parametrize('status', ['completed', 'failed', 'incomplete'])
def test_responses_terminal_snapshot_is_authoritative(status):
    parser = ResponsesSSEReassembler()
    final = {"id": "resp_1", "object": "response", "status": status,
             "output": [{"id": "msg_1", "type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": "你好", "annotations": []}]}],
             "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
             "extra_future_field": {"keep": True}}
    if status == 'incomplete':
        final['incomplete_details'] = {'reason': 'max_output_tokens'}
    if status == 'failed':
        final['error'] = {'message': 'failed'}
    result = feed_split(parser, [
        frame('response.created', response={'id': 'resp_1', 'output': []}),
        frame('response.output_item.added', output_index=0, item={'type': 'message', 'content': []}),
        frame('response.content_part.added', output_index=0, content_index=0, part={'type': 'output_text', 'text': ''}),
        frame('response.output_text.delta', output_index=0, content_index=0, delta='你'),
        frame('response.output_text.delta', output_index=0, content_index=0, delta='好'),
        frame('response.future_event', something={'opaque': True}),
        frame('response.' + status, response=final),
    ])
    assert result == final
    assert parser.done and not parser.errors
    assert parser.model_status == status
    assert parser.events[-2]['data']['something'] == {'opaque': True}


def test_responses_partial_multiple_tools_reasoning_and_output_items():
    parser = ResponsesSSEReassembler()
    result = feed_split(parser, [
        frame('response.created', response={'id': 'resp_1', 'output': [], 'status': 'in_progress'}),
        frame('response.output_item.added', output_index=0, item={'type': 'reasoning', 'id': 'rs_1', 'summary': []}),
        frame('response.reasoning_summary_part.added', output_index=0, summary_index=0, part={'type': 'summary_text', 'text': ''}),
        frame('response.reasoning_summary_text.delta', output_index=0, summary_index=0, delta='Think'),
        frame('response.output_item.added', output_index=2, item={'type': 'function_call', 'call_id': 'c2', 'name': 'write', 'arguments': ''}),
        frame('response.output_item.added', output_index=1, item={'type': 'function_call', 'call_id': 'c1', 'name': 'read', 'arguments': ''}),
        frame('response.function_call_arguments.delta', output_index=1, delta='{"p":'),
        frame('response.function_call_arguments.delta', output_index=2, delta='{}'),
        frame('response.function_call_arguments.delta', output_index=1, delta='"a"}'),
        frame('response.function_call_arguments.done', output_index=1, arguments='{"p":"a"}'),
    ])
    assert not parser.done
    assert result['output'][0]['summary'][0]['text'] == 'Think'
    assert [i['call_id'] for i in result['output'][1:]] == ['c1', 'c2']
    assert result['output'][1]['arguments'] == '{"p":"a"}'


def test_messages_tool_thinking_signature_text_usage_and_unknown_events():
    parser = MessagesSSEReassembler()
    result = feed_split(parser, [
        frame('message_start', message={'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'content': [],
                                       'usage': {'input_tokens': 5, 'output_tokens': 1, 'cache_read_input_tokens': 10}}),
        frame('ping'),
        frame('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': '', 'signature': ''}),
        frame('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'Think'}),
        frame('content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': 'opaque'}),
        frame('content_block_stop', index=0),
        frame('content_block_start', index=1, content_block={'type': 'text', 'text': ''}),
        frame('content_block_delta', index=1, delta={'type': 'text_delta', 'text': '你好'}),
        frame('content_block_stop', index=1),
        frame('content_block_start', index=2, content_block={'type': 'tool_use', 'id': 't1', 'name': 'lookup', 'input': {}}),
        frame('content_block_delta', index=2, delta={'type': 'input_json_delta', 'partial_json': '{"q":'}),
        frame('content_block_delta', index=2, delta={'type': 'input_json_delta', 'partial_json': '"北京"}'}),
        frame('content_block_stop', index=2),
        frame('content_block_start', index=3, content_block={'type': 'tool_use', 'id': 't2', 'name': 'other', 'input': {}}),
        frame('content_block_delta', index=3, delta={'type': 'input_json_delta', 'partial_json': '{}'}),
        frame('content_block_stop', index=3),
        frame('future_event', opaque={'keep': True}),
        frame('message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None}, usage={'output_tokens': 4}),
        frame('message_delta', delta={}, usage={'output_tokens': 9}),
        frame('message_stop'),
    ])
    assert parser.done and not parser.errors
    assert result['content'][0] == {'type': 'thinking', 'thinking': 'Think', 'signature': 'opaque'}
    assert result['content'][1]['text'] == '你好'
    assert result['content'][2]['input'] == {'q': '北京'}
    assert result['content'][3]['input'] == {}
    assert result['usage'] == {'input_tokens': 5, 'output_tokens': 9, 'cache_read_input_tokens': 10}
    assert result['stop_reason'] == 'tool_use'
    assert any(e['data']['type'] == 'future_event' for e in parser.events)


@pytest.mark.parametrize('cls,terminal', [(ResponsesSSEReassembler, frame('response.completed', response={'output': []})),
                                         (MessagesSSEReassembler, frame('message_stop'))])
def test_incomplete_terminal_frame_is_not_completion(cls, terminal):
    parser = cls()
    parser.feed_bytes(terminal[:-2])
    parser.finish()
    assert not parser.done
    assert parser.errors
    assert parser.events[-1]['unterminated']


def test_messages_partial_tool_input_and_stream_error_remain_inspectable():
    parser = MessagesSSEReassembler()
    feed_split(parser, [
        frame('message_start', message={'content': []}),
        frame('content_block_start', index=0, content_block={'type': 'tool_use', 'id': 't1', 'input': {}}),
        frame('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': '{"p":'}),
        b'event: malformed\ndata: not json\n\n',
        frame('error', error={'type': 'overloaded_error', 'message': 'Busy'}),
    ])
    assert not parser.done and parser.model_status == 'failed'
    assert parser.capture_state()['partial_tool_inputs'] == {'0': '{"p":'}
    assert parser.events[-2]['raw_data'] == 'not json'
    assert parser.errors


def test_messages_invalid_tool_json_does_not_block_later_events():
    parser = MessagesSSEReassembler()
    result = feed_split(parser, [
        frame('message_start', message={'content': []}),
        frame('content_block_start', index=0, content_block={'type': 'tool_use', 'input': {}}),
        frame('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': 'broken'}),
        frame('content_block_stop', index=0),
        frame('message_delta', delta={'stop_reason': 'tool_use'}, usage={'output_tokens': 1}),
        frame('message_stop'),
    ])
    assert parser.done and parser.errors
    assert result['usage']['output_tokens'] == 1


def test_invalid_message_shapes_are_recorded_without_poisoning_reconstruction():
    parser = MessagesSSEReassembler()
    feed_split(parser, [
        frame('message_start', message=[]),
        frame('content_block_start', index='bad', content_block={'type': 'text'}),
        frame('message_start', message={'content': []}),
        frame('content_block_start', index=0, content_block={'type': 'text', 'text': 'OK'}),
        frame('content_block_stop', index=0),
        frame('message_stop'),
    ])
    assert parser.final_response()['content'][0]['text'] == 'OK'
    assert len(parser.errors) == 2
