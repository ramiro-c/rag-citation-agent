"""Tests for robust JSON parsing of LLM responses (free models are sloppy)."""

import pytest

from rag_agent.llm import parse_json_response


def test_plain_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_json_in_code_fence():
    content = '```json\n{"answer": "Paris"}\n```'
    assert parse_json_response(content) == {"answer": "Paris"}


def test_json_with_surrounding_prose():
    content = 'Here you go:\n{"answer": "Paris"}\nHope that helps!'
    assert parse_json_response(content) == {"answer": "Paris"}


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="JSON"):
        parse_json_response("no json here at all")
