# SPDX-License-Identifier: Apache-2.0
"""V4.1 DSML terminals on the vLLM streaming parser engine."""

import contextlib
import json
from dataclasses import replace
from functools import cache

import regex as re
from vllm.parser.deepseek_v4 import deepseek_v4_config
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter, ParserEngineToolAdapter
from vllm.parser.engine.parser_engine import ParserEngine

from .tokenizer import thinking_enabled

PARAMETER_PATTERN = re.compile(
    r'<｜DSML｜ parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜ parameter>', re.DOTALL
)
PARTIAL_PARAMETER_PATTERN = re.compile(r'<｜DSML｜ parameter name="([^"]+)" string="(true|false)">(.*)$', re.DOTALL)


def convert_arguments(raw_args, partial):
    params = {}
    end = 0
    for match in PARAMETER_PATTERN.finditer(raw_args):
        name, string, value = match.groups()
        if name in params:
            raise ValueError(f"Duplicate V4.1 tool parameter: {name}")
        params[name] = value if string == "true" else json.loads(value)
        end = match.end()
    if partial and (partial_match := PARTIAL_PARAMETER_PATTERN.search(raw_args, end)):
        name, string, value = partial_match.groups()
        if string == "true":
            params[name] = value
        else:
            with contextlib.suppress(json.JSONDecodeError):
                params[name] = json.loads(value)
    return json.dumps(params, ensure_ascii=False)


@cache
def deepseek_v41_config(thinking):
    config = deepseek_v4_config(thinking=thinking)
    # Override the V4 wire markers explicitly: upstream TOOL_START can be a
    # tuple of V4 aliases, which must not become valid V4.1 tool delimiters.
    terminal_overrides = {
        "TOOL_START": "\n\n<｜DSML｜ calls>",
        "TOOL_END": "</｜DSML｜ calls>",
        "INVOKE_PREFIX": '<｜DSML｜ invoke name="',
        "INVOKE_END": "</｜DSML｜ invoke>",
        "PARAM_START": "<｜DSML｜ parameter",
        "PARAM_CLOSE": "</｜DSML｜ parameter>",
    }
    return replace(
        config,
        name="deepseek_v41",
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            name: terminal_overrides.get(name, text)
            for name, text in config.token_id_terminals.items()
            if name != "TOOL_START"
        },
        arg_converter=convert_arguments,
        strip_trailing_reasoning_whitespace=False,
        drop_whitespace_only_content_before_tools=False,
        strip_content_whitespace_with_tools=False,
        # The DSML marker may be a special token inside a multi-token tag.
        preserve_tokens=config.preserve_tokens | {"｜DSML｜"},
    )


class DeepseekV41Parser(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        super().__init__(
            tokenizer, tools, parser_engine_config=deepseek_v41_config(thinking_enabled(chat_kwargs)), **kwargs
        )

    def _fix_arg_types(self, args_json, func_name):
        # The wire format's string flag is authoritative, including values
        # such as "true" and "42" that happen to resemble another JSON type.
        return args_json


class DeepseekV41ReasoningParser(ParserEngineReasoningAdapter):
    _parser_engine_cls = DeepseekV41Parser


class DeepseekV41ToolParser(ParserEngineToolAdapter):
    _parser_engine_cls = DeepseekV41Parser
    structural_tag_model = "deepseek_v41"

    def get_structural_tag(self, request, *, reasoning=False):
        # Imported lazily: grammar compilation is unnecessary for auto tools.
        from .structural_tag import get_structural_tag

        return get_structural_tag(request, reasoning=reasoning)
