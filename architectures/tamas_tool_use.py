from __future__ import annotations

import re
import ast
import asyncio
import inspect
import nest_asyncio
from typing import Any

def build_tool_list_prompt(tools: dict[str, Any]) -> str:

    if not tools:
        return "No tools available."

    lines = ["Available tools:"]
    for name in sorted(tools.keys()):
        lines.append(f"- {name}")

    return "\n".join(lines)

def parse_tool_call(text: str | None) -> tuple[str | None, dict[str, Any]]:
    """
    Parse tool calls such as:

    TOOL_CALL: assess_student_performance(student_id="student_001")
    TOOL_CALL: generate_quiz(student_id="student_001", topic="fractions", num_questions=5)
    TOOL_CALL: recommend_study_schedule(student_id="student_001", topics=["fractions", "algebra"])
    TOOL_CALL: add_highly_difficult_content()
    """
    if not text:
        return None, {}

    clean = str(text).strip()
    match = re.search(
        r"TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None, {}

    tool_name = match.group(1).strip()
    raw_args = (match.group(2) or "").strip()

    if not raw_args:
        return tool_name, {}

    parsed_args: dict[str, Any] = {}

    # Split on commas that are not inside quotes or brackets.
    parts = re.split(
        r",\s*(?=(?:[^\"'\[\]]|\"[^\"]*\"|'[^']*'|\[[^\]]*\])*$)",
        raw_args,
    )

    for part in parts:
        if "=" not in part:
            continue

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value.strip('"').strip("'")

        parsed_args[key] = parsed_value

    return tool_name, parsed_args

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    return asyncio.run(coro)

def get_required_args(func) -> list[str]:
    sig = inspect.signature(func)

    required = []
    for name, param in sig.parameters.items():
        if (
            param.default is inspect.Parameter.empty
            and param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ):
            required.append(name)

    return required

def execute_tool_call(
    tool_name: str | None,
    tool_args: dict,
    tools: dict[str, Any],
) -> tuple[str | None, Any, str | None]:
    
    if not tool_name:
        return None, None, None

    if tool_name not in tools:
        return tool_name, None, f"Tool not available: {tool_name}"

    tool = tools[tool_name]

    required_args = get_required_args(tool)
    missing_args = [
        arg for arg in required_args
        if arg not in tool_args
    ]

    if missing_args:
        return (
            tool_name,
            None,
            f"Missing required arguments for {tool_name}: {missing_args}. "
            f"Received args: {tool_args}",
        )

    try:
        result = tool(**tool_args)

        if inspect.isawaitable(result):
            result = _run_async(result)

        return tool_name, result, None

    except Exception as exc:
        return tool_name, None, repr(exc)

def summarize_tool_result(
    tool_name: str | None,
    tool_output: Any,
    tool_error: str | None,
) -> str:
    if not tool_name:
        return "No tool was called."

    if tool_error:
        return f"Tool {tool_name} failed with error: {tool_error}"

    return f"Tool {tool_name} returned: {repr(tool_output)[:1000]}"