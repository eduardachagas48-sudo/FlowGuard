from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from architectures.architecture_registry import get_architecture_builder
from datasets.tamas.tamas_utils import (
    load_tamas_jsonl,
    normalize_tamas_tasks,
)
from tools.tamas.tamas_tool_registry import build_tool_mapping_for_scenario
from tools.tamas.tamas_tool_wrappers import (
    wrap_tools_for_agent,
    summarize_tool_events,
)
# from architectures.tools.tamas_langgraph import get_tools_for_agent

# def build_tools_for_task_education(
#     agents: list[dict],
#     tool_events_sink: list[dict],
# ) -> dict:
#     tools_by_agent = {}

#     for agent in agents:
#         agent_name = agent.get("agent_name", "")
#         raw_tools = get_tools_for_agent(agent_name)

#         tools_by_agent[agent_name] = wrap_tools_for_agent(
#             agent_name=agent_name,
#             raw_tools=raw_tools,
#             telemetry_sink=tool_events_sink,
#         )

#     return tools_by_agent

def build_tools_for_task(
    scenario: str,
    agents: list[dict],
    tool_events_sink: list[dict],
    module_prefix: str = "tools.tamas",
) -> dict[str, dict[str, Any]]:
    
    raw_mapping = build_tool_mapping_for_scenario(
        scenario=scenario,
        module_prefix=module_prefix,
    )

    tools_by_agent = {}

    for agent in agents:
        agent_name = agent.get("agent_name", "")
        raw_tools = raw_mapping.get(agent_name, {})

        tools_by_agent[agent_name] = wrap_tools_for_agent(
            agent_name=agent_name,
            raw_tools=raw_tools,
            telemetry_sink=tool_events_sink,
        )

    return tools_by_agent


def run_tamas_task_with_architecture(
    task: dict,
    architecture: str,
    model_name: str,
    scenario: str,
    seed: int | None = None,
    module_prefix: str = "tools.tamas",
    extra_state: dict | None = None,
) -> dict:

    graph_builder = get_architecture_builder(architecture)
    graph = graph_builder()

    tool_events_sink: list[dict] = []

    agents = task.get("agents", [])

    tools_by_agent = build_tools_for_task(
        scenario=scenario,
        agents=agents,
        tool_events_sink=tool_events_sink,
        module_prefix=module_prefix,
    )

    initial_state = {
        "task": task["user_query"],
        "model_name": model_name,
        "seed": seed,

        "benchmark": "tamas",
        "architecture": architecture,
        "attack_type": task.get("attack_type", "benign"),
        "is_attack": task.get("is_attack", False),
        "tamas_task_id": task["task_id"],

        "agents_spec": agents,
        "tools_by_agent": tools_by_agent,
        "tool_events": [],
        "agent_outputs": [],
    }

    if extra_state:
        initial_state.update(extra_state)

    result = graph.invoke(initial_state)

    graph_tool_events = result.get("tool_events", [])
    all_tool_events = tool_events_sink + graph_tool_events

    tool_summary = summarize_tool_events(all_tool_events)

    return {
        "benchmark": "tamas",
        "architecture": architecture,
        "model_name": model_name,
        "seed": seed,
        "task_id": task["task_id"],
        "scenario": scenario,
        "attack_type": task.get("attack_type", "benign"),
        "is_attack": task.get("is_attack", False),
        "expected_label": task.get("expected_label"),
        "user_query": task["user_query"],
        "agents": agents,

        "final_answer": result.get("final_answer", ""),
        "agent_outputs": result.get("agent_outputs", []),
        "tool_events": all_tool_events,
        "tool_summary": tool_summary,
        "workflow_metrics": result.get("workflow_metrics", {}),
        "raw_result": result,
        "error": None,
    }

def run_tamas_batch_for_architecture(
    json_data_path: str | Path,
    architecture: str,
    attack_type: str | None = None,
    scenario: str | None = None,
    limit: int | None = None,
    model_name: str = "gemma2:9b",
    seed: int | None = None,
    output_path: str | None = None,
    module_prefix: str = "tools.tamas",
    extra_state: dict | None = None,
) -> str:
    
    raw_tasks = load_tamas_jsonl(json_data_path)
    tasks = normalize_tamas_tasks(raw_tasks, attack_type)

    is_attack = False
    if attack_type:
        is_attack = True

    if limit:
        tasks = tasks[:limit]

    if output_path is None:
        safe_model = model_name.replace(":", "_").replace("/", "_")
        output_path = (
            f"results/tamas/raw/"
            f"tamas_{architecture}_{safe_model}_seed{seed}.jsonl"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for task in tqdm(tasks, desc=f"TAMAS {architecture}"):
            if extra_state:
                if "forced_attack_type" in extra_state:
                    task["attack_type"] = extra_state["forced_attack_type"]

                if "forced_is_attack" in extra_state:
                    task["is_attack"] = bool(extra_state["forced_is_attack"])
            try:
                record = run_tamas_task_with_architecture(
                    task=task,
                    architecture=architecture,
                    scenario=scenario,
                    model_name=model_name,
                    seed=seed,
                    module_prefix=module_prefix,
                    extra_state=extra_state,
                )

            except Exception as exc:
                print("Exception:", repr(exc))
                print(traceback.format_exc())

                record = {
                    "benchmark": "tamas",
                    "architecture": architecture,
                    "model_name": model_name,
                    "seed": seed,
                    "task_id": task.get("task_id"),
                    "attack_type": attack_type,
                    "is_attack": is_attack,
                    "success": False,
                    "error": repr(exc),
                }

            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return str(output_file)