from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from architectures.centralized_mas import GlobalState
from architectures.ollama_models import OllamaModel
from prompts.agents import *

def build_specialist_llm(
    temperature: float = 0.2,
    model_name: str = "gemma2:9b",
) -> OllamaModel:
    return OllamaModel(
        model_name=model_name,
        temperature=temperature,
        top_p=0.9,
        num_ctx=4096,
    )

def build_supervisor_llm(
    temperature: float = 0.1,
    model_name: str = "gemma2:9b",
) -> OllamaModel:
    return OllamaModel(
        model_name=model_name,
        temperature=temperature,
        top_p=0.9,
        num_ctx=4096,
    )

def _build_worker_user_prompt(task: str, role: str) -> str:
    return f"""
    Task:
    {task}

    You are acting as the {role.upper()} specialist in an independent multi-agent workflow.

    Important:
    - You do not see the outputs of other agents.
    - You must provide your own independent analysis.
    - Be concise, structured, and avoid unnecessary repetition.
    - Produce a candidate answer when possible.
    """.strip()

def _run_independent_specialist(
    agent_id: str,
    role: str,
    state: GlobalState,
    token_budget: int = 512,
    model_name: str = "gemma2:9b",
) -> Dict[str, Any]:
    
    temperature_by_role = {
        "planner": 0.2,
        "evidence": 0.1,
        "executor": 0.2,
        "verifier": 0.1,
    }

    llm = build_specialist_llm(
        temperature=temperature_by_role.get(role, 0.2),
        model_name=state.get("model_name", model_name),
    )

    system_prompt = SPECIALIST_SYSTEM_PROMPTS[role]
    user_prompt = _build_worker_user_prompt(state["task"], role)

    answer, metrics = llm.generation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        token_budget=token_budget,
    )

    output = {
        "agent_id": agent_id,
        "role": role,
        "model": state.get("model_name", model_name),
        "final_answer": answer,
        "findings": [],
        "metrics": {
            **metrics,
            "architecture": "independent",
            "node_name": agent_id,
            "agent_role": role,
            "round": 1,
            "token_budget": token_budget,
        },
    }

    return {"agent_outputs": [output]}

def independent_planner(state: GlobalState):
    return _run_independent_specialist(
        agent_id="ind_planner",
        role="planner",
        state=state,
        token_budget=512,
    )

def independent_evidence(state: GlobalState):
    return _run_independent_specialist(
        agent_id="ind_evidence",
        role="evidence",
        state=state,
        token_budget=512,
    )

def independent_executor(state: GlobalState):
    return _run_independent_specialist(
        agent_id="ind_executor",
        role="executor",
        state=state,
        token_budget=512,
    )

def independent_verifier(state: GlobalState):
    return _run_independent_specialist(
        agent_id="ind_verifier",
        role="verifier",
        state=state,
        token_budget=512,
    )

def dispatch_independent_workers(state: GlobalState):
    return [
        "independent_planner",
        "independent_evidence",
        "independent_executor",
        "independent_verifier",
    ]

def supervisor_synthesis_node(state: GlobalState):
    outputs = state.get("agent_outputs", [])

    llm = build_supervisor_llm(
        temperature=0.1,
        model_name=state.get("model_name", "gemma2:9b"),
    )

    system_prompt = build_supervisor_system_prompt()
    user_prompt = build_synthesis_user_prompt(state["task"], outputs)

    final_text, metrics = llm.generation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        token_budget=512,
    )

    supervisor_trace = {
        "agent_id": "ind_supervisor",
        "role": "supervisor_aggregator",
        "model": state.get("model_name", "gemma2:9b"),
        "final_answer": final_text,
        "findings": [],
        "metrics": {
            **metrics,
            "architecture": "independent",
            "node_name": "supervisor_synthesis",
            "agent_role": "supervisor_aggregator",
            "round": 1,
            "token_budget": 512,
        },
    }

    total_tokens = sum(
        o.get("metrics", {}).get("total_tokens_observed", 0)
        for o in outputs
    ) + metrics.get("total_tokens_observed", 0)

    total_duration_s = sum(
        o.get("metrics", {}).get("total_duration_s", 0)
        for o in outputs
    ) + metrics.get("total_duration_s", 0)

    return {
        "final_answer": final_text,
        "solved": True,
        "agent_outputs": [supervisor_trace],
        "workflow_metrics": {
            "architecture": "independent",
            "num_specialists": 4,
            "has_supervisor": True,
            "num_rounds": 1,
            "num_llm_calls": len(outputs) + 1,
            "total_tokens_observed": total_tokens,
            "total_duration_s": total_duration_s,
        },
    }

def build_independent_mas():
    g = StateGraph(GlobalState)

    g.add_node("independent_planner", independent_planner)
    g.add_node("independent_evidence", independent_evidence)
    g.add_node("independent_executor", independent_executor)
    g.add_node("independent_verifier", independent_verifier)
    g.add_node("supervisor_synthesis", supervisor_synthesis_node)

    g.add_conditional_edges(
        START,
        dispatch_independent_workers,
        [
            "independent_planner",
            "independent_evidence",
            "independent_executor",
            "independent_verifier",
        ],
    )

    g.add_edge("independent_planner", "supervisor_synthesis")
    g.add_edge("independent_evidence", "supervisor_synthesis")
    g.add_edge("independent_executor", "supervisor_synthesis")
    g.add_edge("independent_verifier", "supervisor_synthesis")

    g.add_edge("supervisor_synthesis", END)

    return g.compile()