from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from architectures.centralized_mas import GlobalState
from architectures.ollama_models import OllamaModel

from prompts.agents import (
    SPECIALIST_SYSTEM_PROMPTS,
    build_hybrid_peer_review_user_prompt,
    build_hybrid_initial_user_prompt,
    build_hybrid_supervisor_final_system_prompt,
    build_hybrid_supervisor_final_user_prompt,
)

def build_hybrid_specialist_llm(
    role: str,
    model_name: str = "gemma2:9b",
) -> OllamaModel:
    temperature_by_role = {
        "planner": 0.25,
        "evidence": 0.15,
        "executor": 0.25,
        "verifier": 0.15,
    }

    return OllamaModel(
        model_name=model_name,
        temperature=temperature_by_role.get(role, 0.2),
        top_p=0.9,
        num_ctx=4096,
    )

def build_hybrid_supervisor_llm(
    temperature: float = 0.1,
    model_name: str = "gemma2:9b",
) -> OllamaModel:
    return OllamaModel(
        model_name=model_name,
        temperature=temperature,
        top_p=0.9,
        num_ctx=4096,
    )

def _get_latest_output_for_role(
    agent_outputs: list[dict],
    role: str,
    revision: bool | None = None,
) -> dict | None:
    candidates = [
        o for o in agent_outputs
        if o.get("role") == role
    ]

    if revision is not None:
        candidates = [
            o for o in candidates
            if o.get("metrics", {}).get("revision") == revision
        ]

    if not candidates:
        return None

    return candidates[-1]

def _build_peer_outputs_text(
    agent_outputs: list[dict],
    current_role: str,
    round_filter: int | None = None,
) -> str:
    peers = []

    for output in agent_outputs:
        if output.get("role") == current_role:
            continue

        if round_filter is not None:
            if output.get("metrics", {}).get("round") != round_filter:
                continue

        peers.append(output)

    return "\n\n".join(
        f"""
AGENT_ID: {o.get("agent_id")}
ROLE: {o.get("role")}
OUTPUT:
{o.get("final_answer")}
""".strip()
        for o in peers
    )

def _sum_metric(agent_outputs: list[dict], metric_name: str) -> float:
    return sum(
        o.get("metrics", {}).get(metric_name, 0) or 0
        for o in agent_outputs
    )

def _run_hybrid_specialist(
    agent_id: str,
    role: str,
    state: GlobalState,
    round_id: int,
    token_budget: int,
    revision: bool = False,
) -> Dict[str, Any]:
    model_name = state.get("model_name", "gemma2:9b")

    llm = build_hybrid_specialist_llm(
        role=role,
        model_name=model_name,
    )

    system_prompt = SPECIALIST_SYSTEM_PROMPTS[role]

    if revision:
        own_initial = _get_latest_output_for_role(
            state.get("agent_outputs", []),
            role=role,
            revision=False,
        )

        own_initial_text = own_initial.get("final_answer", "") if own_initial else ""

        peer_outputs_text = _build_peer_outputs_text(
            agent_outputs=state.get("agent_outputs", []),
            current_role=role,
            round_filter=1,
        )

        user_prompt = build_hybrid_peer_review_user_prompt(
            task=state["task"],
            role=role,
            own_initial_output=own_initial_text,
            peer_outputs=peer_outputs_text,
        )
    else:
        user_prompt = build_hybrid_initial_user_prompt(
            task=state["task"],
            role=role,
        )

    answer, metrics = llm.generation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        token_budget=token_budget,
    )

    suffix = "revision" if revision else "initial"

    output = {
        "agent_id": f"{agent_id}_{suffix}",
        "role": role,
        "model": model_name,
        "final_answer": answer,
        "findings": [],
        "metrics": {
            **metrics,
            "architecture": "hybrid",
            "node_name": f"{agent_id}_{suffix}",
            "agent_role": role,
            "round": round_id,
            "token_budget": token_budget,
            "revision": revision,
        },
    }

    return {
        "agent_outputs": [output],
    }

def hybrid_planner_initial(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_planner",
        role="planner",
        state=state,
        round_id=1,
        token_budget=512,
        revision=False,
    )

def hybrid_evidence_initial(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_evidence",
        role="evidence",
        state=state,
        round_id=1,
        token_budget=512,
        revision=False,
    )

def hybrid_executor_initial(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_executor",
        role="executor",
        state=state,
        round_id=1,
        token_budget=512,
        revision=False,
    )

def hybrid_verifier_initial(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_verifier",
        role="verifier",
        state=state,
        round_id=1,
        token_budget=512,
        revision=False,
    )

def hybrid_planner_revision(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_planner",
        role="planner",
        state=state,
        round_id=2,
        token_budget=512,
        revision=True,
    )

def hybrid_evidence_revision(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_evidence",
        role="evidence",
        state=state,
        round_id=2,
        token_budget=512,
        revision=True,
    )

def hybrid_executor_revision(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_executor",
        role="executor",
        state=state,
        round_id=2,
        token_budget=512,
        revision=True,
    )

def hybrid_verifier_revision(state: GlobalState):
    return _run_hybrid_specialist(
        agent_id="hybrid_verifier",
        role="verifier",
        state=state,
        round_id=2,
        token_budget=512,
        revision=True,
    )

def hybrid_supervisor_final_node(state: GlobalState) -> Dict[str, Any]:
    model_name = state.get("model_name", "gemma2:9b")

    llm = build_hybrid_supervisor_llm(
        temperature=0.1,
        model_name=model_name,
    )

    system_prompt = build_hybrid_supervisor_final_system_prompt()
    user_prompt = build_hybrid_supervisor_final_user_prompt(
        task=state["task"],
        outputs=state.get("agent_outputs", []),
    )

    final_text, metrics = llm.generation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        token_budget=512,
    )

    final_output = {
        "agent_id": "hybrid_supervisor_final",
        "role": "supervisor_final_aggregator",
        "model": model_name,
        "final_answer": final_text,
        "findings": [],
        "metrics": {
            **metrics,
            "architecture": "hybrid",
            "node_name": "hybrid_supervisor_final",
            "agent_role": "supervisor_final_aggregator",
            "round": 3,
            "token_budget": 512,
            "revision": False,
        },
    }

    previous_outputs = state.get("agent_outputs", [])

    total_tokens = _sum_metric(
        previous_outputs,
        "total_tokens_observed",
    ) + metrics.get("total_tokens_observed", 0)

    total_duration_s = _sum_metric(
        previous_outputs,
        "total_duration_s",
    ) + metrics.get("total_duration_s", 0)

    return {
        "final_answer": final_text,
        "solved": True,
        "agent_outputs": [final_output],
        "workflow_metrics": {
            "architecture": "hybrid",
            "num_specialists": 4,
            "has_supervisor": True,
            "supervisor_timing": "final_only",
            "peer_review_enabled": True,
            "num_rounds": 2,
            "num_llm_calls": len(previous_outputs) + 1,
            "total_tokens_observed": total_tokens,
            "total_duration_s": total_duration_s,
        },
    }

def build_hybrid_mas():
    g = StateGraph(GlobalState)

    # Initial independent round
    g.add_node("hybrid_planner_initial", hybrid_planner_initial)
    g.add_node("hybrid_evidence_initial", hybrid_evidence_initial)
    g.add_node("hybrid_executor_initial", hybrid_executor_initial)
    g.add_node("hybrid_verifier_initial", hybrid_verifier_initial)

    # Peer-review round
    g.add_node("hybrid_planner_revision", hybrid_planner_revision)
    g.add_node("hybrid_evidence_revision", hybrid_evidence_revision)
    g.add_node("hybrid_executor_revision", hybrid_executor_revision)
    g.add_node("hybrid_verifier_revision", hybrid_verifier_revision)

    # Final supervisor
    g.add_node("hybrid_supervisor_final", hybrid_supervisor_final_node)

    # Fan-out initial round
    g.add_edge(START, "hybrid_planner_initial")
    g.add_edge(START, "hybrid_evidence_initial")
    g.add_edge(START, "hybrid_executor_initial")
    g.add_edge(START, "hybrid_verifier_initial")

    # Fan-in from initial round to each peer-review node
    g.add_edge("hybrid_planner_initial", "hybrid_planner_revision")
    g.add_edge("hybrid_evidence_initial", "hybrid_planner_revision")
    g.add_edge("hybrid_executor_initial", "hybrid_planner_revision")
    g.add_edge("hybrid_verifier_initial", "hybrid_planner_revision")

    g.add_edge("hybrid_planner_initial", "hybrid_evidence_revision")
    g.add_edge("hybrid_evidence_initial", "hybrid_evidence_revision")
    g.add_edge("hybrid_executor_initial", "hybrid_evidence_revision")
    g.add_edge("hybrid_verifier_initial", "hybrid_evidence_revision")

    g.add_edge("hybrid_planner_initial", "hybrid_executor_revision")
    g.add_edge("hybrid_evidence_initial", "hybrid_executor_revision")
    g.add_edge("hybrid_executor_initial", "hybrid_executor_revision")
    g.add_edge("hybrid_verifier_initial", "hybrid_executor_revision")

    g.add_edge("hybrid_planner_initial", "hybrid_verifier_revision")
    g.add_edge("hybrid_evidence_initial", "hybrid_verifier_revision")
    g.add_edge("hybrid_executor_initial", "hybrid_verifier_revision")
    g.add_edge("hybrid_verifier_initial", "hybrid_verifier_revision")

    # Fan-in from peer-review round to final supervisor
    g.add_edge("hybrid_planner_revision", "hybrid_supervisor_final")
    g.add_edge("hybrid_evidence_revision", "hybrid_supervisor_final")
    g.add_edge("hybrid_executor_revision", "hybrid_supervisor_final")
    g.add_edge("hybrid_verifier_revision", "hybrid_supervisor_final")

    g.add_edge("hybrid_supervisor_final", END)

    return g.compile()
