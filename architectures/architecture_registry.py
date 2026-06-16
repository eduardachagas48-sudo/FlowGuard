# architectures/architecture_registry.py

from architectures.centralized_mas import build_centralized_mas

# Mantenha seus imports existentes de PlanCraft e outras arquiteturas.
# from architectures.centralized_plancraft_success_graph import build_centralized_plancraft_success_graph
# ...


ARCHITECTURE_REGISTRY = {
    # Arquiteturas existentes
    # "centralized_plancraft_success": build_centralized_plancraft_success_graph,

    # TAMAS
    "centralized_tamas": build_centralized_mas,
}


def get_architecture_builder(name: str):
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture: {name}. Available architectures: {available}"
        )

    return ARCHITECTURE_REGISTRY[name]