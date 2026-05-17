from __future__ import annotations

import networkx as nx

from ..attacks import run_attack, run_edge_attack
from ..attacks_mix import run_mix_attack

SUPPORTED_NODE_ATTACK_KINDS = (
    "random",
    "degree",
    "betweenness",
    "kcore",
    "richclub_top",
    "richclub_density",
    "low_degree",
    "weak_strength",
)
SUPPORTED_EDGE_ATTACK_KINDS = (
    "weak_edges_by_weight",
    "weak_edges_by_confidence",
    "strong_edges_by_weight",
    "strong_edges_by_confidence",
    "ricci_most_negative",
    "ricci_most_positive",
    "ricci_abs_max",
    "flux_high_rw",
    "flux_high_evo",
    "flux_high_rw_x_neg_ricci",
)
SUPPORTED_MIX_ATTACK_KINDS = (
    "hrish_mix",
    "mix_degree_preserving",
    "mix_weightconf_preserving",
)

NODE_ATTACK_PRESETS = {
    "Random": "random",
    "Degree": "degree",
    "Betweenness": "betweenness",
    "K-core": "kcore",
    "Rich-club top": "richclub_top",
    "Rich-club density": "richclub_density",
    "Low degree": "low_degree",
    "Weak strength": "weak_strength",
}
EDGE_ATTACK_PRESETS = {
    "Weak weight": "weak_edges_by_weight",
    "Weak confidence": "weak_edges_by_confidence",
    "Strong weight": "strong_edges_by_weight",
    "Strong confidence": "strong_edges_by_confidence",
    "Ricci most negative": "ricci_most_negative",
    "Ricci most positive": "ricci_most_positive",
    "Ricci abs max": "ricci_abs_max",
    "Flux high RW": "flux_high_rw",
    "Flux high Evo": "flux_high_evo",
    "Flux x negative Ricci": "flux_high_rw_x_neg_ricci",
}


class AttackService:
    supported_node_kinds = SUPPORTED_NODE_ATTACK_KINDS
    supported_edge_kinds = SUPPORTED_EDGE_ATTACK_KINDS
    supported_mix_kinds = SUPPORTED_MIX_ATTACK_KINDS
    node_presets = NODE_ATTACK_PRESETS
    edge_presets = EDGE_ATTACK_PRESETS

    @staticmethod
    def validate_node_kind(kind: str) -> str:
        use_kind = str(kind)
        if use_kind not in SUPPORTED_NODE_ATTACK_KINDS:
            raise ValueError(f"Unsupported node attack kind: {use_kind}")
        return use_kind

    @staticmethod
    def validate_edge_kind(kind: str) -> str:
        use_kind = str(kind)
        if use_kind not in SUPPORTED_EDGE_ATTACK_KINDS:
            raise ValueError(f"Unsupported edge attack kind: {use_kind}")
        return use_kind

    @staticmethod
    def validate_mix_kind(kind: str) -> str:
        use_kind = str(kind)
        if use_kind not in SUPPORTED_MIX_ATTACK_KINDS:
            raise ValueError(f"Unsupported mix attack kind: {use_kind}")
        return use_kind

    @staticmethod
    def run_node_attack(G: nx.Graph, *args, **kwargs):
        if args:
            args = (AttackService.validate_node_kind(str(args[0])), *args[1:])
        elif "attack_kind" in kwargs:
            kwargs["attack_kind"] = AttackService.validate_node_kind(str(kwargs["attack_kind"]))
        elif "kind" in kwargs:
            kwargs["kind"] = AttackService.validate_node_kind(str(kwargs["kind"]))
        return run_attack(G, *args, **kwargs)

    @staticmethod
    def run_edge_attack(G: nx.Graph, *args, **kwargs):
        if args:
            args = (AttackService.validate_edge_kind(str(args[0])), *args[1:])
        elif "kind" in kwargs:
            kwargs["kind"] = AttackService.validate_edge_kind(str(kwargs["kind"]))
        return run_edge_attack(G, *args, **kwargs)

    @staticmethod
    def run_mix_attack(G: nx.Graph, *args, **kwargs):
        if args:
            args = (AttackService.validate_mix_kind(str(args[0])), *args[1:])
        elif "kind" in kwargs:
            kwargs["kind"] = AttackService.validate_mix_kind(str(kwargs["kind"]))
        return run_mix_attack(G, *args, **kwargs)

    @staticmethod
    def run_node_attack_suite(G: nx.Graph, kinds: list[str] | tuple[str, ...], *args, **kwargs):
        return [(kind, AttackService.run_node_attack(G, kind, *args, **kwargs)) for kind in kinds]

    @staticmethod
    def run_edge_attack_suite(G: nx.Graph, kinds: list[str] | tuple[str, ...], *args, **kwargs):
        return [(kind, AttackService.run_edge_attack(G, kind, *args, **kwargs)) for kind in kinds]
