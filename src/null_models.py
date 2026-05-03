import networkx as nx


def make_er_gnm(n: int, m: int, seed: int) -> nx.Graph:
    return nx.gnm_random_graph(int(n), int(m), seed=int(seed))

def make_configuration_model(G_base: nx.Graph, seed: int) -> nx.Graph:
    """
    Нулевая модель простого графа с сохранением последовательности степеней.

    Метки исходных узлов сохраняются, чтобы сравнения оставались сопоставимыми.
    """
    labels = list(G_base.nodes())
    degs = [int(G_base.degree(n)) for n in labels]
    if not labels:
        return nx.Graph()
    if sum(degs) == 0:
        H = nx.Graph()
        H.add_nodes_from(labels)
        return H

    try:
        H_idx = nx.random_degree_sequence_graph(degs, seed=int(seed), tries=20)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        H_idx = nx.havel_hakimi_graph(degs)
        swaps = max(1, H_idx.number_of_edges() * 3)
        try:
            nx.double_edge_swap(H_idx, nswap=swaps, max_tries=swaps * 20, seed=int(seed))
        except nx.NetworkXError:
            pass

    mapping = {i: labels[i] for i in range(len(labels))}
    H = nx.relabel_nodes(H_idx, mapping, copy=True)
    return H


def rewire_mix(G_base: nx.Graph, p: float, seed: int) -> nx.Graph:
    """
    Постепенная хаотизация через double_edge_swap.
    p=0 -> оригинал
    p=1 -> сильная рандомизация (но сохраняем степени)
    """
    p = float(max(0.0, min(1.0, p)))
    H = G_base.copy()
    if H.number_of_edges() < 2 or H.number_of_nodes() < 4 or p <= 0:
        return H

    swaps = int(p * H.number_of_edges() * 5)  
    swaps = max(1, swaps)
    tries = swaps * 10

    nx.double_edge_swap(H, nswap=swaps, max_tries=tries, seed=seed)

    H.remove_edges_from(nx.selfloop_edges(H))
    return H
