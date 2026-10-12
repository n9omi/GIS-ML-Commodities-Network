"""Network-theory metrics for a bipartite supplier-country x commodity trade network.

Notation: M is a (countries x commodities) matrix of trade values in USD for one year and one
flow; S = M / column sums is the matrix of supplier shares, so every column of S sums to one.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd


# --------------------------------------------------------------------------- construction
def trade_matrix(tidy: pd.DataFrame, year: str, labels: dict | None = None) -> pd.DataFrame:
    """Pivot a tidy USITC table into a countries x commodities matrix of USD values."""
    d = tidy[(tidy.year == str(year)) & (tidy.value_usd > 0)]
    M = d.pivot_table(index="country", columns="hts6", values="value_usd", aggfunc="sum", fill_value=0.0)
    if labels:
        M = M[[c for c in M.columns if c in labels]]
    M = M.loc[M.sum(axis=1) > 0, M.sum(axis=0) > 0]
    return M


def shares(M: pd.DataFrame) -> pd.DataFrame:
    return M / M.sum(axis=0)


def bipartite_graph(M: pd.DataFrame) -> nx.Graph:
    """Weighted bipartite graph; edge attributes: value (USD) and share of the commodity."""
    S = shares(M)
    G = nx.Graph()
    G.add_nodes_from(M.index, bipartite=0, kind="country")
    G.add_nodes_from(M.columns, bipartite=1, kind="commodity")
    for c in M.index:
        for k in M.columns:
            if M.at[c, k] > 0:
                G.add_edge(c, k, value=float(M.at[c, k]), share=float(S.at[c, k]),
                           distance=1.0 / float(S.at[c, k]))
    return G


# --------------------------------------------------------------------------- concentration
def hhi(s: np.ndarray) -> float:
    """Herfindahl-Hirschman index on shares (0-1 scale; 1 = single supplier)."""
    s = np.asarray(s, float)
    s = s[s > 0]
    return float(np.sum(s**2))


def shannon_effective(s: np.ndarray) -> float:
    """exp(Shannon entropy): the Hill number of order 1, an effective supplier count."""
    s = np.asarray(s, float)
    s = s[s > 0]
    return float(np.exp(-np.sum(s * np.log(s))))


def concentration_table(M: pd.DataFrame, labels: dict, min_share: float = 0.01) -> pd.DataFrame:
    """Per-commodity concentration: HHI, effective number of suppliers, top-supplier shares."""
    S = shares(M)
    rows = []
    for k in M.columns:
        s = S[k].sort_values(ascending=False)
        h = hhi(s.values)
        rows.append({
            "hts6": k,
            "commodity": labels.get(k, {}).get("label", k),
            "group": labels.get(k, {}).get("group", ""),
            "value_usd": float(M[k].sum()),
            "n_suppliers": int((s > 0).sum()),
            "n_suppliers_1pct": int((s >= min_share).sum()),
            "hhi": h,
            "n_eff_hhi": 1.0 / h,
            "n_eff_shannon": shannon_effective(s.values),
            "top1_country": s.index[0],
            "top1_share": float(s.iloc[0]),
            "cr3": float(s.iloc[:3].sum()),
        })
    return pd.DataFrame(rows).sort_values("hhi", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- projections and centrality
def country_projection(M: pd.DataFrame) -> nx.Graph:
    """Countries linked by shared commodities; weight = sum_k s_ck * s_c'k (overlap of roles)."""
    S = shares(M).values
    W = S @ S.T
    np.fill_diagonal(W, 0.0)
    G = nx.Graph()
    names = list(M.index)
    G.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if W[i, j] > 0:
                G.add_edge(names[i], names[j], weight=float(W[i, j]))
    return G


def commodity_projection(M: pd.DataFrame) -> pd.DataFrame:
    """Co-dependency between commodities: cosine similarity of their supplier-share vectors.

    Two commodities score near 1 when the same countries supply them in the same proportions,
    i.e. one export ban would hit both.
    """
    S = shares(M)
    X = S.values
    norms = np.linalg.norm(X, axis=0)
    C = (X.T @ X) / np.outer(norms, norms)
    return pd.DataFrame(C, index=S.columns, columns=S.columns)


def country_table(M: pd.DataFrame, min_share: float = 0.01) -> pd.DataFrame:
    """Per-country strength, degree and centrality in the supply network."""
    S = shares(M)
    P = country_projection(M)
    eig = nx.eigenvector_centrality_numpy(P, weight="weight") if P.number_of_edges() else {}
    pr = nx.pagerank(P, weight="weight") if P.number_of_edges() else {}
    B = bipartite_graph(M)
    btw = nx.betweenness_centrality(B, weight="distance", normalized=True)
    core = nx.core_number(nx.Graph(B.edges()))
    out = pd.DataFrame({
        "country": M.index,
        "strength_usd": M.sum(axis=1).values,
        "n_commodities": (S > 0).sum(axis=1).values,
        "n_commodities_1pct": (S >= min_share).sum(axis=1).values,
        "max_share": S.max(axis=1).values,
        "commodities_top1": [(S.idxmax(axis=0) == c).sum() for c in M.index],
    })
    out["eigenvector"] = out.country.map(eig).fillna(0.0)
    out["pagerank"] = out.country.map(pr).fillna(0.0)
    out["betweenness"] = out.country.map(btw).fillna(0.0)
    out["k_core"] = out.country.map(core).fillna(0).astype(int)
    return out.sort_values("strength_usd", ascending=False).reset_index(drop=True)


def commodity_communities(C: pd.DataFrame, threshold: float = 0.3) -> list[set]:
    """Modularity communities of the commodity co-dependency graph (edges where similarity >= threshold)."""
    G = nx.Graph()
    G.add_nodes_from(C.index)
    ks = list(C.index)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            w = float(C.iat[i, j])
            if w >= threshold:
                G.add_edge(ks[i], ks[j], weight=w)
    return [set(c) for c in nx.algorithms.community.greedy_modularity_communities(G, weight="weight")]


# --------------------------------------------------------------------------- nestedness
def nodf(B: np.ndarray) -> float:
    """Nestedness metric based on overlap and decreasing fill (Almeida-Neto et al. 2008), 0-100."""
    B = (np.asarray(B) > 0).astype(int)
    B = B[B.sum(axis=1) > 0][:, B.sum(axis=0) > 0]
    if B.shape[0] < 2 or B.shape[1] < 2:
        return float("nan")

    def _pairs(A):
        deg = A.sum(axis=1)
        order = np.argsort(-deg, kind="stable")
        A, deg = A[order], deg[order]
        vals = []
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                if deg[i] > deg[j] and deg[j] > 0:
                    vals.append(100.0 * (A[i] & A[j]).sum() / deg[j])
                else:
                    vals.append(0.0)
        return vals

    v = _pairs(B) + _pairs(B.T)
    return float(np.mean(v))


def curveball(B: np.ndarray, rng: np.random.Generator, n_swaps: int | None = None) -> np.ndarray:
    """Uniform random matrix with the same row and column sums (Strona et al. 2014)."""
    B = (np.asarray(B) > 0).astype(int)
    rows = [set(np.flatnonzero(r)) for r in B]
    m = len(rows)
    n_swaps = n_swaps or 5 * m
    for _ in range(n_swaps):
        a, b = rng.choice(m, size=2, replace=False)
        ra, rb = rows[a], rows[b]
        only_a, only_b = list(ra - rb), list(rb - ra)
        pool = only_a + only_b
        if not only_a or not only_b:
            continue
        rng.shuffle(pool)
        new_a = set(pool[:len(only_a)])
        new_b = set(pool[len(only_a):])
        rows[a] = (ra & rb) | new_a
        rows[b] = (ra & rb) | new_b
    out = np.zeros_like(B)
    for i, r in enumerate(rows):
        out[i, list(r)] = 1
    return out


def nodf_null(B: np.ndarray, n: int = 300, seed: int = 0) -> dict:
    """Observed NODF vs a fixed-degree null model: z-score and one-sided p-value."""
    rng = np.random.default_rng(seed)
    obs = nodf(B)
    sims = np.array([nodf(curveball(B, rng)) for _ in range(n)])
    sd = sims.std(ddof=1)
    return {"nodf": obs, "null_mean": float(sims.mean()), "null_sd": float(sd),
            "z": float((obs - sims.mean()) / sd) if sd > 0 else float("nan"),
            "p_greater": float((np.sum(sims >= obs) + 1) / (n + 1))}
