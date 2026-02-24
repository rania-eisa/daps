"""
apssa_dap_limited.py

Enhanced APSSA / Clustering for Smart Meter DAP placement
- Limits number of DAPs per district to MAX_DAP_PER_DISTRICT (default 2)
- Ensures no isolated meters by connecting isolated meters to nearest DAP (adds artificial adjacency links)
- Draws polylines between DAPs (in same district)
- Uses BFS for hops, AffinityPropagation for initial exemplars
- Saves map, excel, load-plot, and isolated_meters.log

Requirements:
    pandas, numpy, scikit-learn, matplotlib, folium
"""

import os
import logging
from collections import deque
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import AffinityPropagation
import folium
import matplotlib.pyplot as plt

# ------------------------------
# Parameters (tweak as needed)
# ------------------------------
CSV_PATH = "merged_final_updated.csv"
OUTPUT_DIR = "outputs"

RC = 0.5  # communication range (km)
alpha_w, beta_w = 0.6, 0.4  # for W_i
alpha_s, beta_s, gamma_s = 0.5, 0.3, 0.2  # for similarity Sij

# Path-loss model parameters
PL_d0 = 1.0
d0 = 0.01  # 10m in km
omega = 2.5

# Cost params
A_cost = 1000.0
B_cost = 0.5
C_cost = 10.0

# Fitness weights
w_davg = 0.6
w_loadgap = 0.4

# AP/SSA params
AP_max_iter = 200
SSA_iter = 200

# Penalty / unreachable
UNREACHABLE_PENALTY = 10.0
MISSING_COST_MARGIN = 1.0  # margin when computing missing cost

# Limit DAPs per district
MAX_DAP_PER_DISTRICT = 1  # set 2 or 3 as you like

# Artificial link weight (when connecting isolated meters)
ARTIFICIAL_LINK_COST_KM = 0.001  # very small distance (km) for artificial adjacency

# ------------------------------
# Setup outputs & logging
# ------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "isolated_meters.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(message)s")
# clear previous log
open(LOG_PATH, "w").close()

# ------------------------------
# Utility functions
# ------------------------------
def haversine_km_matrix(coords):
    return haversine_distances(np.radians(coords)) * 6371.0

def bfs_shortest_hops_and_next(adj):
    N = adj.shape[0]
    INF = 10**9
    dist = np.full((N, N), INF, dtype=float)
    next_hop = -np.ones((N, N), dtype=int)
    for s in range(N):
        q = deque([s])
        visited = np.zeros(N, dtype=bool)
        parent = -np.ones(N, dtype=int)
        visited[s] = True
        dist[s, s] = 0
        while q:
            u = q.popleft()
            neighbors = np.where(adj[u] > 0)[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
        for dest in range(N):
            if not visited[dest]:
                continue
            if dest == s:
                next_hop[s, dest] = s
                continue
            cur = dest
            prev = parent[cur]
            while prev != s and prev != -1:
                cur = prev
                prev = parent[cur]
            next_hop[s, dest] = -1 if prev == -1 else cur
    return dist, next_hop

def reconstruct_path(u, v, next_hop):
    if u == v:
        return [u]
    if next_hop[u, v] == -1:
        return []
    path = [u]
    cur = u
    while cur != v:
        cur = next_hop[cur, v]
        if cur == -1:
            return []
        path.append(int(cur))
        if len(path) > next_hop.shape[0] + 5:
            break
    return path

def PL_of_distance(d_km):
    d = np.asarray(d_km)
    zero_mask = (d <= 0)
    d_safe = np.where(d <= d0, d0, d)
    pl = PL_d0 * (d0 / d_safe) ** omega
    pl = np.where(zero_mask, 0.0, pl)
    return pl

# ------------------------------
# 1. Load and prepare data
# ------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

raw = pd.read_csv(CSV_PATH)

hh_cols = [f"hh_{i}" for i in range(48)]
required_cols = ['LCLid', 'latitude', 'longitude', 'district', 'building', 'day'] + hh_cols
missing = [c for c in required_cols if c not in raw.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

raw['day'] = pd.to_datetime(raw['day'], errors='coerce')
raw[hh_cols] = raw[hh_cols].fillna(0.0)
raw['total_consumption'] = raw[hh_cols].sum(axis=1)

# keep latest per LCLid
raw_sorted = raw.sort_values(by='day', ascending=False)
latest = raw_sorted.drop_duplicates(subset='LCLid', keep='first').reset_index(drop=True)

df_unique = latest[['LCLid', 'latitude', 'longitude', 'district', 'building', 'total_consumption']].copy()
df_unique['building'] = df_unique['building'].astype(str).str.lower().str.strip()

# ------------------------------
# 2. Compute W_i
# ------------------------------
building_weights = {
    'apartments': 1.0,
    'schools': 2.0,
    'banks': 1.8,
    'hotels': 2.2,
    'industry': 3.0,
    'hospital': 2.5
}
df_unique['raw_bld_weight'] = df_unique['building'].map(building_weights).fillna(1.0)

scaler = MinMaxScaler()
df_unique[['c_norm', 'b_norm']] = scaler.fit_transform(df_unique[['total_consumption', 'raw_bld_weight']])
df_unique['W_i'] = alpha_w * df_unique['c_norm'] + beta_w * df_unique['b_norm']

# ------------------------------
# 3. Per-district processing
# ------------------------------
results = []
map_center = [df_unique['latitude'].mean(), df_unique['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=13)

for district in df_unique['district'].unique():
    group = df_unique[df_unique['district'] == district].reset_index(drop=True)
    N = len(group)
    if N < 2:
        continue

    coords = group[['latitude', 'longitude']].values
    dist_km = haversine_km_matrix(coords)

    # compute dynamic missing cost: max finite distance + margin
    finite_mask = (dist_km > 0)
    if finite_mask.any():
        missing_cost = dist_km[finite_mask].max() + MISSING_COST_MARGIN
    else:
        missing_cost = 1.0 + MISSING_COST_MARGIN

    # Dij normalized
    Dij = dist_km.copy()
    Dij = (Dij - Dij.min()) / (Dij.max() - Dij.min() + 1e-9)

    # Bij vectorized
    builds = group['building'].astype(str).values
    Bij = (builds[:, None] == builds[None, :]).astype(float)

    # Eij energy similarity
    EC = group['total_consumption'].to_numpy()
    ec_diff = np.abs(EC.reshape(-1,1) - EC.reshape(1,-1))
    max_diff = ec_diff.max() if ec_diff.max() > 0 else 1.0
    Eij = 1.0 - (ec_diff / (max_diff + 1e-9))

    # similarity Sij
    Sij = alpha_s * (1 - Dij) + beta_s * Bij + gamma_s * Eij
    Sij = 0.5 * (Sij + Sij.T)

    # adjacency (initial) based on RC
    adj = (dist_km <= RC).astype(int)
    np.fill_diagonal(adj, 0)

    # BFS hops + next_hop
    hops_matrix, next_hop = bfs_shortest_hops_and_next(adj)

    # AffinityPropagation initial exemplars
    try:
        ap = AffinityPropagation(affinity='precomputed', max_iter=AP_max_iter, convergence_iter=15)
        ap.fit(Sij)
        centers = ap.cluster_centers_indices_
        exemplars = np.unique(centers.astype(int)) if centers is not None else np.array([], dtype=int)
    except Exception:
        exemplars = np.array([], dtype=int)

    if len(exemplars) == 0:
        exemplars = np.array([int(N//2)], dtype=int)

    # cap initial exemplars to MAX_DAP_PER_DISTRICT
    if len(exemplars) > MAX_DAP_PER_DISTRICT:
        exemplars = exemplars[:MAX_DAP_PER_DISTRICT].astype(int)

    # metrics function (uses current next_hop variable)
    def compute_metrics_for_S(S_indices):
        # enforce max cap
        S_indices = list(dict.fromkeys([int(x) for x in S_indices]))
        if len(S_indices) == 0:
            S_indices = [int(N//2)]
        if len(S_indices) > MAX_DAP_PER_DISTRICT:
            S_indices = S_indices[:MAX_DAP_PER_DISTRICT]

        dists_to_S = dist_km[:, S_indices]
        assigned_idx_in_S = np.argmin(dists_to_S, axis=1)
        dap_for_meter = [S_indices[i] for i in assigned_idx_in_S]

        W = group['W_i'].to_numpy()
        distances_to_dap = np.array([dist_km[i, dap_for_meter[i]] for i in range(N)])
        Davg = (W * distances_to_dap).sum() / (W.sum() + 1e-12)

        unique_idxs, counts = np.unique(dap_for_meter, return_counts=True)
        loads = dict(zip(unique_idxs, counts))
        loads_list = np.array([loads.get(idx, 0) for idx in S_indices], dtype=float)
        load_avg = loads_list.mean()
        L_variance = np.mean((loads_list - load_avg)**2)

        # compute PL and hops weighted
        total_PL_weighted = 0.0
        total_hops_weighted = 0.0
        isolated_flags = np.zeros(N, dtype=bool)
        for i in range(N):
            src = i
            dst = dap_for_meter[i]
            if src == dst:
                continue
            path = reconstruct_path(src, dst, next_hop)
            if len(path) == 0:
                # unreachable -> use direct dist with penalty and mark isolated
                d = dist_km[src, dst]
                total_PL_weighted += W[i] * PL_of_distance(d if d>0 else missing_cost) * UNREACHABLE_PENALTY
                est_hops = int(np.ceil((d if d>0 else missing_cost) / (RC + 1e-9)))
                total_hops_weighted += W[i] * est_hops
                isolated_flags[i] = True
            else:
                hops = 0
                for a,b in zip(path[:-1], path[1:]):
                    d = dist_km[int(a), int(b)]
                    total_PL_weighted += W[i] * PL_of_distance(d)
                    hops += 1
                total_hops_weighted += W[i] * hops

        ctrans = B_cost * total_PL_weighted
        cdly = C_cost * total_hops_weighted
        cmain = A_cost * len(S_indices)
        ctotal = cmain + ctrans + cdly

        # normalized fitness
        Davg_norm = Davg / (dist_km.max() + 1e-9)
        L_norm = L_variance / (N**2 + 1e-9)
        fitness = w_davg * Davg_norm + w_loadgap * L_norm

        return {
            'S_indices': S_indices,
            'Davg': Davg,
            'Davg_norm': Davg_norm,
            'L_variance': L_variance,
            'fitness': fitness,
            'ctrans': ctrans,
            'cdly': cdly,
            'cmain': cmain,
            'ctotal': ctotal,
            'dap_for_meter': dap_for_meter,
            'isolated_flags': isolated_flags
        }

    # initial
    initial_S = list(map(int, exemplars.tolist()))
    if len(initial_S) == 0:
        initial_S = [int(N//2)]

    if len(initial_S) > MAX_DAP_PER_DISTRICT:
        initial_S = initial_S[:MAX_DAP_PER_DISTRICT]

    best_S = initial_S.copy()
    best_metrics = compute_metrics_for_S(best_S)

    rng = np.random.default_rng(42)

    # SSA-like search but enforce max DAPs
    for it in range(SSA_iter):
        candidate = best_S.copy()
        op = rng.choice(['replace', 'add', 'remove'], p=[0.6, 0.2, 0.2])
        if op == 'replace' and len(candidate) > 0:
            idx_to_replace = int(rng.integers(0, len(candidate)))
            candidate[idx_to_replace] = int(rng.integers(0, N))
        elif op == 'add':
            if len(candidate) < MAX_DAP_PER_DISTRICT:
                candidate.append(int(rng.integers(0, N)))
            else:
                idx_to_replace = int(rng.integers(0, len(candidate)))
                candidate[idx_to_replace] = int(rng.integers(0, N))
        elif op == 'remove' and len(candidate) > 1:
            rem = int(rng.integers(0, len(candidate)))
            candidate.pop(rem)

        candidate = list(dict.fromkeys(candidate))
        if len(candidate) == 0:
            candidate = [int(rng.integers(0, N))]
        if len(candidate) > MAX_DAP_PER_DISTRICT:
            candidate = candidate[:MAX_DAP_PER_DISTRICT]

        metrics = compute_metrics_for_S(candidate)
        if metrics['fitness'] < best_metrics['fitness']:
            best_S = candidate
            best_metrics = metrics

    # ---------- Ensure no isolated meters ----------
    # If some meters are isolated, connect each isolated meter to its nearest DAP by adding artificial adjacency,
    # then recompute BFS next_hop and metrics. Repeat until no isolated meters remain (or max iterations to be safe).
    max_fix_iters = N  # safe upper bound
    fix_iter = 0
    while True:
        fix_iter += 1
        isolated_flags = best_metrics.get('isolated_flags', np.zeros(N, dtype=bool))
        isolated_indices = np.where(isolated_flags)[0]
        if len(isolated_indices) == 0 or fix_iter > max_fix_iters:
            break
        # For each isolated meter, connect to nearest DAP (by geographic distance)
        for iso in isolated_indices:
            # find assigned dap index (closest by dist_km)
            dists_to_daps = dist_km[iso, best_S]
            nearest_pos = int(np.argmin(dists_to_daps))
            nearest_dap = best_S[nearest_pos]
            # add bidirectional artificial adjacency
            # set adjacency based on small artificial cost: we modify adj matrix to connect them
            adj[iso, nearest_dap] = 1
            adj[nearest_dap, iso] = 1
            # optionally set dist_km entry to very small artificial distance so subsequent Davg uses it
            dist_km[iso, nearest_dap] = ARTIFICIAL_LINK_COST_KM
            dist_km[nearest_dap, iso] = ARTIFICIAL_LINK_COST_KM
            logging.info(f"ARTIFICIAL_LINK,{group.loc[iso,'LCLid']},{district},to_DAP:{group.loc[nearest_dap,'LCLid']}")
        # recompute BFS and metrics
        hops_matrix, next_hop = bfs_shortest_hops_and_next(adj)
        best_metrics = compute_metrics_for_S(best_S)
        # loop will check isolated flags again

    # Draw DAP markers and lines between them
    dap_indices = best_S
    dap_rows = group.iloc[dap_indices]
    # draw lines between each pair of DAPs in this district
    for i_idx in range(len(dap_indices)):
        for j_idx in range(i_idx+1, len(dap_indices)):
            a = dap_rows.iloc[i_idx]
            b = dap_rows.iloc[j_idx]
            folium.PolyLine([[a['latitude'], a['longitude']], [b['latitude'], b['longitude']]],
                            color='purple', weight=2, dash_array='5,5',
                            tooltip=f"DAP-link {a['LCLid']} ↔ {b['LCLid']}").add_to(m)

    # DAP markers
    for idx in dap_indices:
        row = group.iloc[idx]
        folium.Marker(location=[row['latitude'], row['longitude']],
                      icon=folium.Icon(color='red', icon='signal'),
                      popup=f"DAP - {district} - {row['LCLid']}").add_to(m)

    # Meter markers & lines to assigned DAP
    for i in range(N):
        row = group.iloc[i]
        assigned_dap_idx = best_metrics['dap_for_meter'][i]
        dap_row = group.iloc[assigned_dap_idx]
        # after fixes, no isolated meters should remain
        folium.CircleMarker(location=[row['latitude'], row['longitude']],
                            radius=3,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            tooltip=f"{row['LCLid']} -> {dap_row['LCLid']}").add_to(m)
        folium.PolyLine([[row['latitude'], row['longitude']],
                         [dap_row['latitude'], dap_row['longitude']]],
                         color='green', weight=1).add_to(m)

        results.append({
            'district': district,
            'Meter_LCLid': row['LCLid'],
            'DAP_LCLid': dap_row['LCLid'],
            'Distance_km': float(dist_km[i, assigned_dap_idx]),
            'W_i': float(row['W_i'])
        })

    # optional: print status
    remaining_isolated = int(np.sum(best_metrics.get('isolated_flags', np.zeros(N, dtype=bool))))
    print(f"District {district}: N={N}, selected DAPs={len(dap_indices)}, isolated_fixed={remaining_isolated}")

# ------------------------------
# Save outputs
# ------------------------------
map_path = os.path.join(OUTPUT_DIR, "apssa_map_dap_limited_connected1_dap1.html")
m.save(map_path)
print(f"Map saved to: {map_path}")

res_df = pd.DataFrame(results)
excel_path = os.path.join(OUTPUT_DIR, "dap_connections_dap_limited_connected_dap1.xlsx")
res_df.to_excel(excel_path, index=False)
print(f"Excel saved to: {excel_path}")

# global stats & plot
if not res_df.empty:
    numerator = (res_df['W_i'] * res_df['Distance_km']).sum()
    denominator = res_df['W_i'].sum()
    Davg_global = numerator / (denominator + 1e-12)
    load_counts = res_df['DAP_LCLid'].value_counts()
    L_bar = load_counts.mean()
    L_std = load_counts.std()
    print(f"Global D_avg (weighted): {Davg_global:.4f} km")
    print(f"Average load per DAP: {L_bar:.2f}, std: {L_std:.2f}")

    cmain = A_cost * len(res_df['DAP_LCLid'].unique())
    ctrans = B_cost * (res_df['W_i'] * res_df['Distance_km'].apply(lambda d: float(PL_of_distance(d)))).sum()
    cdly = C_cost * (res_df['W_i']).sum() * 1.0
    ctotal = cmain + ctrans + cdly
    print(f"Approx Total Cost: {ctotal:.2f}")

    load_per_dap = res_df.groupby('DAP_LCLid')['W_i'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(load_per_dap.index.astype(str), load_per_dap.values)
    plt.xlabel("DAP LCLid")
    plt.ylabel("Total Weighted Load (W_i)")
    plt.title("Load Distribution per DAP")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dap_load_distribution_dap_limited_connected.png"))
    plt.close()

print("Done. Artificial links (if any) were logged to:", LOG_PATH)
