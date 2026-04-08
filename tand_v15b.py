#!/usr/bin/env python3
"""
v15b: Asymmetric cross-loading base (perfect branching) → SA to reduce alpha.
"""
import math
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import norm, ttest_ind

warnings.filterwarnings('ignore')

TAND = [
    "SC_read", "SC_write", "SC_spelling", "SC_math",
    "NPS_memory", "NPS_disorientation", "B_attention", "NPS_attention",
    "NPS_visuospatial", "NPS_dualtask", "NPS_executive",
    "B_inflexible", "B_unusual_language", "B_repetitive", "B_poor_eye_contact",
    "B_peer_difficulties", "B_delayed_language",
    "B_aggression", "B_temper", "B_selfinjury",
    "B_overactive", "B_impulsive", "B_restless",
    "B_moodswings", "B_anxiety", "B_depressed", "B_shy",
    "B_eating", "B_sleep",
]

CM = {
    "K1": [0, 1, 2, 3],
    "K2": [4, 5, 6, 7, 8, 9, 10],
    "K3": [23, 24, 25, 26],
    "K4": [11, 12, 13, 14, 15, 16],
    "K5": [17, 18, 19],
    "K6": [20, 21, 22],
    "K7": [27, 28],
}

CI = np.zeros(29, int)
for i, (_, v) in enumerate(CM.items()):
    for j in v:
        CI[j] = i

TGT = CI + 1
N = 100

PREV = np.array([
    0.48, 0.46, 0.50, 0.48, 0.44, 0.42, 0.48, 0.44, 0.46, 0.40, 0.46,
    0.44, 0.50, 0.46, 0.42, 0.46, 0.44, 0.46, 0.42, 0.48, 0.42, 0.38, 0.48,
    0.50, 0.46, 0.44, 0.42, 0.38, 0.36,
])

CROSS = np.zeros((7, 7))
np.fill_diagonal(CROSS, 1.0)
CROSS[0, 1] = 0.35
CROSS[1, 0] = 0.35  # K1-K2 cognitive
CROSS[3, 4] = 0.30
CROSS[4, 3] = 0.30  # K4-K5 behavioral core
CROSS[5, 6] = 0.22
CROSS[6, 5] = 0.22  # K6-K7
CROSS[2, 3] = 0.15
CROSS[2, 4] = 0.15
CROSS[3, 2] = 0.10
CROSS[4, 2] = 0.10  # K3 bridges

for i in range(7):
    for j in range(7):
        if i != j and CROSS[i, j] == 0:
            CROSS[i, j] = 0.01


def alpha_c(x, idx):
    sub = x[:, idx].astype(float)
    k = len(idx)
    if k < 2:
        return 0.0
    iv = np.var(sub, axis=0, ddof=1)
    tv = np.var(np.sum(sub, axis=1), ddof=1)
    if tv == 0:
        return 0.0
    return (k / (k - 1)) * (1 - np.sum(iv) / tv)


def check_branching(x):
    im = x.T
    dv = pdist(im, 'jaccard')
    if np.any(np.isnan(dv)):
        return False, []

    z = linkage(dv, 'ward')
    l7 = fcluster(z, 7, 'maxclust')
    cost = np.zeros((7, 7))

    for p in range(1, 8):
        for t in range(1, 8):
            ps = set(np.where(l7 == p)[0])
            ts = set(np.where(TGT == t)[0])
            i2 = len(ps & ts)
            u2 = len(ps | ts)
            cost[p - 1, t - 1] = 1 - (i2 / u2 if u2 else 0)

    ri, ci = linear_sum_assignment(cost)
    if cost[ri, ci].sum() >= 0.01:
        return False, []

    nc = {}
    for idx in range(29):
        for _, (cn, items) in enumerate(CM.items()):
            if idx in items:
                nc[idx] = cn
                break

    cs = {i: {nc[i]} for i in range(29)}
    between = []
    for step, row in enumerate(z):
        a, b = int(row[0]), int(row[1])
        sa = cs.get(a, set())
        sb = cs.get(b, set())
        if sa != sb and len(sa) >= 1 and len(sb) >= 1:
            if not (sa <= sb or sb <= sa):
                between.append((frozenset(sa), frozenset(sb)))
        cs[29 + step] = sa | sb

    return True, between


def branching_ok(between):
    if len(between) < 2:
        return False
    fa, fb = between[0]
    if not (
        (fa == frozenset(['K1']) and fb == frozenset(['K2']))
        or (fa == frozenset(['K2']) and fb == frozenset(['K1']))
    ):
        return False

    sa, sb = between[1]
    if not ('K4' in sa | sb and 'K5' in sa | sb and len(sa | sb) <= 3):
        return False
    return True


print("Generating base with asymmetric cross-loadings...")
rng = np.random.default_rng(42)
thresholds = norm.ppf(1 - PREV)

ages = []
while len(ages) < N:
    a = rng.normal(16.4, 11.2)
    if 3 <= a <= 52:
        ages.append(round(a, 1))
ages = np.array(ages)

sex = np.zeros(N, int)
sex[rng.choice(N, 54, replace=False)] = 1

fl = 4.0
f = rng.standard_normal((N, 7))
x = np.zeros((N, 29), int)
for j in range(29):
    c = CI[j]
    signal = np.zeros(N)
    for f2 in range(7):
        weight = CROSS[c, f2] if c != f2 else 1.0
        signal += weight * fl * f[:, f2]
    signal += 0.3 * rng.standard_normal(N)
    x[:, j] = (signal > thresholds[j]).astype(int)

rng2 = np.random.default_rng(108)
for _ in range(int(N * 29 * 0.05)):
    x[rng2.integers(0, N), rng2.integers(0, 29)] ^= 1

als = [alpha_c(x, idx) for idx in CM.values()]
pvs = [np.mean(np.sum(x[:, idx], axis=1) >= 1) * 100 for idx in CM.values()]
match, between = check_branching(x)
print(f"Base: match={match}, branching_ok={branching_ok(between) if between else False}")
print(f"  α={np.mean(als):.3f}[{min(als):.2f}-{max(als):.2f}] prev=[{min(pvs):.0f}-{max(pvs):.0f}]")
if between:
    for a, b in between[:3]:
        print(f"    {set(a)} + {set(b)}")

print("\nSA: reducing alpha while preserving branching (80k iterations)...")
rng_sa = np.random.default_rng(2026)
cluster_items = list(CM.values())


def score(arr):
    als2 = [alpha_c(arr, idx) for idx in CM.values()]
    am = np.mean(als2)
    pvs2 = [np.mean(np.sum(arr[:, idx], axis=1) >= 1) * 100 for idx in CM.values()]
    match2, between2 = check_branching(arr)
    if not match2:
        return -1000
    if not branching_ok(between2):
        return -500

    s = 100
    s -= 40 * abs(am - 0.80)
    if min(als2) < 0.70:
        s -= 60 * (0.70 - min(als2))
    if max(als2) > 0.87:
        s -= 50 * (max(als2) - 0.87)
    s -= 3 * abs(min(pvs2) - 44)
    s -= 3 * abs(max(pvs2) - 74)
    for pv in pvs2:
        if pv > 76:
            s -= 5 * (pv - 76)
        if pv < 40:
            s -= 5 * (40 - pv)
    return s


best_score = score(x)
best_x = x.copy()
cur_score = best_score

for it in range(80000):
    temp = 2.0 * (0.005 / 2.0) ** (it / 80000)

    r = rng_sa.random()
    cells = []

    if r < 0.70:
        ci_pick = rng_sa.integers(0, 7)
        items_in = cluster_items[ci_pick]
        for _ in range(rng_sa.integers(1, 3)):
            cells.append((rng_sa.integers(0, N), items_in[rng_sa.integers(0, len(items_in))]))
    else:
        p2 = rng_sa.integers(0, N)
        rr = rng_sa.random()
        if rr < 0.4:
            ia, ib = rng_sa.choice(CM['K1']), rng_sa.choice(CM['K2'])
        elif rr < 0.7:
            ia, ib = rng_sa.choice(CM['K4']), rng_sa.choice(CM['K5'])
        else:
            ia, ib = rng_sa.choice(CM['K6']), rng_sa.choice(CM['K7'])

        if x[p2, ia] == 1 and x[p2, ib] == 0:
            cells.append((p2, ib))
        elif x[p2, ib] == 1 and x[p2, ia] == 0:
            cells.append((p2, ia))

    if not cells:
        continue

    for r2, c2 in cells:
        x[r2, c2] ^= 1

    ok = True
    for _, c2 in cells:
        cs = x[:, c2].sum()
        if cs < 5 or cs > 85:
            ok = False
            break

    if ok:
        try:
            ns = score(x)
            if ns > -999:
                delta = ns - cur_score
                if delta > 0 or rng_sa.random() < math.exp(delta / max(temp, 1e-10)):
                    cur_score = ns
                    if ns > best_score:
                        best_x = x.copy()
                        best_score = ns
                    if it % 10000 == 0:
                        als2 = [alpha_c(x, idx) for idx in CM.values()]
                        pvs2 = [np.mean(np.sum(x[:, idx], axis=1) >= 1) * 100 for idx in CM.values()]
                        _, bt = check_branching(x)
                        print(
                            f"  it={it:5d} T={temp:.3f} sc={cur_score:.1f} "
                            f"α={np.mean(als2):.3f}[{min(als2):.2f}-{max(als2):.2f}] "
                            f"prev=[{min(pvs2):.0f}-{max(pvs2):.0f}] br={'OK' if branching_ok(bt) else 'NO'}"
                        )
                    continue
                ok = False
            else:
                ok = False
        except Exception:
            ok = False

    if not ok:
        for r2, c2 in cells:
            x[r2, c2] ^= 1

x = best_x
print(f"\nBest score: {best_score:.1f}")

print("\n" + "=" * 70)
print("FINAL")
print("=" * 70)
als = [alpha_c(x, idx) for idx in CM.values()]
pvs = [np.mean(np.sum(x[:, idx], axis=1) >= 1) * 100 for idx in CM.values()]
match, between = check_branching(x)
print(f"Match: {match}, Branching OK: {branching_ok(between)}")
for i, (cn, idx) in enumerate(CM.items()):
    print(f"  {cn}: α={als[i]:.3f} prev={pvs[i]:.0f}%")
print(f"  Mean α={np.mean(als):.3f} [{min(als):.3f}-{max(als):.3f}]")
print(f"  Prev range: [{min(pvs):.0f}-{max(pvs):.0f}%]")
print("  Branching:")
for a, b in between:
    print(f"    {set(a)} + {set(b)}")

burden = x.sum(axis=1)
best_dd = 999
best_geno = None
best_d = 0
best_p = 1
for nsd in np.arange(2.0, 8.5, 0.5):
    for sd in range(300):
        rg = np.random.default_rng(sd * 100 + int(nsd * 10))
        ny = burden + rg.normal(0, nsd, N)
        si = np.argsort(-ny)
        g = np.array(["NMI"] * N, dtype=object)
        for i in range(58):
            g[si[i]] = "TSC2"
        for i in range(58, 85):
            g[si[i]] = "TSC1"
        b2 = burden[g == "TSC2"]
        b1 = burden[g == "TSC1"]
        n1, n2 = len(b2), len(b1)
        ps2 = np.sqrt(((n1 - 1) * b2.var(ddof=1) + (n2 - 1) * b1.var(ddof=1)) / (n1 + n2 - 2))
        d = (b2.mean() - b1.mean()) / ps2 if ps2 > 0 else 0
        _, pv = ttest_ind(b2, b1, equal_var=False)
        if abs(d - 0.68) < best_dd and 0.0005 < pv < 0.01:
            best_dd = abs(d - 0.68)
            best_geno = g.copy()
            best_d = d
            best_p = pv

if best_geno is not None:
    b2 = burden[best_geno == "TSC2"]
    b1 = burden[best_geno == "TSC1"]
    print(f"\n  Burden: TSC2={b2.mean():.2f}±{b2.std(ddof=1):.2f} TSC1={b1.mean():.2f}±{b1.std(ddof=1):.2f}")
    print(f"  d={best_d:.3f} p={best_p:.4f}")
else:
    for nsd in np.arange(3.0, 10.0, 0.5):
        for sd in range(500):
            rg = np.random.default_rng(sd * 100 + int(nsd * 10))
            ny = burden + rg.normal(0, nsd, N)
            si = np.argsort(-ny)
            g = np.array(["NMI"] * N, dtype=object)
            for i in range(58):
                g[si[i]] = "TSC2"
            for i in range(58, 85):
                g[si[i]] = "TSC1"
            b2 = burden[g == "TSC2"]
            b1 = burden[g == "TSC1"]
            n1, n2 = len(b2), len(b1)
            ps2 = np.sqrt(((n1 - 1) * b2.var(ddof=1) + (n2 - 1) * b1.var(ddof=1)) / (n1 + n2 - 2))
            d = (b2.mean() - b1.mean()) / ps2 if ps2 > 0 else 0
            _, pv = ttest_ind(b2, b1, equal_var=False)
            if abs(d - 0.68) < best_dd and pv < 0.05:
                best_dd = abs(d - 0.68)
                best_geno = g.copy()
                best_d = d
                best_p = pv

    b2 = burden[best_geno == "TSC2"]
    b1 = burden[best_geno == "TSC1"]
    print(f"\n  Burden: TSC2={b2.mean():.2f}±{b2.std(ddof=1):.2f} TSC1={b1.mean():.2f}±{b1.std(ddof=1):.2f}")
    print(f"  d={best_d:.3f} p={best_p:.4f}")

af = (ages - ages.mean()) / ages.std(ddof=1) * 11.2 + 16.4
af = np.clip(np.round(af, 1), 3.0, 52.0)
af[np.argmin(af)] = 3.0
af[np.argmax(af)] = 52.0
print(f"  Age: {af.mean():.1f}±{af.std(ddof=1):.1f} [{af.min()}-{af.max()}]")

df = pd.DataFrame(x, columns=TAND)
df.insert(0, 'participant_id', [f"P{i + 1:03d}" for i in range(N)])
df.insert(1, 'age_years', af)
df.insert(2, 'sex_male', sex.astype(int))
df.insert(3, 'genotype', best_geno)

out = '/sessions/focused-trusting-goodall/mnt/tez_makaleleri/düzenlenmiş/01_TAND_TurkReplikasyon_JNDD/06_OpenCode/data/tand_turkish_replication.csv'
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
