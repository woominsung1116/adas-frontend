#!/usr/bin/env python3
"""Generate result figures for ADAS v16 (30 cls) + v18 (13/30 cls)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/tmp/adas_link/results")
SEEDS = [42, 43, 44]

def load(seed, arm, version="v16"):
    f = ROOT / f"full30_{version}_s{seed}" / f"full30_{version}_s{seed}_{arm}.csv"
    if not f.exists():
        f = ROOT / f"full30_{version}_s{seed}" / f"{arm}_incremental.csv"
    return list(csv.DictReader(open(f)))

def col(rows, k):
    return [float(r[k]) for r in rows]

# ----------------------------------------------------------------------------
# Figure 1: v16 30-class learning curve (baseline vs policy, 3 seeds + mean)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("ADAS v16 — Final 30-class learning curve (baseline vs policy, 3 seeds)",
             fontsize=14, fontweight="bold")

METRICS = [
    ("f1", "F1 score", 0, 0),
    ("sensitivity", "Sensitivity (recall)", 0, 1),
    ("ppv", "PPV (precision)", 1, 0),
    ("avg_identification_turn", "Avg identification turn (lower=faster)", 1, 1),
]

baseline_data = {m: [] for m, _, _, _ in METRICS}
policy_data = {m: [] for m, _, _, _ in METRICS}

for seed in SEEDS:
    base = load(seed, "baseline")
    pol  = load(seed, "policy")
    classes = [int(r["class_id"]) for r in base]
    for m, _, _, _ in METRICS:
        baseline_data[m].append(col(base, m))
        policy_data[m].append(col(pol, m))

for m, label, r, c in METRICS:
    ax = axes[r, c]
    base_arr = np.array(baseline_data[m])
    pol_arr  = np.array(policy_data[m])
    cls = np.arange(1, 31)

    for i, s in enumerate(SEEDS):
        ax.plot(cls, base_arr[i], "o-", color="#4C78A8", alpha=0.25, lw=1, ms=3, label=f"baseline s{s}" if i == 0 else None)
        ax.plot(cls, pol_arr[i],  "o-", color="#E45756", alpha=0.25, lw=1, ms=3, label=f"policy s{s}" if i == 0 else None)
    base_mean = base_arr.mean(axis=0); base_std = base_arr.std(axis=0)
    pol_mean  = pol_arr.mean(axis=0);  pol_std  = pol_arr.std(axis=0)
    ax.plot(cls, base_mean, "-", color="#1F4E79", lw=2.5, label="baseline mean")
    ax.fill_between(cls, base_mean-base_std, base_mean+base_std, color="#4C78A8", alpha=0.15)
    ax.plot(cls, pol_mean, "-", color="#922B21", lw=2.5, label="policy mean")
    ax.fill_between(cls, pol_mean-pol_std, pol_mean+pol_std, color="#E45756", alpha=0.15)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("class index")
    ax.set_ylabel(m)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("/tmp/v16_final_30class.png")
plt.savefig(out, dpi=130)
print(f"saved {out}")

# ----------------------------------------------------------------------------
# Figure 2: v16 baseline vs policy summary bars + delta
# ----------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("v16 final — mean F1 by arm (3 seeds) + Δ", fontsize=13, fontweight="bold")

# Mean F1 grouped bars
ax = axes2[0]
base_means, pol_means, deltas = [], [], []
for seed in SEEDS:
    comp = json.load(open(ROOT / f"full30_v16_s{seed}" / f"full30_v16_s{seed}_comparison.json"))
    base_means.append(comp["baseline"]["mean_f1"])
    pol_means.append(comp["policy"]["mean_f1"])
    deltas.append(comp["delta"]["f1_diff"])

x = np.arange(len(SEEDS))
w = 0.35
ax.bar(x - w/2, base_means, w, label="baseline (rule-based)", color="#4C78A8")
ax.bar(x + w/2, pol_means,  w, label="policy (LLM+memory)",  color="#E45756")
for i, (b, p) in enumerate(zip(base_means, pol_means)):
    ax.text(i - w/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in SEEDS])
ax.set_ylabel("mean F1 (30 classes)")
ax.set_title("Mean F1: baseline vs policy")
ax.grid(alpha=0.3, axis="y"); ax.legend()

# Delta bars
ax = axes2[1]
colors = ["#54A24B" if d > 0 else "#E45756" for d in deltas]
ax.bar(x, deltas, color=colors)
for i, d in enumerate(deltas):
    ax.text(i, d + (0.005 if d > 0 else -0.015), f"{d:+.3f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in SEEDS])
ax.set_ylabel("Δ F1 (policy − baseline)")
ax.set_title("Learning effect (Δ = policy − baseline)")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out2 = Path("/tmp/v16_final_delta.png")
plt.savefig(out2, dpi=130)
print(f"saved {out2}")

# ----------------------------------------------------------------------------
# Figure 3: v16 vs v18 partial comparison (first 13 classes)
# ----------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("v16 (30 cls finished) vs v18 (13/30 — partial)  policy arm F1 by class",
              fontsize=13, fontweight="bold")

for i, seed in enumerate(SEEDS):
    ax = axes3[i]
    v16_pol = load(seed, "policy")
    v18_pol = load(seed, "policy", version="v18")
    cls16 = [int(r["class_id"]) for r in v16_pol]
    cls18 = [int(r["class_id"]) for r in v18_pol]
    f16 = col(v16_pol, "f1")
    f18 = col(v18_pol, "f1")
    ax.plot(cls16, f16, "o-", color="#E45756", alpha=0.8, lw=1.5, ms=4, label="v16 (final 30 cls)")
    ax.plot(cls18, f18, "s-", color="#54A24B", alpha=0.9, lw=2, ms=5, label="v18 (13/30, 7 fixes)")
    ax.axvline(13.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"seed {seed}", fontsize=11, fontweight="bold")
    ax.set_xlabel("class index")
    ax.set_ylabel("F1")
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out3 = Path("/tmp/v16_vs_v18_partial.png")
plt.savefig(out3, dpi=130)
print(f"saved {out3}")

# ----------------------------------------------------------------------------
# Figure 4: Memory growth (case_base + principles) for v16
# ----------------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4.5))
fig4.suptitle("v16 final memory state — accumulated cases & principles", fontsize=13, fontweight="bold")

case_counts, princ_counts, proc_counts = [], [], []
for seed in SEEDS:
    mem = json.load(open(ROOT / f"full30_v16_s{seed}" / "memory_final.json"))
    case_counts.append(len(mem.get("case_base", [])))
    princ_counts.append(len(mem.get("experience_base", [])))
    proc_counts.append(len(mem.get("procedural_memory", [])))

x = np.arange(len(SEEDS))
ax = axes4[0]
ax.bar(x, case_counts, color="#4C78A8")
for i, v in enumerate(case_counts):
    ax.text(i, v + max(case_counts)*0.02, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in SEEDS])
ax.set_ylabel("case count")
ax.set_title("Case base size (episodic memory)")
ax.grid(alpha=0.3, axis="y")

ax = axes4[1]
w = 0.35
ax.bar(x - w/2, princ_counts, w, color="#54A24B", label="principles (semantic)")
ax.bar(x + w/2, proc_counts,  w, color="#F58518", label="procedural patterns")
for i, v in enumerate(princ_counts):
    ax.text(i - w/2, v + 1, str(v), ha="center", fontsize=9)
for i, v in enumerate(proc_counts):
    ax.text(i + w/2, v + 1, str(v), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in SEEDS])
ax.set_ylabel("count")
ax.set_title("Distilled knowledge (principles + procedural)")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out4 = Path("/tmp/v16_memory_growth.png")
plt.savefig(out4, dpi=130)
print(f"saved {out4}")

print()
print(f"v16 memory: case_base={case_counts} principles={princ_counts} procedural={proc_counts}")
