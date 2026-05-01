from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_stats(stats_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for json_path in sorted(stats_dir.glob("*_stats.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        needed = ("gene_id", "mean_reward", "mean_distance", "mean_control_cost")
        if not all(k in data for k in needed):
            continue

        try:
            rewards = data.get("rewards", [])
            distances = data.get("distances", [])
            control_costs = data.get("control_costs", [])

            reward_std = float(data.get("std_reward", np.std(rewards) if rewards else np.nan))
            distance_std = float(np.std(distances)) if distances else np.nan
            cost_std = float(np.std(control_costs)) if control_costs else np.nan

            rows.append(
                {
                    "gene_id": str(data["gene_id"]),
                    "mean_reward": float(data["mean_reward"]),
                    "mean_distance": float(data["mean_distance"]),
                    "mean_control_cost": float(data["mean_control_cost"]),
                    "std_reward": reward_std,
                    "std_distance": distance_std,
                    "std_control_cost": cost_std,
                }
            )
        except Exception:
            continue
    return rows


def nondominated_front(
    rows: Sequence[Dict[str, float]],
    maximize_keys: Sequence[str],
    minimize_keys: Sequence[str],
) -> List[Dict[str, float]]:
    front: List[Dict[str, float]] = []
    n = len(rows)
    for i in range(n):
        a = rows[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            b = rows[j]

            no_worse = True
            strictly_better = False

            for key in maximize_keys:
                if b[key] < a[key]:
                    no_worse = False
                    break
                if b[key] > a[key]:
                    strictly_better = True

            if not no_worse:
                continue

            for key in minimize_keys:
                if b[key] > a[key]:
                    no_worse = False
                    break
                if b[key] < a[key]:
                    strictly_better = True

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            front.append(a)
    return front


def nondominated_front_max_max(rows: Sequence[Dict[str, float]], x_key: str, y_key: str) -> List[Dict[str, float]]:
    return nondominated_front(rows, maximize_keys=[x_key, y_key], minimize_keys=[])


def make_notable_individuals(rows: Sequence[Dict[str, float]], front: Sequence[Dict[str, float]]) -> List[Dict[str, str]]:
    eps = 1e-9
    arr_reward = np.array([r["mean_reward"] for r in rows], dtype=float)
    arr_distance = np.array([r["mean_distance"] for r in rows], dtype=float)
    arr_cost = np.array([r["mean_control_cost"] for r in rows], dtype=float)

    chosen: set[str] = set()

    def as_notable(row: Dict[str, float], reason: str) -> Dict[str, str]:
        return {
            "gene_id": row["gene_id"],
            "reason": reason,
            "mean_reward": f"{row['mean_reward']:.4f}",
            "mean_distance": f"{row['mean_distance']:.4f}",
            "mean_control_cost": f"{row['mean_control_cost']:.6f}",
        }

    def pick_first_unique(candidates: Sequence[Dict[str, float]], reason: str) -> Dict[str, str]:
        for row in candidates:
            gid = row["gene_id"]
            if gid not in chosen:
                chosen.add(gid)
                return as_notable(row, reason)
        fallback = candidates[0]
        return as_notable(fallback, reason)

    novelty = np.column_stack(
        [
            (arr_reward - arr_reward.mean()) / (arr_reward.std() + eps),
            (arr_distance - arr_distance.mean()) / (arr_distance.std() + eps),
            -(arr_cost - arr_cost.mean()) / (arr_cost.std() + eps),
        ]
    )
    novelty_score = np.linalg.norm(novelty, axis=1)
    novelty_ranked = [rows[i] for i in np.argsort(-novelty_score)]

    low_cost_threshold = float(np.quantile(arr_cost, 0.25))
    low_cost_rows = [r for r in rows if r["mean_control_cost"] <= low_cost_threshold]
    reward_given_low_cost_ranked = sorted(low_cost_rows, key=lambda r: r["mean_reward"], reverse=True)

    front_sorted = sorted(front, key=lambda r: r["mean_control_cost"])
    front_high_distance_ranked = sorted(front, key=lambda r: r["mean_distance"], reverse=True)

    notables = [
        pick_first_unique(
            sorted(rows, key=lambda r: r["mean_reward"], reverse=True),
            "Highest mean reward (strong overall performer)",
        ),
        pick_first_unique(
            sorted(rows, key=lambda r: r["mean_distance"], reverse=True),
            "Longest mean distance (exploration/extremal locomotion)",
        ),
        pick_first_unique(
            sorted(rows, key=lambda r: r["mean_control_cost"]),
            "Lowest control cost (efficient/energy-saving behavior)",
        ),
        pick_first_unique(
            front_sorted,
            "Pareto frontier anchor: lowest-cost frontier point",
        ),
        pick_first_unique(
            front_high_distance_ranked,
            "Pareto frontier anchor: highest-distance frontier point",
        ),
        pick_first_unique(
            reward_given_low_cost_ranked,
            "High reward within lowest-cost quartile (efficient high performer)",
        ),
        pick_first_unique(
            novelty_ranked,
            "Most behaviorally unique profile (outlier in reward-distance-cost space)",
        ),
    ]

    return notables


def write_notables_report(notables: Sequence[Dict[str, str]], output_dir: Path) -> None:
    report_path = output_dir / "notable_individuals_for_path_viz.csv"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("gene_id,reason,mean_reward,mean_distance,mean_control_cost\n")
        for row in notables:
            reason = row["reason"].replace(",", ";")
            f.write(
                f"{row['gene_id']},{reason},{row['mean_reward']},{row['mean_distance']},{row['mean_control_cost']}\n"
            )


def percentile_rank(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) > 1:
        ranks = ranks / (len(values) - 1)
    else:
        ranks = np.zeros_like(ranks)
    return ranks if higher_is_better else 1.0 - ranks


def top_candidates_with_reasons(
    rows: Sequence[Dict[str, float]],
    front: Sequence[Dict[str, float]],
    top_k: int = 10,
    force_include_max_reward: bool = True,
) -> List[Dict[str, str]]:
    reward = np.array([r["mean_reward"] for r in rows], dtype=float)
    distance = np.array([r["mean_distance"] for r in rows], dtype=float)
    cost = np.array([r["mean_control_cost"] for r in rows], dtype=float)
    reward_std = np.array([r.get("std_reward", np.nan) for r in rows], dtype=float)

    # Missing std gets neutral robustness score.
    std_filled = np.where(np.isfinite(reward_std), reward_std, np.nanmedian(reward_std[np.isfinite(reward_std)]))

    reward_rank = percentile_rank(reward, higher_is_better=True)
    distance_rank = percentile_rank(distance, higher_is_better=True)
    efficiency_rank = percentile_rank(reward / (cost + 1e-9), higher_is_better=True)
    low_cost_rank = percentile_rank(cost, higher_is_better=False)
    robust_rank = percentile_rank(std_filled, higher_is_better=False)

    front_ids = {r["gene_id"] for r in front}
    front_bonus = np.array([1.0 if r["gene_id"] in front_ids else 0.0 for r in rows], dtype=float)

    # Weighted blend favors performance and distance, while still rewarding efficiency/robustness/frontier status.
    total_score = (
        0.30 * reward_rank
        + 0.30 * distance_rank
        + 0.18 * efficiency_rank
        + 0.12 * low_cost_rank
        + 0.07 * robust_rank
        + 0.03 * front_bonus
    )

    ranked_idx = np.argsort(-total_score)
    selected = list(ranked_idx[: min(top_k, len(rows))])

    if force_include_max_reward and len(selected) > 0:
        max_reward_idx = int(np.argmax(reward))
        if max_reward_idx not in selected:
            selected[-1] = max_reward_idx

    # Preserve uniqueness and order by score descending after forced inclusion.
    selected = list(dict.fromkeys(selected))
    selected = sorted(selected, key=lambda i: total_score[i], reverse=True)

    out: List[Dict[str, str]] = []
    for rank_pos, idx in enumerate(selected, start=1):
        row = rows[int(idx)]

        components = [
            ("very high reward", reward_rank[idx]),
            ("very high distance", distance_rank[idx]),
            ("strong reward-per-cost", efficiency_rank[idx]),
            ("low control cost", low_cost_rank[idx]),
            ("stable across eval episodes", robust_rank[idx]),
        ]
        top_reasons = [name for name, score in sorted(components, key=lambda x: x[1], reverse=True)[:2]]
        reason = f"{top_reasons[0]}; {top_reasons[1]}"
        if row["gene_id"] in front_ids:
            reason += "; on distance-cost Pareto frontier"

        out.append(
            {
                "rank": str(rank_pos),
                "gene_id": row["gene_id"],
                "why": reason,
                "score": f"{total_score[idx]:.4f}",
                "mean_reward": f"{row['mean_reward']:.4f}",
                "mean_distance": f"{row['mean_distance']:.4f}",
                "mean_control_cost": f"{row['mean_control_cost']:.6f}",
                "std_reward": f"{row.get('std_reward', np.nan):.4f}",
            }
        )
    return out


def write_top_candidates_report(top_candidates: Sequence[Dict[str, str]], output_dir: Path, filename: str) -> None:
    report_path = output_dir / filename
    with report_path.open("w", encoding="utf-8") as f:
        f.write("rank,gene_id,why,score,mean_reward,mean_distance,mean_control_cost,std_reward\n")
        for row in top_candidates:
            why = row["why"].replace(",", ";")
            f.write(
                f"{row['rank']},{row['gene_id']},{why},{row['score']},{row['mean_reward']},"
                f"{row['mean_distance']},{row['mean_control_cost']},{row['std_reward']}\n"
            )


def make_plots(rows: Sequence[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    front = nondominated_front(rows, maximize_keys=["mean_distance"], minimize_keys=["mean_control_cost"])
    notables = make_notable_individuals(rows, front)
    top5 = top_candidates_with_reasons(rows, front, top_k=5, force_include_max_reward=True)
    top10 = top_candidates_with_reasons(rows, front, top_k=10, force_include_max_reward=True)

    cost = np.array([r["mean_control_cost"] for r in rows], dtype=float)
    distance = np.array([r["mean_distance"] for r in rows], dtype=float)
    reward = np.array([r["mean_reward"] for r in rows], dtype=float)
    reward_std = np.array([r.get("std_reward", np.nan) for r in rows], dtype=float)
    eps = 1e-9

    efficiency = reward / (cost + eps)
    finite_eff = np.isfinite(efficiency)
    eff_quantile = np.quantile(efficiency[finite_eff], 0.95) if np.any(finite_eff) else 0.0
    clipped_eff = np.clip(efficiency, np.nanpercentile(efficiency, 2), eff_quantile)

    front_sorted_main = sorted(front, key=lambda r: (r["mean_distance"], r["mean_control_cost"]))
    front_distance = np.array([r["mean_distance"] for r in front_sorted_main], dtype=float)
    front_cost = np.array([r["mean_control_cost"] for r in front_sorted_main], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    # Primary requested plot: distance on X, control cost on Y, colored by reward.
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        distance,
        cost,
        c=reward,
        cmap="viridis",
        s=42,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.2,
    )
    ax.step(
        front_distance,
        front_cost,
        where="post",
        color="#111111",
        linewidth=2.4,
        label="Nondominated Pareto Front (step)",
        zorder=4,
    )
    ax.scatter(
        front_distance,
        front_cost,
        s=62,
        facecolors="#111111",
        edgecolors="white",
        linewidths=0.4,
        label="Pareto points",
        zorder=5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Reward")
    ax.set_title("Pareto View: Distance vs Control Cost (colored by Reward)")
    ax.set_xlabel("Mean Distance (higher is better)")
    ax.set_ylabel("Mean Control Cost (lower is better)")
    ax.legend(loc="best")
    ax.margins(x=0.03, y=0.08)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_distance_vs_control_cost_reward.png", dpi=240)
    plt.close(fig)

    # Efficiency view: distance vs reward-per-cost, colored by reward.
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        distance,
        clipped_eff,
        c=reward,
        cmap="viridis",
        s=38,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.2,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Reward")
    ax.set_title("Efficiency Map: Distance vs Reward per Control Cost")
    ax.set_xlabel("Mean Distance")
    ax.set_ylabel("Reward / Control Cost (clipped high tail)")
    ax.margins(x=0.03, y=0.08)
    fig.tight_layout()
    fig.savefig(output_dir / "analysis_efficiency_distance_vs_reward_per_cost.png", dpi=240)
    plt.close(fig)

    # Robustness view: mean reward vs reward std (lower is more stable), colored by cost.
    valid_robust = np.isfinite(reward_std)
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        reward[valid_robust],
        reward_std[valid_robust],
        c=cost[valid_robust],
        cmap="cividis_r",
        s=40,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.2,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Control Cost")
    ax.set_title("Robustness Map: Mean Reward vs Reward Std Dev")
    ax.set_xlabel("Mean Reward (higher is better)")
    ax.set_ylabel("Reward Std Dev (lower is better)")
    ax.margins(x=0.03, y=0.08)
    fig.tight_layout()
    fig.savefig(output_dir / "analysis_robustness_reward_vs_std.png", dpi=240)
    plt.close(fig)

    # Reward vs control cost (reward-max, cost-min) with stepwise frontier in transformed space.
    front_rc = nondominated_front(rows, maximize_keys=["mean_reward"], minimize_keys=["mean_control_cost"])
    x_rc = np.array([r["mean_control_cost"] for r in rows], dtype=float)
    y_rc = np.array([r["mean_reward"] for r in rows], dtype=float)
    front_x_rc = np.array([r["mean_control_cost"] for r in front_rc], dtype=float)
    front_y_rc = np.array([r["mean_reward"] for r in front_rc], dtype=float)
    order = np.argsort(front_x_rc)
    front_x_rc = front_x_rc[order]
    front_y_rc = front_y_rc[order]

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(x_rc, y_rc, c=distance, cmap="plasma", s=36, alpha=0.8, edgecolors="white", linewidths=0.2)
    ax.step(front_x_rc, front_y_rc, where="post", color="#1f1f1f", linewidth=2.2, label="Nondominated Pareto Front")
    ax.scatter(front_x_rc, front_y_rc, s=56, facecolors="#1f1f1f", edgecolors="white", linewidths=0.4, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Distance")
    ax.set_title("Trade-off: Control Cost vs Reward (colored by Distance)")
    ax.set_xlabel("Mean Control Cost (lower is better)")
    ax.set_ylabel("Mean Reward (higher is better)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_control_cost_vs_reward_distance.png", dpi=220)
    plt.close(fig)

    # Distance vs reward (both maximize), with stepwise nondominated line.
    front_dr = nondominated_front_max_max(rows, "mean_distance", "mean_reward")
    front_x_dr = np.array([r["mean_distance"] for r in front_dr], dtype=float)
    front_y_dr = np.array([r["mean_reward"] for r in front_dr], dtype=float)
    idx = np.argsort(front_x_dr)
    front_x_dr = front_x_dr[idx]
    front_y_dr = front_y_dr[idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(distance, reward, c=cost, cmap="cividis_r", s=36, alpha=0.8, edgecolors="white", linewidths=0.2)
    ax.step(front_x_dr, front_y_dr, where="post", color="#111111", linewidth=2.2, label="Nondominated frontier")
    ax.scatter(front_x_dr, front_y_dr, s=56, facecolors="#111111", edgecolors="white", linewidths=0.4, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Control Cost")
    ax.set_title("Distance vs Reward (colored by Control Cost)")
    ax.set_xlabel("Mean Distance (higher is better)")
    ax.set_ylabel("Mean Reward (higher is better)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_distance_vs_reward_control_cost.png", dpi=220)
    plt.close(fig)

    write_notables_report(notables, output_dir)
    write_top_candidates_report(top5, output_dir, "top_5_candidates_for_replay.csv")
    write_top_candidates_report(top10, output_dir, "top_10_candidates_for_replay.csv")

    # Knee point on distance-cost Pareto front.
    if len(front_distance) >= 3:
        x = front_distance
        y = front_cost
        x_n = (x - x.min()) / (x.max() - x.min() + eps)
        y_n = (y - y.min()) / (y.max() - y.min() + eps)
        start = np.array([x_n[0], y_n[0]])
        end = np.array([x_n[-1], y_n[-1]])
        line = end - start
        line_norm = np.linalg.norm(line) + eps
        px = x_n - start[0]
        py = y_n - start[1]
        distances_to_line = np.abs(line[0] * py - line[1] * px) / line_norm
        knee_idx = int(np.argmax(distances_to_line))
        knee_gene = front_sorted_main[knee_idx]["gene_id"]
        knee_distance = float(front_distance[knee_idx])
        knee_cost = float(front_cost[knee_idx])
    else:
        knee_gene = "N/A"
        knee_distance = float("nan")
        knee_cost = float("nan")

    # Correlations among primary metrics.
    corr_reward_distance = float(np.corrcoef(reward, distance)[0, 1])
    corr_reward_cost = float(np.corrcoef(reward, cost)[0, 1])
    corr_distance_cost = float(np.corrcoef(distance, cost)[0, 1])

    # Top robust performers: high reward with low std_reward.
    robust_candidates = [r for r in rows if np.isfinite(r.get("std_reward", np.nan))]
    robust_ranked = sorted(robust_candidates, key=lambda r: (-(r["mean_reward"]), r["std_reward"]))[:5]

    summary = output_dir / "pareto_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("Pareto Analysis Summary\n")
        f.write("=======================\n")
        f.write(f"Total individuals analyzed: {len(rows)}\n")
        f.write(f"Distance-vs-cost Pareto frontier points: {len(front)}\n\n")
        f.write("Metric correlations (Pearson):\n")
        f.write(f"- reward vs distance: {corr_reward_distance:.4f}\n")
        f.write(f"- reward vs control_cost: {corr_reward_cost:.4f}\n")
        f.write(f"- distance vs control_cost: {corr_distance_cost:.4f}\n\n")
        f.write("Pareto knee point (distance-cost front):\n")
        f.write(f"- gene_id: {knee_gene}\n")
        f.write(f"- mean_distance: {knee_distance:.4f}\n")
        f.write(f"- mean_control_cost: {knee_cost:.6f}\n\n")
        f.write("Recommended individuals for path visualization:\n")
        for idx, item in enumerate(notables, start=1):
            f.write(
                f"{idx}. {item['gene_id']} | {item['reason']} | "
                f"reward={item['mean_reward']} distance={item['mean_distance']} cost={item['mean_control_cost']}\n"
            )
        f.write("\nTop 5 candidates for replay/trajectory visualization (forced include max-reward):\n")
        for row in top5:
            f.write(
                f"{row['rank']}. {row['gene_id']} | {row['why']} | score={row['score']} | "
                f"reward={row['mean_reward']} distance={row['mean_distance']} "
                f"cost={row['mean_control_cost']} std_reward={row['std_reward']}\n"
            )

        f.write("\nTop 10 candidates for replay/trajectory visualization (forced include max-reward):\n")
        for row in top10:
            f.write(
                f"{row['rank']}. {row['gene_id']} | {row['why']} | score={row['score']} | "
                f"reward={row['mean_reward']} distance={row['mean_distance']} "
                f"cost={row['mean_control_cost']} std_reward={row['std_reward']}\n"
            )
        f.write("\nTop robust high-reward individuals (reward desc, std asc):\n")
        for idx, row in enumerate(robust_ranked, start=1):
            f.write(
                f"{idx}. {row['gene_id']} | reward={row['mean_reward']:.4f} "
                f"std_reward={row['std_reward']:.4f} distance={row['mean_distance']:.4f} "
                f"cost={row['mean_control_cost']:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pareto-front visualizations from stats JSON files.")
    parser.add_argument("--stats-dir", type=Path, default=Path("stats"), help="Directory containing *_stats.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("pareto_plots"), help="Directory for output plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_stats(args.stats_dir)
    if not rows:
        raise SystemExit(f"No valid stats JSON files found in: {args.stats_dir}")
    make_plots(rows, args.output_dir)
    print(f"Generated Pareto plots and reports in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
