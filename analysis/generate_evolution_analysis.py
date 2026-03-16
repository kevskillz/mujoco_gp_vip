from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis"
SEED_GENE_ID = "xXxjOH38bl0QwZU7tCi3JJAEICx"
DEFAULT_LOGS = [
    ROOT / "slurm-4442153.out",
    ROOT / "slurm-4457323.out",
]

GENERATION_RE = re.compile(r"STARTING GENERATION:\s+(\d+)")
FITNESS_RE = re.compile(
    r"Fitness:\s+\(([^,]+),\s*([^,]+),\s*([^)]+)\), Submission Flag:\s+(True|False)"
)
RUNTIME_RE = re.compile(r"Runtime:\s+(\d+)\s+min,\s+Status:\s+(.+)$")
JOB_RE = re.compile(r"LLM Job-ID:\s+(.+)$")


@dataclass
class SnapshotRecord:
    generation: int
    gene_id: str
    reward: float
    distance: float
    control_cost: float
    submission_flag: bool
    runtime_min: int | None
    status: str
    job_id: str
    source_log: str

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(self.reward)
            and math.isfinite(self.distance)
            and math.isfinite(self.control_cost)
            and self.reward > -1e8
        )

    @property
    def failure_kind(self) -> str:
        if self.is_valid:
            return "valid"
        if not self.submission_flag:
            return "submission_failed"
        if any(not math.isfinite(x) for x in (self.reward, self.distance, self.control_cost)):
            return "runtime_invalid"
        if self.reward <= -1e8:
            return "hard_failure"
        return "other_failure"


def parse_float(text: str) -> float:
    text = text.strip()
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    return float(text)


def parse_generation_snapshots(log_path: Path) -> list[SnapshotRecord]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[SnapshotRecord] = []
    i = 0
    while i < len(lines):
        gen_match = GENERATION_RE.search(lines[i])
        if not gen_match:
            i += 1
            continue

        generation = int(gen_match.group(1))
        i += 1
        while i < len(lines) and "Poplutation Info" not in lines[i]:
            i += 1
        if i >= len(lines):
            break

        i += 1
        while i < len(lines):
            line = lines[i]
            if "Invalid Removal" in line:
                break
            if line.startswith("Gene: "):
                gene_id = line.split("Gene: ", 1)[1].strip()
                fitness_line = lines[i + 1] if i + 1 < len(lines) else ""
                runtime_line = lines[i + 2] if i + 2 < len(lines) else ""
                job_line = lines[i + 3] if i + 3 < len(lines) else ""

                fitness_match = FITNESS_RE.search(fitness_line)
                runtime_match = RUNTIME_RE.search(runtime_line)
                job_match = JOB_RE.search(job_line)

                if fitness_match:
                    reward = parse_float(fitness_match.group(1))
                    distance = parse_float(fitness_match.group(2))
                    control_cost = parse_float(fitness_match.group(3))
                    submission_flag = fitness_match.group(4) == "True"
                    runtime_min = int(runtime_match.group(1)) if runtime_match else None
                    status = runtime_match.group(2) if runtime_match else "unknown"
                    job_id = job_match.group(1).strip() if job_match else "unknown"
                    records.append(
                        SnapshotRecord(
                            generation=generation,
                            gene_id=gene_id,
                            reward=reward,
                            distance=distance,
                            control_cost=control_cost,
                            submission_flag=submission_flag,
                            runtime_min=runtime_min,
                            status=status,
                            job_id=job_id,
                            source_log=log_path.name,
                        )
                    )
                    i += 4
                    continue
            i += 1
    return records


def dedupe_individuals(records: list[SnapshotRecord]) -> list[dict]:
    by_gene: dict[str, dict] = {}
    for rec in records:
        existing = by_gene.get(rec.gene_id)
        if existing is None:
            by_gene[rec.gene_id] = {
                "gene_id": rec.gene_id,
                "first_generation": rec.generation,
                "last_generation": rec.generation,
                "appearances": 1,
                "reward": rec.reward,
                "distance": rec.distance,
                "control_cost": rec.control_cost,
                "runtime_min": rec.runtime_min,
                "submission_flag": rec.submission_flag,
                "status": rec.status,
                "job_id": rec.job_id,
                "source_log": rec.source_log,
                "is_valid": rec.is_valid,
                "failure_kind": rec.failure_kind,
            }
        else:
            existing["last_generation"] = rec.generation
            existing["appearances"] += 1
    return list(by_gene.values())


def is_dominated(candidate: dict, others: list[dict]) -> bool:
    for other in others:
        if other["gene_id"] == candidate["gene_id"]:
            continue
        dominates = (
            other["reward"] >= candidate["reward"]
            and other["distance"] >= candidate["distance"]
            and other["control_cost"] <= candidate["control_cost"]
            and (
                other["reward"] > candidate["reward"]
                or other["distance"] > candidate["distance"]
                or other["control_cost"] < candidate["control_cost"]
            )
        )
        if dominates:
            return True
    return False


def pareto_front(records: list[dict]) -> list[dict]:
    valid = [rec for rec in records if rec["is_valid"]]
    return [rec for rec in valid if not is_dominated(rec, valid)]


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    return (features - mean) / std


def kmeans(features: np.ndarray, k: int, seed: int = 7, steps: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), size=k, replace=False)
    centroids = features[idx].copy()

    for _ in range(steps):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_idx in range(k):
            members = features[labels == cluster_idx]
            if len(members) == 0:
                new_centroids[cluster_idx] = features[rng.integers(0, len(features))]
            else:
                new_centroids[cluster_idx] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    scores = centroids.sum(axis=1)
    order = np.argsort(scores)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[label] for label in labels])


def cluster_labels(snapshot_records: list[SnapshotRecord]) -> tuple[np.ndarray, list[str]]:
    valid = [rec for rec in snapshot_records if rec.is_valid]
    if not valid:
        return np.array([]), []

    features = np.array(
        [[rec.reward, rec.distance, -rec.control_cost] for rec in valid],
        dtype=float,
    )
    features = standardize(features)
    k = min(4, len(valid))
    labels = kmeans(features, k=k)
    names_full = ["Lower", "Developing", "Strong", "Elite"]
    names = names_full[-k:]
    return labels, names


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_generation_summary(records: list[SnapshotRecord]) -> list[dict]:
    by_generation: dict[int, list[SnapshotRecord]] = {}
    for rec in records:
        by_generation.setdefault(rec.generation, []).append(rec)

    summary: list[dict] = []
    for generation in sorted(by_generation):
        bucket = by_generation[generation]
        valid = [rec for rec in bucket if rec.is_valid]
        summary.append(
            {
                "generation": generation,
                "population_size": len(bucket),
                "valid_count": len(valid),
                "invalid_count": len(bucket) - len(valid),
                "mean_reward_valid": (
                    round(float(np.mean([rec.reward for rec in valid])), 6) if valid else ""
                ),
                "max_reward_valid": (
                    round(float(np.max([rec.reward for rec in valid])), 6) if valid else ""
                ),
                "mean_distance_valid": (
                    round(float(np.mean([rec.distance for rec in valid])), 6) if valid else ""
                ),
                "min_control_cost_valid": (
                    round(float(np.min([rec.control_cost for rec in valid])), 6) if valid else ""
                ),
            }
        )
    return summary


def build_markdown_report(
    logs: list[Path],
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    frontier: list[dict],
    generation_summary: list[dict],
) -> str:
    valid_unique = [rec for rec in unique_records if rec["is_valid"]]
    invalid_unique = [rec for rec in unique_records if not rec["is_valid"]]
    top_reward = sorted(valid_unique, key=lambda row: row["reward"], reverse=True)[:15]
    top_distance = sorted(valid_unique, key=lambda row: row["distance"], reverse=True)[:15]

    lines: list[str] = []
    lines.append("# Evolution Run Analysis")
    lines.append("")
    lines.append("## Files Analyzed")
    for log in logs:
        lines.append(f"- `{log.name}`")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Generations covered: `{generation_summary[0]['generation']}` to `{generation_summary[-1]['generation']}`")
    lines.append(f"- Population snapshots parsed: `{len(snapshot_records)}`")
    lines.append(f"- Unique individuals observed: `{len(unique_records)}`")
    lines.append(f"- Unique valid individuals: `{len(valid_unique)}`")
    lines.append(f"- Unique failed / invalid individuals: `{len(invalid_unique)}`")
    lines.append("")
    lines.append("## Top Individuals By Reward")
    lines.append("")
    lines.append("| Rank | Gene ID | First Gen | Reward | Distance | Ctrl Cost | Appearances |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(top_reward, start=1):
        lines.append(
            f"| {idx} | `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {row['distance']:.4f} | {row['control_cost']:.4f} | {row['appearances']} |"
        )
    lines.append("")
    lines.append("## Top Individuals By Distance")
    lines.append("")
    lines.append("| Rank | Gene ID | First Gen | Reward | Distance | Ctrl Cost |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(top_distance, start=1):
        lines.append(
            f"| {idx} | `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {row['distance']:.4f} | {row['control_cost']:.4f} |"
        )
    lines.append("")
    lines.append("## Overall Pareto Frontier")
    lines.append("")
    lines.append("| Gene ID | First Gen | Reward | Distance | Ctrl Cost |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in sorted(frontier, key=lambda item: item["reward"], reverse=True):
        lines.append(
            f"| `{row['gene_id']}` | {row['first_generation']} | "
            f"{row['reward']:.4f} | {row['distance']:.4f} | {row['control_cost']:.4f} |"
        )
    lines.append("")
    lines.append("## Generation Trend")
    lines.append("")
    lines.append("| Gen | Pop Size | Valid | Invalid | Mean Reward | Max Reward | Mean Distance | Min Ctrl Cost |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in generation_summary:
        lines.append(
            f"| {row['generation']} | {row['population_size']} | {row['valid_count']} | {row['invalid_count']} | "
            f"{row['mean_reward_valid']} | {row['max_reward_valid']} | {row['mean_distance_valid']} | {row['min_control_cost_valid']} |"
        )
    return "\n".join(lines)


def prepare_plot_data(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
):
    valid_unique = [rec for rec in unique_records if rec["is_valid"]]
    distances = np.array([rec["distance"] for rec in valid_unique], dtype=float)
    rewards = np.array([rec["reward"] for rec in valid_unique], dtype=float)
    costs = np.array([rec["control_cost"] for rec in valid_unique], dtype=float)
    generations = np.array([rec["first_generation"] for rec in valid_unique], dtype=float)
    cost_norm = (costs.max() - costs) / max(costs.max() - costs.min(), 1e-9)
    sizes = 50 + 180 * cost_norm

    valid_snapshots = [rec for rec in snapshot_records if rec.is_valid]
    labels, cluster_names = cluster_labels(valid_snapshots)
    palette = ["#5E3C99", "#3288BD", "#66C2A5", "#E6AB02"]
    generation_values = sorted({rec.generation for rec in snapshot_records})
    cluster_color = {name: palette[idx] for idx, name in enumerate(cluster_names)}
    grouped_by_generation: dict[int, list[SnapshotRecord]] = {gen: [] for gen in generation_values}
    for rec in valid_snapshots:
        grouped_by_generation[rec.generation].append(rec)

    mean_rewards = []
    max_rewards = []
    for gen in generation_values:
        bucket = grouped_by_generation[gen]
        if bucket:
            mean_rewards.append(np.mean([rec.reward for rec in bucket]))
            max_rewards.append(np.max([rec.reward for rec in bucket]))
        else:
            mean_rewards.append(np.nan)
            max_rewards.append(np.nan)

    return {
        "valid_unique": valid_unique,
        "distances": distances,
        "rewards": rewards,
        "costs": costs,
        "generations": generations,
        "sizes": sizes,
        "valid_snapshots": valid_snapshots,
        "labels": labels,
        "cluster_names": cluster_names,
        "cluster_color": cluster_color,
        "generation_values": generation_values,
        "mean_rewards": mean_rewards,
        "max_rewards": max_rewards,
    }


def get_seed_unique_record(unique_records: list[dict]) -> dict | None:
    for rec in unique_records:
        if rec["gene_id"] == SEED_GENE_ID:
            return rec
    return None


def get_seed_snapshots(snapshot_records: list[SnapshotRecord]) -> list[SnapshotRecord]:
    return [rec for rec in snapshot_records if rec.gene_id == SEED_GENE_ID and rec.is_valid]


def highlight_seed_2d(ax, x: float, y: float, label: str = "Seed baseline") -> None:
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=320,
        color="#D62728",
        edgecolors="black",
        linewidths=1.2,
        zorder=8,
        label=label,
    )
    ax.annotate(
        "seed",
        (x, y),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#8B0000",
    )


def highlight_seed_3d(ax, x: float, y: float, z: float, label: str = "Seed baseline") -> None:
    ax.scatter(
        [x],
        [y],
        [z],
        marker="*",
        s=260,
        color="#D62728",
        edgecolors="black",
        linewidths=1.1,
        depthshade=False,
        zorder=10,
        label=label,
    )
    ax.text(x, y, z, "seed", fontsize=9, color="#8B0000")


def plot_pareto_figure(
    unique_records: list[dict],
    frontier: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_pareto = plt.subplots(figsize=(9, 8), constrained_layout=True)
    plot_data = prepare_plot_data([], unique_records)

    scatter = ax_pareto.scatter(
        plot_data["distances"],
        plot_data["rewards"],
        c=plot_data["generations"],
        s=plot_data["sizes"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )

    frontier_points = sorted(frontier, key=lambda row: row["distance"])
    ax_pareto.plot(
        [row["distance"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="black",
        linewidth=2,
        label="Pareto frontier",
    )
    ax_pareto.scatter(
        [row["distance"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="gold",
        edgecolors="black",
        s=90,
        zorder=5,
    )

    for row in sorted(frontier_points, key=lambda item: item["reward"], reverse=True)[:6]:
        ax_pareto.annotate(
            row["gene_id"][:10],
            (row["distance"], row["reward"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    seed_record = get_seed_unique_record(unique_records)
    if seed_record is not None:
        highlight_seed_2d(ax_pareto, seed_record["distance"], seed_record["reward"])

    ax_pareto.set_title("Pareto View of Unique Valid Individuals")
    ax_pareto.set_xlabel("Mean distance")
    ax_pareto.set_ylabel("Mean reward")
    ax_pareto.set_xlim(left=0)
    ax_pareto.set_ylim(bottom=0)
    ax_pareto.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax_pareto)
    cbar.set_label("First generation seen")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trend_figure(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_trend = plt.subplots(figsize=(10, 8), constrained_layout=True)
    plot_data = prepare_plot_data(snapshot_records, unique_records)

    random.seed(7)
    for rec, label in zip(plot_data["valid_snapshots"], plot_data["labels"]):
        cluster_name = plot_data["cluster_names"][label]
        jitter = random.uniform(-0.22, 0.22)
        ax_trend.scatter(
            rec.generation + jitter,
            rec.reward,
            color=plot_data["cluster_color"][cluster_name],
            alpha=0.7,
            s=28 + 2.2 * max(rec.distance, 0.0),
            edgecolors="none",
        )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["mean_rewards"],
        color="black",
        linewidth=2,
        label="Mean reward",
    )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["max_rewards"],
        color="#D95F02",
        linewidth=2,
        linestyle="--",
        label="Best reward",
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=name, markerfacecolor=plot_data["cluster_color"][name], markersize=8
        )
        for name in plot_data["cluster_names"]
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", linewidth=2, label="Mean reward"),
            plt.Line2D([0], [0], color="#D95F02", linewidth=2, linestyle="--", label="Best reward"),
        ]
    )

    seed_snapshots = get_seed_snapshots(snapshot_records)
    if seed_snapshots:
        ax_trend.plot(
            [rec.generation for rec in seed_snapshots],
            [rec.reward for rec in seed_snapshots],
            color="#D62728",
            linewidth=2,
            linestyle=":",
            label="Seed baseline",
        )
        highlight_seed_2d(ax_trend, seed_snapshots[0].generation, seed_snapshots[0].reward, label="Seed baseline")
        handles.append(plt.Line2D([0], [0], color="#D62728", linewidth=2, linestyle=":", label="Seed baseline"))

    ax_trend.legend(handles=handles, loc="upper left")
    ax_trend.set_title("Generation Trend With Performance Clusters")
    ax_trend.set_xlabel("Generation")
    ax_trend.set_ylabel("Mean reward")
    ax_trend.set_xticks(plot_data["generation_values"])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_3d_figure(
    unique_records: list[dict],
    frontier: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plot_data = prepare_plot_data([], unique_records)

    scatter = ax.scatter(
        plot_data["distances"],
        plot_data["rewards"],
        plot_data["costs"],
        c=plot_data["generations"],
        s=plot_data["sizes"],
        cmap="viridis",
        alpha=0.6,
        depthshade=True,
    )

    frontier_points = sorted(frontier, key=lambda row: (row["distance"], row["reward"]))
    ax.scatter(
        [row["distance"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        [row["control_cost"] for row in frontier_points],
        color="gold",
        edgecolors="black",
        s=120,
        depthshade=False,
        label="Pareto-optimal",
    )

    for row in sorted(frontier_points, key=lambda item: item["reward"], reverse=True)[:6]:
        ax.text(
            row["distance"],
            row["reward"],
            row["control_cost"],
            row["gene_id"][:10],
            fontsize=8,
        )

    seed_record = get_seed_unique_record(unique_records)
    if seed_record is not None:
        highlight_seed_3d(ax, seed_record["distance"], seed_record["reward"], seed_record["control_cost"])

    ax.set_title("3D Objective Space With Pareto-Optimal Individuals Highlighted")
    ax.set_xlabel("Mean distance")
    ax.set_ylabel("Mean reward")
    ax.set_zlabel("Mean control cost\n(lower is better)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("First generation seen")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_generation_3d_figure(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plot_data = prepare_plot_data(snapshot_records, unique_records)
    valid_snapshots = plot_data["valid_snapshots"]

    if not valid_snapshots:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    generations = np.array([rec.generation for rec in valid_snapshots], dtype=float)
    rewards = np.array([rec.reward for rec in valid_snapshots], dtype=float)
    distances = np.array([rec.distance for rec in valid_snapshots], dtype=float)
    costs = np.array([rec.control_cost for rec in valid_snapshots], dtype=float)
    sizes = 28 + 2.2 * np.maximum(distances, 0.0)

    cost_norm = (costs - costs.min()) / max(costs.max() - costs.min(), 1e-9)
    colors = cm.plasma_r(cost_norm)
    ax.scatter(
        generations,
        distances,
        rewards,
        c=colors,
        s=sizes,
        alpha=0.65,
        depthshade=True,
    )

    best_by_generation: list[SnapshotRecord] = []
    for generation in sorted({rec.generation for rec in valid_snapshots}):
        bucket = [rec for rec in valid_snapshots if rec.generation == generation]
        if bucket:
            best_by_generation.append(max(bucket, key=lambda rec: rec.reward))

    ax.plot(
        [rec.generation for rec in best_by_generation],
        [rec.distance for rec in best_by_generation],
        [rec.reward for rec in best_by_generation],
        color="black",
        linewidth=2.2,
        label="Best per generation",
    )
    ax.scatter(
        [rec.generation for rec in best_by_generation],
        [rec.distance for rec in best_by_generation],
        [rec.reward for rec in best_by_generation],
        color="#D95F02",
        s=90,
        depthshade=False,
    )

    for rec in best_by_generation[-4:]:
        ax.text(rec.generation, rec.distance, rec.reward, rec.gene_id[:10], fontsize=8)

    seed_snapshots = get_seed_snapshots(snapshot_records)
    if seed_snapshots:
        ax.plot(
            [rec.generation for rec in seed_snapshots],
            [rec.distance for rec in seed_snapshots],
            [rec.reward for rec in seed_snapshots],
            color="#D62728",
            linewidth=2,
            linestyle=":",
            label="Seed baseline",
        )
        highlight_seed_3d(
            ax,
            seed_snapshots[0].generation,
            seed_snapshots[0].distance,
            seed_snapshots[0].reward,
            label="Seed baseline",
        )

    ax.set_title("3D Generation Trajectory in Objective Space")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean distance")
    ax.set_zlabel("Mean reward")
    ax.view_init(elev=24, azim=-48)
    ax.legend(loc="upper left")

    scalar_map = cm.ScalarMappable(cmap="plasma_r")
    scalar_map.set_array(costs)
    cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("Mean control cost\n(lower is better)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(
    snapshot_records: list[SnapshotRecord],
    unique_records: list[dict],
    frontier: list[dict],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_pareto, ax_trend) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    plot_data = prepare_plot_data(snapshot_records, unique_records)

    scatter = ax_pareto.scatter(
        plot_data["distances"],
        plot_data["rewards"],
        c=plot_data["generations"],
        s=plot_data["sizes"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )

    frontier_points = sorted(frontier, key=lambda row: row["distance"])
    ax_pareto.plot(
        [row["distance"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="black",
        linewidth=2,
        label="Pareto frontier",
    )
    ax_pareto.scatter(
        [row["distance"] for row in frontier_points],
        [row["reward"] for row in frontier_points],
        color="gold",
        edgecolors="black",
        s=90,
        zorder=5,
    )

    for row in sorted(frontier_points, key=lambda item: item["reward"], reverse=True)[:6]:
        ax_pareto.annotate(
            row["gene_id"][:10],
            (row["distance"], row["reward"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    seed_record = get_seed_unique_record(unique_records)
    if seed_record is not None:
        highlight_seed_2d(ax_pareto, seed_record["distance"], seed_record["reward"])

    ax_pareto.set_title("Pareto View of Unique Valid Individuals")
    ax_pareto.set_xlabel("Mean distance")
    ax_pareto.set_ylabel("Mean reward")
    ax_pareto.set_xlim(left=0)
    ax_pareto.set_ylim(bottom=0)
    ax_pareto.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax_pareto)
    cbar.set_label("First generation seen")

    random.seed(7)
    for rec, label in zip(plot_data["valid_snapshots"], plot_data["labels"]):
        cluster_name = plot_data["cluster_names"][label]
        jitter = random.uniform(-0.22, 0.22)
        ax_trend.scatter(
            rec.generation + jitter,
            rec.reward,
            color=plot_data["cluster_color"][cluster_name],
            alpha=0.7,
            s=28 + 2.2 * max(rec.distance, 0.0),
            edgecolors="none",
        )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["mean_rewards"],
        color="black",
        linewidth=2,
        label="Mean reward",
    )
    ax_trend.plot(
        plot_data["generation_values"],
        plot_data["max_rewards"],
        color="#D95F02",
        linewidth=2,
        linestyle="--",
        label="Best reward",
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=name, markerfacecolor=plot_data["cluster_color"][name], markersize=8
        )
        for name in plot_data["cluster_names"]
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", linewidth=2, label="Mean reward"),
            plt.Line2D([0], [0], color="#D95F02", linewidth=2, linestyle="--", label="Best reward"),
        ]
    )
    seed_snapshots = get_seed_snapshots(snapshot_records)
    if seed_snapshots:
        ax_trend.plot(
            [rec.generation for rec in seed_snapshots],
            [rec.reward for rec in seed_snapshots],
            color="#D62728",
            linewidth=2,
            linestyle=":",
            label="Seed baseline",
        )
        highlight_seed_2d(ax_trend, seed_snapshots[0].generation, seed_snapshots[0].reward, label="Seed baseline")
        handles.append(plt.Line2D([0], [0], color="#D62728", linewidth=2, linestyle=":", label="Seed baseline"))
    ax_trend.legend(handles=handles, loc="upper left")
    ax_trend.set_title("Generation Trend With Performance Clusters")
    ax_trend.set_xlabel("Generation")
    ax_trend.set_ylabel("Mean reward")
    ax_trend.set_xticks(plot_data["generation_values"])

    fig.suptitle("Evolution Analysis Across Generations 0-13", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = [path for path in DEFAULT_LOGS if path.exists()]
    records_by_generation: dict[int, list[SnapshotRecord]] = {}
    for log in logs:
        parsed = parse_generation_snapshots(log)
        grouped: dict[int, list[SnapshotRecord]] = {}
        for rec in parsed:
            grouped.setdefault(rec.generation, []).append(rec)
        # Later logs replace earlier generations, which handles checkpoint resumes cleanly.
        records_by_generation.update(grouped)

    snapshot_records = [
        rec
        for generation in sorted(records_by_generation)
        for rec in records_by_generation[generation]
    ]

    unique_records = dedupe_individuals(snapshot_records)
    unique_records.sort(key=lambda row: (not row["is_valid"], -row["reward"] if row["is_valid"] else 0, row["gene_id"]))
    frontier = pareto_front(unique_records)
    generation_summary = build_generation_summary(snapshot_records)

    snapshot_csv_rows = [
        {
            "generation": rec.generation,
            "gene_id": rec.gene_id,
            "reward": rec.reward,
            "distance": rec.distance,
            "control_cost": rec.control_cost,
            "submission_flag": rec.submission_flag,
            "runtime_min": rec.runtime_min,
            "status": rec.status,
            "job_id": rec.job_id,
            "source_log": rec.source_log,
            "is_valid": rec.is_valid,
            "failure_kind": rec.failure_kind,
        }
        for rec in snapshot_records
    ]
    write_csv(
        OUTPUT_DIR / "evolution_population_snapshots.csv",
        snapshot_csv_rows,
        [
            "generation",
            "gene_id",
            "reward",
            "distance",
            "control_cost",
            "submission_flag",
            "runtime_min",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )
    write_csv(
        OUTPUT_DIR / "evolution_unique_individuals.csv",
        unique_records,
        [
            "gene_id",
            "first_generation",
            "last_generation",
            "appearances",
            "reward",
            "distance",
            "control_cost",
            "runtime_min",
            "submission_flag",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )
    write_csv(
        OUTPUT_DIR / "generation_summary.csv",
        generation_summary,
        [
            "generation",
            "population_size",
            "valid_count",
            "invalid_count",
            "mean_reward_valid",
            "max_reward_valid",
            "mean_distance_valid",
            "min_control_cost_valid",
        ],
    )
    write_csv(
        OUTPUT_DIR / "pareto_frontier.csv",
        sorted(frontier, key=lambda row: row["reward"], reverse=True),
        [
            "gene_id",
            "first_generation",
            "last_generation",
            "appearances",
            "reward",
            "distance",
            "control_cost",
            "runtime_min",
            "submission_flag",
            "status",
            "job_id",
            "source_log",
            "is_valid",
            "failure_kind",
        ],
    )

    report_text = build_markdown_report(logs, snapshot_records, unique_records, frontier, generation_summary)
    (OUTPUT_DIR / "evolution_report.md").write_text(report_text, encoding="utf-8")
    plot_pareto_figure(unique_records, frontier, OUTPUT_DIR / "evolution_pareto_plot.png")
    plot_pareto_3d_figure(unique_records, frontier, OUTPUT_DIR / "evolution_pareto_plot_3d.png")
    plot_trend_figure(snapshot_records, unique_records, OUTPUT_DIR / "evolution_trend_plot.png")
    plot_generation_3d_figure(snapshot_records, unique_records, OUTPUT_DIR / "evolution_generation_plot_3d.png")
    plot_combined_figure(snapshot_records, unique_records, frontier, OUTPUT_DIR / "evolution_combined_plot.png")

    print(f"Analyzed {len(snapshot_records)} population snapshot rows across {len(logs)} log files.")
    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
