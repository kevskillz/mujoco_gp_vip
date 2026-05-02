from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_generation_map(map_path: Path) -> Dict[str, int]:
    gene_to_gen: Dict[str, int] = {}
    if not map_path.exists():
        return gene_to_gen

    with map_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            gene_id = row[0].strip()
            try:
                gene_to_gen[gene_id] = int(row[1])
            except Exception:
                continue
    return gene_to_gen


def load_checkpoint_best_by_generation(checkpoints_dir: Path) -> List[Tuple[int, str, float, float, float]]:
    best_rows: List[Tuple[int, str, float, float, float]] = []
    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_gen_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))

    for checkpoint_path in checkpoint_files:
        generation = int(checkpoint_path.stem.split("_")[-1])
        with checkpoint_path.open("rb") as f:
            checkpoint = pickle.load(f)

        population = checkpoint.get("population", [])
        candidates: List[Tuple[str, float, float, float]] = []
        for individual in population:
            if not individual:
                continue

            gene_id = str(individual[0])
            fitness = tuple(getattr(individual.fitness, "values", ())) if hasattr(individual, "fitness") else ()
            valid = bool(getattr(individual.fitness, "valid", False)) if hasattr(individual, "fitness") else False
            finite = len(fitness) == 3 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in fitness)
            if valid and finite:
                candidates.append((gene_id, float(fitness[0]), float(fitness[1]), float(fitness[2])))

        if not candidates:
            continue

        best = max(candidates, key=lambda item: (item[1], item[2], -item[3]))
        best_rows.append((generation, best[0], best[1], best[2], best[3]))

    return best_rows


def write_csv(rows: Sequence[Tuple[int, str, float, float, float]], output_dir: Path) -> None:
    csv_path = output_dir / "gen_best_reward_vs_generation_checkpoint.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "gene_id", "best_reward", "best_distance", "best_control_cost"])
        for generation, gene_id, reward, distance, cost in rows:
            writer.writerow([generation, gene_id, f"{reward:.10f}", f"{distance:.10f}", f"{cost:.10f}"])


def write_summary(rows: Sequence[Tuple[int, str, float, float, float]], output_dir: Path) -> None:
    if not rows:
        return

    generations = np.array([row[0] for row in rows], dtype=int)
    rewards = np.array([row[2] for row in rows], dtype=float)
    distances = np.array([row[3] for row in rows], dtype=float)
    costs = np.array([row[4] for row in rows], dtype=float)

    summary_path = output_dir / "generation_summary_checkpoint.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Checkpoint Generation Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Total generations: {len(rows)}\n")
        f.write(f"Generation range: {generations.min()} to {generations.max()}\n")
        f.write(f"Best reward (max): {rewards.max():.4f} (gen {generations[np.argmax(rewards)]})\n")
        f.write(f"Best reward (min): {rewards.min():.4f} (gen {generations[np.argmin(rewards)]})\n")
        f.write(f"Reward trend: {rewards[0]:.4f} -> {rewards[-1]:.4f}\n")
        f.write(f"Distance trend: {distances[0]:.4f} -> {distances[-1]:.4f}\n")
        f.write(f"Cost trend: {costs[0]:.6f} -> {costs[-1]:.6f}\n")


def make_plots(rows: Sequence[Tuple[int, str, float, float, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No valid checkpoint rows found in the checkpoint directory.")

    generations = np.array([row[0] for row in rows], dtype=int)
    rewards = np.array([row[2] for row in rows], dtype=float)

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

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(
        generations,
        rewards,
        s=85,
        alpha=0.82,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.6,
        label="Checkpoint best reward",
    )
    ax.plot(generations, rewards, color="#1f77b4", alpha=0.35, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Mean Reward (checkpoint population)")
    ax.set_title("Best Reward vs Generation (Checkpoint Method, Origin Anchored)", fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "gen_best_reward_vs_generation_checkpoint.png", dpi=240)
    fig.savefig(output_dir / "gen_best_reward_vs_generation_checkpoint_origin.png", dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate checkpoint-based generation visualizations.")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"), help="Directory containing checkpoint_gen_*.pkl files")
    parser.add_argument("--generation-map", type=Path, default=Path("models_generation_map.csv"), help="CSV mapping gene IDs to generations")
    parser.add_argument("--output-dir", type=Path, default=Path("pareto_plots"), help="Directory for output plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoints_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {args.checkpoints_dir}")

    generation_map = load_generation_map(args.generation_map)
    if generation_map:
        print(f"Loaded {len(generation_map)} generation mappings from {args.generation_map}")
    else:
        print(f"Warning: no generation map found at {args.generation_map}")

    rows = load_checkpoint_best_by_generation(args.checkpoints_dir)
    if not rows:
        raise SystemExit(f"No valid checkpoint population rows found in: {args.checkpoints_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir)
    write_summary(rows, args.output_dir)
    make_plots(rows, args.output_dir)
    print(f"Generated checkpoint-based generation plots in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
