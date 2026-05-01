from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_stats(stats_dir: Path) -> List[Dict[str, float]]:
    """Load all stats JSON files."""
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


def extract_generation_from_models(models_dir: Path, checkpoints_dir: Optional[Path] = None) -> Dict[str, int]:
    """
    Map gene_id to generation by checking checkpoints and model file order.
    """
    gene_to_gen: Dict[str, int] = {}
    
    # First, determine how many generations exist from checkpoints
    num_generations = 0
    if checkpoints_dir and checkpoints_dir.exists():
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_gen_*.pkl"))
        num_generations = len(checkpoint_files)
    
    # Collect all model files sorted by name
    model_files = sorted(models_dir.glob("network_*.py"))
    
    if num_generations == 0:
        # Fallback: use all files
        num_generations = len(model_files)
    
    # Distribute genes evenly across generations based on sorted order
    genes_per_gen = len(model_files) / max(num_generations, 1)
    
    for idx, model_file in enumerate(model_files):
        # Extract gene_id from filename
        match = re.search(r"network_(.*?)\.py$", model_file.name)
        if match:
            gene_id = match.group(1)
            # Map to generation based on position
            gen = int(idx / genes_per_gen)
            gen = min(gen, num_generations - 1)  # Clamp to max generation
            gene_to_gen[gene_id] = gen
    
    return gene_to_gen


def assign_generations_to_rows(rows: List[Dict[str, float]], gene_to_gen: Dict[str, int]) -> None:
    """Add generation info to each row if available, otherwise use a default."""
    for row in rows:
        gene_id = row["gene_id"]
        if gene_id in gene_to_gen:
            row["generation"] = gene_to_gen[gene_id]
        else:
            # Fallback: use index in sorted list as proxy for generation
            row["generation"] = -1


def make_generation_plots(rows: List[Dict[str, float]], output_dir: Path) -> None:
    """Create plots showing metrics across generations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter rows that have generation info
    rows_with_gen = [r for r in rows if r.get("generation", -1) >= 0]
    
    if not rows_with_gen:
        print("Warning: No generation information found. Skipping generation plots.")
        return
    
    # Group by generation
    gen_groups: Dict[int, List[Dict]] = {}
    for row in rows_with_gen:
        gen = row["generation"]
        if gen not in gen_groups:
            gen_groups[gen] = []
        gen_groups[gen].append(row)
    
    # Extract statistics per generation
    gens = sorted(gen_groups.keys())
    best_rewards = []
    mean_rewards = []
    best_distances = []
    mean_distances = []
    best_costs = []
    mean_costs = []
    pop_sizes = []
    
    for gen in gens:
        group = gen_groups[gen]
        rewards = [r["mean_reward"] for r in group]
        distances = [r["mean_distance"] for r in group]
        costs = [r["mean_control_cost"] for r in group]
        
        best_rewards.append(max(rewards))
        mean_rewards.append(np.mean(rewards))
        best_distances.append(max(distances))
        mean_distances.append(np.mean(distances))
        best_costs.append(min(costs))  # Best cost = minimum
        mean_costs.append(np.mean(costs))
        pop_sizes.append(len(group))
    
    best_rewards = np.array(best_rewards)
    mean_rewards = np.array(mean_rewards)
    best_distances = np.array(best_distances)
    mean_distances = np.array(mean_distances)
    best_costs = np.array(best_costs)
    mean_costs = np.array(mean_costs)
    pop_sizes = np.array(pop_sizes)
    gens = np.array(gens)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    
    # Plot 1: Best Reward vs Generation
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gens, best_rewards, s=80, alpha=0.7, label="Best Reward", color="#1f77b4", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Mean Reward", fontsize=13)
    ax.set_title("Evolution of Best Reward Across Generations", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "gen_best_reward_vs_generation.png", dpi=240)
    plt.close(fig)
    
    # Plot 2: Mean Reward vs Generation (with best overlaid)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gens, mean_rewards, s=70, alpha=0.6, label="Mean Reward", color="#ff7f0e", edgecolors="white", linewidths=0.5)
    ax.scatter(gens, best_rewards, s=90, alpha=0.8, label="Best Reward", color="#1f77b4", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.set_title("Reward Evolution: Best vs Population Mean", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "gen_reward_mean_vs_best.png", dpi=240)
    plt.close(fig)
    
    # Plot 3: Best Distance vs Generation
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gens, best_distances, s=90, alpha=0.7, label="Best Distance", color="#2ca02c", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Mean Distance", fontsize=13)
    ax.set_title("Evolution of Best Distance Across Generations", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "gen_best_distance_vs_generation.png", dpi=240)
    plt.close(fig)
    
    # Plot 4: Distance and Reward together (dual axis)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = "#1f77b4"
    ax1.set_xlabel("Generation", fontsize=13)
    ax1.set_ylabel("Best Reward", fontsize=13, color=color1)
    ax1.scatter(gens, best_rewards, s=90, alpha=0.8, color=color1, label="Best Reward", edgecolors="white", linewidths=0.5)
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Best Distance", fontsize=13, color=color2)
    ax2.scatter(gens, best_distances, s=90, alpha=0.8, color=color2, label="Best Distance", edgecolors="white", linewidths=0.5)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    ax1.set_title("Reward and Distance Evolution Across Generations", fontsize=16, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.patches import Patch
    patches = [
        plt.scatter([], [], s=90, color=color1, edgecolors="white", linewidths=0.5, label="Best Reward"),
        plt.scatter([], [], s=90, color=color2, edgecolors="white", linewidths=0.5, label="Best Distance")
    ]
    ax1.legend(fontsize=11, loc="upper left")
    
    fig.tight_layout()
    fig.savefig(output_dir / "gen_reward_distance_dual_axis.png", dpi=240)
    plt.close(fig)
    
    # Plot 5: Best Control Cost vs Generation
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gens, best_costs, s=80, alpha=0.7, label="Best (Lowest) Cost", color="#d62728", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Mean Control Cost", fontsize=13)
    ax.set_title("Evolution of Lowest Control Cost Across Generations", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "gen_best_cost_vs_generation.png", dpi=240)
    plt.close(fig)
    
    # Plot 6: Population size by generation
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(gens, pop_sizes, color="#9467bd", alpha=0.7, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Population Size", fontsize=13)
    ax.set_title("Population Size Across Generations", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "gen_population_size.png", dpi=240)
    plt.close(fig)
    
    # Plot 7: All three metrics together (normalized)
    rewards_norm = (best_rewards - best_rewards.min()) / (best_rewards.max() - best_rewards.min() + 1e-9)
    distances_norm = (best_distances - best_distances.min()) / (best_distances.max() - best_distances.min() + 1e-9)
    costs_norm = 1 - (best_costs - best_costs.min()) / (best_costs.max() - best_costs.min() + 1e-9)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(gens, rewards_norm, s=80, alpha=0.7, label="Reward (normalized)", color="#1f77b4", edgecolors="white", linewidths=0.5)
    ax.scatter(gens, distances_norm, s=80, alpha=0.7, label="Distance (normalized)", color="#2ca02c", edgecolors="white", linewidths=0.5)
    ax.scatter(gens, costs_norm, s=80, alpha=0.7, label="Cost Efficiency (1-normalized)", color="#d62728", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Normalized Value", fontsize=13)
    ax.set_title("Normalized Fitness Metrics Across Generations", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    fig.tight_layout()
    fig.savefig(output_dir / "gen_normalized_metrics.png", dpi=240)
    plt.close(fig)
    
    # Write summary
    summary_path = output_dir / "generation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Generation Analysis Summary\n")
        f.write("============================\n\n")
        f.write(f"Total generations: {len(gens)}\n")
        f.write(f"Generation range: {gens.min()} to {gens.max()}\n")
        f.write(f"Total individuals: {len(rows_with_gen)}\n\n")
        
        f.write("Reward Statistics:\n")
        f.write(f"  - Best reward (max): {best_rewards.max():.4f} (gen {gens[np.argmax(best_rewards)]})\n")
        f.write(f"  - Best reward (min): {best_rewards.min():.4f} (gen {gens[np.argmin(best_rewards)]})\n")
        f.write(f"  - Improvement: {(best_rewards[-1] - best_rewards[0]):.4f}\n")
        f.write(f"  - Mean reward trend: {mean_rewards[0]:.4f} -> {mean_rewards[-1]:.4f}\n\n")
        
        f.write("Distance Statistics:\n")
        f.write(f"  - Best distance (max): {best_distances.max():.4f} (gen {gens[np.argmax(best_distances)]})\n")
        f.write(f"  - Best distance (min): {best_distances.min():.4f} (gen {gens[np.argmin(best_distances)]})\n")
        f.write(f"  - Improvement: {(best_distances[-1] - best_distances[0]):.4f}\n\n")
        
        f.write("Control Cost Statistics:\n")
        f.write(f"  - Best (lowest) cost: {best_costs.min():.6f} (gen {gens[np.argmin(best_costs)]})\n")
        f.write(f"  - Worst (highest) cost: {best_costs.max():.6f} (gen {gens[np.argmax(best_costs)]})\n")
        f.write(f"  - Mean cost trend: {mean_costs[0]:.6f} -> {mean_costs[-1]:.6f}\n\n")
        
        f.write("Population Statistics:\n")
        f.write(f"  - Average population size: {pop_sizes.mean():.1f}\n")
        f.write(f"  - Min population: {pop_sizes.min()}\n")
        f.write(f"  - Max population: {pop_sizes.max()}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate generation-wise visualizations from stats JSON files.")
    parser.add_argument("--stats-dir", type=Path, default=Path("stats"), help="Directory containing *_stats.json files")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory containing network_*.py model files")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"), help="Directory containing checkpoint_gen_*.pkl files")
    parser.add_argument("--output-dir", type=Path, default=Path("pareto_plots"), help="Directory for output plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_stats(args.stats_dir)
    
    if not rows:
        raise SystemExit(f"No valid stats JSON files found in: {args.stats_dir}")
    
    # Try to extract generation information from model files and checkpoints
    if args.models_dir.exists():
        gene_to_gen = extract_generation_from_models(args.models_dir, args.checkpoints_dir)
        print(f"Mapped {len(gene_to_gen)} genes to generations")
        # Count unique generations
        num_gens = len(set(gene_to_gen.values()))
        print(f"Total generations: {num_gens}")
    else:
        gene_to_gen = {}
        print(f"Warning: models directory not found at {args.models_dir}")
    
    # Assign generation info to rows
    assign_generations_to_rows(rows, gene_to_gen)
    
    # Create plots
    make_generation_plots(rows, args.output_dir)
    print(f"Generated generation plots in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
