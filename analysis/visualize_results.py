#!/usr/bin/env python3
"""
Visualize GPT-OSS ARC Results
Creates visualizations for correct and incorrect solutions
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: maroon
]

def plot_grid(ax, grid, title=""):
    """Plot a single ARC grid"""
    grid = np.array(grid)
    height, width = grid.shape

    # Create colormap from ARC colors
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(ARC_COLORS)

    # Plot grid with colormap
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal', interpolation='nearest')

    # Add grid lines
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

def visualize_task(task_id, solution_data, challenges_data, output_path):
    """Visualize a single task with input examples, ground truth, and model output"""

    # Get challenge data
    challenge = challenges_data.get(task_id)
    if not challenge:
        print(f"Warning: Challenge {task_id} not found in challenges data")
        return

    train_examples = challenge.get('train', [])
    test_example = challenge.get('test', [{}])[0]

    # Get solution data
    ground_truth = solution_data['ground_truth']
    first_solution = solution_data['first_solutions'][0] if solution_data['first_solutions'] else None
    is_correct = solution_data['correct']

    # Calculate number of rows needed
    n_train = min(len(train_examples), 3)  # Show up to 3 training examples
    n_rows = n_train + 1  # +1 for test row

    # Create figure
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 5, hspace=0.3, wspace=0.2)

    # Plot training examples (up to 3)
    for i, example in enumerate(train_examples[:3]):
        ax_in = fig.add_subplot(gs[i, 0])
        ax_out = fig.add_subplot(gs[i, 1])
        plot_grid(ax_in, example['input'], f"Train {i+1} Input")
        plot_grid(ax_out, example['output'], f"Train {i+1} Output")

        # Empty columns for alignment
        for j in range(2, 5):
            ax_empty = fig.add_subplot(gs[i, j])
            ax_empty.axis('off')

    # Test row
    test_row = n_train
    ax_test_in = fig.add_subplot(gs[test_row, 0])
    ax_test_gt = fig.add_subplot(gs[test_row, 1])
    ax_test_pred = fig.add_subplot(gs[test_row, 2])
    ax_test_diff = fig.add_subplot(gs[test_row, 3])

    plot_grid(ax_test_in, test_example.get('input', []), "Test Input")
    plot_grid(ax_test_gt, ground_truth, "Ground Truth")

    if first_solution:
        plot_grid(ax_test_pred, first_solution, "Model Output")

        # Difference visualization
        if len(first_solution) == len(ground_truth) and len(first_solution[0]) == len(ground_truth[0]):
            diff = np.array(first_solution) != np.array(ground_truth)
            ax_test_diff.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='equal')
            ax_test_diff.set_title("Differences (Red=Wrong)", fontsize=10, fontweight='bold')
            ax_test_diff.set_xticks([])
            ax_test_diff.set_yticks([])
        else:
            ax_test_diff.text(0.5, 0.5, 'Shape\nMismatch',
                            ha='center', va='center', fontsize=12)
            ax_test_diff.set_xticks([])
            ax_test_diff.set_yticks([])
            ax_test_diff.set_title("Shape Mismatch", fontsize=10, fontweight='bold')
    else:
        ax_test_pred.text(0.5, 0.5, 'No\nSolution',
                         ha='center', va='center', fontsize=12)
        ax_test_pred.set_xticks([])
        ax_test_pred.set_yticks([])
        ax_test_diff.axis('off')

    # Status indicator in last column
    ax_status = fig.add_subplot(gs[test_row, 4])
    ax_status.axis('off')
    status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = 'green' if is_correct else 'red'
    ax_status.text(0.5, 0.5, status_text,
                   ha='center', va='center', fontsize=20, fontweight='bold',
                   color=status_color)

    # Overall title
    fig.suptitle(f"Task {task_id}", fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")

def main():
    # Paths
    solutions_dir = Path("/data/dreamlang/gptoss_sequential_20251029_090617/solutions")
    challenges_path = Path("/home/ubuntu/arc_agi_jeremy/arc-prize-2024/arc-agi_evaluation_challenges.json")
    output_dir = Path("/home/ubuntu/arc_agi_jeremy/analysis/images")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load challenges
    with open(challenges_path) as f:
        challenges_data = json.load(f)

    # Selected tasks (5 correct, 5 incorrect with actual solutions)
    correct_tasks = ['833dafe3', '140c817e', '64a7c07e', '310f3251', '332efdb3']
    incorrect_tasks = ['1a6449f1', '414297c0', '22a4bbc2', '7bb29440', '136b0064']

    print("Generating visualizations...")
    print("=" * 60)

    # Visualize correct tasks
    print("\nCorrect Solutions:")
    for task_id in correct_tasks:
        solution_path = solutions_dir / f"{task_id}.json"
        if solution_path.exists():
            with open(solution_path) as f:
                solution_data = json.load(f)
            output_path = output_dir / f"correct_{task_id}.png"
            visualize_task(task_id, solution_data, challenges_data, output_path)
        else:
            print(f"Warning: Solution file not found for {task_id}")

    # Visualize incorrect tasks
    print("\nIncorrect Solutions:")
    for task_id in incorrect_tasks:
        solution_path = solutions_dir / f"{task_id}.json"
        if solution_path.exists():
            with open(solution_path) as f:
                solution_data = json.load(f)
            output_path = output_dir / f"incorrect_{task_id}.png"
            visualize_task(task_id, solution_data, challenges_data, output_path)
        else:
            print(f"Warning: Solution file not found for {task_id}")

    print("\n" + "=" * 60)
    print(f"Done! Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
