#!/usr/bin/env python3
"""Extract LLM solutions for incorrect tasks"""
import json
from pathlib import Path

# Paths
solutions_dir = Path("/data/dreamlang/gptoss_sequential_20251029_090617/solutions")
challenges_path = Path("/home/ubuntu/arc_agi_jeremy/arc-prize-2024/arc-agi_evaluation_challenges.json")
output_dir = Path("/home/ubuntu/arc_agi_jeremy/analysis")

# Load challenges
with open(challenges_path) as f:
    challenges_data = json.load(f)

# Incorrect tasks (with solutions)
incorrect_tasks = ['1a6449f1', '414297c0', '22a4bbc2', '7bb29440', '136b0064']

# Extract solutions
all_solutions = {}

for task_id in incorrect_tasks:
    print(f"Processing {task_id}...")

    # Load solution
    solution_path = solutions_dir / f"{task_id}.json"
    with open(solution_path) as f:
        solution_data = json.load(f)

    # Load challenge
    challenge = challenges_data.get(task_id, {})

    # Extract data
    all_solutions[task_id] = {
        "task_id": task_id,
        "test_input": challenge.get('test', [{}])[0].get('input', []),
        "ground_truth": solution_data.get('ground_truth', []),
        "llm_first_solution": solution_data.get('first_solutions', [[]])[0] if solution_data.get('first_solutions') else [],
        "llm_second_solution": solution_data.get('second_solutions', [[]])[0] if solution_data.get('second_solutions') else [],
        "correct": solution_data.get('correct', False),
        "timestamp": solution_data.get('timestamp', ''),
        "train_examples": challenge.get('train', [])
    }

# Save to JSON
output_file = output_dir / "incorrect_solutions.json"
with open(output_file, 'w') as f:
    json.dump(all_solutions, f, indent=2)

print(f"\n✓ Saved solutions to: {output_file}")
print(f"Total tasks: {len(all_solutions)}")
