#!/usr/bin/env python3
"""Analyze solution statistics and find incorrect tasks with actual solutions"""
import json
from pathlib import Path

solutions_dir = Path("/data/dreamlang/gptoss_sequential_20251029_090617/solutions")

correct_count = 0
incorrect_with_solution = []
incorrect_no_solution = []
total = 0

for solution_file in solutions_dir.glob("*.json"):
    with open(solution_file) as f:
        data = json.load(f)

    total += 1
    is_correct = data.get("correct", False)
    has_first = len(data.get("first_solutions", [])) > 0
    has_second = len(data.get("second_solutions", [])) > 0

    if is_correct:
        correct_count += 1
    elif has_first or has_second:
        # Incorrect but has solution
        first_len = len(data["first_solutions"][0]) if has_first else 0
        second_len = len(data["second_solutions"][0]) if has_second else 0
        grid_size = max(first_len, second_len)
        incorrect_with_solution.append((data["challenge_id"], grid_size))
    else:
        # Incorrect with no solution
        incorrect_no_solution.append(data["challenge_id"])

# Sort by grid size (descending) for complex examples
incorrect_with_solution.sort(key=lambda x: x[1], reverse=True)

print("="*60)
print("SOLUTION STATISTICS")
print("="*60)
print(f"Total processed: {total}")
print(f"Correct: {correct_count} ({correct_count/total*100:.1f}%)")
print(f"Incorrect with solution: {len(incorrect_with_solution)}")
print(f"Incorrect with no solution: {len(incorrect_no_solution)}")
print()
print("Top 10 incorrect tasks with solutions (by grid size):")
for task_id, grid_size in incorrect_with_solution[:10]:
    print(f"  {task_id}: {grid_size}×{grid_size}")
