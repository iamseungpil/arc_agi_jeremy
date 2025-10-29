#!/usr/bin/env python3
"""
Run GPT-OSS using ORIGINAL experiments.sonnet_writeup_deep configuration
Only changing the model to GPT-OSS
"""
import asyncio
import json
import os
import time
import typing as T
from pathlib import Path

# Set GPUs before importing
os.environ["GPT_OSS_GPU_IDS"] = "2,3"

from pydantic import BaseModel, TypeAdapter

from src import logfire
from src.data import build_challenges
from src.logic import solve_challenge
from src.models import GRID, Challenge, Library, Model
from src.trees import experiments


class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID


async def solve_and_write(
    solutions_d: dict[str, list[ChallengeSolution]],
    challenge: Challenge,
    tree,
    library: Library,
    solutions_dir: Path,
) -> None:
    start = time.time()
    print(f"[{challenge.id}] starting challenge...")

    first_solutions, second_solutions = await solve_challenge(
        challenge=challenge,
        tree=tree,
        library=library,
    )
    solutions_d[challenge.id] = []
    for i in range(len(first_solutions)):
        solutions_d[challenge.id].append(
            ChallengeSolution(
                attempt_1=first_solutions[i],
                attempt_2=second_solutions[i],
            )
        )

    # Write individual solution
    open(solutions_dir / f"{challenge.id}.json", "w").write(
        TypeAdapter(list[ChallengeSolution])
        .dump_json(solutions_d[challenge.id])
        .decode("utf-8")
    )

    took_secs = time.time() - start
    print(f"[{challenge.id}] took {took_secs:.2f} secs | correct: {len(first_solutions) > 0}")


async def run_gpt_oss_validation():
    print("=" * 80)
    print("GPT-OSS with ORIGINAL experiments.sonnet_writeup_deep configuration")
    print("=" * 80)

    # Load validation challenges
    challenges_path = Path("arc-prize-2024/arc-agi_evaluation_challenges.json")
    truth_solutions_path = Path("arc-prize-2024/arc-agi_evaluation_solutions.json")

    challenges = build_challenges(
        challenges_path=challenges_path,
        solutions_path=truth_solutions_path,
    )

    print(f"Loaded {len(challenges)} validation challenges")

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/data/dreamlang/validation_original_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(exist_ok=True)

    # Use ORIGINAL sonnet_writeup_deep configuration but change model to GPT-OSS
    # This has: 50 root attempts + deep fix hierarchy (3 layers)
    original_tree = experiments.sonnet_writeup_deep

    # Modify ONLY the model in the tree
    gpt_oss_tree = []
    for root_config in original_tree:
        # Create new config with GPT-OSS model
        new_config = root_config.model_copy(deep=True)
        new_config.llm_config.model = Model.gpt_oss_20b

        # Also update fix layer models
        if new_config.fixes:
            for fix_edge in new_config.fixes:
                for fix_config in fix_edge.configs:
                    fix_config.llm_config.model = Model.gpt_oss_20b
                    # Update nested fixes too
                    if fix_config.fixes:
                        for nested_edge in fix_config.fixes:
                            for nested_config in nested_edge.configs:
                                nested_config.llm_config.model = Model.gpt_oss_20b

        gpt_oss_tree.append(new_config)

    print(f"Configuration: {gpt_oss_tree[0].attempts} root attempts + deep fix layers")
    print(f"Output: {output_dir}")
    print()

    # Empty library
    empty_library = Library(primitives=[])

    # Track results
    solutions_d: dict[str, list[ChallengeSolution]] = {}
    results = {
        "start_time": datetime.now().isoformat(),
        "total": len(challenges),
        "completed": 0,
        "correct": 0,
    }

    start_time = time.time()

    # Process challenges sequentially (to avoid OOM)
    for idx, (challenge_id, challenge) in enumerate(challenges.items(), 1):
        print(f"\n[{idx}/{len(challenges)}] Processing {challenge_id}...")

        try:
            await solve_and_write(
                solutions_d=solutions_d,
                challenge=challenge,
                tree=gpt_oss_tree,
                library=empty_library,
                solutions_dir=solutions_dir,
            )

            # Check correctness
            is_correct = False
            if solutions_d[challenge_id] and challenge.test and challenge.test[0].output:
                first_solution = solutions_d[challenge_id][0].attempt_1
                is_correct = (first_solution == challenge.test[0].output)

            results["completed"] += 1
            if is_correct:
                results["correct"] += 1

            accuracy = (results["correct"] / results["completed"] * 100)
            print(f"  Status: {'✓ CORRECT' if is_correct else '✗ wrong'} | Accuracy: {accuracy:.1f}% ({results['correct']}/{results['completed']})")

        except Exception as e:
            print(f"  Error: {e}")
            results["completed"] += 1

        # Save progress every 10
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta_mins = (len(challenges) - idx) * avg_time / 60
            print(f"\n  Progress: {idx}/{len(challenges)} | Avg: {avg_time:.1f}s | ETA: {eta_mins:.1f}min\n")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"COMPLETE: {results['correct']}/{results['completed']} correct ({results['correct']/results['completed']*100:.2f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results: {output_dir}")
    print("=" * 80)

    # Save final results
    open(output_dir / "results.json", "w").write(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(run_gpt_oss_validation())
