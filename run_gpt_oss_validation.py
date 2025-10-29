#!/usr/bin/env python3
"""
Run GPT-OSS on ARC validation set (400 problems)
Results saved to /data/dreamlang/
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Set GPUs before importing anything
os.environ["GPT_OSS_GPU_IDS"] = "2,3"

from src.trees import experiments
from src.models import Model, Library
from src.logic import solve_challenge
from src.data import build_challenges
from pydantic import TypeAdapter


async def run_validation():
    print("=" * 80)
    print("GPT-OSS ARC Validation Run")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('GPT_OSS_GPU_IDS', '2,3')}")
    print()

    # Load validation challenges
    challenges_path = Path("arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("arc-prize-2024/arc-agi_evaluation_solutions.json")

    if not challenges_path.exists():
        print(f"❌ Challenges file not found: {challenges_path}")
        sys.exit(1)

    challenges = build_challenges(
        challenges_path=challenges_path,
        solutions_path=solutions_path if solutions_path.exists() else None,
    )

    total_challenges = len(challenges)
    print(f"Loaded {total_challenges} validation challenges")
    print()

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/data/dreamlang/validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(exist_ok=True)

    log_file = output_dir / "run.log"
    results_file = output_dir / "results.json"

    # GPT-OSS configuration
    gpt_oss_tree = [
        experiments.RootAttemptConfig(
            include_all_attempts_in_fixes=False,
            attempts=1,  # 1 attempt per problem for initial run
            llm_config=experiments.LLMConfig(
                model=Model.gpt_oss_20b,
                temperature=0.7,
            ),
            prompt_config=experiments.RootPromptConfig(
                base_prompt=experiments.Prompt.REASONING,
                use_examples=True,
                use_diffs=False,
                use_images=False,
                use_ascii=True,
                use_array=True,
                use_image=False,
            ),
            fixes=[],  # No fixes for initial run
        )
    ]

    # Create empty library
    empty_library = Library(primitives=[])

    # Results tracking
    results = {
        "start_time": datetime.now().isoformat(),
        "total_challenges": total_challenges,
        "completed": 0,
        "correct": 0,
        "challenges": {}
    }

    def save_results():
        """Save current results to file"""
        results["end_time"] = datetime.now().isoformat()
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    def log(msg):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    # Process all challenges
    log(f"Starting validation run on {total_challenges} challenges")
    log(f"Output directory: {output_dir}")
    log("")

    start_time = time.time()

    for idx, (challenge_id, challenge) in enumerate(challenges.items(), 1):
        challenge_start = time.time()

        log(f"[{idx}/{total_challenges}] Processing {challenge_id}...")

        try:
            first_solutions, second_solutions = await solve_challenge(
                challenge=challenge,
                tree=gpt_oss_tree,
                library=empty_library,
            )

            # Check correctness
            is_correct = False
            if first_solutions and challenge.test and challenge.test[0].output:
                is_correct = (first_solutions[0] == challenge.test[0].output)

            # Save solution
            solution_data = {
                "challenge_id": challenge_id,
                "timestamp": datetime.now().isoformat(),
                "first_solutions": first_solutions,
                "second_solutions": second_solutions,
                "ground_truth": challenge.test[0].output if challenge.test else None,
                "correct": is_correct,
            }

            solution_file = solutions_dir / f"{challenge_id}.json"
            with open(solution_file, "w") as f:
                json.dump(solution_data, f, indent=2)

            # Update results
            results["challenges"][challenge_id] = {
                "correct": is_correct,
                "time_secs": time.time() - challenge_start,
            }
            results["completed"] += 1
            if is_correct:
                results["correct"] += 1

            # Save progress
            save_results()

            elapsed = time.time() - challenge_start
            accuracy = (results["correct"] / results["completed"] * 100) if results["completed"] > 0 else 0

            log(f"  ✓ Completed in {elapsed:.1f}s | Correct: {is_correct} | Accuracy: {accuracy:.1f}% ({results['correct']}/{results['completed']})")

        except Exception as e:
            log(f"  ✗ Error: {type(e).__name__}: {e}")
            results["challenges"][challenge_id] = {
                "correct": False,
                "error": str(e),
            }
            results["completed"] += 1
            save_results()

        # Log progress every 10 challenges
        if idx % 10 == 0:
            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / idx
            remaining = total_challenges - idx
            eta_secs = remaining * avg_time
            eta_mins = eta_secs / 60

            log("")
            log(f"Progress: {idx}/{total_challenges} ({idx/total_challenges*100:.1f}%)")
            log(f"Accuracy: {results['correct']}/{results['completed']} ({results['correct']/results['completed']*100:.1f}%)")
            log(f"Average time: {avg_time:.1f}s per challenge")
            log(f"ETA: {eta_mins:.1f} minutes")
            log("")

    # Final summary
    total_time = time.time() - start_time
    final_accuracy = (results["correct"] / results["completed"] * 100) if results["completed"] > 0 else 0

    log("")
    log("=" * 80)
    log("VALIDATION RUN COMPLETE")
    log("=" * 80)
    log(f"Total challenges: {total_challenges}")
    log(f"Completed: {results['completed']}")
    log(f"Correct: {results['correct']}")
    log(f"Accuracy: {final_accuracy:.2f}%")
    log(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    log(f"Average time per challenge: {total_time/results['completed']:.1f}s")
    log(f"Results saved to: {output_dir}")
    log("")

    save_results()


if __name__ == "__main__":
    asyncio.run(run_validation())
