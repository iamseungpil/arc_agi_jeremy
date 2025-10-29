#!/usr/bin/env python3
"""
GPT-OSS Evaluation with Pre-trained Library
Using saved_library_1000.pkl (538 primitives from Grok-4/Sonnet training)
Following the exact structure from main.py for proper library integration
"""
import asyncio
import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Set GPUs before importing (critical: must be before any CUDA initialization)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["GPT_OSS_GPU_IDS"] = "2,3"

from src.data import build_challenges
from src.logic import solve_challenge
from src.models import (
    Library,
    Model,
    LLMConfig,
    RootAttemptConfig,
    RootPromptConfig,
    Prompt,
    AttemptEdge,
    FixAttemptConfig,
    FixPromptConfig,
    KTopConfig,
)


def load_library(filename="saved_library_1000.pkl"):
    """Load pre-trained library from pickle file"""
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️  Library file {filename} not found, starting with empty library")
        return Library(primitives=[])


def save_library(library, filename):
    """Save library to pickle file"""
    with open(filename, "wb") as f:
        pickle.dump(library, f)


async def run_gpt_oss_with_library():
    """
    Run GPT-OSS evaluation using pre-trained library
    Following main.py structure for proper integration
    """
    print("=" * 80)
    print("GPT-OSS Evaluation with Pre-trained Library")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('GPT_OSS_GPU_IDS', '2,3')}")
    print()

    # Load validation challenges (400 problems)
    challenges_path = Path("arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("arc-prize-2024/arc-agi_evaluation_solutions.json")

    challenges = build_challenges(
        challenges_path=challenges_path,
        solutions_path=solutions_path if solutions_path.exists() else None,
    )

    print(f"Loaded {len(challenges)} validation challenges")

    # Load pre-trained library (538 primitives from Grok-4/Sonnet training on 1000 problems)
    library_path = "saved_library_1000.pkl"
    library = load_library(library_path)
    print(f"✓ Loaded library with {len(library.primitives)} primitives from {library_path}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/data/dreamlang/gptoss_with_library_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(exist_ok=True)

    log_file = output_dir / "run.log"
    results_file = output_dir / "results.json"

    def log(msg):
        """Log to both console and file"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp_str}] {msg}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    # GPT-OSS tree configuration
    # Based on experiments.sonnet_writeup_deep but with GPT-OSS model
    # 5 root attempts + 1 layer of fixes (following README: 2 rounds × 5 programs = 10 total)
    gpt_oss_tree = [
        RootAttemptConfig(
            include_all_attempts_in_fixes=False,
            attempts=5,  # 5 attempts per problem
            llm_config=LLMConfig(
                model=Model.gpt_oss_20b,
                temperature=0.7,
            ),
            prompt_config=RootPromptConfig(
                base_prompt=Prompt.REASONING,
                use_examples=True,
                use_diffs=True,  # Show input/output diffs
                use_images=False,
                use_ascii=True,
                use_array=True,
                use_image=False,
            ),
            fixes=[
                AttemptEdge(
                    k_top_config=KTopConfig(
                        k_top=3,  # Top 3 failed attempts
                        unique_code=False,
                        unique_output=False
                    ),
                    configs=[
                        FixAttemptConfig(
                            attempts=2,  # 2 fix attempts per failed code
                            llm_config=LLMConfig(
                                model=Model.gpt_oss_20b,
                                temperature=0.7,
                            ),
                            prompt_config=FixPromptConfig(
                                base_prompt=Prompt.REASONING,
                                use_ascii=True,
                                use_array=True,
                                use_image=False,
                                use_fix_reasoning_tags=True,
                                use_fix_fail_line=True,
                                use_typical_issue_text=True,
                                include_diffs=True,
                            ),
                            fixes=[],
                        )
                    ],
                )
            ],
        )
    ]

    log(f"Configuration:")
    log(f"  - Library: {len(library.primitives)} primitives")
    log(f"  - Root attempts: 5")
    log(f"  - Fix attempts: 3 × 2 = 6")
    log(f"  - Total max attempts: 11")
    log(f"  - Rounds: 2")
    log(f"  - Output: {output_dir}")
    log("")

    # Track results
    solved_challenges = set()
    results = {
        "start_time": datetime.now().isoformat(),
        "library_primitives": len(library.primitives),
        "total_challenges": len(challenges),
        "rounds": [],
    }

    # Following main.py: 2 rounds with batch processing
    eval_ids_to_test = list(challenges.keys())

    # Dictionary to store primitive accuracy scores (following main.py structure)
    challenge_primitive_accuracy_scores = defaultdict(dict)

    async def try_solve_challenge(challenge_id: str) -> tuple[bool, float]:
        """Try to solve a single challenge, return (is_correct, time_secs)"""
        start_time = time.time()

        try:
            challenge = challenges[challenge_id]

            # Call solve_challenge with library (critical for using primitives!)
            first_solutions, second_solutions = await solve_challenge(
                challenge=challenge,
                tree=gpt_oss_tree,
                library=library,  # ← Key: pass library so primitives are used in prompts
                use_primitives_weighed_by_score=True,  # Use best primitives from library
                challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
            )

            # Check correctness
            is_correct = False
            if first_solutions and challenge.test and challenge.test[0].output:
                is_correct = (first_solutions[0] == challenge.test[0].output)

            if not is_correct and second_solutions and challenge.test and challenge.test[0].output:
                is_correct = (second_solutions[0] == challenge.test[0].output)

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

            elapsed = time.time() - start_time
            return is_correct, elapsed

        except Exception as e:
            log(f"  ✗ Error on {challenge_id}: {type(e).__name__}: {e}")
            elapsed = time.time() - start_time
            return False, elapsed

    # Run 2 rounds (following main.py)
    start_time = time.time()

    for round_num in range(2):
        log("")
        log("=" * 80)
        log(f"ROUND {round_num + 1}/2")
        log("=" * 80)
        log("")

        round_start = time.time()
        round_results = {
            "round": round_num + 1,
            "solved": 0,
            "total": 0,
            "challenges": {}
        }

        # Process in batches of 60 (following main.py)
        batch_size = 60
        for batch_idx in range(0, len(eval_ids_to_test), batch_size):
            batch_ids = eval_ids_to_test[batch_idx:batch_idx + batch_size]

            log(f"Batch {batch_idx//batch_size + 1}: Processing {len(batch_ids)} challenges...")

            # Process batch in parallel
            tasks = [try_solve_challenge(cid) for cid in batch_ids]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Update results
            for challenge_id, result in zip(batch_ids, batch_results):
                if isinstance(result, Exception):
                    log(f"  {challenge_id}: Exception - {result}")
                    round_results["challenges"][challenge_id] = {
                        "correct": False,
                        "error": str(result),
                    }
                else:
                    is_correct, elapsed = result
                    round_results["challenges"][challenge_id] = {
                        "correct": is_correct,
                        "time_secs": elapsed,
                    }
                    round_results["total"] += 1
                    if is_correct:
                        round_results["solved"] += 1
                        solved_challenges.add(challenge_id)
                        log(f"  ✓ {challenge_id}: CORRECT ({elapsed:.1f}s)")
                    else:
                        log(f"  ✗ {challenge_id}: wrong ({elapsed:.1f}s)")

            # Log batch progress
            batch_end_idx = min(batch_idx + batch_size, len(eval_ids_to_test))
            accuracy = (round_results["solved"] / round_results["total"] * 100) if round_results["total"] > 0 else 0
            log(f"  Progress: {batch_end_idx}/{len(eval_ids_to_test)} | Accuracy: {accuracy:.1f}% ({round_results['solved']}/{round_results['total']})")

        # Round summary
        round_elapsed = time.time() - round_start
        round_accuracy = (round_results["solved"] / round_results["total"] * 100) if round_results["total"] > 0 else 0

        log("")
        log(f"Round {round_num + 1} Complete:")
        log(f"  Solved: {round_results['solved']}/{round_results['total']} ({round_accuracy:.2f}%)")
        log(f"  Total solved so far: {len(solved_challenges)}/{len(challenges)}")
        log(f"  Round time: {round_elapsed/60:.1f} minutes")
        log("")

        # Save library after each round (new primitives added during solving)
        library_save_path = output_dir / f"library_after_round_{round_num + 1}.pkl"
        save_library(library, library_save_path)
        log(f"✓ Saved library ({len(library.primitives)} primitives) to {library_save_path}")

        results["rounds"].append(round_results)

        # Save intermediate results
        results["end_time"] = datetime.now().isoformat()
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    total_elapsed = time.time() - start_time
    final_accuracy = (len(solved_challenges) / len(challenges) * 100)

    log("")
    log("=" * 80)
    log("EVALUATION COMPLETE")
    log("=" * 80)
    log(f"Final Results:")
    log(f"  Total challenges: {len(challenges)}")
    log(f"  Solved: {len(solved_challenges)}")
    log(f"  Accuracy: {final_accuracy:.2f}%")
    log(f"  Library growth: {len(library.primitives)} primitives")
    log(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    log(f"  Results: {output_dir}")
    log("")

    # Save final results
    results["final_solved"] = len(solved_challenges)
    results["final_accuracy"] = final_accuracy
    results["final_library_size"] = len(library.primitives)
    results["end_time"] = datetime.now().isoformat()

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save final library
    final_library_path = output_dir / "final_library.pkl"
    save_library(library, final_library_path)
    log(f"✓ Saved final library to {final_library_path}")


if __name__ == "__main__":
    asyncio.run(run_gpt_oss_with_library())
