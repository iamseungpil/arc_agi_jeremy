#!/usr/bin/env python3
"""
GPT-OSS Evaluation with Pre-trained Library - Sequential Processing
Using saved_library_1000.pkl (538 primitives)
Processing challenges one by one to avoid GPU contention
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


async def run_gpt_oss_sequential():
    """
    Run GPT-OSS evaluation sequentially (one challenge at a time)
    """
    print("=" * 80)
    print("GPT-OSS Sequential Evaluation with Pre-trained Library")
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

    # Load pre-trained library
    library_path = "saved_library_1000.pkl"
    library = load_library(library_path)
    print(f"✓ Loaded library with {len(library.primitives)} primitives")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/data/dreamlang/gptoss_sequential_{timestamp}")
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

    # GPT-OSS tree configuration (5 attempts + fix layer)
    gpt_oss_tree = [
        RootAttemptConfig(
            include_all_attempts_in_fixes=False,
            attempts=5,
            llm_config=LLMConfig(
                model=Model.gpt_oss_20b,
                temperature=0.7,
            ),
            prompt_config=RootPromptConfig(
                base_prompt=Prompt.REASONING,
                use_examples=True,
                use_diffs=True,
                use_images=False,
                use_ascii=True,
                use_array=True,
                use_image=False,
            ),
            fixes=[
                AttemptEdge(
                    k_top_config=KTopConfig(
                        k_top=3,
                        unique_code=False,
                        unique_output=False
                    ),
                    configs=[
                        FixAttemptConfig(
                            attempts=2,
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
    log(f"  - Processing: Sequential (one at a time)")
    log(f"  - Output: {output_dir}")
    log("")

    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "library_primitives": len(library.primitives),
        "total_challenges": len(challenges),
        "challenges": {}
    }

    # Dictionary to store primitive accuracy scores
    challenge_primitive_accuracy_scores = defaultdict(dict)

    # Process challenges sequentially
    start_time = time.time()
    solved_count = 0

    for idx, (challenge_id, challenge) in enumerate(challenges.items(), 1):
        challenge_start = time.time()

        log(f"[{idx}/{len(challenges)}] Processing {challenge_id}...")

        try:
            # Solve challenge with library
            first_solutions, second_solutions = await solve_challenge(
                challenge=challenge,
                tree=gpt_oss_tree,
                library=library,
                use_primitives_weighed_by_score=True,
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

            elapsed = time.time() - challenge_start

            if is_correct:
                solved_count += 1
                log(f"  ✓ CORRECT ({elapsed:.1f}s) | Accuracy: {solved_count}/{idx} = {solved_count/idx*100:.1f}%")
            else:
                log(f"  ✗ wrong ({elapsed:.1f}s) | Accuracy: {solved_count}/{idx} = {solved_count/idx*100:.1f}%")

            results["challenges"][challenge_id] = {
                "correct": is_correct,
                "time_secs": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - challenge_start
            log(f"  ✗ Error: {type(e).__name__}: {e} ({elapsed:.1f}s)")
            results["challenges"][challenge_id] = {
                "correct": False,
                "error": str(e),
                "time_secs": elapsed,
            }

        # Save progress every 10
        if idx % 10 == 0:
            results["end_time"] = datetime.now().isoformat()
            results["solved"] = solved_count
            results["completed"] = idx
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / idx
            eta_secs = (len(challenges) - idx) * avg_time
            log(f"")
            log(f"Progress: {idx}/{len(challenges)} ({idx/len(challenges)*100:.1f}%)")
            log(f"Accuracy: {solved_count}/{idx} ({solved_count/idx*100:.1f}%)")
            log(f"Avg time: {avg_time:.1f}s | ETA: {eta_secs/60:.1f} min")
            log(f"")

    # Final summary
    total_elapsed = time.time() - start_time
    final_accuracy = (solved_count / len(challenges) * 100)

    log("")
    log("=" * 80)
    log("EVALUATION COMPLETE")
    log("=" * 80)
    log(f"Final Results:")
    log(f"  Total challenges: {len(challenges)}")
    log(f"  Solved: {solved_count}")
    log(f"  Accuracy: {final_accuracy:.2f}%")
    log(f"  Library: {len(library.primitives)} primitives")
    log(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    log(f"  Avg per challenge: {total_elapsed/len(challenges):.1f}s")
    log(f"  Results: {output_dir}")
    log("")

    # Save final results
    results["final_solved"] = solved_count
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
    asyncio.run(run_gpt_oss_sequential())
