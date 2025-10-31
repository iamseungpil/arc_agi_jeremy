#!/usr/bin/env python3
"""
Properly recover Python code using the full arc_jeremy framework.
This calls the actual solve_challenge logic with library primitives.
"""
import asyncio
import json
import pickle
from pathlib import Path
from datetime import datetime

# Must set GPUs before importing CUDA-dependent modules
# Use GPU 0,1 (NOT 2,3 - those are running the main experiment!)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["GPT_OSS_GPU_IDS"] = "0,1"

from src.data import build_challenges
from src.logic import run_tree, get_best_attempts, dedup_attempts
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
    with open(filename, "rb") as f:
        return pickle.load(f)


async def solve_and_save_python_code(challenge_id: str, challenge, library, tree, output_dir: Path):
    """
    Solve a challenge using the full framework and save the Python code.

    This replicates the logic from solve_challenge() but captures python_code_str.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {challenge_id}")
    print(f"{'='*60}")

    start_time = datetime.now()

    # Run the full tree (this is what the actual experiment does)
    print(f"[{challenge_id}] Running tree with {len(library.primitives)} primitives...")
    attempts = await run_tree(
        tree=tree,
        challenge=challenge,
        library=library,
        warm_cache_root=True,
        warm_cache_fix=False,
        use_primitives_weighed_by_score=False,
        lpn_model=None,
        evaluator=None,
        key=None,
        challenge_primitive_lpn_scores=None,
        challenge_primitive_accuracy_scores=None,
    )

    print(f"[{challenge_id}] Generated {len(attempts)} attempts")

    # Deduplicate
    attempts = dedup_attempts(attempts)
    print(f"[{challenge_id}] After dedup: {len(attempts)} attempts")

    if len(attempts) == 0:
        print(f"[{challenge_id}] ERROR: No attempts generated!")
        return {
            "challenge_id": challenge_id,
            "error": "No attempts generated",
            "timestamp": datetime.now().isoformat(),
        }

    # Get best 2 attempts (same as solve_challenge)
    top_two = get_best_attempts(
        attempts=attempts, k_top=2, unique_code=True, unique_output=True
    )

    if len(top_two) == 0:
        print(f"[{challenge_id}] ERROR: No valid attempts after filtering!")
        return {
            "challenge_id": challenge_id,
            "error": "No valid attempts",
            "timestamp": datetime.now().isoformat(),
        }

    if len(top_two) == 1:
        top_two.append(top_two[0])

    first_solution = top_two[0]
    second_solution = top_two[1]

    # Extract Python code from Attempt objects
    first_code = first_solution.python_code_str
    second_code = second_solution.python_code_str

    print(f"[{challenge_id}] First solution code length: {len(first_code) if first_code else 0} chars")
    print(f"[{challenge_id}] Second solution code length: {len(second_code) if second_code else 0} chars")
    print(f"[{challenge_id}] First has 'def transform': {'def transform' in (first_code or '')}")
    print(f"[{challenge_id}] Second has 'def transform': {'def transform' in (second_code or '')}")

    # Save Python codes
    if first_code:
        first_code_file = output_dir / f"{challenge_id}_first.py"
        with open(first_code_file, "w") as f:
            f.write(first_code)
        print(f"[{challenge_id}] ✓ Saved first solution: {first_code_file}")

    if second_code:
        second_code_file = output_dir / f"{challenge_id}_second.py"
        with open(second_code_file, "w") as f:
            f.write(second_code)
        print(f"[{challenge_id}] ✓ Saved second solution: {second_code_file}")

    # Also save grids for comparison
    first_grids = first_solution.get_grids()
    second_grids = second_solution.get_grids()

    # Save metadata
    elapsed = (datetime.now() - start_time).total_seconds()
    metadata = {
        "challenge_id": challenge_id,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "num_attempts": len(attempts),
        "first_solution": {
            "has_code": first_code is not None,
            "code_length": len(first_code) if first_code else 0,
            "has_transform": "def transform" in (first_code or ""),
            "test_grid": first_grids[0] if first_grids else None,
        },
        "second_solution": {
            "has_code": second_code is not None,
            "code_length": len(second_code) if second_code else 0,
            "has_transform": "def transform" in (second_code or ""),
            "test_grid": second_grids[0] if second_grids else None,
        },
    }

    metadata_file = output_dir / f"{challenge_id}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{challenge_id}] ✓ Completed in {elapsed:.1f}s")

    return metadata


async def main():
    """Recover Python code for 1 correct + 1 incorrect task"""
    print("=" * 80)
    print("Python Code Recovery Using Full Framework")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('GPT_OSS_GPU_IDS', '0,1')} (NOT touching 2,3!)")
    print()

    # Load challenges
    challenges_path = Path("arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("arc-prize-2024/arc-agi_evaluation_solutions.json")

    challenges = build_challenges(
        challenges_path=challenges_path,
        solutions_path=solutions_path if solutions_path.exists() else None,
    )

    print(f"✓ Loaded {len(challenges)} challenges")

    # Load library (same as experiment)
    library_path = "saved_library_1000.pkl"
    library = load_library(library_path)
    print(f"✓ Loaded library with {len(library.primitives)} primitives")
    print()

    # Same tree configuration as experiment
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

    # Output directory
    output_dir = Path("/home/ubuntu/arc_agi_jeremy/analysis/python_codes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tasks to recover (1 correct + 1 incorrect)
    tasks_to_recover = [
        ("140c817e", "correct"),
        ("136b0064", "incorrect"),
    ]

    # Run both tasks in PARALLEL using asyncio.gather
    async def process_task(challenge_id, status):
        challenge = challenges.get(challenge_id)
        if challenge is None:
            print(f"✗ Challenge {challenge_id} not found!")
            return {
                "challenge_id": challenge_id,
                "status": status,
                "error": "Challenge not found",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            result = await solve_and_save_python_code(
                challenge_id=challenge_id,
                challenge=challenge,
                library=library,
                tree=gpt_oss_tree,
                output_dir=output_dir,
            )
            result["status"] = status
            return result

        except Exception as e:
            print(f"✗ Error processing {challenge_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "challenge_id": challenge_id,
                "status": status,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # Execute both tasks in parallel
    print("Starting PARALLEL execution of 2 tasks...")
    tasks = [process_task(cid, status) for cid, status in tasks_to_recover]
    results = await asyncio.gather(*tasks)

    # Save summary
    summary_file = output_dir / "recovery_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        cid = result["challenge_id"]
        status = result.get("status", "unknown")
        if "error" in result:
            print(f"✗ {cid} ({status}): {result['error']}")
        else:
            first = result.get("first_solution", {})
            second = result.get("second_solution", {})
            print(f"✓ {cid} ({status}):")
            print(f"  First:  code={first.get('has_code')}, transform={first.get('has_transform')}")
            print(f"  Second: code={second.get('has_code')}, transform={second.get('has_transform')}")

    print()
    print(f"✓ All results saved to: {output_dir}")
    print(f"✓ Summary: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
