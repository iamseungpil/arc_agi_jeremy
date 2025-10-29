#!/usr/bin/env python3
"""
Test GPT-OSS with a real ARC problem
Verifies that parsing works correctly with the existing codebase
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Set GPUs before importing anything
os.environ["GPT_OSS_GPU_IDS"] = "2,3"

from src.trees import experiments
from src.models import Model, Library
from src.logic import solve_challenge
from src.data import build_challenges


async def test_arc_problem():
    print("=" * 80)
    print("GPT-OSS ARC Problem Test")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('GPT_OSS_GPU_IDS', '2,3')}")
    print()

    # Load one ARC challenge
    challenges_path = Path("arc-prize-2024/arc-agi_training_challenges.json")
    truth_solutions_path = Path("arc-prize-2024/arc-agi_training_solutions.json")

    if not challenges_path.exists():
        print(f"❌ Challenges file not found: {challenges_path}")
        sys.exit(1)

    challenges = build_challenges(
        challenges_path=challenges_path,
        solutions_path=truth_solutions_path if truth_solutions_path.exists() else None,
    )

    # Get the first challenge
    challenge_id = list(challenges.keys())[0]
    challenge = challenges[challenge_id]

    print(f"Testing with challenge: {challenge_id}")
    print(f"Number of training examples: {len(challenge.train)}")
    print(f"Number of test examples: {len(challenge.test)}")
    print()

    # Show training examples
    print("Training examples:")
    print("-" * 80)
    for i, example in enumerate(challenge.train[:2], 1):  # Show first 2
        print(f"Example {i}:")
        print(f"  Input shape:  {len(example.input)}x{len(example.input[0]) if example.input else 0}")
        print(f"  Output shape: {len(example.output)}x{len(example.output[0]) if example.output else 0}")
    print()

    # Create GPT-OSS tree configuration
    gpt_oss_tree = [
        experiments.RootAttemptConfig(
            include_all_attempts_in_fixes=False,
            attempts=1,  # Just one attempt for testing
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
            fixes=[],  # No fixes for simple test
        )
    ]

    print("Running GPT-OSS on the challenge...")
    print("(This will take a few minutes)")
    print()

    try:
        # Create empty library (required by solve_challenge)
        empty_library = Library(primitives=[])

        first_solutions, second_solutions = await solve_challenge(
            challenge=challenge,
            tree=gpt_oss_tree,
            library=empty_library,
        )

        print("SUCCESS!")
        print("=" * 80)
        print(f"Generated {len(first_solutions)} solution(s)")
        print()

        # Show first solution
        if first_solutions:
            print("First solution (attempt 1):")
            print("-" * 80)
            solution = first_solutions[0]
            print(f"Shape: {len(solution)}x{len(solution[0]) if solution else 0}")
            print("Grid:")
            for row in solution:
                print("  " + " ".join(str(x) for x in row))
            print()

            # Check if it matches the ground truth
            if challenge.test and challenge.test[0].output:
                truth = challenge.test[0].output
                matches = (solution == truth)
                print(f"Matches ground truth: {matches}")
                if not matches:
                    print()
                    print("Ground truth:")
                    print("-" * 80)
                    for row in truth:
                        print("  " + " ".join(str(x) for x in row))

        print()
        print("✅ Parsing and generation test PASSED")
        print()

        # Save results
        os.makedirs("/data/dreamlang/results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"/data/dreamlang/results/test_arc_{challenge_id}_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump({
                "challenge_id": challenge_id,
                "timestamp": datetime.now().isoformat(),
                "first_solutions": [s for s in first_solutions],
                "second_solutions": [s for s in second_solutions],
                "ground_truth": challenge.test[0].output if challenge.test else None,
            }, f, indent=2)

        print(f"Results saved to: {result_file}")

    except Exception as e:
        print(f"❌ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_arc_problem())
