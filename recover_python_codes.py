#!/usr/bin/env python3
"""
Recover Python code from LLM responses for selected tasks.
Runs in parallel without interfering with the main experiment.
"""
import json
import asyncio
import re
from pathlib import Path
from datetime import datetime

import httpx

# Import from existing codebase
from src.data import build_challenges
from src.logic import challenge_to_messages
from src.models import Challenge, Prompt

# Configuration
VLLM_BASE_URL = "http://localhost:8001/v1"
OUTPUT_DIR = Path("/home/ubuntu/arc_agi_jeremy/analysis/python_codes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tasks to recover (1 correct + 1 incorrect)
TASKS_TO_RECOVER = {
    "140c817e": "correct",
    "136b0064": "incorrect",
}


def parse_python_backticks(response: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try to find code in ```python ``` blocks
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find code in ``` ``` blocks
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    return response.strip()


async def recover_python_code_for_task(challenge_id: str, status: str) -> dict:
    """
    Recover Python code for a single task by re-running it through vLLM.

    Args:
        challenge_id: The challenge ID to recover
        status: "correct" or "incorrect" (for logging only)

    Returns:
        Dictionary with challenge_id, status, python_code, and full_response
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {challenge_id} ({status})...")

    # Load challenge
    challenges_path = Path("arc-prize-2024/arc-agi_evaluation_challenges.json")
    solutions_path = Path("arc-prize-2024/arc-agi_evaluation_solutions.json")

    challenges_dict = build_challenges(
        challenges_path=challenges_path,
        solutions_path=solutions_path if solutions_path.exists() else None,
    )

    challenge = challenges_dict.get(challenge_id)
    if challenge is None:
        # Try alternative lookup
        for c in challenges_dict.values():
            if c.id == challenge_id:
                challenge = c
                break

    if challenge is None:
        print(f"  ERROR: Challenge {challenge_id} not found!")
        return {
            "challenge_id": challenge_id,
            "status": status,
            "error": "Challenge not found",
        }

    # Generate messages using same logic as main experiment
    messages = challenge_to_messages(
        challenge=challenge,
        add_examples=True,
        include_diffs=True,
        prompt=Prompt.REASONING,
        include_image=False,
        use_ascii=True,
        use_array=True,
    )

    print(f"  Generated {len(messages)} messages")
    print(f"  Last message length: {len(messages[-1]['content'])} chars")

    # Send request to vLLM
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 8192,  # Increased to allow full code generation
                    # Removed reasoning_effort to get actual code output
                },
            )

            if response.status_code != 200:
                print(f"  ERROR: HTTP {response.status_code}")
                print(f"  Response: {response.text[:500]}")
                return {
                    "challenge_id": challenge_id,
                    "status": status,
                    "error": f"HTTP {response.status_code}",
                    "response_text": response.text[:1000],
                }

            response_data = response.json()
            print(f"  [DEBUG] Response structure: {list(response_data.keys())}")
            print(f"  [DEBUG] Number of choices: {len(response_data.get('choices', []))}")

            # Extract LLM response
            if not response_data.get("choices"):
                print(f"  ERROR: No choices in response")
                print(f"  [DEBUG] Full response: {json.dumps(response_data, indent=2)}")
                return {
                    "challenge_id": challenge_id,
                    "status": status,
                    "error": "No choices in response",
                    "response_data": response_data,
                }

            choice = response_data["choices"][0]
            print(f"  [DEBUG] Choice keys: {list(choice.keys())}")
            print(f"  [DEBUG] Message keys: {list(choice.get('message', {}).keys())}")

            llm_response = choice["message"].get("content")
            reasoning_content = choice["message"].get("reasoning_content", "")

            # If content is None but reasoning_content exists, use reasoning_content
            if llm_response is None and reasoning_content:
                print(f"  Content is None, using reasoning_content instead")
                llm_response = reasoning_content
            elif llm_response is None:
                print(f"  ERROR: Both content and reasoning_content are None")
                print(f"  [DEBUG] Message: {choice['message']}")
                print(f"  [DEBUG] Finish reason: {choice.get('finish_reason')}")
                return {
                    "challenge_id": challenge_id,
                    "status": status,
                    "error": "Both content and reasoning_content are None",
                    "finish_reason": choice.get("finish_reason"),
                    "message": choice["message"],
                }

            print(f"  LLM response length: {len(llm_response)} chars")
            print(f"  Finish reason: {choice.get('finish_reason')}")

            # Extract Python code
            python_code = parse_python_backticks(llm_response)
            print(f"  Extracted Python code length: {len(python_code)} chars")

            # Check if we actually got Python code
            has_transform = "def transform" in python_code
            print(f"  Contains 'def transform': {has_transform}")

            return {
                "challenge_id": challenge_id,
                "status": status,
                "python_code": python_code,
                "full_response": llm_response,
                "response_length": len(llm_response),
                "code_length": len(python_code),
                "has_transform_function": has_transform,
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return {
            "challenge_id": challenge_id,
            "status": status,
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def main():
    """Main recovery function - runs tasks in parallel."""
    print(f"=== Python Code Recovery ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"vLLM URL: {VLLM_BASE_URL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tasks to recover: {len(TASKS_TO_RECOVER)}")
    print()

    # Run tasks in parallel
    tasks = [
        recover_python_code_for_task(challenge_id, status)
        for challenge_id, status in TASKS_TO_RECOVER.items()
    ]

    results = await asyncio.gather(*tasks)

    print()
    print("=== Results Summary ===")

    # Save results
    for result in results:
        challenge_id = result["challenge_id"]
        status = result.get("status", "unknown")

        if "error" in result:
            print(f"✗ {challenge_id} ({status}): ERROR - {result['error']}")
            continue

        # Save Python code to separate file
        code_file = OUTPUT_DIR / f"{challenge_id}.py"
        with open(code_file, "w") as f:
            f.write(result["python_code"])

        # Save full response to separate file
        response_file = OUTPUT_DIR / f"{challenge_id}_full_response.txt"
        with open(response_file, "w") as f:
            f.write(result["full_response"])

        # Save metadata
        metadata_file = OUTPUT_DIR / f"{challenge_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "challenge_id": result["challenge_id"],
                "status": result["status"],
                "timestamp": result["timestamp"],
                "response_length": result["response_length"],
                "code_length": result["code_length"],
                "has_transform_function": result["has_transform_function"],
            }, f, indent=2)

        print(f"✓ {challenge_id} ({status}):")
        print(f"  - Python code: {code_file}")
        print(f"  - Full response: {response_file}")
        print(f"  - Has transform function: {result['has_transform_function']}")

    # Save combined results
    combined_file = OUTPUT_DIR / "recovery_results.json"
    with open(combined_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✓ All results saved to: {OUTPUT_DIR}")
    print(f"✓ Combined results: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
