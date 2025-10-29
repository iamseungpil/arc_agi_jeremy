#!/usr/bin/env python3
"""
Simple test script for GPT-OSS 20B integration
Tests basic inference on a simple ARC-like problem
"""

import asyncio
import os
import sys
from datetime import datetime

# Set GPUs before importing anything
os.environ["GPT_OSS_GPU_IDS"] = "2,3"

from src.llms import get_next_message
from src.models import Model


async def test_gpt_oss():
    print("=" * 80)
    print("GPT-OSS 20B Integration Test")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"GPUs: {os.environ.get('GPT_OSS_GPU_IDS', '2,3')}")
    print()

    # Simple test messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves puzzles."
        },
        {
            "role": "user",
            "content": """Look at this simple pattern:

Input:
1 1 1
1 1 1

Output:
2 2 2
2 2 2

Now apply the same pattern to:

Input:
3 3 3
3 3 3

What is the output?"""
        }
    ]

    print("Test prompt: Simple pattern recognition")
    print("-" * 80)

    try:
        print("Loading GPT-OSS model and generating response...")
        print("(First run will take time to load the model)")
        print()

        response, usage = await get_next_message(
            messages=messages,
            model=Model.gpt_oss_20b,
            temperature=0.7,
        )

        print("SUCCESS!")
        print("=" * 80)
        print("Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print()
        print("Usage:")
        print(f"  Input tokens:  {usage.input_tokens}")
        print(f"  Output tokens: {usage.output_tokens}")
        print(f"  Total tokens:  {usage.input_tokens + usage.output_tokens}")
        print()
        print("✅ GPT-OSS integration test PASSED")
        print()

    except Exception as e:
        print(f"❌ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_gpt_oss())
