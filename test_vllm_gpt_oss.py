#!/usr/bin/env python3
"""Test vLLM server connection for GPT-OSS 20B on port 8001."""

import asyncio
from openai import AsyncOpenAI

async def test_connection():
    """Test basic chat completion."""
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1",
        timeout=60.0,
    )

    print("=" * 60)
    print("Testing vLLM GPT-OSS 20B Connection")
    print("=" * 60)

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Hello! What is 2+2? Answer briefly."}
            ],
            max_tokens=50,
            temperature=0.7,
        )

        print(f"✅ Success!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    exit(0 if success else 1)
