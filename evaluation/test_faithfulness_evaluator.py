"""Test script cho FaithfulnessEvaluator.

Chạy: python test_faithfulness_evaluator.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from openai import AzureOpenAI
from metrics import JudgeLLM, FaithfulnessEvaluator

SYSTEM_PROMPT = (
    "Bạn là copywriter marketing người Việt, viết phong cách Facebook-native, "
    "giọng founder/operator, thuyết phục bằng specifics (số liệu, KPI, USP), "
    "đoạn ngắn, có CTA cụ thể. KHÔNG hype rỗng. Trả về Markdown."
)

GENERATION_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.4"))


def build_user_prompt(title: str, seed: str) -> str:
    return (
        f"Viết bài marketing Markdown từ tiêu đề và mồi:\n\n"
        f"Tiêu đề: {title}\n\n"
        f"Mồi:\n{seed}"
    )


def create_azure_client() -> AzureOpenAI:
    """Create an AzureOpenAI client from environment variables."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ.get("OPENAI_API_VERSION", "2024-12-01")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def generate_llm_output(
    client: AzureOpenAI, model_id: str, title: str, seed: str
) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(title, seed)},
        ],
        temperature=GENERATION_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip()


def load_test_case(case_index=0):
    """Load a single test case by zero-based index."""
    data_path = Path(__file__).parent.parent / "dataset" / "fnb_dataset_test.json"
    with open(data_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    if not 0 <= case_index < len(all_cases):
        raise IndexError(
            f"case_index out of range: {case_index} (must be 0..{len(all_cases)-1})"
        )
    return all_cases[case_index]


def main():
    print("=" * 80)
    print("TEST FAITHFULNESS EVALUATOR")
    print("=" * 80)

    # Load test data
    case_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_case = load_test_case(case_index=case_index)
    print(f"\n✓ Loaded test case index={case_index}\n")

    # Create judge LLM
    client = create_azure_client()
    generation_model_id = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")
    judge_model_id = os.environ.get("JUDGE_MODEL", generation_model_id)
    judge_llm = JudgeLLM(client, judge_model_id)
    print(f"✓ Created JudgeLLM using AzureOpenAI (judge_model={judge_model_id})\n")

    # Create evaluator
    evaluator = FaithfulnessEvaluator(judge_llm)
    print("✓ Created FaithfulnessEvaluator\n")

    # Test the case
    print("-" * 80)
    case_id = test_case.get("case_id", f"case_{case_index}")
    instruction = test_case.get("instruction", "")
    seed_content = test_case.get("input", "")
    actual_output = test_case.get("response", "")

    print(f"\n[Case {case_index}] {case_id}")
    print(f"  Input Title: {instruction}")
    print(f"  Seed Content: {seed_content}")
    print(f"  Actual Output: {actual_output}")

    generated_output = generate_llm_output(
        client, generation_model_id, instruction, seed_content
    )
    print(f"  Output from LLM: {generated_output}\n")

    try:
        result = evaluator.evaluate_one(
            input_title=instruction,
            seed_content=seed_content,
            actual_output=generated_output,
        )

        print(f"\n  Results:")
        print(f"    - Rule Entity Score: {result.rule_entity_score:.3f}")
        print(f"    - LLM Faithfulness Score: {result.llm_faithfulness_score:.3f}")
        print(f"    - Combined Score: {result.combined_score:.3f}")
        print(f"    - LLM Reason: {result.llm_reason[:100]}...")

    except Exception as e:
        print(f"\n  ✗ Error: {str(e)}")

    print("\n" + "=" * 80)
    print("✓ TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
