import json
import os
import sys
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.inference import build_prompt, generate_response
from eval.score import score_single, aggregate_scores

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "models/pr_baked_sft/final_adapter"


def load_pr_baked_model():
    """
    Same loading pattern as conditions C and D but loads the
    PR-baked adapter instead of the vanilla SFT adapter.
    The base model is identical — only the adapter weights differ.
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading PR-baked LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("PR-baked model loaded.")
    return model, tokenizer


def run_condition(model, tokenizer, test_data, condition_name, repeat=False):
    print(f"\n{'='*50}")
    print(f"Running Condition {condition_name} ({'repeated' if repeat else 'normal'} prompt)")
    print(f"{'='*50}")

    results = []
    total_input_tokens = 0

    for i, example in enumerate(test_data):
        if i % 25 == 0:
            print(f"  Progress: {i}/{len(test_data)}")

        messages = build_prompt(
            instruction=example["instruction"],
            input_text=example["input"],
            repeat=repeat
        )

        response, input_tokens = generate_response(model, tokenizer, messages)
        total_input_tokens += input_tokens

        score = score_single(response, example["output"])
        score["condition"] = condition_name
        score["example_id"] = example["id"]
        results.append(score)

    agg = aggregate_scores(results)
    agg["avg_input_tokens"] = total_input_tokens / len(test_data)
    agg["condition"] = condition_name

    print(f"\nCondition {condition_name} Results:")
    print(f"  Format validity:    {agg['format_validity']:.3f}")
    print(f"  Field completeness: {agg['field_completeness']:.3f}")
    print(f"  Value correctness:  {agg['value_correctness']:.3f}")
    print(f"  Avg input tokens:   {agg['avg_input_tokens']:.1f}")

    return results, agg


def main():
    print("Loading test set...")
    with open("dataset/splits/test.json") as f:
        test_data = json.load(f)
    print(f"Test examples: {len(test_data)}")

    wandb.init(
        project="slm-sft-study",
        name="pr-baked-sft-conditions-EF",
        config={
            "model": MODEL_ID,
            "adapter": ADAPTER_DIR,
            "conditions": ["E", "F"],
            "test_size": len(test_data)
        }
    )

    model, tokenizer = load_pr_baked_model()

    # condition E — PR-baked model, normal prompt at inference
    results_e, agg_e = run_condition(model, tokenizer, test_data, "E", repeat=False)

    # condition F — PR-baked model, repeated prompt at inference
    # this is the wildcard — model was trained on doubled input,
    # now it receives it again at inference. compound or redundant?
    results_f, agg_f = run_condition(model, tokenizer, test_data, "F", repeat=True)

    wandb.log({
        "E/format_validity":    agg_e["format_validity"],
        "E/field_completeness": agg_e["field_completeness"],
        "E/value_correctness":  agg_e["value_correctness"],
        "E/avg_input_tokens":   agg_e["avg_input_tokens"],

        "F/format_validity":    agg_f["format_validity"],
        "F/field_completeness": agg_f["field_completeness"],
        "F/value_correctness":  agg_f["value_correctness"],
        "F/avg_input_tokens":   agg_f["avg_input_tokens"],
    })

    os.makedirs("results", exist_ok=True)
    with open("results/condition_E_raw.json", "w") as f:
        json.dump(results_e, f, indent=2)
    with open("results/condition_F_raw.json", "w") as f:
        json.dump(results_f, f, indent=2)

    print("\n" + "="*50)
    print("CONDITIONS E & F SUMMARY")
    print("="*50)
    print(f"{'Metric':<25} {'Condition E':>12} {'Condition F':>12}")
    print("-"*50)
    print(f"{'Format validity':<25} {agg_e['format_validity']:>12.3f} {agg_f['format_validity']:>12.3f}")
    print(f"{'Field completeness':<25} {agg_e['field_completeness']:>12.3f} {agg_f['field_completeness']:>12.3f}")
    print(f"{'Value correctness':<25} {agg_e['value_correctness']:>12.3f} {agg_f['value_correctness']:>12.3f}")
    print(f"{'Avg input tokens':<25} {agg_e['avg_input_tokens']:>12.1f} {agg_f['avg_input_tokens']:>12.1f}")

    wandb.finish()
    print("\nResults saved to results/ and logged to W&B.")


if __name__ == "__main__":
    main()