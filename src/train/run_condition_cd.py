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
ADAPTER_DIR = "models/vanilla_sft/final_adapter"


def load_finetuned_model():
    """
    Loads the base model in 4-bit then layers the LoRA adapter on top.
    This is how you use a fine-tuned QLoRA model at inference time —
    the frozen base + the trained adapter together = your fine-tuned model.
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

    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("Fine-tuned model loaded.")
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
        name="vanilla-sft-conditions-CD",
        config={
            "model": MODEL_ID,
            "adapter": ADAPTER_DIR,
            "conditions": ["C", "D"],
            "test_size": len(test_data)
        }
    )

    model, tokenizer = load_finetuned_model()

    results_c, agg_c = run_condition(model, tokenizer, test_data, "C", repeat=False)
    results_d, agg_d = run_condition(model, tokenizer, test_data, "D", repeat=True)

    wandb.log({
        "C/format_validity":    agg_c["format_validity"],
        "C/field_completeness": agg_c["field_completeness"],
        "C/value_correctness":  agg_c["value_correctness"],
        "C/avg_input_tokens":   agg_c["avg_input_tokens"],

        "D/format_validity":    agg_d["format_validity"],
        "D/field_completeness": agg_d["field_completeness"],
        "D/value_correctness":  agg_d["value_correctness"],
        "D/avg_input_tokens":   agg_d["avg_input_tokens"],
    })

    os.makedirs("results", exist_ok=True)
    with open("results/condition_C_raw.json", "w") as f:
        json.dump(results_c, f, indent=2)
    with open("results/condition_D_raw.json", "w") as f:
        json.dump(results_d, f, indent=2)

    print("\n" + "="*50)
    print("CONDITIONS C & D SUMMARY")
    print("="*50)
    print(f"{'Metric':<25} {'Condition C':>12} {'Condition D':>12}")
    print("-"*50)
    print(f"{'Format validity':<25} {agg_c['format_validity']:>12.3f} {agg_d['format_validity']:>12.3f}")
    print(f"{'Field completeness':<25} {agg_c['field_completeness']:>12.3f} {agg_d['field_completeness']:>12.3f}")
    print(f"{'Value correctness':<25} {agg_c['value_correctness']:>12.3f} {agg_d['value_correctness']:>12.3f}")
    print(f"{'Avg input tokens':<25} {agg_c['avg_input_tokens']:>12.1f} {agg_d['avg_input_tokens']:>12.1f}")

    wandb.finish()
    print("\nResults saved to results/ and logged to W&B.")


if __name__ == "__main__":
    main()