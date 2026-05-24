import json
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.inference import load_model_and_tokenizer, build_prompt, generate_response
from eval.score import score_single, aggregate_scores


def run_condition(model, tokenizer, test_data, condition_name, repeat=False):
    """
    Runs a full evaluation pass over the test set for one condition.
    condition_name: "A" or "B"
    repeat: whether to repeat the prompt at inference
    """
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
    with open("dataset/splits/test.json", "r") as f:
        test_data = json.load(f)
    print(f"Test examples: {len(test_data)}")
    
    wandb.init(
        project="slm-sft-study",
        name="baseline-conditions-AB",
        config={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "conditions": ["A", "B"],
            "test_size": len(test_data),
            "quantization": "4bit-nf4"
        }
    )
    
    model, tokenizer = load_model_and_tokenizer()
    # run condition A — base model, normal prompt
    results_a, agg_a = run_condition(model, tokenizer, test_data, "A", repeat=False)
    # run condition B — base model, repeated prompt
    results_b, agg_b = run_condition(model, tokenizer, test_data, "B", repeat=True)
    
    wandb.log({
        "A/format_validity":    agg_a["format_validity"],
        "A/field_completeness": agg_a["field_completeness"],
        "A/value_correctness":  agg_a["value_correctness"],
        "A/avg_input_tokens":   agg_a["avg_input_tokens"],
        
        "B/format_validity":    agg_b["format_validity"],
        "B/field_completeness": agg_b["field_completeness"],
        "B/value_correctness":  agg_b["value_correctness"],
        "B/avg_input_tokens":   agg_b["avg_input_tokens"],
    })
    
    os.makedirs("results", exist_ok=True)
    with open("results/condition_A_raw.json", "w") as f:
        json.dump(results_a, f, indent=2)
    with open("results/condition_B_raw.json", "w") as f:
        json.dump(results_b, f, indent=2)
    
    print("\n" + "="*50)
    print("BASELINE SUMMARY")
    print("="*50)
    print(f"{'Metric':<25} {'Condition A':>12} {'Condition B':>12}")
    print("-"*50)
    print(f"{'Format validity':<25} {agg_a['format_validity']:>12.3f} {agg_b['format_validity']:>12.3f}")
    print(f"{'Field completeness':<25} {agg_a['field_completeness']:>12.3f} {agg_b['field_completeness']:>12.3f}")
    print(f"{'Value correctness':<25} {agg_a['value_correctness']:>12.3f} {agg_b['value_correctness']:>12.3f}")
    print(f"{'Avg input tokens':<25} {agg_a['avg_input_tokens']:>12.1f} {agg_b['avg_input_tokens']:>12.1f}")
    
    wandb.finish()
    print("\nResults saved to results/ and logged to W&B.")


if __name__ == "__main__":
    main()