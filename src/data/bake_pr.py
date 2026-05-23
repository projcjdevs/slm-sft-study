import json
import os

def bake_prompt_repetition(filepath, output_path):
    """
    Takes a training set and creates a PR-baked version where 
    the instruction + input is repeated before the output.
    
    Normal format:
        instruction: "Extract the following fields..."
        input: "John needs to finish..."
        output: {...}
    
    PR-baked format:
        instruction: "Extract the following fields... [input] 
                      Extract the following fields... [input]"
        input: ""
        output: {...}
    """
    
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    baked = []
    for ex in data:
        full_prompt = f"{ex['instruction']}\n\n{ex['input']}"
        repeated_prompt = f"{full_prompt}\n\n{full_prompt}"
        
        baked_ex = {
            "id": ex["id"],
            "instruction": repeated_prompt,
            "input": "",           # already embedded in instruction
            "output": ex["output"] # output is identical
        }
        baked.append(baked_ex)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baked, f, indent=2)
    
    print(f"PR-baked dataset saved → {output_path}")
    print(f"Total examples: {len(baked)}")
    
    # show one example so you can visually verify
    print("\n--- SAMPLE PR-BAKED EXAMPLE ---")
    sample = baked[0]
    print(f"Instruction preview (first 300 chars):\n{sample['instruction'][:300]}...")
    print(f"\nOutput: {json.dumps(sample['output'], indent=2)}")

if __name__ == "__main__":
    bake_prompt_repetition(
        filepath="dataset/splits/train.json",
        output_path="dataset/splits/train_pr_baked.json"
    )