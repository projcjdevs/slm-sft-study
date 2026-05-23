import json
import random
import os

def split_dataset(filepath, train=1700, val=150, test=150, seed=42):
    print(f"Loading {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    assert len(data) >= train + val + test, \
        f"Not enough examples. Need {train+val+test}, have {len(data)}"
    
    # shuffle with fixed seed so splits are reproducible
    random.seed(seed)
    random.shuffle(data)
    
    train_data = data[:train]
    val_data   = data[train:train+val]
    test_data  = data[train+val:train+val+test]
    
    os.makedirs("dataset/splits", exist_ok=True)
    
    for split_name, split_data in [("train", train_data), 
                                    ("val", val_data), 
                                    ("test", test_data)]:
        out_path = f"dataset/splits/{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} examples → {out_path}")
    
    print("\nSplit complete.")
    print("IMPORTANT: Do not touch data/splits/test.json until final evaluation.")

if __name__ == "__main__":
    split_dataset("dataset/dataset.json")