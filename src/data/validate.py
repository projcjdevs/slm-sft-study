import json
import sys

def validate_dataset(filepath):
    print(f"Loading {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total examples found: {len(data)}")
    
    required_top_keys = {"id", "instruction", "input", "output"}
    required_output_keys = {"task_name", "priority", "assignee", "deadline", "dependencies"}
    valid_priorities = {"low", "medium", "high", "urgent", "not specified"}
    
    issues = []
    
    for i, ex in enumerate(data):
        # check top level keys
        missing_top = required_top_keys - set(ex.keys())
        if missing_top:
            issues.append(f"Example {i} (id={ex.get('id','?')}): missing keys {missing_top}")
            continue
        
        # check output is a dict not a string
        if isinstance(ex["output"], str):
            issues.append(f"Example {i} (id={ex.get('id','?')}): output is a string, should be a dict")
            continue
        
        # check output fields
        missing_out = required_output_keys - set(ex["output"].keys())
        extra_out = set(ex["output"].keys()) - required_output_keys
        if missing_out:
            issues.append(f"Example {i} (id={ex.get('id','?')}): missing output fields {missing_out}")
        if extra_out:
            issues.append(f"Example {i} (id={ex.get('id','?')}): unexpected output fields {extra_out}")
        
        # check priority values
        priority = ex["output"].get("priority", "").lower()
        if priority not in valid_priorities:
            issues.append(f"Example {i} (id={ex.get('id','?')}): invalid priority '{priority}'")
        
        # check no empty strings
        for field, val in ex["output"].items():
            if isinstance(val, str) and val.strip() == "":
                issues.append(f"Example {i} (id={ex.get('id','?')}): empty value for '{field}'")
        
        # check input not empty
        if not ex["input"].strip():
            issues.append(f"Example {i} (id={ex.get('id','?')}): empty input passage")
    
    print(f"\nValidation complete.")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nFirst 20 issues:")
        for issue in issues[:20]:
            print(f"  {issue}")
        return False
    else:
        print("All examples passed validation.")
        return True

if __name__ == "__main__":
    path = "dataset/dataset.json"
    valid = validate_dataset(path)
    sys.exit(0 if valid else 1)