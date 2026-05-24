import json

def format_example_for_training(example):
    """
    Converts a dataset example into the text format SFTTrainer expects.
    
    SFTTrainer wants a single string per example that looks like a 
    complete conversation — system message, user turn, assistant turn.
    The model learns to predict the assistant turn given the rest.
    
    This is called a "completion" format — the model sees everything
    up to and including the assistant prefix, then predicts the output.
    """
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    # convert output dict to JSON string for training
    if isinstance(output, dict):
        output_str = json.dumps(output, indent=2)
    else:
        output_str = str(output)

    text = f"""<|im_start|>system
You are an information extraction assistant. Extract the requested fields and return only valid JSON.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output_str}<|im_end|>"""
    
    return {"text": text}


def format_dataset(dataset):
    """
    Applies formatter to every example in the dataset.
    Returns a list of dicts with a single "text" key.
    """
    return [format_example_for_training(ex) for ex in dataset]