import json
import re

REQUIRED_FIELDS = {"task_name", "priority", "assignee", "deadline", "dependencies"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent", "not specified"}


def extract_json_from_response(response_text):
    """
    Tries to parse a JSON object from model output.
    Models often wrap JSON in markdown code blocks or add extra text,
    so we try a few strategies before giving up.
    """
    # strategy 1: direct parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # strategy 2: find JSON block inside markdown code fences
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    
    # strategy 3: find anything that looks like a JSON object
    json_like = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_like:
        try:
            return json.loads(json_like.group())
        except json.JSONDecodeError:
            pass
    
    return None  # complete failure to parse


def score_single(predicted_text, ground_truth):
    """
    Scores one model output against ground truth.
    Returns a dict of individual metric scores.
    
    format_valid:      did we get parseable JSON at all? (0 or 1)
    field_completeness: fraction of required fields present (0.0 to 1.0)
    value_correctness:  fraction of fields with correct values (0.0 to 1.0)
    """
    predicted = extract_json_from_response(predicted_text)
    
    if predicted is None:
        return {
            "format_valid": 0,
            "field_completeness": 0.0,
            "value_correctness": 0.0,
            "raw_predicted": predicted_text
        }
    
    # format is valid
    format_valid = 1
    
    # field completeness: how many required fields are present
    present_fields = REQUIRED_FIELDS.intersection(set(predicted.keys()))
    field_completeness = len(present_fields) / len(REQUIRED_FIELDS)
    
    # value correctness: for present fields, how many match ground truth
    correct = 0
    for field in present_fields:
        pred_val = str(predicted.get(field, "")).lower().strip()
        true_val = str(ground_truth.get(field, "")).lower().strip()
        if pred_val == true_val:
            correct += 1
    
    value_correctness = correct / len(REQUIRED_FIELDS)
    
    return {
        "format_valid": format_valid,
        "field_completeness": field_completeness,
        "value_correctness": value_correctness,
        "raw_predicted": predicted_text,
        "parsed_predicted": predicted
    }


def aggregate_scores(results):
    """
    Takes a list of individual scores and returns averages.
    This is what gets logged to W&B.
    """
    n = len(results)
    return {
        "format_validity": sum(r["format_valid"] for r in results) / n,
        "field_completeness": sum(r["field_completeness"] for r in results) / n,
        "value_correctness": sum(r["value_correctness"] for r in results) / n,
        "n_examples": n
    }