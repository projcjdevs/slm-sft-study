import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

def load_model_and_tokenizer(model_id="Qwen/Qwen2.5-1.5B-Instruct", quantize=True):
    """
    Loads Qwen2.5-1.5B with 4-bit quantization for inference.
    quantize=True means we compress weights to fit in 4GB VRAM.
    """
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if quantize:
        print("Setting up 4-bit quantization config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        # best for inference
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,    # saves a bit more memory
        )
        print("Loading model in 4-bit mode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",                 # auto puts it on GPU if it fits
            trust_remote_code=True
        )
    else:
        print("Loading model in full precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.eval()  # puts model in inference mode, disables dropout
    print("Model loaded successfully.")
    return model, tokenizer


def build_prompt(instruction, input_text, repeat=False):
    """
    Builds the prompt string for a single example.
    repeat=False → Condition A or C (normal prompt)
    repeat=True  → Condition B or D (repeated prompt at inference)
    
    For PR-baked examples, instruction already contains the repetition,
    so repeat=False is used even for conditions E and F at the call site.
    """
    if repeat:
        full = f"{instruction}\n\n{input_text}"
        prompt = f"{full}\n\n{full}"
    else:
        prompt = f"{instruction}\n\n{input_text}"
    
    # Qwen uses a chat template, we wrap it properly
    messages = [
        {"role": "system", "content": "You are an information extraction assistant. Extract the requested fields and return only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    return messages


def generate_response(model, tokenizer, messages, max_new_tokens=200):
    """
    Runs inference and returns the raw text response.
    max_new_tokens=200 is enough for a 5-field JSON object.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]  # for token efficiency metric
    
    with torch.no_grad():  # no_grad saves memory since we're not training
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # deterministic output, important for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # decode only the newly generated tokens, not the input
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response.strip(), input_token_count