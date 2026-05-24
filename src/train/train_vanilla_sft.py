import json
import os
import sys
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.formatter import format_dataset

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH    = "dataset/splits/train.json"
VAL_PATH      = "dataset/splits/val.json"
OUTPUT_DIR    = "models/vanilla_sft"
WANDB_PROJECT = "slm-sft-study"
WANDB_RUN     = "vanilla-sft-training"

# ── QLoRA config ─────────────────────────────────────────────────────────────

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16,                          # rank — controls trainable param count
    lora_alpha=32,                 # scaling factor, conventionally 2x rank
    target_modules=[               # which layers LoRA attaches to in Qwen
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,             # small dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# ── Training arguments ────────────────────────────────────────────────────────

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8, # effective batch = 2×8 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",    # cosine decay feels more natural than linear
    warmup_ratio=0.05,             # 5% of steps are warmup
    fp16=True,                     # mixed precision, saves VRAM during training
    logging_steps=10,              # log loss every 10 steps to W&B
    eval_strategy="steps",
    eval_steps=100,                # run validation every 100 steps
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,   # keeps the best checkpoint, not just last
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",             # sends all metrics to W&B automatically
    run_name=WANDB_RUN,
    dataloader_num_workers=0,      # Windows needs this at 0
                                   # reduces padding waste, speeds up training
)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config={
            "model": MODEL_ID,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
            "learning_rate": 2e-4,
            "effective_batch_size": 16,
            "train_examples": 1700,
            "val_examples": 150,
            "training_type": "vanilla_sft"
        }
    )

    print("Loading and formatting datasets...")
    with open(TRAIN_PATH) as f:
        train_raw = json.load(f)
    with open(VAL_PATH) as f:
        val_raw = json.load(f)

    train_formatted = format_dataset(train_raw)
    val_formatted   = format_dataset(val_raw)

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset   = Dataset.from_list(val_formatted)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")
    print(f"\nSample formatted text (first 400 chars):")
    print(train_formatted[0]["text"][:400])
    print("...")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True
    )

    print("Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} ({100*trainable/total:.2f}% of total)")

    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    print("\nStarting training...")
    print("Watch your W&B dashboard for live loss curves.")
    print("Training will take roughly 45-90 minutes on your GPU.\n")
    trainer.train()

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}/final_adapter...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

    print("\nTraining complete.")
    wandb.finish()


if __name__ == "__main__":
    main()