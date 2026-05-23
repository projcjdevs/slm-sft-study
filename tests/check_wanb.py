import wandb

wandb.init(project="sft-study-test", name="setup-check")
wandb.log({"test_metric": 1.0})
wandb.finish()

print("W&B working correctly")