# SFT Study: Prompt Repetition in Small Language Models

Investigating whether instruction fine-tuning on SLMs can internalize 
the prompt repetition effect on structured information extraction tasks.

## Research Question
Does baking prompt repetition into training data substitute for, compound 
with, or outperform inference-time prompt repetition on a fine-tuned SLM?

## Conditions
| Condition | Fine-tuning | Inference |
|---|---|---|
| A | Base | Normal |
| B | Base | Repeated |
| C | Vanilla SFT | Normal |
| D | Vanilla SFT | Repeated |
| E | PR-baked SFT | Normal |
| F | PR-baked SFT | Repeated |

## Status
🔄 Sprint 2 — Dataset preparation