# Qwen3 Sinhala Training Audit Report

## Dataset details

- Train file: `/workspace/train_small.jsonl`
- Train instances: 28000
- Validation file: `/workspace/val_small.jsonl`
- Validation instances: 2000
- Total instances: 30000
- Train per-dataset breakdown: not found in JSONL metadata
- Validation per-dataset breakdown: not found in JSONL metadata

## Training environment

- timestamp_local: 2026-03-25T02:26:21
- hostname: 7f83b23f5fd0
- platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.35
- python_version: 3.11.10
- torch_version: 2.4.1+cu124
- transformers_version: 5.3.0
- cuda_version: 12.4
- cuda_available: True
- gpu_name: NVIDIA A40
- gpu_count: 1
- gpu_vram_gb: 44.43
- bf16_supported: True
- cpu_count_logical: 96
- ram_gb: 503.51

## Hyperparameters

- base_path: /workspace/
- batch_size: 2
- derived_steps_per_epoch: 875
- derived_total_steps: 2625
- derived_warmup_steps: 131
- effective_batch_size: 32
- epochs: 3
- eval_steps: 500
- grad_accum: 16
- learning_rate: 0.0002
- logging_steps: 25
- lora_alpha: 32
- lora_dropout: 0.05
- lora_r: 16
- max_length: 512
- model_id: Qwen/Qwen3-4B-Instruct-2507
- num_proc: 8
- num_workers: 8
- output_dir: sinhala-qwen3-4b-lora
- save_steps: 500
- seed: 42
- train_file: train_small.jsonl
- val_file: val_small.jsonl
- warmup_ratio: 0.05
- weight_decay: 0.01

## Training period

- training_step8_clock: 10:52:13
- training_complete_clock: 17:07:33
- estimated_start_local: 2026-03-24T10:53:23
- estimated_end_local: 2026-03-24T17:08:16

## Training metrics

- total_flos: 8.54465309835817e+17
- train_loss: 0.45352137683686755
- train_runtime: 22492.485
- train_samples_per_second: 3.735
- train_steps_per_second: 0.117
- epoch: 3.0
- global_step: 2625
- best_metric: 0.45217809081077576
- best_model_checkpoint: /workspace/sinhala-qwen3-4b-lora/checkpoint-2500

## Culture generation outputs

- Saved to: `/workspace/qwen3_audit_outputs/sri_lankan_culture_outputs.json`
