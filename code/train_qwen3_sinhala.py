#!/usr/bin/env python3
"""
Sinhala Qwen3-4B SFT Script
=====================================
Target GPU : RunPod A40 (44 GB VRAM, 9 vCPUs)
Model      : Qwen/Qwen3-4B-Instruct-2507 (4B params)
Dataset    : 28,000 train / 2,000 val (messages format)
Method     : Full bf16 + LoRA rank 16

Setup:
    pip install "transformers>=4.51.0" trl peft accelerate \
                datasets sentencepiece huggingface_hub psutil

Run:
    export HF_TOKEN=hf_your_token_here
    python train_qwen3_sinhala.py

Monitor:
    tail -f /workspace/console.log
"""

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
CONFIG = {
    "model_id"      : "Qwen/Qwen3-4B-Instruct-2507",
    "base_path"     : "/workspace/",
    "train_file"    : "train_small.jsonl",     # 28,000 rows
    "val_file"      : "val_small.jsonl",       # 2,000 rows
    "output_dir"    : "sinhala-qwen3-4b-lora",

    "epochs"        : 3,
    "batch_size"    : 2,       # safe on A40 without liger
    "grad_accum"    : 16,      # effective batch = 32
    "max_length"    : 512,     # safe — avoids cross_entropy OOM
    "learning_rate" : 2e-4,
    "warmup_ratio"  : 0.05,
    "weight_decay"  : 0.01,

    "lora_r"        : 16,
    "lora_alpha"    : 32,
    "lora_dropout"  : 0.05,

    "num_workers"   : 8,       # 8 of 9 vCPUs for dataloading
    "num_proc"      : 8,       # parallel preprocessing

    "logging_steps" : 25,
    "eval_steps"    : 500,
    "save_steps"    : 500,
    "seed"          : 42,
}

# ─────────────────────────────────────────────────────────────────────────────
import os, sys, gc, time, json, logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "true"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(CONFIG["base_path"], "training_qwen3.log"),
            mode="a"
        ),
    ],
)
log = logging.getLogger(__name__)

# ── 1. SYSTEM CHECK ───────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 1 — System Check")
log.info("=" * 60)

import transformers

tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if tv < (4, 51):
    log.error(f"transformers {transformers.__version__} too old. Need >= 4.51.0")
    log.error("Run:  pip install 'transformers>=4.51.0'")
    sys.exit(1)
log.info(f"transformers : {transformers.__version__}  OK")

if not torch.cuda.is_available():
    log.error("No GPU found.")
    sys.exit(1)

gpu_name  = torch.cuda.get_device_name(0)
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
log.info(f"GPU  : {gpu_name}")
log.info(f"VRAM : {total_mem:.1f} GB")

try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    log.info(f"RAM  : {ram_gb:.1f} GB")
except ImportError:
    log.info("RAM  : psutil not installed, skipping RAM check")

# ── 2. HF LOGIN ───────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 2 — HuggingFace Login")
log.info("=" * 60)

from huggingface_hub import login, whoami

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    log.error("HF_TOKEN not set.  Run:  export HF_TOKEN=hf_your_token_here")
    sys.exit(1)

login(token=hf_token, add_to_git_credential=False)
log.info(f"Authenticated as: {whoami()['name']}")

# ── 3. PATHS ──────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 3 — Paths & Format Check")
log.info("=" * 60)

BASE       = CONFIG["base_path"]
TRAIN_FILE = os.path.join(BASE, CONFIG["train_file"])
VAL_FILE   = os.path.join(BASE, CONFIG["val_file"])
OUTPUT_DIR = os.path.join(BASE, CONFIG["output_dir"])
MODEL_ID   = CONFIG["model_id"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for path, label in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
    if not os.path.exists(path):
        log.error(f"{label} file not found: {path}")
        log.error("Run the dataset reduction script first:")
        log.error("  python -c \"import json,random,os; ...")
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1024**2
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    try:
        sample = json.loads(first)
        fmt = "messages format OK" if "messages" in sample else "WRONG FORMAT"
    except Exception as e:
        fmt = f"parse error: {e}"
    log.info(f"  {label:6s} : {path}  ({size_mb:.1f} MB)  [{fmt}]")

# ── 4. TOKENIZER ──────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 4 — Tokenizer")
log.info("=" * 60)

from transformers import AutoTokenizer

log.info(f"Loading: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    log.info("  pad_token set to eos_token")

tokenizer.padding_side = "right"
log.info(f"  Vocab size : {tokenizer.vocab_size:,}")
log.info(f"  EOS token  : '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

# Verify chat template with system role
test_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": "test /no_think"},
     {"role": "user",   "content": "hi"}],
    tokenize=False,
    add_generation_prompt=True,
)
log.info(f"  Chat template OK — sample: {repr(test_text[:60])}")

# ── 5. DATASET ────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 5 — Load & Preprocess Dataset")
log.info("=" * 60)

from datasets import load_dataset

log.info(f"Loading with {CONFIG['num_proc']} parallel workers...")
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    num_proc=CONFIG["num_proc"],
)

n_train = len(dataset["train"])
n_val   = len(dataset["validation"])
log.info(f"  Train      : {n_train:,}")
log.info(f"  Validation : {n_val:,}")

# Add /no_think to every system message
# This disables Qwen3 chain-of-thought during training
# so model learns to give direct answers instead of <think> blocks
def add_no_think(example):
    messages = []
    for msg in example["messages"]:
        if msg["role"] == "system" and "/no_think" not in msg["content"]:
            messages.append({
                "role"   : "system",
                "content": msg["content"] + " /no_think"
            })
        else:
            messages.append(msg)
    example["messages"] = messages
    return example

log.info("Adding /no_think to system messages...")
dataset = dataset.map(
    add_no_think,
    num_proc=CONFIG["num_proc"],
    desc="Adding /no_think",
)
log.info("Dataset ready.")

# ── 6. MODEL ──────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 6 — Load Model (bf16, no quantization)")
log.info("=" * 60)

from transformers import AutoModelForCausalLM

log.info(f"Loading: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache      = True
model.config.pretraining_tp = 1

alloc = torch.cuda.memory_allocated(0) / 1024**3
free  = total_mem - torch.cuda.memory_reserved(0) / 1024**3
log.info(f"Model loaded.  Allocated: {alloc:.2f} GB | Free: {free:.2f} GB")
log.info(f"dtype: {model.dtype}")

# ── 7. LORA ───────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 7 — LoRA Adapters (rank 16)")
log.info("=" * 60)

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Force gradient checkpointing OFF — was root cause of 0.07 it/s
model.gradient_checkpointing_disable()
if hasattr(model, "base_model"):
    model.base_model.gradient_checkpointing_disable()
model.config.use_cache = True

grad_ckpt = getattr(model, "is_gradient_checkpointing", False)
log.info(f"Gradient checkpointing : {grad_ckpt}  (must be False)")

alloc = torch.cuda.memory_allocated(0) / 1024**3
free  = total_mem - torch.cuda.memory_reserved(0) / 1024**3
log.info(f"After LoRA — allocated: {alloc:.2f} GB | free: {free:.2f} GB")

# ── 8. TRAINING ───────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 8 — Training")
log.info("=" * 60)

from trl import SFTTrainer, SFTConfig

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

free = (torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_reserved(0)) / 1024**3
log.info(f"GPU before training: {total_mem:.1f} GB total | {free:.1f} GB free")

bs           = CONFIG["batch_size"]
accum        = CONFIG["grad_accum"]
total_steps  = (n_train // (bs * accum)) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
steps_ep     = n_train // (bs * accum)
est_min      = total_steps / 1.5 / 60

log.info(f"Effective batch   : {bs * accum}")
log.info(f"Steps per epoch   : {steps_ep:,}")
log.info(f"Total steps       : {total_steps:,}")
log.info(f"Warmup steps      : {warmup_steps:,}")
log.info(f"Estimated time    : ~{est_min:.0f} min (~{est_min/60:.1f} hrs)")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=CONFIG["epochs"],

    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    gradient_accumulation_steps=accum,

    optim="adamw_torch",
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    weight_decay=CONFIG["weight_decay"],

    max_length=CONFIG["max_length"],

    bf16=True,
    fp16=False,

    gradient_checkpointing=False,

    dataloader_num_workers=CONFIG["num_workers"],
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    torch_empty_cache_steps=50,

    logging_steps=CONFIG["logging_steps"],
    report_to="none",

    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    seed=CONFIG["seed"],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

log.info("-" * 60)
log.info(f"Model          : {MODEL_ID}")
log.info("Thinking mode  : DISABLED via /no_think")
log.info("Grad ckpt      : OFF (full speed)")
log.info(f"Epochs         : {CONFIG['epochs']}")
log.info(f"Effective batch: {bs * accum}")
log.info(f"Max length     : {CONFIG['max_length']}")
log.info(f"Estimated time : ~{est_min:.0f} min total")
log.info("-" * 60)

start = time.time()
train_result = trainer.train()
elapsed = time.time() - start
hours, mins = int(elapsed // 3600), int((elapsed % 3600) // 60)

log.info("=" * 60)
log.info(f"Training complete!  Time: {hours}h {mins}m")
log.info(f"Final training loss: {train_result.training_loss:.4f}")
log.info("=" * 60)

# ── 9. SAVE ───────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 9 — Save")
log.info("=" * 60)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

log.info(f"Saved to: {OUTPUT_DIR}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath):
        log.info(f"  {fname}  ({os.path.getsize(fpath)/1024**2:.1f} MB)")

# ── 10. MERGE ─────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 10 — Merge LoRA into Base Model")
log.info("=" * 60)

from peft import PeftModel

MERGED_DIR = os.path.join(BASE, CONFIG["output_dir"] + "-merged")
os.makedirs(MERGED_DIR, exist_ok=True)

del trainer
gc.collect()
torch.cuda.empty_cache()

log.info("Loading base model for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

log.info("Loading LoRA adapters...")
peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

log.info("Merging...")
merged = peft_model.merge_and_unload()

log.info(f"Saving merged model to: {MERGED_DIR}")
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

log.info("Merged model saved.")
del base_model, peft_model, merged
torch.cuda.empty_cache()

log.info("=" * 60)
log.info("ALL DONE")
log.info(f"  Adapter : {OUTPUT_DIR}")
log.info(f"  Merged  : {MERGED_DIR}")
log.info(f"  Log     : {os.path.join(BASE, 'training_qwen3.log')}")
log.info("=" * 60)
