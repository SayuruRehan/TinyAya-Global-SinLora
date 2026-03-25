#!/usr/bin/env python3
"""
Benchmark a local or Hub model (including PEFT adapters) on SinhalaMMLU and report:
- overall accuracy
- per-category/domain accuracy
- per-subject accuracy
- per-difficulty accuracy
- Sri Lankan culture slice accuracy (heuristic subject/keyword slice)
- optional comparison vs a baseline model
- saved predictions for >=50 cultural questions

Notes:
- This script mirrors the public SinhalaMMLU repo's evaluation style by:
  1) building a chat prompt
  2) asking the model to output only the answer number
  3) scoring the next-token logits over the choice labels 1..N
- The HF dataset is gated. You must accept the dataset terms on Hugging Face first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# -----------------------------
# Prompt construction
# -----------------------------
def generate_instruction_prompt(
    level: str,
    subject: str,
    subject_original: str,
    question: str,
    choices: List[str],
    language: str = "sinhala",
    intro_type: bool = True,
) -> str:
    sinhala_intro = f"මෙය {subject_original} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n" if intro_type else ""
    english_intro = (
        f"This is a multiple choice question related to the subject {subject}. Given in Sinhala Language.\n"
        if intro_type
        else ""
    )

    sinhala_instructions = {
        "easy": "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n",
        "medium": "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n",
        "hard": "පහත ප්‍රශ්නයට 1, 2, 3, 4, 5 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n",
    }
    english_instructions = {
        "easy": "Choose the correct or most appropriate answer from 1, 2, 3, 4 for the question below.\n\n",
        "medium": "Choose the correct or most appropriate answer from 1, 2, 3, 4 for the question below.\n\n",
        "hard": "Select the correct answer from 1, 2, 3, 4, 5 for the given question.\n\n",
    }

    level = (level or "medium").strip().lower()
    if level not in sinhala_instructions:
        level = "medium"

    instruction_text = sinhala_instructions if language.lower() == "sinhala" else english_instructions
    intro = sinhala_intro if language.lower() == "sinhala" else english_intro

    question_label = "ප්‍රශ්නය" if language.lower() == "sinhala" else "Question"
    choice_lines = []
    for i, choice in enumerate(choices, start=1):
        choice_lines.append(f"{i}. {choice}")

    prompt = (
        f"{intro}"
        f"{instruction_text[level]}"
        f"{question_label}: {question}\n"
        + "\n".join(choice_lines)
        + "\n\n"
        + ("පිළිතුර: " if language.lower() == "sinhala" else "Answer: ")
    )
    return prompt


# -----------------------------
# Dataset loading
# -----------------------------
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Common patterns: {"data": [...]}, {"test": [...]}
        for key in ["data", "test", "eval", "validation", "items", "questions"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def maybe_download_hf_dataset(repo_id: str, token: Optional[str], local_dir: Optional[str], force: bool = False) -> Path:
    if local_dir:
        p = Path(local_dir)
        if not p.exists():
            raise FileNotFoundError(f"dataset_dir not found: {p}")
        return p

    cache_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=None if not force else None,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(cache_dir)


def collect_candidate_files(dataset_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".jsonl"}:
            continue
        name = p.name.lower()
        # skip obvious helper/example files
        if "few_shot" in name or "sample" in name:
            continue
        files.append(p)
    return sorted(files)


def normalize_record(record: Dict[str, Any], source_file: Path) -> Optional[Dict[str, Any]]:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}

    question = record.get("question")
    choices = record.get("choices")
    answer = record.get("answer")

    if not question or not isinstance(choices, list) or len(choices) < 2 or answer is None:
        return None

    # Normalize answer to string label "1".."5"
    answer_str = str(answer).strip()
    m = re.search(r"([1-5])", answer_str)
    if m:
        answer_str = m.group(1)

    subject = str(record.get("subject", metadata.get("subject", "Unknown Subject"))).strip()
    category = str(record.get("category", metadata.get("category", "unknown"))).strip()
    difficulty = str(metadata.get("difficulty", record.get("difficulty", "unknown"))).strip().lower()
    subject_original = str(metadata.get("subject_original", record.get("subject_original", subject))).strip()
    q_type = str(metadata.get("type", record.get("type", "unknown"))).strip()
    grade = str(metadata.get("grade", record.get("grade", "unknown"))).strip()
    source = str(metadata.get("source", record.get("source", source_file.name))).strip()
    year = str(metadata.get("year", record.get("year", ""))).strip()
    province = str(metadata.get("province", record.get("province", ""))).strip()
    q_no = str(record.get("q_no", record.get("id", ""))).strip()

    return {
        "q_no": q_no,
        "subject": subject,
        "subject_original": subject_original,
        "category": category,
        "difficulty": difficulty,
        "type": q_type,
        "grade": grade,
        "source": source,
        "year": year,
        "province": province,
        "question": question,
        "choices": [str(c) for c in choices],
        "answer": answer_str,
        "source_file": str(source_file),
    }


def load_sinhalammlu_records(dataset_root: Path) -> List[Dict[str, Any]]:
    files = collect_candidate_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found under {dataset_root}")

    records: List[Dict[str, Any]] = []
    for fp in files:
        try:
            rows = load_json_or_jsonl(fp)
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rec = normalize_record(row, fp)
            if rec:
                records.append(rec)

    if not records:
        raise RuntimeError(
            "No valid benchmark records found. Make sure you downloaded the SinhalaMMLU dataset files, not only the code repo."
        )

    # Deduplicate conservatively using q_no + question + subject
    deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in records:
        key = (r.get("q_no", ""), r["subject"], r["question"])
        deduped[key] = r
    return list(deduped.values())


# -----------------------------
# Culture slice heuristics
# -----------------------------
CULTURE_SUBJECT_KEYWORDS = {
    "history of sri lanka",
    "history",
    "buddhism",
    "buddhist civilization",
    "sinhala language and literature",
    "drama and theatre",
    "dancing",
    "dancing indigenous",
    "eastern music",
    "oriental music",
    "arts",
    "citizenship education",
    "geography",
}

CULTURE_TEXT_PATTERNS = [
    r"\bsri\s*lanka\b",
    r"\blankan\b",
    r"\bsinhala\b",
    r"\bbuddh",
    r"\banuradhapur",
    r"\bpolonnaru",
    r"\bkandy\b",
    r"\bmahavams",
    r"\bsigiriya\b",
    r"\bruhuna\b",
    r"\btheravada\b",
    r"ශ්‍?රී\s*ලංක",
    r"සිංහල",
    r"බෞද්ධ",
    r"අනුරාධපුර",
    r"පොළොන්නරුව",
    r"මහනුවර",
    r"සිගිරිය",
    r"මහාවංශ",
]


def is_culture_item(rec: Dict[str, Any]) -> bool:
    subject = f"{rec.get('subject', '')} {rec.get('subject_original', '')}".lower()
    category = str(rec.get("category", "")).strip().lower()
    source = str(rec.get("source", "")).lower()
    question = str(rec.get("question", "")).lower()
    text_blob = f"{subject}\n{category}\n{source}\n{question}"

    # Strong subject signal
    if any(k in subject for k in CULTURE_SUBJECT_KEYWORDS):
        return True

    # Humanities / Language are culturally rich in the SinhalaMMLU paper,
    # but we avoid marking all of them by default unless text also contains local cues.
    for pattern in CULTURE_TEXT_PATTERNS:
        if re.search(pattern, text_blob, flags=re.IGNORECASE):
            return True

    # Explicitly include core culture-heavy domains when subject strongly suggests it
    if category in {"humanities", "language"} and any(
        x in subject for x in ["history", "music", "dance", "drama", "art", "buddh", "sinhala"]
    ):
        return True

    return False


# -----------------------------
# Model inference
# -----------------------------
@dataclass
class LoadedModel:
    name: str
    tokenizer: Any
    model: Any
    device: str



def _read_adapter_base_model_name(model_ref: str, hf_token: Optional[str]) -> Optional[str]:
    """
    If `model_ref` is a PEFT adapter repo/folder, read adapter_config.json and return
    the recorded base_model_name_or_path when available.
    """
    try:
        local_path = Path(model_ref)
        if local_path.exists() and local_path.is_dir():
            cfg = local_path / "adapter_config.json"
            if cfg.exists():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                return data.get("base_model_name_or_path")
        else:
            cfg_path = hf_hub_download(
                repo_id=model_ref,
                filename="adapter_config.json",
                token=hf_token,
            )
            data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            return data.get("base_model_name_or_path")
    except Exception:
        return None
    return None


def _load_base_causal_lm(
    model_ref: str,
    hf_token: Optional[str],
    load_in_4bit: bool = False,
):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    base_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "token": hf_token,
    }

    def _load_with_kwargs(extra_kwargs: Dict[str, Any]):
        model = AutoModelForCausalLM.from_pretrained(model_ref, **{**base_kwargs, **extra_kwargs})
        if not use_cuda:
            model = model.to(device)
        model.eval()
        return model

    if use_cuda and load_in_4bit:
        try:
            model = _load_with_kwargs({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                "device_map": "auto",
            })
            return model, device
        except Exception as e:
            print("WARNING: 4-bit loading failed; falling back to bfloat16. Error was:")
            print(repr(e))

    if use_cuda:
        model = _load_with_kwargs({
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        })
    else:
        model = _load_with_kwargs({
            "torch_dtype": torch.float32,
        })

    return model, device


def load_model(
    model_ref: str,
    hf_token: Optional[str],
    load_in_4bit: bool = False,
    base_model_override: Optional[str] = None,
) -> LoadedModel:
    """
    Load either:
    1) a full local/Hub causal LM, or
    2) a PEFT adapter repo/folder plus its base model.

    If adapter_config.json is found, we treat model_ref as an adapter and try to read
    base_model_name_or_path automatically. You can override that with --base_model.
    """
    detected_base_model = _read_adapter_base_model_name(model_ref, hf_token)
    is_adapter = detected_base_model is not None or base_model_override is not None
    base_model_name = base_model_override or detected_base_model

    tokenizer_source = base_model_name if is_adapter else model_ref

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        if not base_model_name:
            raise ValueError(
                f"{model_ref} looks like an adapter, but no base model could be determined. "
                "Pass --base_model explicitly."
            )
        print(f"Detected adapter model. Base model: {base_model_name}")
        base_model, device = _load_base_causal_lm(
            base_model_name,
            hf_token=hf_token,
            load_in_4bit=load_in_4bit,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_ref,
            token=hf_token,
            is_trainable=False,
        )
        model.eval()
        return LoadedModel(
            name=f"{model_ref} (adapter on {base_model_name})",
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

    model, device = _load_base_causal_lm(
        model_ref,
        hf_token=hf_token,
        load_in_4bit=load_in_4bit,
    )
    return LoadedModel(name=model_ref, tokenizer=tokenizer, model=model, device=device)


@torch.inference_mode()
def predict_choice_number(
    lm: LoadedModel,
    prompt: str,
    num_choices: int,
    system_prompt: str = "You must output only the answer number.",
) -> Tuple[str, List[float]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    full_prompt = lm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = lm.tokenizer(full_prompt, return_tensors="pt")
    if hasattr(lm.model.config, "model_type") and lm.model.config.model_type == "falcon":
        inputs.pop("token_type_ids", None)
    inputs = {k: v.to(lm.device) for k, v in inputs.items()}

    outputs = lm.model(**inputs)
    first_token_logits = outputs.logits[:, -1, :].to(torch.float32)

    choice_labels = [str(i) for i in range(1, num_choices + 1)]
    choice_ids = [lm.tokenizer.encode(label, add_special_tokens=False)[-1] for label in choice_labels]
    choice_tensor = torch.tensor(choice_ids, dtype=torch.long, device=first_token_logits.device)
    choice_logits = first_token_logits[:, choice_tensor][0]
    probs = torch.softmax(choice_logits, dim=-1)
    pred_idx = int(torch.argmax(choice_logits).item())
    pred = choice_labels[pred_idx]
    return pred, probs.detach().cpu().tolist()


# -----------------------------
# Metrics and reporting
# -----------------------------
def accuracy_from_df(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float((df["pred"] == df["answer"]).mean())


def group_accuracy(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df.columns:
        return pd.DataFrame(columns=[by, "n", "accuracy"])
    out = (
        df.groupby(by, dropna=False)
        .apply(lambda g: pd.Series({"n": len(g), "accuracy": accuracy_from_df(g)}))
        .reset_index()
        .sort_values(["accuracy", "n"], ascending=[False, False])
    )
    return out


def bootstrap_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    if len(series) == 0:
        return float("nan"), float("nan")
    g = torch.Generator().manual_seed(seed)
    arr = torch.tensor(series.astype(float).to_numpy())
    n = arr.numel()
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        vals.append(float(arr[idx].mean().item()))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def evaluate_model(
    lm: LoadedModel,
    records: List[Dict[str, Any]],
    language: str = "sinhala",
    intro: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    iterable = records[:limit] if limit else records

    start = time.time()
    for idx, rec in enumerate(iterable, start=1):
        prompt = generate_instruction_prompt(
            level=rec.get("difficulty", "medium"),
            subject=rec.get("subject", "Unknown Subject"),
            subject_original=rec.get("subject_original", rec.get("subject", "Unknown Subject")),
            question=rec["question"],
            choices=rec["choices"],
            language=language,
            intro_type=intro,
        )
        pred, probs = predict_choice_number(lm, prompt, len(rec["choices"]))
        conf = float(max(probs)) if probs else float("nan")
        culture = is_culture_item(rec)
        rows.append(
            {
                **rec,
                "model": lm.name,
                "prompt": prompt,
                "pred": pred,
                "correct": int(str(pred).strip() == str(rec["answer"]).strip()),
                "confidence": conf,
                "culture_slice": culture,
            }
        )
        if idx % 100 == 0:
            elapsed = time.time() - start
            rate = idx / max(elapsed, 1e-9)
            print(f"[{lm.name}] processed {idx}/{len(iterable)} | {rate:.2f} q/s", flush=True)

    df = pd.DataFrame(rows)
    return df


def make_summary(df: pd.DataFrame) -> Dict[str, Any]:
    overall = accuracy_from_df(df)
    overall_lo, overall_hi = bootstrap_ci(df["correct"])

    culture_df = df[df["culture_slice"] == True]
    nonculture_df = df[df["culture_slice"] == False]

    culture_acc = accuracy_from_df(culture_df)
    nonculture_acc = accuracy_from_df(nonculture_df)

    culture_lo, culture_hi = bootstrap_ci(culture_df["correct"]) if len(culture_df) else (float("nan"), float("nan"))
    nonculture_lo, nonculture_hi = bootstrap_ci(nonculture_df["correct"]) if len(nonculture_df) else (float("nan"), float("nan"))

    return {
        "model": str(df["model"].iloc[0]) if len(df) else "",
        "num_questions": int(len(df)),
        "overall_accuracy": overall,
        "overall_ci95": [overall_lo, overall_hi],
        "culture_questions": int(len(culture_df)),
        "culture_accuracy": culture_acc,
        "culture_ci95": [culture_lo, culture_hi],
        "nonculture_questions": int(len(nonculture_df)),
        "nonculture_accuracy": nonculture_acc,
        "nonculture_ci95": [nonculture_lo, nonculture_hi],
        "culture_minus_nonculture": culture_acc - nonculture_acc if not math.isnan(culture_acc) and not math.isnan(nonculture_acc) else float("nan"),
        "avg_confidence": float(df["confidence"].mean()) if len(df) else float("nan"),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a model on SinhalaMMLU with an extra Sri Lankan culture slice report.")
    parser.add_argument("--model", required=True, help="Local model path or HF model id for the model under test")
    parser.add_argument("--baseline_model", default=None, help="Optional baseline model path or HF model id for side-by-side comparison")
    parser.add_argument("--base_model", default=None, help="Optional base model id/path when --model is a PEFT adapter")
    parser.add_argument("--baseline_base_model", default=None, help="Optional base model id/path when --baseline_model is a PEFT adapter")
    parser.add_argument("--dataset_dir", default=None, help="Local directory containing downloaded SinhalaMMLU dataset files")
    parser.add_argument("--dataset_repo", default="naist-nlp/SinhalaMMLU", help="HF dataset repo id if downloading via Hugging Face")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"), help="HF token for gated dataset / private model")
    parser.add_argument("--output_dir", default="sinhalammlu_benchmark_outputs")
    parser.add_argument("--language", choices=["sinhala", "english"], default="sinhala")
    parser.add_argument("--intro", action="store_true", help="Include subject intro in the prompt, matching the repo's intro=True setting")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load models in 4-bit to save VRAM")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of questions for a quick run")
    parser.add_argument("--min_culture_examples", type=int, default=50, help="Save at least this many culture-slice predictions if available")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset...")
    dataset_root = maybe_download_hf_dataset(args.dataset_repo, args.hf_token, args.dataset_dir)
    records = load_sinhalammlu_records(dataset_root)
    print(f"Loaded {len(records)} usable questions from {dataset_root}")

    # Load test model
    print(f"Loading model: {args.model}")
    test_lm = load_model(args.model, args.hf_token, load_in_4bit=args.load_in_4bit, base_model_override=args.base_model)
    test_df = evaluate_model(test_lm, records, language=args.language, intro=args.intro, limit=args.limit)
    test_summary = make_summary(test_df)

    test_df.to_csv(out_dir / "predictions_test_model.csv", index=False, encoding="utf-8")
    group_accuracy(test_df, "category").to_csv(out_dir / "accuracy_by_category_test_model.csv", index=False, encoding="utf-8")
    group_accuracy(test_df, "subject").to_csv(out_dir / "accuracy_by_subject_test_model.csv", index=False, encoding="utf-8")
    group_accuracy(test_df, "difficulty").to_csv(out_dir / "accuracy_by_difficulty_test_model.csv", index=False, encoding="utf-8")

    culture_examples = test_df[test_df["culture_slice"] == True].copy()
    culture_examples = culture_examples.sort_values(["correct", "confidence"], ascending=[True, False])
    culture_examples.head(max(args.min_culture_examples, 50)).to_csv(
        out_dir / "culture_slice_examples_test_model.csv", index=False, encoding="utf-8"
    )

    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "dataset_questions": int(len(test_df)),
        "culture_questions_detected": int(test_df["culture_slice"].sum()),
        "test_model_summary": test_summary,
    }

    if args.baseline_model:
        print(f"Loading baseline model: {args.baseline_model}")
        baseline_lm = load_model(args.baseline_model, args.hf_token, load_in_4bit=args.load_in_4bit, base_model_override=args.baseline_base_model)
        baseline_df = evaluate_model(baseline_lm, records, language=args.language, intro=args.intro, limit=args.limit)
        baseline_summary = make_summary(baseline_df)

        baseline_df.to_csv(out_dir / "predictions_baseline_model.csv", index=False, encoding="utf-8")
        group_accuracy(baseline_df, "category").to_csv(out_dir / "accuracy_by_category_baseline_model.csv", index=False, encoding="utf-8")
        group_accuracy(baseline_df, "subject").to_csv(out_dir / "accuracy_by_subject_baseline_model.csv", index=False, encoding="utf-8")
        group_accuracy(baseline_df, "difficulty").to_csv(out_dir / "accuracy_by_difficulty_baseline_model.csv", index=False, encoding="utf-8")

        merged = test_df.merge(
            baseline_df[["q_no", "subject", "question", "pred", "correct"]].rename(
                columns={"pred": "baseline_pred", "correct": "baseline_correct"}
            ),
            on=["q_no", "subject", "question"],
            how="left",
        )
        improvement = merged.copy()
        improvement["delta_correct"] = improvement["correct"] - improvement["baseline_correct"]

        # Delta summaries
        culture_mask = improvement["culture_slice"] == True
        overall_delta = float(improvement["delta_correct"].mean())
        culture_delta = float(improvement.loc[culture_mask, "delta_correct"].mean()) if culture_mask.any() else float("nan")
        nonculture_delta = float(improvement.loc[~culture_mask, "delta_correct"].mean()) if (~culture_mask).any() else float("nan")

        cat_delta = (
            improvement.groupby("category", dropna=False)["delta_correct"]
            .mean()
            .reset_index()
            .sort_values("delta_correct", ascending=False)
        )
        subj_delta = (
            improvement.groupby("subject", dropna=False)["delta_correct"]
            .mean()
            .reset_index()
            .sort_values("delta_correct", ascending=False)
        )
        cat_delta.to_csv(out_dir / "delta_by_category_test_minus_baseline.csv", index=False, encoding="utf-8")
        subj_delta.to_csv(out_dir / "delta_by_subject_test_minus_baseline.csv", index=False, encoding="utf-8")

        report["baseline_model_summary"] = baseline_summary
        report["comparison"] = {
            "overall_accuracy_delta_test_minus_baseline": overall_delta,
            "culture_accuracy_delta_test_minus_baseline": culture_delta,
            "nonculture_accuracy_delta_test_minus_baseline": nonculture_delta,
        }

    with (out_dir / "benchmark_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Simple markdown summary
    lines = [
        "# SinhalaMMLU Benchmark Report",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Questions evaluated: **{len(test_df)}**",
        f"- Culture-slice questions detected: **{int(test_df['culture_slice'].sum())}**",
        "",
        "## Test model",
        f"- Model: `{test_summary['model']}`",
        f"- Overall accuracy: **{test_summary['overall_accuracy']:.4f}**",
        f"- Culture-slice accuracy: **{test_summary['culture_accuracy']:.4f}**",
        f"- Non-culture accuracy: **{test_summary['nonculture_accuracy']:.4f}**",
        f"- Culture minus non-culture: **{test_summary['culture_minus_nonculture']:.4f}**",
    ]

    if args.baseline_model and "baseline_model_summary" in report:
        b = report["baseline_model_summary"]
        c = report["comparison"]
        lines += [
            "",
            "## Baseline model",
            f"- Model: `{b['model']}`",
            f"- Overall accuracy: **{b['overall_accuracy']:.4f}**",
            f"- Culture-slice accuracy: **{b['culture_accuracy']:.4f}**",
            f"- Non-culture accuracy: **{b['nonculture_accuracy']:.4f}**",
            "",
            "## Test minus baseline",
            f"- Overall delta: **{c['overall_accuracy_delta_test_minus_baseline']:.4f}**",
            f"- Culture delta: **{c['culture_accuracy_delta_test_minus_baseline']:.4f}**",
            f"- Non-culture delta: **{c['nonculture_accuracy_delta_test_minus_baseline']:.4f}**",
        ]

    with (out_dir / "benchmark_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nDone.")
    print(f"Saved outputs to: {out_dir.resolve()}")
    print(f"- {out_dir / 'benchmark_report.json'}")
    print(f"- {out_dir / 'benchmark_report.md'}")
    print(f"- {out_dir / 'accuracy_by_category_test_model.csv'}")
    print(f"- {out_dir / 'accuracy_by_subject_test_model.csv'}")
    print(f"- {out_dir / 'accuracy_by_difficulty_test_model.csv'}")
    print(f"- {out_dir / 'culture_slice_examples_test_model.csv'}")
    if args.baseline_model:
        print(f"- {out_dir / 'delta_by_category_test_minus_baseline.csv'}")
        print(f"- {out_dir / 'delta_by_subject_test_minus_baseline.csv'}")


if __name__ == "__main__":
    main()
