#!/usr/bin/env python3
import os
import re
import ast
import csv
import json
import math
import time
import platform
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict


def safe_read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def file_exists(path):
    return path and os.path.exists(path)


def parse_config_from_script(script_path):
    text = safe_read_text(script_path)
    if not text:
        return {}

    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        pass
    return {}


def infer_field_value(row, key_candidates):
    for key in key_candidates:
        if key in row and row[key] not in (None, ""):
            return key, row[key]

    for parent in ("meta", "metadata", "info"):
        sub = row.get(parent)
        if isinstance(sub, dict):
            for key in key_candidates:
                if key in sub and sub[key] not in (None, ""):
                    return f"{parent}.{key}", sub[key]

    return None, None


def analyze_jsonl(path, sample_limit=2000):
    result = {
        "path": path,
        "exists": False,
        "instances": 0,
        "file_size_mb": None,
        "first_row_keys": [],
        "messages_format": False,
        "dataset_breakdown_key": None,
        "dataset_breakdown": {},
        "avg_messages_per_instance": None,
        "avg_user_chars": None,
        "avg_assistant_chars": None,
    }

    if not file_exists(path):
        return result

    result["exists"] = True
    result["file_size_mb"] = round(os.path.getsize(path) / 1024**2, 3)

    key_candidates = ["dataset", "source", "origin", "subset", "task", "category"]
    field_counters = defaultdict(Counter)
    total_messages = 0
    total_user_chars = 0
    total_assistant_chars = 0
    sampled = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            result["instances"] += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            if idx == 1:
                result["first_row_keys"] = sorted(list(row.keys()))
                result["messages_format"] = isinstance(row.get("messages"), list)

            if sampled < sample_limit:
                key_path, field_value = infer_field_value(row, key_candidates)
                if key_path and field_value is not None:
                    field_counters[key_path][str(field_value)] += 1

                messages = row.get("messages", [])
                if isinstance(messages, list):
                    total_messages += len(messages)
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if role == "user":
                                total_user_chars += len(content)
                            elif role == "assistant":
                                total_assistant_chars += len(content)
                sampled += 1

    if sampled > 0:
        result["avg_messages_per_instance"] = round(total_messages / sampled, 3)
        result["avg_user_chars"] = round(total_user_chars / sampled, 3)
        result["avg_assistant_chars"] = round(total_assistant_chars / sampled, 3)

    if field_counters:
        best_key, best_counts = max(field_counters.items(), key=lambda kv: sum(kv[1].values()))
        result["dataset_breakdown_key"] = best_key
        result["dataset_breakdown"] = dict(best_counts)

    return result


def get_runtime_environment():
    info = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        info["ram_gb"] = None

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
            try:
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                info["bf16_supported"] = None
        else:
            info["gpu_name"] = None
            info["gpu_count"] = 0
            info["gpu_vram_gb"] = None
            info["bf16_supported"] = None
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = None
        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_count"] = None
        info["gpu_vram_gb"] = None
        info["bf16_supported"] = None

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None

    return info


def try_load_training_args_bin(path):
    if not file_exists(path):
        return None
    try:
        import torch
        obj = torch.load(path, map_location="cpu")
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
        return str(obj)
    except Exception as e:
        return {"load_error": str(e)}


def extract_times_from_log(log_path, train_runtime_seconds=None):
    result = {
        "log_exists": file_exists(log_path),
        "training_step8_clock": None,
        "training_complete_clock": None,
        "estimated_start_local": None,
        "estimated_end_local": None,
    }
    if not file_exists(log_path):
        return result

    text = safe_read_text(log_path) or ""
    step8_match = re.search(r"(\d{2}:\d{2}:\d{2}).*STEP 8 .*Training", text)
    complete_match = re.search(r"(\d{2}:\d{2}:\d{2}).*Training complete!", text)

    if step8_match:
        result["training_step8_clock"] = step8_match.group(1)
    if complete_match:
        result["training_complete_clock"] = complete_match.group(1)

    if train_runtime_seconds is not None:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
            result["estimated_end_local"] = mtime.isoformat(timespec="seconds")
            result["estimated_start_local"] = (mtime - timedelta(seconds=float(train_runtime_seconds))).isoformat(timespec="seconds")
        except Exception:
            pass

    return result


def extract_metrics_from_log_text(text):
    metrics = {}
    if not text:
        return metrics

    patterns = {
        "total_flos_log": r"total_flos\s*=\s*([^\n]+)",
        "train_loss_log": r"train_loss\s*=\s*([^\n]+)",
        "train_runtime_log": r"train_runtime\s*=\s*([^\n]+)",
        "train_samples_per_second_log": r"train_samples_per_second\s*=\s*([^\n]+)",
        "train_steps_per_second_log": r"train_steps_per_second\s*=\s*([^\n]+)",
        "final_training_loss_log": r"Final training loss:\s*([0-9.]+)",
        "human_time_log": r"Training complete!\s*Time:\s*([^\n]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = m.group(1).strip()

    eval_match = re.findall(r"'eval_loss':\s*'([^']+)'.*?'eval_runtime':\s*'([^']+)'.*?'eval_samples_per_second':\s*'([^']+)'.*?'eval_steps_per_second':\s*'([^']+)'.*?'eval_mean_token_accuracy':\s*'([^']+)'.*?'epoch':\s*'([^']+)'", text)
    if eval_match:
        last = eval_match[-1]
        metrics["last_eval_from_log"] = {
            "eval_loss": last[0],
            "eval_runtime": last[1],
            "eval_samples_per_second": last[2],
            "eval_steps_per_second": last[3],
            "eval_mean_token_accuracy": last[4],
            "epoch": last[5],
        }

    return metrics


def collect_training_metrics(adapter_dir, log_path, train_results_path=None, trainer_state_path=None, all_results_path=None):
    report = {
        "train_results_json": safe_load_json(train_results_path) if train_results_path else None,
        "all_results_json": safe_load_json(all_results_path) if all_results_path else None,
        "trainer_state_summary": None,
        "log_metrics": None,
    }

    if trainer_state_path and file_exists(trainer_state_path):
        trainer_state = safe_load_json(trainer_state_path)
        if isinstance(trainer_state, dict):
            summary = {
                "best_metric": trainer_state.get("best_metric"),
                "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
                "global_step": trainer_state.get("global_step"),
                "epoch": trainer_state.get("epoch"),
                "num_train_epochs": trainer_state.get("num_train_epochs"),
                "max_steps": trainer_state.get("max_steps"),
                "log_history_tail": trainer_state.get("log_history", [])[-8:],
            }
            report["trainer_state_summary"] = summary

    if file_exists(log_path):
        report["log_metrics"] = extract_metrics_from_log_text(safe_read_text(log_path))

    merged = {}
    for src_key in ("train_results_json", "all_results_json"):
        src = report.get(src_key)
        if isinstance(src, dict):
            for k, v in src.items():
                merged[k] = v

    if isinstance(report.get("trainer_state_summary"), dict):
        ts = report["trainer_state_summary"]
        for k in ("best_metric", "best_model_checkpoint", "global_step", "epoch", "num_train_epochs", "max_steps"):
            if k in ts and ts[k] is not None:
                merged[k] = ts[k]

    if isinstance(report.get("log_metrics"), dict):
        merged.update(report["log_metrics"])

    report["merged_metrics"] = merged
    return report


def derive_hyperparameter_summary(config, train_count=None):
    summary = dict(config) if config else {}
    if config and train_count is not None:
        bs = config.get("batch_size")
        accum = config.get("grad_accum")
        epochs = config.get("epochs")
        warmup_ratio = config.get("warmup_ratio")
        if isinstance(bs, int) and isinstance(accum, int):
            summary["effective_batch_size"] = bs * accum
        if isinstance(bs, int) and isinstance(accum, int) and isinstance(epochs, int) and bs > 0 and accum > 0:
            total_steps = (train_count // (bs * accum)) * epochs
            summary["derived_total_steps"] = total_steps
            summary["derived_steps_per_epoch"] = train_count // (bs * accum)
            if isinstance(warmup_ratio, (int, float)):
                summary["derived_warmup_steps"] = int(total_steps * warmup_ratio)
    return summary


def build_cultural_questions():
    return [
        "ශ්‍රී ලංකාවේ සිංහල හා දෙමළ අලුත් අවුරුද්දේ සංස්කෘතික වැදගත්කම කුමක්ද?",
        "අවුරුදු චාරිත්‍රවල නැකත් මත කරන කටයුතු කිහිපයක් පැහැදිලි කරන්න.",
        "කිරිබත් අවුරුදු උත්සවවල වැදගත් ආහාරයක් වන්නේ ඇයි?",
        "කොහොමද ශ්‍රී ලංකාවේ අවුරුදු උදාවේදී ගෙවල් පිරිසිදු කිරීම හා අලුත් ආරම්භයක් අතර සම්බන්ධය?",
        "ශ්‍රී ලංකාවේ වෙසක් උත්සවයේ ආගමික හා සමාජීය අර්ථය කුමක්ද?",
        "වෙසක් කූඩු, තොරණ, දන්සැල් වැනි දේවල් වෙසක් සමයේ කෙරෙන්නේ ඇයි?",
        "පොසොන් පෝය ශ්‍රී ලංකාවේ බෞද්ධ ඉතිහාසයේ වැදගත් වන්නේ කෙසේද?",
        "කැන්ඩි පෙරහර ශ්‍රී ලංකාවේ සංස්කෘතිය නියෝජනය කරන ආකාරය පැහැදිලි කරන්න.",
        "දළදා පෙරහරේ ඇතුළත් සාම්ප්‍රදායික අංග කිහිපයක් නම් කරන්න.",
        "ශ්‍රී ලංකාවේ කලා සහ සංස්කෘතිය තුළ උඩරට නර්තනයේ ස්ථානය කුමක්ද?",
        "පහතරට නර්තනය සහ සබරගමුව නර්තනය අතර වෙනස්කම් මොනවාද?",
        "කොහොමද යක් බෙර, දවුල්, තම්මැටම වැනි වාදන ශ්‍රී ලාංකික උත්සවවල භාවිත වන්නේ?",
        "ශ්‍රී ලංකාවේ ජනකලා අතර වෙස් මුහුණු නිර්මාණයේ සංස්කෘතික වැදගත්කම කුමක්ද?",
        "අම්බලන්ගොඩ වෙස් මුහුණු කලාව ගැන කෙටි හැඳින්වීමක් දෙන්න.",
        "ශ්‍රී ලංකාවේ සාම්ප්‍රදායික ඇඳුම් ලෙස ඔසරිය සහ සාරිය භාවිතය ගැන පැහැදිලි කරන්න.",
        "නිලමේ ඇඳුම සහ කන්දියන් ඇඳුම් සම්ප්‍රදාය ගැන කියන්න.",
        "ශ්‍රී ලංකාවේ විවාහ චාරිත්‍රවල පෝරුwa උත්සවයේ අර්ථය කුමක්ද?",
        "පොරුwa චාරිත්‍රයේදී කරන සම්ප්‍රදායික ක්‍රියා කිහිපයක් කියන්න.",
        "ශ්‍රී ලංකාවේ දෙමළ විවාහ සම්ප්‍රදායන්හි විශේෂ ලක්ෂණ මොනවාද?",
        "මුස්ලිම් ප්‍රජාවගේ උත්සව හා ආගමික චාරිත්‍ර ශ්‍රී ලංකා සංස්කෘතියට දායක වන්නේ කෙසේද?",
        "ක්‍රිස්තියානි උත්සව වන නත්තල් හා පාස්කු ශ්‍රී ලංකාවේ සැමරෙන ආකාරය ගැන කියන්න.",
        "ශ්‍රී ලංකාවේ බහු ආගමික හා බහු සංස්කෘතික සමාජයක් ලෙස ඇති විශේෂත්වය කුමක්ද?",
        "ශ්‍රී ලංකාවේ ආයුබෝවන් කියන ආචාරයේ සංස්කෘතික අර්ථය කුමක්ද?",
        "වැඩිහිටියන්ට ගරු කිරීම ශ්‍රී ලංකා පවුල් සංස්කෘතියේ කෙසේ පෙන්නුම් කරන්නේද?",
        "ගමේ පන්සල ශ්‍රී ලංකාවේ සමාජ ජීවිතයේ මධ්‍යස්ථානයක් වන්නේ කෙසේද?",
        "ගමේ දේවාල හා කොවිල් උත්සව ගම්මාන ජීවිතයට බලපාන ආකාරය කියන්න.",
        "ශ්‍රී ලංකාවේ සාම්ප්‍රදායික ආහාර සංස්කෘතියේ බත් සහ කරි වැදගත් වන්නේ ඇයි?",
        "හොප්පර්ස්, ඉඳිආප්ප, පොල් සම්බෝල වැනි ආහාර ශ්‍රී ලාංකික අනන්‍යතාවයට සම්බන්ධ වන්නේ කෙසේද?",
        "ලැම්ප්රයිස්, කොත්තු, වට්ටලප්පන් වැනි ආහාර ශ්‍රී ලංකාවේ බහු සංස්කෘතික බලපෑම් පෙන්නුම් කරන්නේ කෙසේද?",
        "ශ්‍රී ලංකාවේ තේ සංස්කෘතිය සහ අමුත්තන් පිළිගැනීම අතර සම්බන්ධය කුමක්ද?",
        "ගෙදරට අමුත්තන් ආවම කෑමපාන දීමේ සම්ප්‍රදාය ගැන කියන්න.",
        "ශ්‍රී ලංකාවේ ජනකතා හා ජනපද ගීවල සංස්කෘතික වටිනාකම කුමක්ද?",
        "කොහොමද නාඩගම, කොලම්, සොකාරි වැනි රංග කලාවන් ඉතිහාසය සහ සමාජය පිළිබිඹු කරන්නේ?",
        "ශ්‍රී ලංකාවේ දරුවන්ට කුඩා වියේ සිට උගන්වන සාම්ප්‍රදායික සිරිත් විරිත් කිහිපයක් කියන්න.",
        "පන්සල්, කොවිල්, පල්ලිය, පල්ලිය වැනි ආගමික ස්ථාන සම්බන්ධයෙන් ගරු කළ යුතු සම්ප්‍රදායන් මොනවාද?",
        "ශ්‍රී ලංකාවේ පෝය දිනවල සාමාන්‍යයෙන් ජන ජීවිතයේ වෙනස්කම් මොනවාද?",
        "ශ්‍රී ලංකාවේ ගොවි සංස්කෘතිය හා වැපිරීම, අස්වැන්න, උත්සව අතර සම්බන්ධය පැහැදිලි කරන්න.",
        "කිරි ඉතිරවීම, මුල් ගෙට පිවිසීම වැනි චාරිත්‍රවල අර්ථය කුමක්ද?",
        "ශ්‍රී ලංකාවේ උපන් දින, නමකරණ, පළමු අකුරු වැනි පවුල් චාරිත්‍ර ගැන කියන්න.",
        "කතරගම යාත්‍රාව වැනි ආගමික සංචාර ශ්‍රී ලංකාවේ සංස්කෘතික එකමුතුව පෙන්නුම් කරන්නේ කෙසේද?",
        "ශ්‍රී ලංකාවේ මුහුදුබඩ, කඳුකර, වියළි කලාප සංස්කෘතික වෙනස්කම් ගැන කෙටි පැහැදිලි කිරීමක් දෙන්න.",
        "ජනවාරි 4 නොව, පෙබරවාරි 4 වන නිදහස් දිනයේ සංස්කෘතික අර්ථය කුමක්ද?",
        "ශ්‍රී ලංකාවේ ජාතික කොඩිය තුළ සංස්කෘතික සහ ආගමික නිරූපණයන් මොනවාද?",
        "අනුරාධපුර, පොළොන්නරුව, මහනුවර වැනි ඓතිහාසික නගර සංස්කෘතික උරුමයට දායක වන්නේ කෙසේද?",
        "සිගිරිය සහ දඹුල්ල වැනි ස්ථාන ශ්‍රී ලංකාවේ කලාත්මක හා සංස්කෘතික උරුමය පිළිබඳ මොනවද කියන්නේ?",
        "ශ්‍රී ලංකාවේ හින්දු කොවිල් උත්සවවල වර්ණවත් සැරසිලි සහ පෙරහැරවල අර්ථය කුමක්ද?",
        "රමසාන් සහ ඊද් සැමරුම් ශ්‍රී ලාංකික මුස්ලිම් සංස්කෘතිය තුළ කෙසේ අත්විඳිනවාද?",
        "බතික්, හස්ත කර්මාන්ත, ලක්ෂා කර්මාන්ත වැනි දේවල් ශ්‍රී ලංකාවේ සංස්කෘතික ආර්ථිකයට දායක වන්නේ කෙසේද?",
        "ශ්‍රී ලංකාවේ පැරණි ගෘහ නිර්මාණ ශිල්පයේ සංස්කෘතික ලක්ෂණ කිහිපයක් පැහැදිලි කරන්න.",
        "ගමේ උත්සව, කානිවල්, ක්‍රීඩා සහ ජන සහභාගීත්වය අතර ඇති සම්බන්ධය කුමක්ද?",
        "අවුරුදු කාලයේ කබඩි, කොට්ටපොර, කණාමුට්ටි බිඳීම වැනි ක්‍රීඩා සංස්කෘතිකව වැදගත් වන්නේ ඇයි?",
        "ශ්‍රී ලංකාවේ බොහෝ පවුල්වල ආහාර වේලකදී අත්වලින් කෑම කෑමට ඇති සංස්කෘතික හේතු මොනවාද?",
        "ශ්‍රී ලංකාවේ භාෂා විවිධත්වය සංස්කෘතියට බලපාන ආකාරය පැහැදිලි කරන්න.",
        "සිංහල, දෙමළ, ඉංග්‍රීසි භාෂා එකට භාවිත වීම නාගරික සංස්කෘතියේ පෙනෙන ආකාරය කියන්න.",
        "ශ්‍රී ලංකාවේ අතින් කරන සුබ පැතුම්, ආශිර්වාද, සහ ආචාර විධිවල සමාජීය වටිනාකම කුමක්ද?"
    ]


SYSTEM_PROMPT = "ඔබ ශ්‍රී ලංකා සංස්කෘතිය පිළිබඳ හොඳ දැනුමක් ඇති, පැහැදිලි හා ගෞරවනීය AI සහයකයෙකි. සරල සහ නිවැරදි සිංහලෙන් පිළිතුරු දෙන්න. /no_think"


def load_local_model(model_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if not use_cuda:
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, user_text, max_new_tokens=220, temperature=0.0, top_p=0.9):
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(path, report, culture_outputs_path):
    cfg = report.get("hyperparameters", {})
    ds = report.get("dataset", {})
    env = report.get("runtime_environment", {})
    tr = report.get("training_metrics", {}).get("merged_metrics", {})
    tp = report.get("training_period", {})

    lines = []
    lines.append("# Qwen3 Sinhala Training Audit Report\n")
    lines.append("## Dataset details\n")
    lines.append(f"- Train file: `{ds.get('train', {}).get('path')}`")
    lines.append(f"- Train instances: {ds.get('train', {}).get('instances')}")
    lines.append(f"- Validation file: `{ds.get('validation', {}).get('path')}`")
    lines.append(f"- Validation instances: {ds.get('validation', {}).get('instances')}")
    lines.append(f"- Total instances: {ds.get('total_instances')}")

    train_breakdown = ds.get("train", {}).get("dataset_breakdown") or {}
    val_breakdown = ds.get("validation", {}).get("dataset_breakdown") or {}
    if train_breakdown:
        lines.append(f"- Train breakdown key: `{ds.get('train', {}).get('dataset_breakdown_key')}`")
        for k, v in train_breakdown.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("- Train per-dataset breakdown: not found in JSONL metadata")

    if val_breakdown:
        lines.append(f"- Validation breakdown key: `{ds.get('validation', {}).get('dataset_breakdown_key')}`")
        for k, v in val_breakdown.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("- Validation per-dataset breakdown: not found in JSONL metadata")

    lines.append("\n## Training environment\n")
    for key in [
        "timestamp_local", "hostname", "platform", "python_version", "torch_version",
        "transformers_version", "cuda_version", "cuda_available", "gpu_name", "gpu_count",
        "gpu_vram_gb", "bf16_supported", "cpu_count_logical", "ram_gb"
    ]:
        lines.append(f"- {key}: {env.get(key)}")

    lines.append("\n## Hyperparameters\n")
    for key in sorted(cfg.keys()):
        lines.append(f"- {key}: {cfg[key]}")

    lines.append("\n## Training period\n")
    for key in [
        "training_step8_clock", "training_complete_clock",
        "estimated_start_local", "estimated_end_local"
    ]:
        lines.append(f"- {key}: {tp.get(key)}")

    lines.append("\n## Training metrics\n")
    preferred_keys = [
        "total_flos", "total_flos_log", "train_loss", "train_loss_log", "train_runtime",
        "train_runtime_log", "train_samples_per_second", "train_samples_per_second_log",
        "train_steps_per_second", "train_steps_per_second_log", "epoch", "global_step",
        "best_metric", "best_model_checkpoint"
    ]
    seen = set()
    for key in preferred_keys:
        if key in tr:
            lines.append(f"- {key}: {tr[key]}")
            seen.add(key)

    if tr.get("last_eval_from_log"):
        lines.append("- last_eval_from_log:")
        for k, v in tr["last_eval_from_log"].items():
            lines.append(f"  - {k}: {v}")

    lines.append("\n## Culture generation outputs\n")
    lines.append(f"- Saved to: `{culture_outputs_path}`")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit Qwen3 Sinhala training + generate 50 Sri Lankan culture outputs")
    parser.add_argument("--base_path", type=str, default="/workspace")
    parser.add_argument("--model_path", type=str, default="/workspace/sinhala-qwen3-4b-lora-merged")
    parser.add_argument("--adapter_dir", type=str, default="/workspace/sinhala-qwen3-4b-lora")
    parser.add_argument("--train_script", type=str, default="/workspace/train_qwen3_sinhala.py")
    parser.add_argument("--train_file", type=str, default="/workspace/train_small.jsonl")
    parser.add_argument("--val_file", type=str, default="/workspace/val_small.jsonl")
    parser.add_argument("--log_file", type=str, default="/workspace/training_qwen3.log")
    parser.add_argument("--console_log", type=str, default="/workspace/console.log")
    parser.add_argument("--questions_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--out_dir", type=str, default="/workspace/qwen3_audit_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_results_path = os.path.join(args.adapter_dir, "train_results.json")
    trainer_state_path = os.path.join(args.adapter_dir, "trainer_state.json")
    all_results_path = os.path.join(args.adapter_dir, "all_results.json")
    training_args_bin_path = os.path.join(args.adapter_dir, "training_args.bin")

    config = parse_config_from_script(args.train_script)
    train_ds = analyze_jsonl(args.train_file)
    val_ds = analyze_jsonl(args.val_file)

    dataset_summary = {
        "train": train_ds,
        "validation": val_ds,
        "total_instances": (train_ds.get("instances", 0) or 0) + (val_ds.get("instances", 0) or 0),
    }

    runtime_env = get_runtime_environment()
    hyperparameters = derive_hyperparameter_summary(config, train_ds.get("instances"))
    training_args_bin = try_load_training_args_bin(training_args_bin_path)
    training_metrics = collect_training_metrics(
        args.adapter_dir,
        args.log_file if file_exists(args.log_file) else args.console_log,
        train_results_path=train_results_path,
        trainer_state_path=trainer_state_path,
        all_results_path=all_results_path,
    )

    merged_metrics = training_metrics.get("merged_metrics", {})
    train_runtime_seconds = merged_metrics.get("train_runtime")
    training_period = extract_times_from_log(
        args.log_file if file_exists(args.log_file) else args.console_log,
        train_runtime_seconds=train_runtime_seconds,
    )

    if args.questions_file and file_exists(args.questions_file):
        custom_questions = safe_load_json(args.questions_file)
        if isinstance(custom_questions, list) and custom_questions:
            questions = [str(q) for q in custom_questions]
        else:
            questions = build_cultural_questions()
    else:
        questions = build_cultural_questions()

    culture_outputs = []
    if file_exists(args.model_path):
        print(f"Loading model from: {args.model_path}")
        model, tokenizer = load_local_model(args.model_path)
        for i, question in enumerate(questions, start=1):
            answer = generate_answer(
                model,
                tokenizer,
                question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = {
                "id": i,
                "question": question,
                "answer": answer,
                "answer_chars": len(answer),
                "answer_words_approx": len(answer.split()),
            }
            culture_outputs.append(row)
            print(f"[{i}/{len(questions)}] done")
    else:
        print(f"WARNING: model path not found: {args.model_path}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "base_path": args.base_path,
            "model_path": args.model_path,
            "adapter_dir": args.adapter_dir,
            "train_script": args.train_script,
            "train_file": args.train_file,
            "val_file": args.val_file,
            "log_file": args.log_file,
            "console_log": args.console_log,
        },
        "dataset": dataset_summary,
        "runtime_environment": runtime_env,
        "hyperparameters": hyperparameters,
        "training_args_bin": training_args_bin,
        "training_period": training_period,
        "training_metrics": training_metrics,
        "culture_questions_count": len(questions),
        "culture_outputs_count": len(culture_outputs),
    }

    report_json = os.path.join(args.out_dir, "qwen3_sinhala_audit_report.json")
    report_md = os.path.join(args.out_dir, "qwen3_sinhala_audit_report.md")
    culture_json = os.path.join(args.out_dir, "sri_lankan_culture_outputs.json")
    culture_csv = os.path.join(args.out_dir, "sri_lankan_culture_outputs.csv")

    save_json(report_json, report)
    save_json(culture_json, culture_outputs)
    save_csv(culture_csv, culture_outputs, ["id", "question", "answer", "answer_chars", "answer_words_approx"])
    write_markdown_report(report_md, report, culture_json)

    print("\nSaved files:")
    print(f"- {report_json}")
    print(f"- {report_md}")
    print(f"- {culture_json}")
    print(f"- {culture_csv}")


if __name__ == "__main__":
    main()
