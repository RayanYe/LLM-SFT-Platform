from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------
# Data models
# ----------------------------

@dataclass
class TrainConfig:
    base_model: str
    dataset_path: str
    finetune_method: str
    output_dir: str
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    lora_rank: int = 8
    max_length: int = 512


@dataclass
class TrainingResult:
    success: bool
    message: str
    output_dir: str
    log_path: Optional[str] = None
    metrics_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    loss_history: Optional[List[Dict[str, float]]] = None


# ----------------------------
# 1) Validate dataset
# ----------------------------

def validate_dataset(file_path: str) -> Dict[str, Any]:
    """
    Validate a JSONL dataset for fine-tuning.

    Expected format per line:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }

    Returns:
        {
            "valid": bool,
            "message": str,
            "num_samples": int,
            "sample_keys": list[str]
        }
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "valid": False,
            "message": f"Dataset file does not exist: {file_path}",
            "num_samples": 0,
            "sample_keys": []
        }

    if path.suffix.lower() != ".jsonl":
        return {
            "valid": False,
            "message": "Only .jsonl dataset format is supported in the MVP.",
            "num_samples": 0,
            "sample_keys": []
        }

    required_fields = {"instruction", "input", "output"}
    num_samples = 0
    sample_keys: List[str] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as e:
                    return {
                        "valid": False,
                        "message": f"Invalid JSON on line {line_no}: {e}",
                        "num_samples": num_samples,
                        "sample_keys": sample_keys
                    }

                if not isinstance(item, dict):
                    return {
                        "valid": False,
                        "message": f"Line {line_no} is not a JSON object.",
                        "num_samples": num_samples,
                        "sample_keys": sample_keys
                    }

                if num_samples == 0:
                    sample_keys = sorted(list(item.keys()))

                missing = required_fields - set(item.keys())
                if missing:
                    return {
                        "valid": False,
                        "message": f"Missing required fields on line {line_no}: {sorted(list(missing))}",
                        "num_samples": num_samples,
                        "sample_keys": sample_keys
                    }

                num_samples += 1

    except Exception as e:
        return {
            "valid": False,
            "message": f"Failed to read dataset: {e}",
            "num_samples": num_samples,
            "sample_keys": sample_keys
        }

    if num_samples == 0:
        return {
            "valid": False,
            "message": "Dataset is empty.",
            "num_samples": 0,
            "sample_keys": []
        }

    return {
        "valid": True,
        "message": "Dataset validation passed.",
        "num_samples": num_samples,
        "sample_keys": sample_keys
    }


# ----------------------------
# 2) Build train config
# ----------------------------

def build_train_config(form_data: Dict[str, Any]) -> TrainConfig:
    """
    Convert UI form data into a typed training config.
    This function also applies basic defaults and validation.
    """
    base_model = str(form_data.get("base_model", "")).strip()
    dataset_path = str(form_data.get("dataset_path", "")).strip()
    finetune_method = str(form_data.get("finetune_method", "LoRA")).strip()
    output_dir = str(form_data.get("output_dir", "./outputs/finetuned_model")).strip()

    if not base_model:
        raise ValueError("base_model is required.")
    if not dataset_path:
        raise ValueError("dataset_path is required.")
    if not output_dir:
        raise ValueError("output_dir is required.")

    learning_rate = float(form_data.get("learning_rate", 2e-4))
    batch_size = int(form_data.get("batch_size", 4))
    epochs = int(form_data.get("epochs", 3))
    lora_rank = int(form_data.get("lora_rank", 8))
    max_length = int(form_data.get("max_length", 512))

    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be > 0.")
    if max_length <= 0:
        raise ValueError("max_length must be > 0.")

    return TrainConfig(
        base_model=base_model,
        dataset_path=dataset_path,
        finetune_method=finetune_method,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        lora_rank=lora_rank,
        max_length=max_length,
    )


# ----------------------------
# 3) Start training
# ----------------------------

def start_training(config: TrainConfig) -> TrainingResult:
    """
    Start training workflow.

    Current version is a mock training loop:
    - creates output dirs
    - writes logs/metrics
    - simulates loss decrease

    Later you can replace the inner loop with:
    - Transformers + PEFT training
    - subprocess call to a real train script
    """
    output_dir = Path(config.output_dir)
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train.log"
    metrics_path = metrics_dir / "metrics.json"

    # Re-validate dataset before training
    dataset_check = validate_dataset(config.dataset_path)
    if not dataset_check["valid"]:
        return TrainingResult(
            success=False,
            message=f"Training aborted: {dataset_check['message']}",
            output_dir=str(output_dir),
            log_path=str(log_path),
            metrics_path=str(metrics_path),
            checkpoint_dir=str(ckpt_dir),
            loss_history=[],
        )

    loss_history: List[Dict[str, float]] = []
    global_step = 0

    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write("=== Training Started ===\n")
            log_f.write(f"Config: {json.dumps(asdict(config), ensure_ascii=False, indent=2)}\n")
            log_f.write(f"Dataset samples: {dataset_check['num_samples']}\n\n")

            for epoch in range(1, config.epochs + 1):
                log_f.write(f"[Epoch {epoch}/{config.epochs}] started\n")
                log_f.flush()

                # Simulate a fixed number of steps per epoch
                steps_per_epoch = max(5, min(20, dataset_check["num_samples"]))
                for step in range(1, steps_per_epoch + 1):
                    global_step += 1

                    # Mock loss curve
                    loss = max(0.05, 2.0 * (0.92 ** global_step))
                    loss_history.append(
                        {
                            "step": float(global_step),
                            "epoch": float(epoch),
                            "loss": float(loss),
                        }
                    )

                    log_f.write(
                        f"Epoch {epoch}/{config.epochs} | Step {step}/{steps_per_epoch} | "
                        f"global_step={global_step} | loss={loss:.4f}\n"
                    )
                    log_f.flush()

                    time.sleep(0.05)  # mock training latency

                # Simulate checkpoint saving
                epoch_ckpt = ckpt_dir / f"epoch_{epoch}"
                epoch_ckpt.mkdir(parents=True, exist_ok=True)
                with (epoch_ckpt / "model.txt").open("w", encoding="utf-8") as ckpt_f:
                    ckpt_f.write(f"Mock checkpoint for epoch {epoch}\n")
                    ckpt_f.write(f"base_model={config.base_model}\n")

                log_f.write(f"[Epoch {epoch}/{config.epochs}] completed\n\n")
                log_f.flush()

            log_f.write("=== Training Finished Successfully ===\n")

        with metrics_path.open("w", encoding="utf-8") as metrics_f:
            json.dump(
                {
                    "config": asdict(config),
                    "dataset": dataset_check,
                    "loss_history": loss_history,
                    "final_loss": loss_history[-1]["loss"] if loss_history else None,
                },
                metrics_f,
                ensure_ascii=False,
                indent=2,
            )

        return TrainingResult(
            success=True,
            message="Training completed successfully.",
            output_dir=str(output_dir),
            log_path=str(log_path),
            metrics_path=str(metrics_path),
            checkpoint_dir=str(ckpt_dir),
            loss_history=loss_history,
        )

    except Exception as e:
        # Best-effort error logging
        try:
            with log_path.open("a", encoding="utf-8") as log_f:
                log_f.write(f"\n[ERROR] Training failed: {e}\n")
        except Exception:
            pass

        return TrainingResult(
            success=False,
            message=f"Training failed: {e}",
            output_dir=str(output_dir),
            log_path=str(log_path),
            metrics_path=str(metrics_path),
            checkpoint_dir=str(ckpt_dir),
            loss_history=loss_history,
        )


# ----------------------------
# Example usage
# ----------------------------
# if __name__ == "__main__":
#     form_data = {
#         "base_model": "Qwen2.5-0.5B",
#         "dataset_path": "./data/demo_dataset.jsonl",
#         "finetune_method": "LoRA",
#         "output_dir": "./outputs/finetuned_model",
#         "learning_rate": 2e-4,
#         "batch_size": 4,
#         "epochs": 3,
#         "lora_rank": 8,
#         "max_length": 512,
#     }

#     config = build_train_config(form_data)
#     result = start_training(config)
#     print(result)