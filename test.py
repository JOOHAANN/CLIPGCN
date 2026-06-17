import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from model import build_model, load_action_descriptions
from train import (
    TrimodalContrastiveDataset,
    get_device,
    get_path_from_config,
    load_config,
    move_batch_to_device,
    print_device_info,
    progress_bar,
)


def load_split_classes(split_dir, scope):
    metadata_path = Path(split_dir) / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if scope == "unseen":
        return metadata["unseen_classes"]
    if scope == "seen":
        return metadata["seen_classes"]
    if scope == "all":
        return sorted(metadata["seen_classes"] + metadata["unseen_classes"])
    raise ValueError(f"Unsupported candidate scope: {scope}")


def load_latest_run_info(config, config_path):
    output_config = config.get("outputs", {})
    work_dir = output_config.get("work_dir")
    if not work_dir:
        return None

    latest_path = os.path.join(get_path_from_config(config_path, work_dir), "latest_run.json")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_text_bank(model, config, config_path, candidate_labels):
    text_config = config["data"]["text"]
    labels, texts, _records = load_action_descriptions(
        xlsx_path=get_path_from_config(config_path, text_config["xlsx"]),
        text_column=text_config.get("text_column", "global_description"),
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
        prompt_template=text_config.get("prompt_template", "{global_description}"),
    )

    if candidate_labels is not None:
        candidate_set = {int(label) for label in candidate_labels}
        filtered = [(label, text) for label, text in zip(labels, texts) if int(label) in candidate_set]
        labels = [label for label, _text in filtered]
        texts = [text for _label, text in filtered]

    if len(labels) == 0:
        raise ValueError("No candidate action descriptions were selected for evaluation.")

    model.set_action_texts(texts, labels, batch_size=text_config.get("batch_size", 64))
    return labels, texts


def load_model(config, config_path, device, checkpoint_path):
    model = build_model(
        text_model_name=config["model"]["text_encoder"]["name"],
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(model, data_loader, candidate_labels, device, use_amp=False):
    candidate_tensor = torch.as_tensor(candidate_labels, dtype=torch.long, device=device)
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0
    inference_time_seconds = 0.0
    predictions = []

    with torch.no_grad():
        for batch in progress_bar(data_loader, "Test"):
            batch = move_batch_to_device(batch, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    batch["video"],
                    batch["pose"],
                    batch["object"],
                    batch["joint_xy"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time_seconds += time.perf_counter() - start_time

            pred_indices = torch.argmax(logits, dim=1)
            pred_labels = candidate_tensor[pred_indices]
            labels = batch["label"].long()
            targets = model.build_target_indices(labels)
            loss = F.cross_entropy(logits.float(), targets)
            correct1 += int((pred_labels == labels).sum().item())

            if logits.shape[1] >= 5:
                top5_indices = torch.topk(logits, k=5, dim=1).indices
                top5_labels = candidate_tensor[top5_indices]
                correct5 += int((top5_labels == labels[:, None]).any(dim=1).sum().item())

            batch_size = labels.shape[0]
            total += batch_size
            loss_sum += float(loss.detach().cpu()) * batch_size
            predictions.extend(
                {
                    "label": int(label),
                    "pred": int(pred),
                }
                for label, pred in zip(labels.detach().cpu().tolist(), pred_labels.detach().cpu().tolist())
            )

    metrics = {
        "num_samples": total,
        "loss": loss_sum / max(total, 1),
        "top1_acc": correct1 / max(total, 1),
        "top5_acc": correct5 / max(total, 1) if len(candidate_labels) >= 5 else None,
        "candidate_labels": [int(label) for label in candidate_labels],
        "inference_time_seconds": inference_time_seconds,
        "avg_inference_time_seconds_per_sample": inference_time_seconds / max(total, 1),
        "avg_inference_time_ms_per_sample": (inference_time_seconds / max(total, 1)) * 1000.0,
        "samples_per_second": total / inference_time_seconds if inference_time_seconds > 0 else None,
    }
    return metrics, predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIPGCN zero-shot action recognition.")
    parser.add_argument("--config", default="config.yaml", help="Path to CLIPGCN YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint. Defaults to outputs.best_model.")
    parser.add_argument("--split-dir", default=None, help="Directory containing unseen_*.npy. Defaults to data.train.data_dir.")
    parser.add_argument("--prefix", default="unseen", help="Dataset prefix, usually unseen.")
    parser.add_argument(
        "--candidate-scope",
        choices=["unseen", "seen", "all"],
        default="unseen",
        help="Which text labels are valid predictions. Use unseen for standard ZSL accuracy.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)

    latest_run = load_latest_run_info(config, config_path)
    if args.checkpoint:
        checkpoint_path = get_path_from_config(config_path, args.checkpoint)
    elif latest_run and latest_run.get("best_model"):
        checkpoint_path = latest_run["best_model"]
    else:
        checkpoint_path = get_path_from_config(config_path, config["outputs"]["best_model"])
    split_dir = args.split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    split_dir = get_path_from_config(config_path, split_dir)

    candidate_labels = load_split_classes(split_dir, args.candidate_scope)
    model = load_model(config, config_path, device, checkpoint_path)
    candidate_labels, _texts = build_text_bank(model, config, config_path, candidate_labels)

    dataset = TrimodalContrastiveDataset(
        data_dir=split_dir,
        prefix=args.prefix,
        mmap=True,
    )
    loader_config = config["data"]["dataloader"]
    data_loader = Data.DataLoader(
        dataset,
        batch_size=args.batch_size or loader_config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else loader_config.get("num_workers", 4),
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=False,
    )

    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    metrics, predictions = evaluate(model, data_loader, candidate_labels, device, use_amp=use_amp)

    print("Unseen action recognition results")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split_dir: {split_dir}")
    print(f"  prefix: {args.prefix}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {metrics['candidate_labels']}")
    print(f"  num_samples: {metrics['num_samples']}")
    print(f"  loss: {metrics['loss']:.4f}")
    print(f"  top1_acc: {metrics['top1_acc']:.4f}")
    if metrics["top5_acc"] is not None:
        print(f"  top5_acc: {metrics['top5_acc']:.4f}")
    print(f"  total_inference_time: {metrics['inference_time_seconds']:.4f}s")
    print(f"  avg_inference_time: {metrics['avg_inference_time_ms_per_sample']:.4f} ms/sample")
    if metrics["samples_per_second"] is not None:
        print(f"  throughput: {metrics['samples_per_second']:.2f} samples/s")

    output_config_path = args.output
    if output_config_path is None and latest_run and latest_run.get("test_results"):
        output_config_path = latest_run["test_results"]
    if output_config_path is None:
        output_config_path = config.get("outputs", {}).get("test_results")
    if output_config_path:
        output_path = get_path_from_config(config_path, output_config_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": metrics,
                    "predictions": predictions,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved test results to {output_path}")


if __name__ == "__main__":
    main()
