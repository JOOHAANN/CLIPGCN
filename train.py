import argparse
import copy
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, Subset

from model import build_model, load_action_descriptions

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_config(config_path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML first: pip install pyyaml") from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_path_from_config(config_path, path):
    if path is None or os.path.isabs(str(path)):
        return path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(config_dir, str(path))


def get_device(device_name=None):
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    if str(device_name).startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")
        if ":" in str(device_name):
            gpu_id = int(str(device_name).split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} does not exist. Available GPU count: {torch.cuda.device_count()}")
    return torch.device(device_name)


def print_device_info(device):
    if torch.cuda.is_available():
        print("Available GPUs:")
        for gpu_id in range(torch.cuda.device_count()):
            print(f"  cuda:{gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
    print(f"Using device: {device}")


def progress_bar(data_loader, desc):
    if tqdm is None:
        print(f"{desc}...")
        return data_loader
    return tqdm(data_loader, desc=desc, leave=False)


class TrimodalContrastiveDataset(Dataset):
    def __init__(self, data_dir, prefix="trimodal_train", mmap=True):
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        mmap_mode = "r" if mmap else None

        self.video = np.load(self.data_dir / f"{prefix}_video.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.pose = np.load(self.data_dir / f"{prefix}_pose.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.object = np.load(self.data_dir / f"{prefix}_object.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.joint_xy = np.load(self.data_dir / f"{prefix}_joint_xy.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.labels = np.load(self.data_dir / f"{prefix}_labels.npy", mmap_mode=mmap_mode, allow_pickle=False)

        sample_path = self.data_dir / f"{prefix}_sample_names.npy"
        self.sample_names = np.load(sample_path, allow_pickle=True) if sample_path.exists() else None

        lengths = {len(self.video), len(self.pose), len(self.object), len(self.joint_xy), len(self.labels)}
        if len(lengths) != 1:
            raise ValueError(
                "Modality lengths are not aligned: "
                f"video={len(self.video)}, pose={len(self.pose)}, object={len(self.object)}, "
                f"joint_xy={len(self.joint_xy)}, labels={len(self.labels)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "video": torch.from_numpy(np.array(self.video[index], copy=True)).float(),
            "pose": torch.from_numpy(np.array(self.pose[index], copy=True)).float(),
            "object": torch.from_numpy(np.array(self.object[index], copy=True)).float(),
            "joint_xy": torch.from_numpy(np.array(self.joint_xy[index], copy=True)).float(),
            "label": torch.as_tensor(int(self.labels[index]), dtype=torch.long),
        }


def stratified_split_indices(labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices = []
    val_indices = []

    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        if val_fraction > 0 and len(indices) > 1:
            val_count = min(max(val_count, 1), len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_dataloaders(config, config_path):
    data_config = config["data"]
    loader_config = data_config["dataloader"]
    train_config = data_config["train"]
    val_config = data_config.get("val", {})
    split_config = data_config.get("validation_split", {})

    train_dataset = TrimodalContrastiveDataset(
        data_dir=get_path_from_config(config_path, train_config["data_dir"]),
        prefix=train_config.get("prefix", "trimodal_train"),
        mmap=train_config.get("mmap", True),
    )

    if val_config.get("data_dir"):
        val_dataset = TrimodalContrastiveDataset(
            data_dir=get_path_from_config(config_path, val_config["data_dir"]),
            prefix=val_config.get("prefix", "trimodal_train"),
            mmap=val_config.get("mmap", True),
        )
    else:
        val_fraction = float(split_config.get("fraction", 0.1))
        train_indices, val_indices = stratified_split_indices(
            train_dataset.labels,
            val_fraction=val_fraction,
            seed=int(split_config.get("seed", 20260616)),
        )
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config.get("train_shuffle", True),
        num_workers=loader_config.get("num_workers", 4),
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=loader_config.get("drop_last", True),
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config.get("val_shuffle", False),
        num_workers=loader_config.get("num_workers", 4),
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer(model, optimizer_config):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = optimizer_config["name"].lower()
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")


def move_batch_to_device(batch, device):
    return {
        "video": batch["video"].to(device, non_blocking=True),
        "pose": batch["pose"].to(device, non_blocking=True),
        "object": batch["object"].to(device, non_blocking=True),
        "joint_xy": batch["joint_xy"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def run_one_epoch(model, data_loader, optimizer, scaler, device, use_amp, train, epoch, num_epochs, grad_clip_norm=None):
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    mode = "Train" if train else "Val"
    bar = progress_bar(data_loader, f"{mode} epoch {epoch + 1}/{num_epochs}")

    for batch in bar:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = model.contrastive_loss(
                    batch["video"],
                    batch["pose"],
                    batch["object"],
                    batch["joint_xy"],
                    batch["label"],
                )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1} during {mode.lower()}: {float(loss.detach().cpu())}"
            )

        if train:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = batch["label"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        if tqdm is not None:
            bar.set_postfix(loss=total_loss / max(total_samples, 1))

    return total_loss / max(total_samples, 1)


def prepare_action_text_bank(model, config, config_path):
    text_config = config["data"]["text"]
    xlsx_path = get_path_from_config(config_path, text_config["xlsx"])
    labels, texts, records = load_action_descriptions(
        xlsx_path=xlsx_path,
        text_column=text_config.get("text_column", "global_description"),
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
        prompt_template=text_config.get("prompt_template", "{global_description}"),
    )
    model.set_action_texts(texts, labels, batch_size=text_config.get("batch_size", 64))

    output_path = text_config.get("cache_output")
    if output_path:
        output_path = get_path_from_config(config_path, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "embeddings": model.text_features.detach().cpu(),
                "labels": labels,
                "texts": texts,
                "records": records,
            },
            output_path,
        )
        with open(os.path.splitext(output_path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "output": output_path,
                    "shape": list(model.text_features.shape),
                    "labels": labels,
                    "texts": texts,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    print(f"Loaded {len(texts)} action descriptions from {xlsx_path}")


def prepare_output_paths(config, config_path):
    output_config = config["outputs"]
    base_work_dir = get_path_from_config(config_path, output_config["work_dir"])

    if output_config.get("auto_run_dir", False):
        run_name = output_config.get("run_name")
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        work_dir = os.path.join(base_work_dir, run_name)
        if os.path.exists(work_dir):
            suffix = 1
            while os.path.exists(f"{work_dir}_{suffix:02d}"):
                suffix += 1
            work_dir = f"{work_dir}_{suffix:02d}"

        os.makedirs(work_dir, exist_ok=False)
        output_config["work_dir"] = work_dir
        output_config["best_model"] = os.path.join(work_dir, "best_model.pth")
        output_config["last_model"] = os.path.join(work_dir, "last_model.pth")
        output_config["history"] = os.path.join(work_dir, "history.json")
        output_config["train_curve"] = os.path.join(work_dir, "train_curve.png")
        output_config["test_results"] = os.path.join(work_dir, "unseen_test_results.json")

        latest_path = os.path.join(base_work_dir, "latest_run.json")
        os.makedirs(base_work_dir, exist_ok=True)
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "work_dir": work_dir,
                    "best_model": output_config["best_model"],
                    "last_model": output_config["last_model"],
                    "history": output_config["history"],
                    "train_curve": output_config["train_curve"],
                    "test_results": output_config["test_results"],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        shutil.copy2(config_path, os.path.join(work_dir, "source_config.yaml"))
        with open(os.path.join(work_dir, "resolved_config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
    else:
        os.makedirs(base_work_dir, exist_ok=True)

    return output_config["work_dir"]


def train_model(model, train_loader, val_loader, config, config_path, device):
    train_config = config["train"]
    output_config = config["outputs"]
    optimizer = build_optimizer(model, train_config["optimizer"])

    epochs = int(train_config["epochs"])
    val_interval = int(train_config.get("val_interval", 1))
    if val_interval <= 0:
        raise ValueError(f"val_interval must be positive, got {val_interval}")
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip_norm = train_config.get("grad_clip_norm", None)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        train_loss = run_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, True, epoch, epochs, grad_clip_norm
        )
        should_validate = (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
        if should_validate:
            val_loss = run_one_epoch(
                model, val_loader, optimizer, scaler, device, use_amp, False, epoch, epochs, None
            )
        else:
            val_loss = None

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"train_loss: {train_loss:.4f}")
        if val_loss is None:
            print(f"val_loss: skipped (val_interval={val_interval})")
        else:
            print(f"val_loss: {val_loss:.4f}")

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start_time
        print("Training Time {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))

    work_dir = get_path_from_config(config_path, output_config["work_dir"])
    os.makedirs(work_dir, exist_ok=True)

    best_path = get_path_from_config(config_path, output_config["best_model"])
    last_path = get_path_from_config(config_path, output_config["last_model"])
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    os.makedirs(os.path.dirname(last_path), exist_ok=True)

    torch.save(best_model_wts, best_path)
    torch.save(model.state_dict(), last_path)

    history_path = get_path_from_config(config_path, output_config["history"])
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    model.load_state_dict(best_model_wts)
    print(f"Best model saved to {best_path}")
    print(f"Last model saved to {last_path}")
    return history


def plot_loss(history, save_path=None, show=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip plotting.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], "ro-", label="train_loss")
    val_loss = [np.nan if value is None else value for value in history["val_loss"]]
    plt.plot(history["epoch"], val_loss, "bo-", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CLIPGCN Contrastive Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Training curve saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIPGCN contrastive model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    set_seed(int(config["runtime"].get("seed", 20260616)))
    run_dir = prepare_output_paths(config, config_path)
    print(f"Run output directory: {run_dir}")

    device = get_device(config["runtime"].get("device"))
    print_device_info(device)

    model = build_model(
        text_model_name=config["model"]["text_encoder"]["name"],
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )
    prepare_action_text_bank(model, config, config_path)
    model = model.to(device)

    train_loader, val_loader = build_dataloaders(config, config_path)
    history = train_model(model, train_loader, val_loader, config, config_path, device)

    curve_path = get_path_from_config(config_path, config["outputs"].get("train_curve"))
    if curve_path:
        plot_loss(history, save_path=curve_path, show=not config["runtime"].get("no_show", True))


if __name__ == "__main__":
    main()
