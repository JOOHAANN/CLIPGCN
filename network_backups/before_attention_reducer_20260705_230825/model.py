"""CLIPGCN contrastive model.

The visual side is the tri-modal ADL feature encoder in ``fusion.py``. The text
side is a frozen CLIP text encoder that turns action descriptions from the xlsx
file into 512-D targets.
"""

import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .fusion import TriModalFusion
except ImportError:
    from fusion import TriModalFusion

_CLIP_MODULE = None


def get_clip_module():
    global _CLIP_MODULE
    if _CLIP_MODULE is None:
        import clip

        _CLIP_MODULE = clip
    return _CLIP_MODULE


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _read_xlsx_rows(path: str) -> List[Dict[str, str]]:
    """Read the first worksheet using only the Python standard library."""

    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", namespace)))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", namespace):
            values: Dict[int, str] = {}
            for cell in row.findall("a:c", namespace):
                ref = cell.get("r")
                if ref is None:
                    continue
                value = ""
                value_node = cell.find("a:v", namespace)
                inline_node = cell.find("a:is/a:t", namespace)
                if value_node is not None:
                    value = value_node.text or ""
                    if cell.get("t") == "s" and value != "":
                        value = shared_strings[int(value)]
                elif inline_node is not None:
                    value = inline_node.text or ""
                values[_column_index(ref)] = value
            rows.append(values)

    if not rows:
        return []

    header = [rows[0].get(index, "").strip() for index in range(max(rows[0]) + 1)]
    records = []
    for row in rows[1:]:
        record = {}
        for index, column in enumerate(header):
            if column:
                record[column] = row.get(index, "").strip()
        if any(record.values()):
            records.append(record)
    return records


def load_action_descriptions(
    xlsx_path: str,
    text_column: str = "global_description",
    id_column: str = "ID",
    label_offset: int = 1,
    prompt_template: str = "{global_description}",
) -> Tuple[List[int], List[str], List[Dict[str, str]]]:
    """Load action descriptions ordered by label id.

    The provided xlsx uses ``ID=1..55`` while the dataset labels are ``0..54``.
    ``label_offset=1`` converts those xlsx IDs into dataset labels.
    """

    records = _read_xlsx_rows(xlsx_path)
    if not records:
        raise ValueError(f"No rows found in {xlsx_path}")

    label_text_pairs = []
    for row_idx, record in enumerate(records):
        if id_column in record and record[id_column] != "":
            label = int(float(record[id_column])) - label_offset
        else:
            label = row_idx

        if text_column not in record or record[text_column] == "":
            raise ValueError(f"Missing text column {text_column!r} for label {label}")

        prompt_values = {key: value for key, value in record.items()}
        prompt_values.setdefault("description", record[text_column])
        text = prompt_template.format(**prompt_values).strip()
        label_text_pairs.append((label, text, record))

    label_text_pairs.sort(key=lambda item: item[0])
    labels = [item[0] for item in label_text_pairs]
    texts = [item[1] for item in label_text_pairs]
    sorted_records = [item[2] for item in label_text_pairs]
    return labels, texts, sorted_records


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", device=None, download_root: Optional[str] = None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        clip_module = get_clip_module()
        self.clip_model, _ = clip_module.load(
            model_name,
            device=device,
            jit=False,
            download_root=download_root,
        )
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        self.embed_dim = int(self.clip_model.text_projection.shape[1])

    def get_device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    def forward(self, captions: Sequence[str]) -> torch.Tensor:
        if isinstance(captions, str):
            captions = [captions]
        with torch.no_grad():
            clip_module = get_clip_module()
            tokens = clip_module.tokenize(list(captions), truncate=True).to(self.get_device())
            return self.clip_model.encode_text(tokens).float()

    def encode_batches(self, captions: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        outputs = []
        for start in range(0, len(captions), batch_size):
            outputs.append(self(captions[start:start + batch_size]))
        return torch.cat(outputs, dim=0)


def encode_xlsx_action_descriptions(
    xlsx_path: str,
    model_name: str = "ViT-B/32",
    device=None,
    download_root: Optional[str] = None,
    text_column: str = "global_description",
    id_column: str = "ID",
    label_offset: int = 1,
    prompt_template: str = "{global_description}",
    batch_size: int = 64,
    normalize: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """Encode xlsx action descriptions into 512-D CLIP/ADL text embeddings."""

    labels, texts, records = load_action_descriptions(
        xlsx_path=xlsx_path,
        text_column=text_column,
        id_column=id_column,
        label_offset=label_offset,
        prompt_template=prompt_template,
    )
    encoder = FrozenCLIPTextEncoder(model_name=model_name, device=device, download_root=download_root)
    embeddings = encoder.encode_batches(texts, batch_size=batch_size)
    if normalize:
        embeddings = l2_normalize(embeddings)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings.cpu(),
                "labels": labels,
                "texts": texts,
                "records": records,
                "model_name": model_name,
                "text_column": text_column,
                "prompt_template": prompt_template,
            },
            output_path,
        )
        metadata_path = os.path.splitext(output_path)[0] + ".json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "output": output_path,
                    "shape": list(embeddings.shape),
                    "labels": labels,
                    "texts": texts,
                    "model_name": model_name,
                    "text_column": text_column,
                    "prompt_template": prompt_template,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    return embeddings, labels, texts


class TriModalModel(nn.Module):
    def __init__(self, fusion_dropout: float = 0.2):
        super().__init__()
        self.fusion_module = TriModalFusion(reducer_dropout=fusion_dropout)

    def forward(
        self,
        video_feature_raw: torch.Tensor,
        pose_feature_raw: torch.Tensor,
        object_feature_raw: torch.Tensor,
        joint_location_raw: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion_module(
            video_feature_raw,
            pose_feature_raw,
            object_feature_raw,
            joint_location_raw,
        )


class CLIPGCNContrastiveModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "ViT-B/32",
        device=None,
        download_root: Optional[str] = None,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()

        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=text_model_name,
            device=device,
            download_root=download_root,
        )
        self.visual_encoder = TriModalModel(fusion_dropout=fusion_dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.register_buffer("text_features", torch.empty(0, self.text_encoder.embed_dim), persistent=False)
        self.register_buffer("text_label_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.to(self.text_encoder.get_device())

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.clip_model.eval()
        return self

    def set_text_features(self, text_features: torch.Tensor, label_ids: Sequence[int]):
        device = self.text_encoder.get_device()
        self.text_features = text_features.detach().float().to(device)
        self.text_label_ids = torch.as_tensor(label_ids, dtype=torch.long, device=device)

    def set_action_texts(self, texts: Sequence[str], label_ids: Sequence[int], batch_size: int = 64):
        self.set_text_features(self.text_encoder.encode_batches(texts, batch_size=batch_size), label_ids)

    def build_target_indices(self, labels: torch.Tensor) -> torch.Tensor:
        if self.text_label_ids.numel() == 0:
            raise RuntimeError("Text feature bank is empty. Call set_text_features or set_action_texts first.")
        max_label = int(torch.max(torch.cat([labels.detach(), self.text_label_ids])).item())
        label_to_index = torch.full((max_label + 1,), -1, dtype=torch.long, device=labels.device)
        label_to_index[self.text_label_ids.to(labels.device)] = torch.arange(
            self.text_label_ids.numel(),
            device=labels.device,
        )
        targets = label_to_index[labels.long()]
        if torch.any(targets < 0):
            missing = torch.unique(labels[targets < 0]).detach().cpu().tolist()
            raise ValueError(f"Labels not found in text bank: {missing}")
        return targets

    def encode_visual(self, video, pose, obj, joint_xy) -> torch.Tensor:
        device = self.text_encoder.get_device()
        return self.visual_encoder(
            video.to(device=device, dtype=torch.float32),
            pose.to(device=device, dtype=torch.float32),
            obj.to(device=device, dtype=torch.float32),
            joint_xy.to(device=device, dtype=torch.float32),
        )

    def forward(self, video, pose, obj, joint_xy) -> torch.Tensor:
        visual_features = l2_normalize(self.encode_visual(video, pose, obj, joint_xy))
        text_features = l2_normalize(self.text_features.to(visual_features.device))
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * visual_features @ text_features.t()

    def contrastive_loss(self, video, pose, obj, joint_xy, labels) -> torch.Tensor:
        labels = labels.to(self.text_encoder.get_device()).long()
        logits = self(video, pose, obj, joint_xy)
        targets = self.build_target_indices(labels)
        return F.cross_entropy(logits, targets)


def build_model(
    text_model_name: str = "ViT-B/32",
    device=None,
    download_root: Optional[str] = None,
    fusion_dropout: float = 0.2,
):
    return CLIPGCNContrastiveModel(
        text_model_name=text_model_name,
        device=device,
        download_root=download_root,
        fusion_dropout=fusion_dropout,
    )
