"""Train a compact learned three-frame RGB encoder on GeoFusion video clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived" / "clip_encoder"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipDataset(Dataset):
    def __init__(self, clips: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.clips, self.labels, self.augment = clips, labels, augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        clip = torch.from_numpy(np.array(self.clips[index], copy=True)).permute(3, 0, 1, 2).float().div(255.0)
        if self.augment and torch.rand(()) < 0.5:
            clip = torch.flip(clip, dims=[3])
        return clip, torch.tensor(self.labels[index], dtype=torch.long)


class CompactClipEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.25))
        self.head = nn.Linear(64, 2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(self.features(value))
        return self.head(embedding), embedding


def infer(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); probabilities = []; embeddings = []
    with torch.no_grad():
        for clips, _ in loader:
            logits, embedding = model(clips.to(DEVICE))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.asarray(probabilities), np.vstack(embeddings)


def metric_values(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(target)), "positive_rows": int(target.sum()), "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)), "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)), "average_precision": float(average_precision_score(target, probability)), "roc_auc": float(roc_auc_score(target, probability)),
    }


def main() -> None:
    torch.manual_seed(20260923); np.random.seed(20260923)
    manifest = pd.read_csv(DATA / "source_manifest.csv")
    clips = np.load(DATA / "clips_uint8.npy", mmap_mode="r")
    train_mask = manifest["split"].eq("train").to_numpy(); validation_mask = manifest["split"].eq("validation").to_numpy()
    train_labels = manifest.loc[train_mask, "target_anomaly"].to_numpy(dtype=int); validation_labels = manifest.loc[validation_mask, "target_anomaly"].to_numpy(dtype=int)
    train_loader = DataLoader(ClipDataset(clips[train_mask], train_labels, augment=True), batch_size=32, shuffle=True, num_workers=0)
    train_eval = DataLoader(ClipDataset(clips[train_mask], train_labels), batch_size=64, shuffle=False, num_workers=0)
    validation_loader = DataLoader(ClipDataset(clips[validation_mask], validation_labels), batch_size=64, shuffle=False, num_workers=0)
    model = CompactClipEncoder().to(DEVICE)
    weights = torch.tensor([1.0, len(train_labels) / max(1, train_labels.sum())], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights); optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_f1 = None, -1.0
    for epoch in range(20):
        model.train()
        for batch, labels in train_loader:
            optimiser.zero_grad(); logits, _ = model(batch.to(DEVICE)); loss = criterion(logits, labels.to(DEVICE)); loss.backward(); optimiser.step()
        probability, _ = infer(model, validation_loader); score = f1_score(validation_labels, probability >= 0.5, zero_division=0)
        if score > best_f1:
            best_f1 = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch + 1:02d}/20 validation_f1={score:.4f}")
    model.load_state_dict(best_state)
    train_probability, train_embedding = infer(model, train_eval)
    validation_probability, validation_embedding = infer(model, validation_loader)
    all_embedding = np.empty((len(manifest), 64), dtype=np.float32); all_embedding[train_mask] = train_embedding; all_embedding[validation_mask] = validation_embedding
    np.savez_compressed(PROCESSED / "clip_encoder_embeddings.npz", event_id=manifest["event_id"].to_numpy(dtype=str), embeddings=all_embedding)
    prediction = manifest.loc[validation_mask, ["event_id", "session", "target_anomaly"]].copy(); prediction["clip_event_probability"] = validation_probability
    prediction.to_csv(PROCESSED / "clip_validation_predictions.csv", index=False)
    payload = {"model": "compact_3dcnn_3frame_96px", "selection_split": "validation_only", "device": str(DEVICE), **metric_values(validation_labels, validation_probability)}
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": "CompactClipEncoder", "frame_offsets_seconds": [-1, 0, 1]}, MODELS / "compact_clip_encoder.pt")
    (MODELS / "compact_clip_encoder_validation_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); print("Test clips and test metrics were not used.")


if __name__ == "__main__":
    main()
