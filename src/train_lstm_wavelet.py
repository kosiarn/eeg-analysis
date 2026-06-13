import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from model_WaveletSequenceDataset import WaveletSequenceDataset
from models.lstm_wavelet_classifier import LSTMWaveletClassifier
from preprocessing_DataReader import PreprocessingDataReader
from preprocessing_Transform import Transformer, Wavelet
from utils.channel_names import CHANNEL_NAMES
from utils.data_specs import NUM_PATIENTS
from utils.training import get_run_title


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    experiment: int
    seq_len: int
    stride: int
    wavelet: str
    min_scale: float
    max_scale: float
    n_scales: int
    batch_size: int
    lr: float
    epochs: int
    seed: int
    hidden_size: int
    num_layers: int
    dropout: float
    head_hidden_size: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_patients(patients: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _build_dataset(
    data_dir: str,
    patients: list[int],
    experiment: int,
    wavelet: Wavelet,
    scales: np.ndarray,
    seq_len: int,
    stride: int,
) -> WaveletSequenceDataset:
    reader = PreprocessingDataReader(path=data_dir)
    reader.load(patient=patients, experiment=experiment)
    reader.normalize()
    transformer = Transformer(wavelet)
    transformed = transformer.CWTTransform(reader.get(), scales=scales.tolist(), channels="all")
    return WaveletSequenceDataset(transformed=transformed, seq_len=seq_len, stride=stride)


def _eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    return float(acc), float(f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--experiment", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--wavelet", type=str, default=Wavelet.CGAU4.value)
    parser.add_argument("--min-scale", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float, default=64.0)
    parser.add_argument("--n-scales", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-hidden-size", type=int, default=128)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        experiment=args.experiment,
        seq_len=args.seq_len,
        stride=args.stride,
        wavelet=args.wavelet,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        n_scales=args.n_scales,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_size=args.head_hidden_size,
    )

    _set_seed(cfg.seed)

    scales = np.logspace(np.log10(cfg.min_scale), np.log10(cfg.max_scale), cfg.n_scales)
    input_dim = len(CHANNEL_NAMES) * len(scales)

    all_patients = [p for p in range(1, NUM_PATIENTS + 1)]
    train_patients, val_patients, test_patients = _split_patients(all_patients, cfg.seed)

    wavelet_enum = Wavelet(cfg.wavelet)
    train_ds = _build_dataset(cfg.data_dir, train_patients, cfg.experiment, wavelet_enum, scales, cfg.seq_len, cfg.stride)
    val_ds = _build_dataset(cfg.data_dir, val_patients, cfg.experiment, wavelet_enum, scales, cfg.seq_len, cfg.stride)
    test_ds = _build_dataset(cfg.data_dir, test_patients, cfg.experiment, wavelet_enum, scales, cfg.seq_len, cfg.stride)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWaveletClassifier(
        input_dim=input_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        head_hidden_size=cfg.head_hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    run_title = get_run_title("lstm_wavelet")
    run_dir = os.path.join("models", run_title)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    best_val_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc, val_f1 = _eval(model, val_loader, device)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "input_dim": input_dim,
            "scales": scales.tolist(),
        }
        torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(run_dir, "best.pt"))

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} best_val_acc={best_val_acc:.4f}"
        )

    test_acc, test_f1 = _eval(model, test_loader, device)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test_acc": test_acc, "test_f1": test_f1}, f, ensure_ascii=False, indent=2)
    print(f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
