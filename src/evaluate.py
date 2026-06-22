"""Valutazione di un checkpoint su uno split di test.

Carica un modello allenato, calcola le metriche (accuracy/precision/recall/F1/
ROC-AUC) su un CSV di test e salva metrics.json + confusion matrix PNG.

Esempio:
    python -m src.evaluate --config experiments/cnn_custom_baseline/config.yaml \
        --data-root data/genimage_subset \
        --checkpoint experiments/cnn_custom_baseline/best.pt \
        --split splits/baseline_misto/test_in_distribution.csv

Se --split e --checkpoint non sono dati, usa data.test_csv del config e
experiments/<name>/best.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import utils
from src.datasets import build_dataloader
from src.metrics import (
    compute_metrics,
    confusion,
    format_metrics,
    save_confusion_matrix,
)
from src.models import build_model
from src.train import collect_predictions
from src.transforms import build_eval_transforms


def main() -> None:
    ap = argparse.ArgumentParser(description="Valutazione GenImage detector.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="default: experiments/<name>/best.pt")
    ap.add_argument("--split", default=None,
                    help="CSV di test relativo a --data-root; default: data.test_csv")
    ap.add_argument("--out-dir", default=None,
                    help="dove salvare i risultati; default: <run>/eval")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    data_root = Path(args.data_root)
    exp_dir = Path("experiments") / cfg["experiment"]["name"]

    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "best.pt"
    split_csv = data_root / (args.split or cfg["data"]["test_csv"])
    out_dir = utils.ensure_dir(Path(args.out_dir) if args.out_dir else exp_dir / "eval")
    workers = args.num_workers if args.num_workers is not None else cfg["data"].get("num_workers", 0)

    device = utils.get_device()
    print(f"Device: {utils.describe_device(device)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test split: {split_csv}")

    # --- modello + pesi allenati ---
    model = build_model(cfg).to(device)
    ckpt = utils.load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # --- dati di test ---
    loader = build_dataloader(split_csv, data_root, build_eval_transforms(cfg),
                              cfg["train"]["batch_size"], shuffle=False,
                              num_workers=workers)

    # --- inferenza + metriche ---
    preds = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds["y_true"], preds["y_pred"], preds["y_prob"])
    cm = confusion(preds["y_true"], preds["y_pred"])

    print("\n" + format_metrics(metrics))
    print("confusion matrix [righe=vero, colonne=predetto] ordine [nature, ai]:")
    print(cm)

    # --- salvataggio ---
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                          title=f"{cfg['experiment']['name']} — test")
    print(f"\nRisultati salvati in {out_dir}/")


if __name__ == "__main__":
    main()
