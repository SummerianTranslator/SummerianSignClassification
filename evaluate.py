import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch

from training_pipeline import (
    SignClassifier,
    SignsDataModule,
    find_best_checkpoint,
    load_config,
)


def parse_limit(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sign classifier.")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to YAML config file. model_name supports torchvision "
            "(resnet50, efficientnet_b0, vit_b_16, swin_t) and HF ConvNeXT "
            "(convnext:facebook/convnext-tiny-224)."
        ),
    )
    parser.add_argument(
        "--limit-test-batches",
        type=parse_limit,
        default=1.0,
        help="Fraction or integer number of test batches to run (Lightning option).",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Optional path to CSV file where per-class metrics "
            "(precision, recall, f1, support, class_accuracy) are saved."
        ),
    )
    return parser.parse_args()


def build_label_mapping(config: Any, num_classes: int) -> dict[int, str]:
    frames = []
    for path in [config.train_path, config.val_path, config.test_path]:
        csv_path = Path(path)
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path, usecols=["label", "sign_name"]))

    if not frames:
        return {index: f"class_{index}" for index in range(num_classes)}

    mapping_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    counts = mapping_df.groupby("label")["sign_name"].nunique()
    inconsistent = counts[counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Found multiple sign_name values for the same label in split CSV files."
        )

    mapping = (
        mapping_df.sort_values("label")
        .drop_duplicates(subset=["label"])
        .set_index("label")["sign_name"]
        .to_dict()
    )
    return {index: mapping.get(index, f"class_{index}") for index in range(num_classes)}


def resolve_max_batches(limit_value: int | float, total_batches: int) -> int:
    if isinstance(limit_value, int):
        return min(total_batches, max(1, limit_value))
    if limit_value <= 0:
        return 0
    return min(total_batches, max(1, int(math.ceil(total_batches * limit_value))))


def collect_predictions(
    model: SignClassifier,
    datamodule: SignsDataModule,
    limit_test_batches: int | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    total_batches = len(test_loader)
    max_batches = resolve_max_batches(limit_test_batches, total_batches)

    if max_batches == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    device = next(model.parameters()).device
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            y_true.append(labels.cpu())
            y_pred.append(predictions.cpu())

    if not y_true:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(y_true), torch.cat(y_pred)


def build_per_class_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    label_mapping: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for class_id in range(num_classes):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        tp = int((true_mask & pred_mask).sum().item())
        fp = int((~true_mask & pred_mask).sum().item())
        support = int(true_mask.sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        class_accuracy = recall

        rows.append(
            {
                "class_id": class_id,
                "class_name": label_mapping.get(class_id, f"class_{class_id}"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "class_accuracy": class_accuracy,
            }
        )
    return pd.DataFrame(rows)


def resolve_checkpoint(output_dir: Path) -> Path:
    best_checkpoint_file = output_dir / "best_checkpoint.txt"
    if best_checkpoint_file.exists():
        checkpoint_path = Path(best_checkpoint_file.read_text(encoding="utf-8").strip())
        if checkpoint_path.exists():
            return checkpoint_path

    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory is missing: {checkpoints_dir}. "
            "Train the model first with train.py."
        )
    return find_best_checkpoint(checkpoints_dir)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config.output_folder)

    datamodule = SignsDataModule(config)
    datamodule.setup("fit")
    checkpoint_path = resolve_checkpoint(output_dir)

    model = SignClassifier.load_from_checkpoint(
        str(checkpoint_path),
        model_name=config.model_name,
        num_classes=datamodule.num_classes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        scheduler_name=config.scheduler_name,
        scheduler_t_max=config.scheduler_t_max,
        scheduler_eta_min=config.scheduler_eta_min,
    )
    trainer = pl.Trainer(logger=False, limit_test_batches=args.limit_test_batches)
    metrics = trainer.test(model=model, datamodule=datamodule)

    if not metrics:
        raise RuntimeError("No test metrics returned by trainer.test().")

    values = metrics[0]
    print(f"accuracy: {values.get('test_accuracy')}")
    print(f"precision: {values.get('test_precision')}")
    print(f"recall: {values.get('test_recall')}")
    print(f"f1: {values.get('test_f1')}")

    if args.metrics:
        y_true, y_pred = collect_predictions(
            model=model,
            datamodule=datamodule,
            limit_test_batches=args.limit_test_batches,
        )
        label_mapping = build_label_mapping(config=config, num_classes=datamodule.num_classes)
        per_class_df = build_per_class_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=datamodule.num_classes,
            label_mapping=label_mapping,
        )
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        per_class_df.to_csv(metrics_path, index=False)
        print(f"Per-class metrics saved to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
