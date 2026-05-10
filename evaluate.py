import argparse
from pathlib import Path

import pytorch_lightning as pl

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
    return parser.parse_args()


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
