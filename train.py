import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from training_pipeline import SignClassifier, SignsDataModule, load_config


def parse_limit(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sign classification model.")
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
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch through train/val/test for quick smoke checks.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=parse_limit,
        default=1.0,
        help="Fraction or integer number of train batches to run (Lightning option).",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=parse_limit,
        default=1.0,
        help="Fraction or integer number of validation batches to run (Lightning option).",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=parse_limit,
        default=1.0,
        help="Fraction or integer number of test batches to run (Lightning option).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config.output_folder)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    datamodule = SignsDataModule(config)
    datamodule.setup("fit")
    model = SignClassifier(
        model_name=config.model_name,
        num_classes=datamodule.num_classes,
        learning_rate=config.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best-epoch={epoch:02d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    logger = CSVLogger(save_dir=str(logs_dir), name=config.model_name)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    trainer.fit(model=model, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path
    if best_path:
        print(f"Best checkpoint: {best_path}")
        (output_dir / "best_checkpoint.txt").write_text(best_path, encoding="utf-8")
    elif not args.fast_dev_run:
        raise RuntimeError("No best checkpoint was saved.")

    if args.fast_dev_run:
        print("Fast dev run completed; checkpoint selection is skipped.")
        return 0

    test_metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    if test_metrics:
        metrics = test_metrics[0]
        print(f"accuracy: {metrics.get('test_accuracy')}")
        print(f"precision: {metrics.get('test_precision')}")
        print(f"recall: {metrics.get('test_recall')}")
        print(f"f1: {metrics.get('test_f1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
