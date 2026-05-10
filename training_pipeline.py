from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision import models, transforms as T

try:
    from transformers import ConvNextForImageClassification

    TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    ConvNextForImageClassification = None
    TRANSFORMERS_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    A = None
    ToTensorV2 = None
    ALBUMENTATIONS_AVAILABLE = False


REQUIRED_CONFIG_KEYS = {
    "model_name",
    "train_path",
    "val_path",
    "test_path",
    "image_folder",
    "batch_size",
    "learning_rate",
    "num_epochs",
    "output_folder",
}


@dataclass(slots=True)
class TrainConfig:
    model_name: str
    train_path: str
    val_path: str
    test_path: str
    image_folder: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    output_folder: str
    weight_decay: float
    scheduler_name: str | None
    scheduler_t_max: int
    scheduler_eta_min: float


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    missing_keys = REQUIRED_CONFIG_KEYS.difference(raw_config.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required config keys: {missing}")

    return TrainConfig(
        model_name=str(raw_config["model_name"]),
        train_path=str(raw_config["train_path"]),
        val_path=str(raw_config["val_path"]),
        test_path=str(raw_config["test_path"]),
        image_folder=str(raw_config["image_folder"]),
        batch_size=int(raw_config["batch_size"]),
        learning_rate=float(raw_config["learning_rate"]),
        num_epochs=int(raw_config["num_epochs"]),
        output_folder=str(raw_config["output_folder"]),
        weight_decay=float(raw_config.get("weight_decay", 0.0)),
        scheduler_name=(
            str(raw_config["scheduler_name"]).strip().lower()
            if raw_config.get("scheduler_name") is not None
            else None
        ),
        scheduler_t_max=int(raw_config.get("scheduler_t_max", raw_config["num_epochs"])),
        scheduler_eta_min=float(raw_config.get("scheduler_eta_min", 0.0)),
    )


def build_transforms(image_size: int = 224) -> tuple[Any, Any]:
    if ALBUMENTATIONS_AVAILABLE:
        train_transform = A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        eval_transform = A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        return train_transform, eval_transform

    # Fallback for restricted environments where Albumentations cannot be installed.
    train_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


class SignsDataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, transform: Any):
        self.data = pd.read_csv(csv_path)
        self.image_folder = Path(image_folder)
        self.transform = transform

        required_columns = {"image_name", "label"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"{csv_path} must contain columns: {sorted(required_columns)}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        image_path = self._resolve_image_path(str(row["image_name"]))
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")
        if ALBUMENTATIONS_AVAILABLE:
            transformed = self.transform(image=np.array(image))
            image_tensor = transformed["image"]
        else:
            image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

    def _resolve_image_path(self, image_name: str) -> Path:
        candidate = self.image_folder / image_name
        if candidate.exists():
            return candidate

        stem = Path(image_name).stem
        for extension in (".jpeg", ".jpg", ".png"):
            alt = self.image_folder / f"{stem}{extension}"
            if alt.exists():
                return alt

        raise FileNotFoundError(f"Image not found: {candidate}")


class SignsDataModule(pl.LightningDataModule):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_dataset: SignsDataset | None = None
        self.val_dataset: SignsDataset | None = None
        self.test_dataset: SignsDataset | None = None
        self.num_classes = 0

    def setup(self, stage: str | None = None) -> None:
        train_transform, eval_transform = build_transforms()
        if stage in (None, "fit"):
            self.train_dataset = SignsDataset(
                self.config.train_path, self.config.image_folder, train_transform
            )
            self.val_dataset = SignsDataset(
                self.config.val_path, self.config.image_folder, eval_transform
            )

        if stage in (None, "test", "validate", "predict"):
            self.test_dataset = SignsDataset(
                self.config.test_path, self.config.image_folder, eval_transform
            )

        labels = []
        for csv_path in (
            self.config.train_path,
            self.config.val_path,
            self.config.test_path,
        ):
            frame = pd.read_csv(csv_path, usecols=["label"])
            labels.extend(frame["label"].tolist())
        self.num_classes = int(max(labels)) + 1

    def _loader_kwargs(self) -> dict[str, Any]:
        # Use single-process loading by default to avoid shared-memory issues.
        workers = 0
        return {
            "batch_size": self.config.batch_size,
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup('fit').")
        return DataLoader(self.train_dataset, shuffle=True, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup('fit').")
        return DataLoader(self.val_dataset, shuffle=False, **self._loader_kwargs())

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Call setup('test').")
        return DataLoader(self.test_dataset, shuffle=False, **self._loader_kwargs())


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name.startswith("convnext:"):
        if not TRANSFORMERS_AVAILABLE:
            raise ModuleNotFoundError(
                "transformers is required for ConvNeXT models. "
                "Install it and use model_name like convnext:facebook/convnext-tiny-224."
            )
        hf_model_id = model_name.split(":", maxsplit=1)[1].strip()
        if not hf_model_id:
            raise ValueError(
                "ConvNeXT model_name must be in format convnext:<hf_repo_id>, "
                "for example convnext:facebook/convnext-tiny-224."
            )
        model = ConvNextForImageClassification.from_pretrained(
            hf_model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        model.train()
        return model

    if not hasattr(models, model_name):
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            "Use torchvision names like resnet50/efficientnet_b0/vit_b_16/swin_t or "
            "ConvNeXT from Hugging Face as convnext:<hf_repo_id>."
        )

    if not (
        model_name.startswith("resnet")
        or model_name.startswith("efficientnet")
        or model_name.startswith("vit")
        or model_name.startswith("swin")
    ):
        raise ValueError(
            f"model_name='{model_name}' is not supported. "
            "Only ResNet, EfficientNet, ViT, and Swin families are allowed."
        )

    model_factory = getattr(models, model_name)
    model = model_factory(weights=None)

    if model_name.startswith("resnet"):
        if not hasattr(model, "fc"):
            raise ValueError(f"Cannot find fc layer for model: {model_name}")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        if not hasattr(model, "classifier"):
            raise ValueError(f"Cannot find classifier layer for model: {model_name}")
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("vit"):
        if not hasattr(model, "heads") or not hasattr(model.heads, "head"):
            raise ValueError(f"Cannot find heads.head layer for model: {model_name}")
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_name.startswith("swin"):
        if not hasattr(model, "head"):
            raise ValueError(f"Cannot find head layer for model: {model_name}")
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model


class SignClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        scheduler_name: str | None = None,
        scheduler_t_max: int = 1,
        scheduler_eta_min: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_name=model_name, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_t_max = max(1, scheduler_t_max)
        self.scheduler_eta_min = scheduler_eta_min

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

        self.train_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.val_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.test_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )

        self.train_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return get_model_logits(output)

    def configure_optimizers(self) -> Any:
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_name is None:
            return optimizer

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_t_max,
                eta_min=self.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        raise ValueError(
            f"Unsupported scheduler_name='{self.scheduler_name}'. "
            "Supported values: null, cosine."
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_accuracy",
            self.train_accuracy(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_precision",
            self.train_precision(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_recall",
            self.train_recall(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log("train_f1", self.train_f1(logits, labels), on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy",
            self.val_accuracy(logits, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_precision",
            self.val_precision(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_recall",
            self.val_recall(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.val_f1(logits, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test_accuracy",
            self.test_accuracy(logits, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_precision",
            self.test_precision(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_recall",
            self.test_recall(logits, labels),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1",
            self.test_f1(logits, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss


def find_best_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoint_files = sorted(
        checkpoints_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")

    best_candidates = [path for path in checkpoint_files if "best" in path.name.lower()]
    if best_candidates:
        return best_candidates[0]
    return checkpoint_files[0]


def get_model_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    raise TypeError(
        "Model output does not contain logits. "
        "Expected Tensor or output with .logits attribute."
    )
