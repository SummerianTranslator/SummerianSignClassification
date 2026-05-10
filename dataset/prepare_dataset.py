import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def collect_most_common_sign_annotations(signs_annotations: dict, count_annotations: int = 50) -> list:
    sign_names_count = Counter()

    for sign_annotation in signs_annotations:
        sign_name = sign_annotation["signName"]
        sign_names_count[sign_name] += 1
    
    return sign_names_count.most_common(count_annotations)


def create_signs_dataset_annotations(
    signs_annotations: dict,
    most_common_sign_annotations: list,
    output_dir: str,
    *,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
):
    sign_names_for_dataset = pl.DataFrame(most_common_sign_annotations, schema=["sign_name", "sign_count"], orient="row")["sign_name"].to_list()
    signs_dataset = {
        "image_name": [],
        "sign_name": [],
    }

    le = LabelEncoder()
    le.fit(sign_names_for_dataset)
    count_missing_signs =  0

    for sign_annotation in signs_annotations:
        sign_name = sign_annotation["signName"]
        image_id = sign_annotation["_id"]
        image_name = f"{image_id}.jpeg"
        if sign_name in sign_names_for_dataset:
            signs_dataset["image_name"].append(image_name)
            signs_dataset["sign_name"].append(sign_name)
        else:
            count_missing_signs += 1

    signs_dataset["label"] = le.transform(signs_dataset["sign_name"])

    df = pl.DataFrame(signs_dataset)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    indices = np.arange(df.height)
    labels = df["label"].to_numpy()
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    labels_tv = labels[train_val_idx]
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        stratify=labels_tv,
        random_state=random_state,
    )

    df = (
        df.with_columns(
            pl.int_range(pl.len()).alias("index"),
        )
    )

    train_df = (
        df.filter(
            pl.col("index").is_in(train_idx),
        )
        .drop("index")
    )

    val_df = (
        df.filter(
            pl.col("index").is_in(val_idx),
        )
        .drop("index")
    )

    test_df = (
        df.filter(
            pl.col("index").is_in(test_idx),
        )
        .drop("index")
    )

    train_df.write_csv(out / "train.csv")
    val_df.write_csv(out / "val.csv")
    test_df.write_csv(out / "test.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input json file with dataset annotations", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="Directory where train.csv, val.csv, and test.csv are written",
        type=str,
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        sign_annotations = json.load(f)

    most_common_sign_annotations = collect_most_common_sign_annotations(sign_annotations)
    create_signs_dataset_annotations(sign_annotations, most_common_sign_annotations, args.output)
    print(f"Create dataset split into train.csv, val.csv, and test.csv in folder: {args.output}")
    

if __name__ == "__main__":
    exit(main())
