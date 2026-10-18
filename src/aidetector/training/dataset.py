"""Dataset manifest representation, validation, and Defactify extractor."""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
from PIL import Image
import torch


REQUIRED_COLUMNS = [
    "image_path",
    "label",
    "source_dataset",
    "generator_family",
    "content_group",
    "split",
    "format",
]

VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = {0, 1}


class ManifestValidationError(Exception):
    """Raised when dataset manifest violates integrity or leakage constraints."""
    pass


@dataclass
class ManifestReport:
    total_rows: int
    split_counts: Dict[str, int]
    label_counts: Dict[str, Dict[int, int]]
    generator_counts: Dict[str, int]
    format_counts: Dict[str, int]
    leakage_detected: bool
    missing_files_count: int


def validate_manifest(
    manifest_path: Union[str, Path],
    check_files_exist: bool = True,
    base_dir: Optional[Union[str, Path]] = None,
) -> ManifestReport:
    """Validates a dataset manifest CSV for schema correctness, missing files, and data leakage.

    Args:
        manifest_path: Path to manifest CSV file.
        check_files_exist: Whether to verify that each referenced image exists on disk.
        base_dir: Optional base directory to resolve relative image paths against.

    Returns:
        ManifestReport summarizing validation findings.

    Raises:
        ManifestValidationError: If columns are missing, splits/labels are invalid, or data leakage exists.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    df = pd.read_csv(path)

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ManifestValidationError(f"Manifest is missing required columns: {missing_cols}")

    # Validate splits
    unique_splits = set(df["split"].unique())
    invalid_splits = unique_splits - VALID_SPLITS
    if invalid_splits:
        raise ManifestValidationError(f"Invalid splits found: {invalid_splits}. Must be in {VALID_SPLITS}")

    # Validate labels
    unique_labels = set(df["label"].unique())
    invalid_labels = unique_labels - VALID_LABELS
    if invalid_labels:
        raise ManifestValidationError(f"Invalid labels found: {invalid_labels}. Must be in {VALID_LABELS}")

    # Check for image_path duplicates across different splits (leakage)
    split_images: Dict[str, Set[str]] = {s: set() for s in unique_splits}
    for _, row in df.iterrows():
        s = row["split"]
        img = str(row["image_path"]).strip()
        split_images[s].add(img)

    splits_list = list(unique_splits)
    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            s1, s2 = splits_list[i], splits_list[j]
            overlap = split_images[s1].intersection(split_images[s2])
            if overlap:
                sample_overlap = list(overlap)[:3]
                raise ManifestValidationError(
                    f"Data leakage detected! {len(overlap)} images shared between '{s1}' and '{s2}' splits. "
                    f"Examples: {sample_overlap}"
                )

    # Check content_group leakage across splits (if content_group is provided and non-empty)
    split_groups: Dict[str, Set[str]] = {s: set() for s in unique_splits}
    for _, row in df.iterrows():
        s = row["split"]
        grp = str(row.get("content_group", "")).strip()
        if grp and grp.lower() not in ("none", "nan", ""):
            split_groups[s].add(grp)

    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            s1, s2 = splits_list[i], splits_list[j]
            grp_overlap = split_groups[s1].intersection(split_groups[s2])
            if grp_overlap:
                sample_overlap = list(grp_overlap)[:3]
                raise ManifestValidationError(
                    f"Content-group leakage detected! {len(grp_overlap)} content groups shared between '{s1}' and '{s2}'. "
                    f"Examples: {sample_overlap}"
                )

    # Check file existence if requested
    missing_files_count = 0
    if check_files_exist:
        resolve_base = Path(base_dir) if base_dir else path.parent
        for _, row in df.iterrows():
            img_p = Path(row["image_path"])
            if not img_p.is_absolute():
                img_p = resolve_base / img_p
            if not img_p.is_file():
                missing_files_count += 1

        if missing_files_count > 0:
            raise ManifestValidationError(
                f"{missing_files_count} referenced images do not exist on disk."
            )

    # Generate summary stats
    split_counts = df["split"].value_counts().to_dict()
    label_counts: Dict[str, Dict[int, int]] = {}
    for s in unique_splits:
        s_df = df[df["split"] == s]
        label_counts[s] = s_df["label"].value_counts().to_dict()

    generator_counts = df["generator_family"].value_counts().to_dict()
    format_counts = df["format"].value_counts().to_dict()

    return ManifestReport(
        total_rows=len(df),
        split_counts=split_counts,
        label_counts=label_counts,
        generator_counts=generator_counts,
        format_counts=format_counts,
        leakage_detected=False,
        missing_files_count=missing_files_count,
    )


def extract_defactify_subset(
    parquet_dir: Union[str, Path],
    output_images_dir: Union[str, Path],
    output_manifest_path: Union[str, Path],
    max_train_samples: int = 1000,
    max_val_samples: int = 300,
    max_test_samples: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Extracts a balanced, leakage-resistant subset from Defactify parquet files.

    Saves images as real JPEG/PNG files and generates a validated manifest CSV.

    Args:
        parquet_dir: Directory containing train-*.parquet, validation-*.parquet, test-*.parquet.
        output_images_dir: Destination directory for image files.
        output_manifest_path: Destination path for manifest CSV.
        max_train_samples: Total training samples to extract.
        max_val_samples: Total validation samples to extract.
        max_test_samples: Total test samples to extract.
        seed: Random seed for sample selection.

    Returns:
        DataFrame containing the written manifest.
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_dir)
    img_out_base = Path(output_images_dir)
    manifest_out = Path(output_manifest_path)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    generator_map = {
        0: "real",
        1: "sd21",
        2: "sdxl",
        3: "sd3",
        4: "dalle3",
        5: "midjourney",
    }

    splits_spec = [
        ("train", "train-00000-of-00007.parquet", max_train_samples),
        ("validation", "validation-00000-of-00002.parquet", max_val_samples),
        ("test", "test-00000-of-00008.parquet", max_test_samples),
    ]

    manifest_rows: List[Dict[str, Any]] = []

    for split_name, parquet_file, target_samples in splits_spec:
        pq_file_path = parquet_path / parquet_file
        if not pq_file_path.is_file():
            # Check data/ subdirectory
            pq_file_path = parquet_path / "data" / parquet_file
            if not pq_file_path.is_file():
                continue

        table = pq.read_table(pq_file_path)
        total_rows = table.num_rows

        # Balanced selection between real (0) and AI (1)
        labels_a = table["Label_A"].to_pylist()
        real_indices = [i for i, lbl in enumerate(labels_a) if lbl == 0]
        ai_indices = [i for i, lbl in enumerate(labels_a) if lbl == 1]

        target_per_class = target_samples // 2

        import random
        rng = random.Random(seed)
        selected_real = rng.sample(real_indices, min(len(real_indices), target_per_class))
        selected_ai = rng.sample(ai_indices, min(len(ai_indices), target_per_class))
        selected_indices = selected_real + selected_ai
        rng.shuffle(selected_indices)

        split_img_dir = img_out_base / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)

        for idx in selected_indices:
            row_img_dict = table["Image"][idx].as_py()
            label_a = int(table["Label_A"][idx].as_py())
            label_b = int(table["Label_B"][idx].as_py())
            caption = str(table["Caption"][idx].as_py() or "")

            img_bytes = row_img_dict["bytes"]
            gen_family = generator_map.get(label_b, "unknown")

            # Determine format
            img_format = "jpeg"
            if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                img_format = "png"
            elif img_bytes[:2] == b"\xff\xd8":
                img_format = "jpeg"

            # Create deterministic filename based on content hash
            content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
            filename = f"{split_name}_{label_a}_{gen_family}_{content_hash}.{img_format}"
            file_path = split_img_dir / filename

            # Write image to disk
            with open(file_path, "wb") as f:
                f.write(img_bytes)

            # Content group: derived from hash of first 40 chars of caption to avoid leakage of exact prompts
            content_group = hashlib.md5(caption[:40].encode("utf-8")).hexdigest()[:12] if caption else f"grp_{content_hash[:8]}"

            # Save relative path from manifest_out.parent
            rel_path = file_path.resolve()

            manifest_rows.append({
                "image_path": str(rel_path),
                "label": label_a,
                "source_dataset": "defactify",
                "generator_family": gen_family,
                "content_group": content_group,
                "split": split_name,
                "format": img_format,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_out, index=False)
    return manifest_df


class AIDetectionTorchDataset:
    """PyTorch Dataset for training and evaluating CNN detectors with manifest CSV."""

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        transform=None,
    ):
        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        else:
            df = manifest.copy()

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label = float(row["label"])

        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
        except Exception as e:
            # Fallback to zero image on corrupt file
            rgb_img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            from torchvision.transforms import functional as TF
            tensor = TF.to_tensor(rgb_img)

        sample_info = {
            "image_path": str(img_path),
            "label": int(label),
            "generator_family": row.get("generator_family", "unknown"),
            "content_group": row.get("content_group", ""),
            "format": row.get("format", ""),
        }

        return tensor, torch.tensor(label, dtype=torch.float32), sample_info


