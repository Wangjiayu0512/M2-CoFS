# check_val_label_info.py

import os
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageSequence


SUPPORTED_EXTS = [
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff",
    ".npy", ".npz",
    ".nii", ".nii.gz"
]


CLASS_NAMES = {
    0: "BK",
    1: "CSF",
    2: "GM",
    3: "WM",
    4: "MS",
}


def is_supported_file(path):
    name = path.name.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTS)


def load_label(path):
    """
    支持读取：
    1. png / jpg / bmp
    2. tif / tiff，包括多页 tiff
    3. npy / npz
    4. nii / nii.gz，需要安装 nibabel
    """
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".npy"):
        arr = np.load(path)
        return arr

    if name.endswith(".npz"):
        data = np.load(path)
        key = list(data.keys())[0]
        arr = data[key]
        return arr

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "读取 .nii/.nii.gz 需要安装 nibabel: pip install nibabel"
            )
        img = nib.load(str(path))
        arr = img.get_fdata()
        return arr

    if name.endswith(".tif") or name.endswith(".tiff"):
        img = Image.open(path)
        frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]

        if len(frames) == 1:
            return frames[0]
        else:
            return np.stack(frames, axis=0)

    img = Image.open(path)
    arr = np.array(img)
    return arr


def maybe_convert_onehot_to_label(arr, expected_labels):
    """
    如果 label 已经是 one-hot，转成类别图。
    常见情况：
    [D, H, W, C]
    [C, D, H, W]
    """
    arr = np.asarray(arr)

    num_classes = len(expected_labels)

    if arr.ndim == 4:
        unique_vals = np.unique(arr)

        is_binary = set(unique_vals.tolist()).issubset({0, 1})

        if is_binary and arr.shape[-1] == num_classes:
            arr = np.argmax(arr, axis=-1)
            return arr, "onehot_last_dim"

        if is_binary and arr.shape[0] == num_classes:
            arr = np.argmax(arr, axis=0)
            return arr, "onehot_first_dim"

    return arr, "scalar"


def is_rgb_label(arr):
    """
    判断是否可能是 RGB 彩色标签。
    注意：
    2D RGB label 通常是 [H, W, 3] 或 [H, W, 4]
    3D 医学 label 通常是 [D, H, W]，不会满足最后一维为 3/4 且 ndim=3 的情况。
    """
    return arr.ndim == 3 and arr.shape[-1] in [3, 4]


def format_counter(counter):
    return "; ".join([f"{k}:{v}" for k, v in sorted(counter.items())])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="验证集 label 文件夹路径，例如 ./dataset/validation/label"
    )
    parser.add_argument(
        "--expected",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="期望的类别值，默认 0 1 2 3 4"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="label_report.csv",
        help="保存统计结果的 csv 文件名"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否打印每个文件的详细信息"
    )

    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    expected_labels = set(args.expected)

    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    label_paths = sorted([
        p for p in label_dir.rglob("*")
        if p.is_file() and is_supported_file(p)
    ])

    if len(label_paths) == 0:
        raise RuntimeError(f"No supported label files found in: {label_dir}")

    print("=" * 80)
    print(f"Label dir: {label_dir}")
    print(f"Number of label files: {len(label_paths)}")
    print(f"Expected labels: {sorted(expected_labels)}")
    print("=" * 80)

    global_counter = Counter()
    shape_counter = Counter()
    dtype_counter = Counter()
    bad_files = []
    rgb_files = []
    onehot_files = []

    rows = []

    for idx, path in enumerate(label_paths):
        try:
            arr = load_label(path)
        except Exception as e:
            bad_files.append((str(path), f"read_error: {e}"))
            continue

        raw_shape = tuple(arr.shape)
        raw_dtype = str(arr.dtype)

        if is_rgb_label(arr):
            rgb_files.append(str(path))

            flat_color = arr.reshape(-1, arr.shape[-1])
            unique_colors = np.unique(flat_color, axis=0)
            num_unique_colors = unique_colors.shape[0]

            rows.append({
                "file": str(path),
                "raw_shape": raw_shape,
                "used_shape": raw_shape,
                "dtype": raw_dtype,
                "mode": "rgb_or_rgba_label",
                "min": "",
                "max": "",
                "unique_values": f"{num_unique_colors} unique colors",
                "class_counts": "",
                "bad_values": "RGB label, not scalar class index",
            })

            bad_files.append((str(path), "RGB/RGBA label detected, expected scalar label values 0~4"))

            if args.verbose:
                print(f"[{idx+1}/{len(label_paths)}] {path.name}")
                print(f"  RGB/RGBA label detected. shape={raw_shape}, dtype={raw_dtype}")
                print(f"  unique colors number: {num_unique_colors}")

            continue

        arr, mode = maybe_convert_onehot_to_label(arr, expected_labels)

        if mode != "scalar":
            onehot_files.append(str(path))

        arr = np.asarray(arr)

        # 如果是 float，但实际是 0.0/1.0/2.0 这种整数标签，转成 int 检查
        if np.issubdtype(arr.dtype, np.floating):
            if np.allclose(arr, np.round(arr)):
                arr = np.round(arr).astype(np.int64)

        used_shape = tuple(arr.shape)
        shape_counter[used_shape] += 1
        dtype_counter[raw_dtype] += 1

        unique_vals, counts = np.unique(arr, return_counts=True)
        unique_vals_list = unique_vals.tolist()
        counts_list = counts.tolist()

        file_counter = Counter({
            int(k): int(v)
            for k, v in zip(unique_vals_list, counts_list)
            if isinstance(k, (int, np.integer)) or float(k).is_integer()
        })

        global_counter.update(file_counter)

        current_values = set([int(v) for v in unique_vals_list])
        bad_values = sorted(list(current_values - expected_labels))

        if len(bad_values) > 0:
            bad_files.append((str(path), f"unexpected values: {bad_values}"))

        min_val = np.min(arr)
        max_val = np.max(arr)

        class_counts_str = "; ".join([
            f"{c}({CLASS_NAMES.get(c, 'UNK')}):{file_counter.get(c, 0)}"
            for c in sorted(expected_labels)
        ])

        rows.append({
            "file": str(path),
            "raw_shape": raw_shape,
            "used_shape": used_shape,
            "dtype": raw_dtype,
            "mode": mode,
            "min": min_val,
            "max": max_val,
            "unique_values": unique_vals_list,
            "class_counts": class_counts_str,
            "bad_values": bad_values,
        })

        if args.verbose:
            print(f"[{idx+1}/{len(label_paths)}] {path.name}")
            print(f"  raw_shape: {raw_shape}, used_shape: {used_shape}, dtype: {raw_dtype}, mode: {mode}")
            print(f"  min/max: {min_val}/{max_val}")
            print(f"  unique: {unique_vals_list}")
            print(f"  counts: {class_counts_str}")
            if len(bad_values) > 0:
                print(f"  WARNING bad values: {bad_values}")

    # 保存 CSV
    save_csv = Path(args.save_csv)
    with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "file",
            "raw_shape",
            "used_shape",
            "dtype",
            "mode",
            "min",
            "max",
            "unique_values",
            "class_counts",
            "bad_values",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\nShape distribution:")
    for shape, num in shape_counter.items():
        print(f"  {shape}: {num}")

    print("\nDtype distribution:")
    for dtype, num in dtype_counter.items():
        print(f"  {dtype}: {num}")

    print("\nGlobal class distribution:")
    total_voxels = sum(global_counter.values())

    for c in sorted(expected_labels):
        count = global_counter.get(c, 0)
        ratio = count / total_voxels * 100 if total_voxels > 0 else 0
        name = CLASS_NAMES.get(c, "UNK")
        print(f"  class {c} ({name}): {count}  ({ratio:.4f}%)")

    print(f"\nTotal voxels / pixels: {total_voxels}")

    if len(onehot_files) > 0:
        print(f"\nOne-hot labels detected and converted: {len(onehot_files)}")

    if len(rgb_files) > 0:
        print(f"\nRGB/RGBA label files detected: {len(rgb_files)}")
        for p in rgb_files[:10]:
            print(f"  {p}")

    if len(bad_files) > 0:
        print("\nPotential problems:")
        for p, reason in bad_files[:30]:
            print(f"  {p}")
            print(f"    -> {reason}")

        if len(bad_files) > 30:
            print(f"  ... and {len(bad_files) - 30} more")
    else:
        print("\nNo obvious label-value problem found.")

    print(f"\nCSV report saved to: {save_csv.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()