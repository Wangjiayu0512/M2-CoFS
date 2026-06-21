import os
import shutil
from pathlib import Path


# =========================
# 修改这里
# =========================
src_root = Path(r"TrainingData")
dst_root = Path(r"test_img/whole")

train_ids = ["1", "2", "3", "5"]
val_ids = ["4"]

# 根据你的原始文件名关键词匹配
# 如果你的文件名不是这些关键词，改这里即可
modality_patterns = {
    "t1": ["t1", "T1"],
    "t2-flair": ["flair", "FLAIR", "t2-flair", "T2-FLAIR"],
    "label": ["label", "seg", "mask", "Seg", "Label"]
}


def make_dirs():
    for split in ["training", "validation"]:
        for folder in ["t1", "t2-flair", "label"]:
            (dst_root / split / folder).mkdir(parents=True, exist_ok=True)


def find_file(case_dir, keywords):
    """
    在一个病例文件夹中，根据关键词查找 nii 或 nii.gz 文件
    """
    nii_files = list(case_dir.rglob("*.nii")) + list(case_dir.rglob("*.nii.gz"))

    matched_files = []
    for file in nii_files:
        name = file.name
        for kw in keywords:
            if kw in name:
                matched_files.append(file)
                break

    if len(matched_files) == 0:
        raise FileNotFoundError(
            f"在 {case_dir} 中没有找到关键词 {keywords} 对应的文件"
        )

    if len(matched_files) > 1:
        print(f"[Warning] 在 {case_dir} 中找到多个匹配文件：")
        for f in matched_files:
            print("   ", f)
        print(f"默认使用第一个：{matched_files[0]}")

    return matched_files[0]


def get_suffix(file_path):
    """
    兼容 .nii 和 .nii.gz
    """
    name = file_path.name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    elif name.endswith(".nii"):
        return ".nii"
    else:
        return file_path.suffix


def copy_case(case_id, split):
    case_dir = src_root / case_id

    if not case_dir.exists():
        raise FileNotFoundError(f"病例文件夹不存在：{case_dir}")

    print(f"\nProcessing case {case_id} -> {split}")

    for modality, keywords in modality_patterns.items():
        src_file = find_file(case_dir, keywords)
        suffix = get_suffix(src_file)

        dst_file = dst_root / split / modality / f"{case_id}{suffix}"

        shutil.copy2(src_file, dst_file)

        print(f"  {modality}:")
        print(f"    from: {src_file}")
        print(f"    to  : {dst_file}")


def main():
    make_dirs()

    for case_id in train_ids:
        copy_case(case_id, "training")

    for case_id in val_ids:
        copy_case(case_id, "validation")

    print("\nDataset organization finished!")
    print(f"Saved to: {dst_root}")


if __name__ == "__main__":
    main()