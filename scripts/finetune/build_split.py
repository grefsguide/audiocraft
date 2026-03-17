import argparse
import hashlib
import json
import shutil
from pathlib import Path


def split_by_hash(key: str, valid_pct: int) -> str:
    """
    Deterministically assign one item to the train or valid split.

    We use hash-based splitting so that the same source item always goes to
    the same split across reruns, which makes dataset preparation reproducible.
    """
    hash_value = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    return "valid" if hash_value < valid_pct else "train"


def get_group_key(wav_path: Path) -> str:
    """
    Extract a stable grouping key for one audio file.

    We prefer the YouTube ID from the sidecar source JSON, because it gives us
    a consistent identity for the original MusicCaps item. If it is missing,
    we fall back to the file stem.
    """
    source_json_path = wav_path.with_suffix(".source.json")
    if source_json_path.exists():
        try:
            payload = json.loads(source_json_path.read_text(encoding="utf-8"))
            ytid = payload.get("ytid")
            if ytid:
                return str(ytid)
        except Exception:
            pass
    return wav_path.stem


def copy_if_exists(src_path: Path, dst_path: Path) -> None:
    """
    Copy a file only if it exists.

    We use this helper to move WAV files and their sidecar JSON metadata into
    the final split folders without failing on optional files.
    """
    if src_path.exists():
        shutil.copy2(src_path, dst_path)


def main() -> None:
    """
    Build train/valid folders from the raw downloaded dataset.

    This script scans all WAV files, assigns each item to a split with a stable
    hash rule, and copies the audio plus its metadata files into the final
    structured dataset layout used later by AudioCraft.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/raw",
        help="Source dataset root with wav/json files",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="data/musiccaps_struct",
        help="Destination dataset root",
    )
    parser.add_argument(
        "--valid_pct",
        type=int,
        default=20,
        help="Validation percentage, integer from 1 to 99",
    )
    args = parser.parse_args()

    if not (1 <= args.valid_pct <= 99):
        raise ValueError("--valid_pct must be between 1 and 99")

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    (dst_root / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "valid").mkdir(parents=True, exist_ok=True)

    wav_files = sorted(src_root.rglob("*.wav"))

    train_count = 0
    valid_count = 0

    for wav_path in wav_files:
        group_key = get_group_key(wav_path)
        split_name = split_by_hash(group_key, valid_pct=args.valid_pct)
        split_dir = dst_root / split_name

        json_path = wav_path.with_suffix(".json")
        source_json_path = wav_path.with_suffix(".source.json")

        copy_if_exists(wav_path, split_dir / wav_path.name)
        copy_if_exists(json_path, split_dir / json_path.name)
        copy_if_exists(source_json_path, split_dir / source_json_path.name)

        if split_name == "train":
            train_count += 1
        else:
            valid_count += 1

    print(f"train={train_count}, valid={valid_count}")
    print(f"Output: {dst_root.resolve()}")


if __name__ == "__main__":
    main()