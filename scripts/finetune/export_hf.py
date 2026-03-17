import argparse
import subprocess
from pathlib import Path

from audiocraft.utils import export


def main() -> None:
    """
    Export a trained AudioCraft checkpoint and upload it to Hugging Face Hub.

    We use this script to convert the training checkpoint into the inference
    format expected by MusicGen and then publish the exported files to HF.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig", type=str, required=True, help="You can find it in audiocraft_runs/xps")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="hf_model/musicgen_ft")
    parser.add_argument("--runs_dir", type=str, default="audiocraft_runs/xps")
    args = parser.parse_args()

    xp_folder = Path(args.runs_dir) / args.sig
    checkpoint_path = xp_folder / "checkpoint.th"
    out_dir = Path(args.out_dir)

    if not xp_folder.exists():
        raise FileNotFoundError(f"XP folder not found: {xp_folder}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint.th not found: {checkpoint_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    lm_out = out_dir / "state_dict.bin"
    compression_out = out_dir / "compression_state_dict.bin"

    export.export_lm(checkpoint_path, lm_out)
    export.export_pretrained_compression_model("facebook/encodec_32khz", compression_out)

    cmd = ["hf", "upload", args.repo_id, str(out_dir), "."]
    subprocess.run(cmd, check=True)

    print(f"hf.py: done -- https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()