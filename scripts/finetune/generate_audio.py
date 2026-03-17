import argparse
import json
from pathlib import Path
from typing import Any

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from huggingface_hub import snapshot_download


FIELD_ORDER = [
    "description",
    "general_mood",
    "genre_tags",
    "lead_instrument",
    "accompaniment",
    "tempo_and_rhythm",
    "vocal_presence",
    "production_quality",
]


def struct_to_text(obj: dict[str, Any]) -> str:
    """
    Convert one structured prompt object into one flat text prompt.

    We trained the model with structured metadata merged into text, so for
    inference we rebuild the same textual format from the JSON fields.
    """
    parts = []

    for key in FIELD_ORDER:
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        value = str(value).strip()
        if value:
            parts.append(f"{key}: {value}")

    return ". ".join(parts)


def main() -> None:
    """
    Generate WAV files from structured JSON prompts with a fine-tuned MusicGen model.

    This script loads either a local exported model or a Hugging Face repo,
    converts structured prompts into text conditioning, and saves the generated
    audio files into the output folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="greygreef/musicgen-small_ft", help="Local exported folder or HF repo_id")
    parser.add_argument("--prompts_json", type=str, default="data/gen_sound/prompts.json")
    parser.add_argument("--out_dir",type=str, default="data/gen_sound")
    parser.add_argument("--prefix", type=str, default="")

    parser.add_argument("--num_variants", type=int, default=1)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=250)

    parser.add_argument("--cfg_coef", type=float, default=3.0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    args = parser.parse_args()

    model_path = args.model_path
    local_model_path = Path(model_path)

    if not local_model_path.exists():
        model_path = snapshot_download(repo_id=model_path)

    model = MusicGen.get_pretrained(str(model_path))
    model.set_generation_params(
        duration=args.duration,
        use_sampling=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        cfg_coef=args.cfg_coef,
        extend_stride=9,
    )

    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts, start=1):
        prompt_text = struct_to_text(prompt)
        texts = [prompt_text] * args.num_variants

        print(f"Generating prompt {i} with {args.num_variants} variants...")
        wavs = model.generate(texts)

        for j, wav in enumerate(wavs, start=1):
            if args.num_variants == 1:
                out_stem = out_dir / f"{args.prefix}prompt_{i}"
            else:
                out_stem = out_dir / f"{args.prefix}prompt_{i}_var_{j}"
            audio_write(
                str(out_stem),
                wav.cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )
            print(f"Saved: {out_stem}.wav")


if __name__ == "__main__":
    main()