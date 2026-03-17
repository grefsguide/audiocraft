import argparse
import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset


log_lock = threading.Lock()


def run_cmd(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """
    Run one external shell command and capture its output.

    We use this helper for yt-dlp and ffmpeg calls so that download errors can
    be handled in Python and written to logs instead of crashing the whole run.
    """
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def get_audio_url(
    youtube_url: str,
    yt_dlp_path: str,
    cookies_file: str,
    js_runtimes: str = "deno",
    extractor_args: str = "youtube:player_client=default",
    ytdlp_timeout: int = 120,
) -> str:
    """
    Resolve a direct audio stream URL for one YouTube video.

    We use yt-dlp in URL-only mode because the assignment requires downloading
    only the needed audio fragment instead of the full video file.
    """
    cmd = [
        yt_dlp_path,
        "--cookies",
        cookies_file,
        "--js-runtimes",
        js_runtimes,
        "--extractor-args",
        extractor_args,
        "-g",
        "-f",
        "bestaudio/best",
        youtube_url,
    ]

    result = run_cmd(cmd, timeout=ytdlp_timeout)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Empty direct media URL from yt-dlp")

    return lines[0]


def download_clip(
    direct_url: str,
    start_s: float,
    out_wav: Path,
    ffmpeg_path: str,
    ffmpeg_timeout: int = 120,
) -> None:
    """
    Download and convert one 10-second audio fragment into WAV format.

    We call ffmpeg directly on the stream URL so we save only the required clip
    for MusicCaps instead of downloading full media files.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    temp_wav = out_wav.with_suffix(".part.wav")
    if temp_wav.exists():
        temp_wav.unlink()

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(start_s),
        "-i",
        direct_url,
        "-t",
        "10",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "32000",
        "-sample_fmt",
        "s16",
        str(temp_wav),
    ]

    result = run_cmd(cmd, timeout=ffmpeg_timeout)
    if result.returncode != 0:
        if temp_wav.exists():
            temp_wav.unlink()
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    if not temp_wav.exists() or temp_wav.stat().st_size == 0:
        if temp_wav.exists():
            temp_wav.unlink()
        raise RuntimeError("ffmpeg finished but output file is missing or empty")

    temp_wav.replace(out_wav)


def process_row(
    row: Dict[str, Any],
    idx: int,
    total: int,
    args: argparse.Namespace,
    out_root: Path,
    failed_log_path: Path,
) -> str:
    """
    Process one MusicCaps row from start to finish.

    This function resolves the stream URL, downloads the audio clip, saves the
    raw source metadata, and records failures so the full dataset build can
    continue even if some videos are unavailable.
    """
    ytid = row["ytid"]
    caption = row["caption"]
    start_s = float(row["start_s"])
    end_s = float(row.get("end_s", start_s + 10))
    musiccaps_id = row.get("musiccaps_id", idx)

    base_name = f"{musiccaps_id}_{ytid}"
    wav_path = out_root / f"{base_name}.wav"
    source_json_path = out_root / f"{base_name}.source.json"

    if wav_path.exists() and source_json_path.exists():
        return f"[{idx + 1}/{total}] skip existing: {wav_path.name}"

    youtube_url = f"https://www.youtube.com/watch?v={ytid}"

    try:
        if args.sleep > 0:
            time.sleep(random.uniform(0.0, args.sleep))

        direct_url = get_audio_url(
            youtube_url=youtube_url,
            yt_dlp_path=args.yt_dlp_path,
            cookies_file=args.cookies_file,
            js_runtimes=args.js_runtimes,
            extractor_args=args.extractor_args,
            ytdlp_timeout=args.ytdlp_timeout,
        )

        download_clip(
            direct_url=direct_url,
            start_s=start_s,
            out_wav=wav_path,
            ffmpeg_path=args.ffmpeg_path,
            ffmpeg_timeout=args.ffmpeg_timeout,
        )

        source_payload = {
            "musiccaps_id": musiccaps_id,
            "ytid": ytid,
            "youtube_url": youtube_url,
            "start_s": start_s,
            "end_s": end_s,
            "raw_caption": caption,
        }
        source_json_path.write_text(
            json.dumps(source_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return f"[{idx + 1}/{total}] OK -- {wav_path}"

    except Exception as exc:
        record = {
            "musiccaps_id": musiccaps_id,
            "ytid": ytid,
            "youtube_url": youtube_url,
            "start_s": start_s,
            "error": str(exc),
        }
        with log_lock:
            with failed_log_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
        return f"[{idx + 1}/{total}] FAIL -- {ytid}: {exc}"


def main() -> None:
    """
    Download raw MusicCaps audio clips into the local project folder.

    This is the entry point for dataset collection: it loads MusicCaps from
    Hugging Face, runs parallel downloads, and stores WAV files together with
    source metadata needed for later enrichment and splitting.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--sleep", type=float, default=0.5)

    parser.add_argument("--yt_dlp_path", type=str, required=True)
    parser.add_argument("--cookies_file", type=str, required=True)
    parser.add_argument("--js_runtimes", type=str, required=True)
    parser.add_argument("--ffmpeg_path", type=str, required=True)

    parser.add_argument("--extractor_args", type=str, default="youtube:player_client=default")
    parser.add_argument("--ytdlp_timeout", type=int, default=600)
    parser.add_argument("--ffmpeg_timeout", type=int, default=600)

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failed_log_path = logs_dir / "failed_downloads.jsonl"

    print("yt-dlp exists:", Path(args.yt_dlp_path).exists(), args.yt_dlp_path)
    print("cookies exists:", Path(args.cookies_file).exists(), args.cookies_file)
    print("ffmpeg exists:", Path(args.ffmpeg_path).exists(), args.ffmpeg_path)
    print("js_runtimes:", args.js_runtimes)

    dataset = load_dataset("google/MusicCaps", split="train")
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    rows = list(dataset)
    total = len(rows)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_row, row, idx, total, args, out_root, failed_log_path)
            for idx, row in enumerate(rows)
        ]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()