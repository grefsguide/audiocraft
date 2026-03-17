import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Tuple

import requests

SCHEMA = {
    "name": "music_metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "One short generation-ready sentence describing audible musical content only, in a clean text-to-music prompt style."
            },
            "general_mood": {
                "type": "string",
                "description": "A short comma-separated phrase with 3 to 5 musically meaningful mood descriptors."
            },
            "genre_tags": {
                "type": "array",
                "description": "1 to 5 normalized genre or style tags, ordered from more specific to more general. Use musically useful conditioning tags only. Tags like Soundtrack and Instrumental are allowed when they are clearly useful style labels.",
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "maxItems": 5
            },
            "lead_instrument": {
                "type": "string",
                "description": "The dominant melodic, harmonic, vocal, or timbral driver."
            },
            "accompaniment": {
                "type": "string",
                "description": "Supporting instruments, backing layers, drums, or textures in one concise musical phrase."
            },
            "tempo_and_rhythm": {
                "type": "string",
                "description": "Tempo, pulse, groove, or meter in concise musical wording; do not invent exact BPM unless explicitly stated."
            },
            "vocal_presence": {
                "type": "string",
                "description": "Short musical wording describing vocals or their absence, for example None, Male vocals, Female vocals, Mixed vocals, Choir, Background choir, Wordless choir, Vocal samples, Chopped vocal samples, or Unknown."
            },
            "production_quality": {
                "type": "string",
                "description": "Sonic character, recording style, ambience, or mix texture useful for music generation, in one concise phrase, for example high fidelity, wide stereo, warm tape saturation, natural room acoustics, club-ready mix, or retro 80s texture."
            }
        },
        "required": [
            "description",
            "general_mood",
            "genre_tags",
            "lead_instrument",
            "accompaniment",
            "tempo_and_rhythm",
            "vocal_presence",
            "production_quality"
        ],
        "additionalProperties": False
    }
}


SYSTEM_PROMPT = """
You convert short raw music captions into strict structured JSON metadata for text-to-music training.

Goal:
Produce compact, musically useful, generation-oriented metadata that helps a text-to-music model generate better audio.
The output should sound like clean conditioning text: short, concrete, musical, and reusable.

Core rules:
1. Output only valid JSON matching the schema.
2. Do not add fields outside the schema.
3. Preserve only audible or strongly implied musical information from the raw caption.
4. Do not invent story, scene, artist, brand, era, language, lyrics, BPM, key, or song structure unless clearly stated.
5. Remove hashtags, usernames, emojis, file names, video/platform context, and irrelevant text.
6. Prefer broad but musically useful labels over "Unknown" whenever safe.
7. Keep wording standardized across examples.
8. Be musically informative, not literary.

Priority information:
- genre and style
- dominant lead sound source
- accompaniment and backing layers
- tempo, groove, pulse, or meter
- vocal type or absence of vocals
- sonic texture, recording character, and production cues
- mood only when musically relevant

General style rules:
- The metadata should read like clean conditioning text for a text-to-music model.
- Prefer short reusable phrases over creative paraphrasing.
- Use concise musical wording.
- Similar captions should produce similarly phrased outputs.
- Better broad and useful than overly specific and wrong.

"description":
- Write exactly one short, clean, generation-ready sentence.
- Target length: 18 to 40 words.
- It should read like a good text-to-music prompt.
- Preferred order:
  style/genre -> lead element -> accompaniment -> tempo/rhythm -> vocals -> production cue
- Use only audible or strongly implied musical information.
- Avoid empty words like amazing, beautiful, cool, awesome.
- Do not mention tutorial, demonstration, teaching, explanation, or platform context unless it clearly changes the audible musical sound.

"general_mood":
- Return a short comma-separated phrase with 3 to 5 musically meaningful mood descriptors.
- Use only musical or emotional descriptors such as:
  energetic, uplifting, melancholic, dark, warm, intimate, dreamy, aggressive, epic, peaceful, tense, nostalgic, playful, reflective, soulful, raw, vibrant
- Do not include visual story elements.
- Do not include functional or social descriptors such as:
  informative, educational, engaging, instructional, urban
- Do not over-dramatize.

"genre_tags":
- Return 1 to 5 short normalized genre/style tags.
- Prefer tags from more specific to more general.
- Use musically useful conditioning tags only.
- Good examples:
  Progressive House, Lo-Fi Hip Hop, Boom Bap, Orchestral, Cinematic, Soundtrack, Folk, Acoustic, Indie, Synthwave, Darkwave, Ambient, Trap, Jazz, Rock, EDM, Ballad, Soul, R&B, Gospel, Choral, Instrumental
- Avoid duplicate or near-duplicate tags.
- Do not invent niche subgenres unless clearly supported.
- Do not use non-musical metadata tags such as:
  Tutorial, Cover, Religious, Spiritual, Vocal, Live
- Soundtrack is allowed when it functions as a musical style label.
- Instrumental is allowed only when it is musically useful and there are no vocals.
- If vocal_presence is not "None", do not use "Instrumental" in genre_tags.

"lead_instrument":
- Name exactly one dominant melodic, harmonic, vocal, or timbral driver.
- This field may contain a vocal source if vocals are clearly the lead element.
- It must describe one main leading sound source only, not multiple sources joined together.
- Good examples:
  Fingerpicked acoustic guitar
  Bright detuned synth lead
  Distorted analog bass synth
  Muted trumpet
  Rhodes electric piano
  Male lead vocals
  Female lead vocals
  Choir
- If unclear, use "Unknown".

"accompaniment":
- Describe the supporting musical layers in one concise phrase.
- Focus on instruments, drums, textures, or backing layers.
- Do not repeat the lead element unless needed.
- Good examples:
  Sweeping strings, cinematic percussion, and timpani
  Dusty vinyl crackle, sub-bass, and soft boom-bap drums
  Arpeggiated synth plucks and retro drum machine
  Light tambourine, upright bass, and room ambience
- Do not describe video context or non-audible information.

"tempo_and_rhythm":
- Describe speed and rhythmic feel, not exact BPM unless explicitly stated.
- Prefer short musical phrases like:
  Slow, laid-back groove
  Mid-tempo, steady pulse
  Fast, driving four-on-the-floor beat
  Gentle waltz-like triple meter
  March-like rhythm
  Driving, mid-tempo pulse
  Slow, free rhythm
  Upbeat, syncopated groove
- If rhythm is unclear, still prefer a broad useful label over "Unknown" when possible.
- Avoid vague phrases like:
  varying tempos
  different rhythms
  changing beat
- Use more musical wording whenever possible.

"vocal_presence":
- Use exactly one short standardized musical value.
- Preferred values:
  None
  Male vocals
  Female vocals
  Mixed vocals
  Choir
  Background choir
  Wordless choir
  Vocal samples
  Chopped vocal samples
  Unknown
- Use None when there are no vocals.
- Do not use Instrumental in this field.
- Do not output multiple values in this field.
- Keep special vocal roles concise.

"production_quality":
- Describe sonic character, recording style, ambience, or mix texture in one concise phrase.
- Make it useful for generation.
- Good examples:
  High fidelity, wide stereo, large hall reverb
  Lo-fi, dusty, warm tape saturation
  Raw acoustic recording, natural room ambience
  Club-ready, punchy, sidechained
  Retro 80s texture, compressed, synthetic
  Clean studio, polished, wide stereo
  Raw, noisy recording
  Lo-fi, hissy, narrow-band
- Avoid weak generic labels such as:
  Poor audio quality
  Mediocre audio quality
- Prefer musically useful wording over generic judgment.
- Avoid over-describing defects.
- Use "Unknown" only when truly unclear.

Normalization rules:
- "no vocals", "without vocals", "instrumental only" -> "None"
- "female singer", "woman singing", "girl vocals" -> "Female vocals"
- "male singer", "man singing" -> "Male vocals"
- "male and female vocals", "female lead with male backing vocals" -> "Mixed vocals"
- "choir", "choral", "chanting choir" -> "Choir" or "Background choir"
- "wordless choir", "wordless chanting" -> "Wordless choir"
- "vocal chops", "chopped vox", "sampled vocals" -> "Vocal samples" or "Chopped vocal samples"
- "lofi", "lo-fi", "dusty", "cassette", "vinyl crackle" -> production should reflect lo-fi texture
- "sidechained", "pumping" -> mention in production_quality or accompaniment if musically relevant
- "live", "crowd", "stage", "concert hall" -> mention room/live character only if clearly supported
- If vocals are the main lead element, lead_instrument may be:
  Male lead vocals
  Female lead vocals
  Choir

Inference policy:
- Infer only what is audible or strongly implied.
- Never invent exact BPM, key, language, singer identity, or harmonic structure unless explicit.
- If the caption is underspecified, stay broad and safe rather than hallucinating detail.
- Better broad and musically useful than falsely specific.
- Prefer reusable conditioning phrases over creative paraphrasing.

Consistency policy:
- Similar inputs should produce similarly phrased outputs.
- Prefer the same normalized vocabulary for repeated concepts.
- Keep each field concise, musically informative, and useful for conditioning a text-to-music model.
"""


def normalize_text(value: Any, default: str = "Unknown") -> str:
    """
    Normalize one value into a clean non-empty text string.

    We use this to keep structured metadata fields consistent before saving
    them into JSON sidecar files for training.
    """
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def normalize_payload(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize one LLM response against our metadata schema.

    This step protects the pipeline from malformed JSON, duplicate tags, and
    inconsistent values before the metadata is written next to each WAV file.
    """
    required = [
        "description",
        "general_mood",
        "genre_tags",
        "lead_instrument",
        "accompaniment",
        "tempo_and_rhythm",
        "vocal_presence",
        "production_quality",
    ]

    for key in required:
        if key not in obj:
            raise ValueError(f"Missing key: {key}")

    if not isinstance(obj["genre_tags"], list):
        raise ValueError("genre_tags must be a list")

    for key in required:
        if key != "genre_tags":
            obj[key] = normalize_text(obj[key])

    tags: list[str] = []
    seen: set[str] = set()

    for value in obj["genre_tags"]:
        tag = normalize_text(value, default="").strip()
        if not tag:
            continue

        tag_key = tag.casefold()
        if tag_key in seen:
            continue

        seen.add(tag_key)
        tags.append(tag)

    if not tags:
        tags = ["Unknown"]

    obj["genre_tags"] = tags[:5]

    words = obj["description"].split()
    if len(words) < 8:
        raise ValueError(f"description too short: {obj['description']}")
    if len(words) > 60:
        obj["description"] = " ".join(words[:60]).strip()

    return obj


def append_jsonl(path: Path, record: dict[str, Any], lock: Lock) -> None:
    """
    Append one JSON record to a shared JSONL log file in a thread-safe way.

    We use this for failure logs so parallel workers can safely report errors
    without corrupting the output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with lock:
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def openrouter_request(api_key: str, model: str, caption: str) -> dict[str, Any]:
    """
    Send one raw MusicCaps caption to OpenRouter and parse the structured reply.

    This is the core LLM enrichment call that converts plain captions into the
    JSON metadata schema used for structured conditioning.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "music-caption-structuring",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Raw caption:\n{caption}\n\nReturn only valid JSON."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": SCHEMA,
        },
        "plugins": [
            {"id": "response-healing"},
        ],
        "provider": {
            "require_parameters": True,
        },
        "temperature": 0.1,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError(f"Unexpected message content type: {type(content)}")

    obj = json.loads(content)
    return normalize_payload(obj)


def call_openrouter(
    api_key: str,
    model: str,
    caption: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Call OpenRouter with retry logic for transient API or parsing failures.

    We need retries because large batch enrichment can fail on rate limits,
    temporary server errors, or occasional malformed responses.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return openrouter_request(api_key=api_key, model=model, caption=caption)
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {429, 500, 502, 503, 504}:
                raise
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc

        sleep_s = min(2 ** attempt, 20)
        print(f"retry {attempt + 1}/{max_retries} after error: {last_error}")
        time.sleep(sleep_s)

    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries: {last_error}")


def process_source_file(
    src_path: Path,
    api_key: str,
    model: str,
    max_retries: int,
    failed_log_path: Path,
    failed_lock: Lock,
) -> Tuple[str, str]:
    """
    Enrich one source metadata file with structured LLM-generated fields.

    This function reads the raw caption, requests structured metadata from
    OpenRouter, writes the final JSON next to the audio file, and logs errors
    if processing fails.
    """
    try:
        payload = json.loads(src_path.read_text(encoding="utf-8"))
        caption = payload["raw_caption"]

        final_json_path = src_path.with_suffix("").with_suffix(".json")
        if final_json_path.exists():
            return "skip", str(final_json_path)

        enriched = call_openrouter(
            api_key=api_key,
            model=model,
            caption=caption,
            max_retries=max_retries,
        )

        final_json_path.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return "ok", str(final_json_path)

    except Exception as exc:
        append_jsonl(
            failed_log_path,
            {
                "source_file": str(src_path),
                "error": str(exc),
            },
            lock=failed_lock,
        )
        return "fail", f"{src_path.name}: {exc}"


def main() -> None:
    """
    Enrich all downloaded MusicCaps items with structured metadata in parallel.

    This is the batch entry point for the LLM stage: it scans source JSON files,
    sends raw captions to OpenRouter, writes final sidecar metadata, and stores
    failure logs for later reruns.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/musiccaps_struct")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_files = sorted(data_root.rglob("*.source.json"))

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failed_log_path = logs_dir / "openrouter_failed.jsonl"

    failed_lock = Lock()
    total = len(source_files)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_source_file,
                src_path,
                args.api_key,
                args.model,
                args.max_retries,
                failed_log_path,
                failed_lock,
            ): src_path
            for src_path in source_files
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            src_path = futures[future]
            try:
                status, message = future.result()
                if status == "ok":
                    print(f"[{idx}/{total}] OK -- {message}")
                elif status == "skip":
                    print(f"[{idx}/{total}] Skip existing -- {message}")
                else:
                    print(f"[{idx}/{total}] FAIL -- {message}")
            except Exception as exc:
                print(f"[{idx}/{total}] CRASH -- {src_path.name}: {exc}")


if __name__ == "__main__":
    main()