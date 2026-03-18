# Summary

## This document summarizes the main practical outcomes of the project:

## 1. What difficulties did we face?

### 1.1 Dataset collection issues
The MusicCaps dataset does not contain raw audio files. Instead, it provides YouTube IDs and captions, so audio had to be downloaded separately.

Main issues:

- some YouTube items were unavailable or removed
- stream extraction sometimes failed in `yt-dlp`
- direct stream download had to be implemented through `yt-dlp` + `ffmpeg` without downloading full videos

### 1.2 Throughput and parallelization issues

Another practical difficulty was preprocessing speed.

The initial version of `download_musiccaps.py` was too slow when executed sequentially, because stream resolution and clip extraction for a large number of MusicCaps items took too much time on Windows.  
To make dataset collection practical, multithreading was added to the download stage.

The same idea was later applied to `musiccaps_openrouter.py` in order to speed up metadata enrichment through OpenRouter.  
However, this introduced a different trade-off: when the number of worker threads was set too high, the total annotation cost increased significantly, in our case by up to roughly **x3**.

As a result, the final pipeline used parallel processing in both scripts, but with a moderate number of workers in order to balance.

### 1.3 Metadata enrichment issues

The original MusicCaps captions are unstructured free text. For MusicGen fine-tuning, it was important to convert them into a cleaner and more consistent schema.

Main issues:

- LLM outputs were sometimes incomplete or malformed
- genre tags could be duplicated or overly noisy
- batch processing required retry logic for transient API failures

### 1.4 AudioCraft adaptation issues

AudioCraft does not natively expect the new structured fields from the custom JSON schema, so the training pipeline had to be adapted.

Main issues:

- `MusicInfo` had to be extended with new fields
- sidecar metadata loading had to be patched
- structured fields had to be merged into `description` to remain compatible with pretrained MusicGen conditioning
- required/default metadata fields had to be adjusted for MusicCaps-style sidecar JSON files

### 1.5 Windows-specific training issues

The project was developed and tested on Windows 11, while AudioCraft is significantly more Linux-oriented.

Main issues:

- `xformers` was incompatible with the chosen Windows + PyTorch stack
- several AudioCraft internals assumed Linux 
- `forkserver` multiprocessing is not supported on Windows
- Windows file replacement behavior required patching `os.rename(...)` to `os.replace(...)`
- `ffmpeg` path resolution during sample generation was less reliable than on Linux
- Comet ML required a `setuptools<82` pin because `pkg_resources` was removed from newer setuptools

### 1.6 Training efficiency

The default AudioCraft training setup is too heavy for a single consumer GPU if used unchanged.

Main issues:

- default epoch length was too large
- validation was larger than needed for a homework-scale project
- segment duration had to be aligned with the actual 10-second clips (by default its 30-second clips)
- a compact training regime was needed to make experiments practical
- increasing the number of epochs could improve the metrics, but worsen the quality of melody generation

---
## 2. Which LLM was used for parsing, and which system prompt worked best?

### LLM used

We used an LLM through **OpenRouter**.

Models used:
- `openai/gpt-4o-mini`
- `google/gemini-2.5-flash` (best results, and final run)

### Best-performing prompting strategy

The most reliable setup was:

- strict JSON schema output
- detailed field-by-field instructions
- explicit normalization rules
- compact, generation-oriented wording
- low temperature (`0.1`)
- retry logic for transient failures

The best system prompt was the long schema-guided prompt from `scripts/finetune/musiccaps_openrouter.py`, because it:

- forced valid JSON
- reduced hallucinations
- encouraged reusable conditioning-style phrases
- normalized vocal labels, genre tags, and production descriptors
- preserved only audible or strongly implied musical information

In practice, this prompt worked better than short generic prompts because it made the outputs:
- more stable,
- more homogeneous across the dataset,
- and more suitable for text-to-music conditioning.

---

## 3. Which training hyperparameters were used?

Main training setup used at the time of writing the summary :

- model: `musicgen-small`
- learning rate: `1e-5`
- batch size: `4`
- epochs: `5`
- updates per epoch: `100`
- segment duration: `10`
- valid samples: `100`
- generation samples during training: `4`
- optimizer: `AdamW`
- updates per epoch: `300`

Additional important runtime settings:

- `continue_from=//pretrained/facebook/musicgen-small`
- `transformer_lm.custom=true`
- `transformer_lm.memory_efficient=false`
- `transformer_lm.checkpointing=none`
- `efficient_attention_backend=torch`

These settings were chosen to keep training stable and feasible on a single GPU while remaining compatible with the pretrained MusicGen checkpoint.

---
## 4. Training logs

Experiment tracking was uploaded to **CometML**.

Comet run URL:
- https://www.comet.com/gaby/musicgen-finetune/49a03ac69320411fb5b760ca088e11eb
---
