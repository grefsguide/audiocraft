# Fine-tuning MusicGen on Structured MusicCaps

This repository contains a full pipeline for fine-tuning **MusicGen** with **structured metadata** built from the **MusicCaps** dataset

The project includes:

- raw audio download from MusicCaps / YouTube
- caption enrichment with an LLM through OpenRouter
- deterministic train/valid split building
- AudioCraft fine-tuning
- model export to Hugging Face
- prompt-based inference
- experiment logging to Comet ML

For experiment notes, difficulties, chosen hyperparameters, and reporting guidance, see [SUMMARY.md](docs/SUMMARY.md)

This project was tested on:

- Windows 11
- Python 3.9.13
- PyTorch + CUDA
- ffmpeg
- yt-dlp
- 64 GB RAM
- RTX 3080Ti 16 GB VRAM

You can see the generation result by prompts in: `data/gen_sound`
### Table of contents

1. [Environment](#1-environment)
2. [Data preparation](#2-data-preparation) 
   * [Download a prepared dataset](#21-download-a-prepared-dataset)
   * [Build the dataset from scratch](#22-build-the-dataset-from-scratch)
3. [Training](#3-training)
4. [Log the run to CometML](#4-log-the-run-to-cometml)
5. [Export model to Hugging Face](#5-export-model-to-hugging-face)
6. [Generate audio from structured prompts](#6-generate-audio-from-structured-prompts)
7. [Recommended run order](#7-recommended-run-order)


---
## 1. Environment

### Windows
Create and activate a virtual environment:
```ps1
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```
Install PyTorch with CUDA:

```ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
Install project dependencies:
```ps1
python -m pip install av==12.2.0 einops flashy hydra-core hydra_colorlog julius num2words numpy<2.0.0 sentencepiece spacy==3.7.6 huggingface_hub tqdm transformers>=4.31.0 librosa soundfile torchmetrics encodec protobuf torchdiffeq datasets yt-dlp requests orjson tensorboard comet_ml
python -m pip install -e . --no-deps
```
### Linux
Recommended for a cleaner setup than Windows

Create and activate a virtual environment:
```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```
Install ffmpeg:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```
Install PyTorch:
```bash
python -m pip install torch==2.1.0 torchvision torchaudio
```

Install project dependencies:
```bash
python -m pip install -e .
python -m pip install comet_ml datasets yt-dlp requests
```
---
## 2. Data preparation
You can either:

1. download a prepared archive into `data/`
2. build the dataset from scratch

### 2.1 Download a prepared dataset
You can skip the full dataset build and download a prepared archive from Google Drive directly into `data/`

Google Drive archive:
- https://drive.google.com/file/d/1KGR8f2D5NyOEVeMVEzov3rRtcYzaMbHA/view?usp=sharing

Install `gdown` and download:
#### Windows
```ps1
python -m pip install gdown
New-Item -ItemType Directory -Force -Path data | Out-Null
gdown --id 1KGR8f2D5NyOEVeMVEzov3rRtcYzaMbHA -O data\musiccaps_struct.zip
Expand-Archive "data\musiccaps_struct.zip" -DestinationPath "data" -Force
```
#### Linux
```bash
python -m pip install gdown
mkdir -p data
gdown --id 1KGR8f2D5NyOEVeMVEzov3rRtcYzaMbHA -O data/musiccaps_struct.zip
unzip -o data/musiccaps_struct.zip -d data
```
### 2.2 Build the dataset from scratch
#### Step 1. Download 10-sec clips and source json
##### Windows
```ps1
python scripts/finetune/download_musiccaps.py `
  --out_dir data/raw `
  --yt_dlp_path "<YOUR PATH TO yt-dlp.exe>" `
  --cookies_file "<YOUR PATH TO cookies.txt>" `
  --js_runtimes "deno:<YOUR PATH TO deno.exe>" `
  --ffmpeg_path <YOUR PATH TO ffmpeg.exe>" `
  --max_workers 4
```
##### Linux
```bash
python scripts/finetune/download_musiccaps.py \
  --out_dir data/raw \
  --yt_dlp_path "$(which yt-dlp)" \
  --cookies_file "<YOUR PATH TO cookies.txt>" \
  --js_runtimes "deno" \
  --ffmpeg_path "$(which ffmpeg)" \
  --max_workers 4 
```
#### Step 2. Enrich captions with structured metadata
##### Windows
```ps1
python scripts/finetune/musiccaps_openrouter.py `
  --data_root data/raw `
  --api_key <OPENROUTER_API_KEY> `
  --model <OPENROUTER_MODEL_ID> `
  --max_workers 4
```
##### Linux
```bash
python scripts/finetune/musiccaps_openrouter.py \
  --data_root data/raw \
  --api_key <OPENROUTER_API_KEY> \
  --model <OPENROUTER_MODEL_ID> \
  --max_workers 4
```
#### Step 3. Build train/valid split
##### Windows
```ps1
python scripts/finetune/build_split.py `
  --src_root data/raw `
  --dst_root data/musiccaps_struct `
  --valid_pct 20 # This parameter is responsible for the proportion of valid (train/val)
```
##### Linux
```bash
python scripts/finetune/build_split.py \
  --src_root data/raw \
  --dst_root data/musiccaps_struct \
  --valid_pct 20 # This parameter is responsible for the proportion of valid (train/val)
```
#### Step 4. Create AudioCraft manifests
```bash
python -m audiocraft.data.audio_dataset data/musiccaps_struct/train egs/musiccaps_struct/train/data.jsonl.gz
python -m audiocraft.data.audio_dataset data/musiccaps_struct/valid egs/musiccaps_struct/valid/data.jsonl.gz
```
---
## 3. Training
### Windows
Set environment variables:
```ps1
$env:USER = $env:USERNAME
$env:AUDIOCRAFT_TEAM = "default"
$env:AUDIOCRAFT_CLUSTER = "default"
$env:AUDIOCRAFT_DORA_DIR = "$PWD\audiocraft_runs"
$env:COMET_API_KEY = "<COMET_API_KEY>"
$env:COMET_WORKSPACE = "<COMET_WORKSPACE>"
$env:COMET_PROJECT_NAME = "musicgen-finetune"
```
Run training:
```ps1
dora run `
  solver=musicgen/musicgen_base_32khz `
  dset=audio/musiccaps `
  conditioner=text2music `
  model/lm/model_scale=small `
  continue_from=//pretrained/facebook/musicgen-small `
  mp_start_method=spawn `
  efficient_attention_backend=torch `
  transformer_lm.custom=true `
  transformer_lm.memory_efficient=false `
  transformer_lm.checkpointing=none `
  dataset.num_workers=0 `
  dataset.batch_size=4 `
  dataset.segment_duration=10 `
  dataset.min_segment_ratio=0.95 `
  dataset.valid.num_samples=100 `
  dataset.generate.num_samples=4 `
  optim.lr=1e-5 `
  optim.epochs=20 `
  optim.updates_per_epoch=100 `
  deadlock.use=false `
  generate.every=999 `
  evaluate.every=999
```
Configs:
- `config/conditioner/text2music_struct.yaml`
```yaml
# @package __global__

classifier_free_guidance:
  training_dropout: 0.3
  inference_coef: 3.0

attribute_dropout: {}

fuser:
  cross_attention_pos_emb: false
  cross_attention_pos_emb_scale: 1
  sum: []
  prepend: []
  cross: [description]
  input_interpolate: []

conditioners:
  description:
    model: t5
    t5:
      name: t5-base
      finetune: false
      word_dropout: 0.3
      normalize_text: false

dataset:
  train:
    merge_text_p: 0.7
    drop_desc_p: 0.1
    drop_other_p: 0.5
```
- `config/dset/audio/musiccaps_struct.yaml`
```yaml
# @package __global__

datasource:
  max_sample_rate: 32000
  max_channels: 1
  train: egs/musiccaps_struct/train
  valid: egs/musiccaps_struct/valid
  evaluate: egs/musiccaps_struct/valid
  generate: egs/musiccaps_struct/valid
```
### Linux
It's the same run, without the Windows-specific workarounds:
```bash
export AUDIOCRAFT_TEAM=default
export AUDIOCRAFT_DORA_DIR=$PWD/audiocraft_runs
export COMET_API_KEY=<COMET_API_KEY>
export COMET_WORKSPACE=<COMET_WORKSPACE>
export COMET_PROJECT_NAME=musicgen-finetune
```
Run training:
```bash
dora run \
  solver=musicgen/musicgen_base_32khz \
  dset=audio/musiccaps \
  conditioner=text2music \
  model/lm/model_scale=small \
  continue_from=//pretrained/facebook/musicgen-small \
  dataset.batch_size=4 \
  dataset.segment_duration=10 \
  dataset.min_segment_ratio=0.95 \
  dataset.valid.num_samples=100 \
  dataset.generate.num_samples=4 \
  optim.lr=1e-5 \
  optim.epochs=20 \
  optim.updates_per_epoch=100 \
  generate.every=999 \
  evaluate.every=999
```
Training artifacts are stored in:
`audiocraft_runs/xps/<SIG>/`
---
## 4. Log the run to CometML
After training finishes:
### Windows
```ps1
python scripts/finetune/train_log_cometml.py `
  --sig <SIG> `
  --project_name <YOUR COMET PROJECT NAME> `
  --workspace <COMET_WORKSPACE> `
  --run_name <YOUR RUN NAME>
```
### Linux
```bash
python scripts/finetune/train_log_cometml.py \
  --sig <SIG> \
  --project_name <YOUR COMET PROJECT NAME> \
  --workspace <COMET_WORKSPACE> \
  --run_name <YOUR RUN NAME>
```
---
## 5. Export model to Hugging Face
### Windows
```ps1
python scripts/finetune/export_hf.py `
  --sig <SIG> `
  --repo_id <HF_USERNAME>/<HF_REPO_NAME> `
  --out_dir hf_model/<YOUR MODEL NAME>
```
### Linux
```bash
python scripts/finetune/export_hf.py \
  --sig <SIG> \
  --repo_id <HF_USERNAME>/<HF_REPO_NAME> \
  --out_dir hf_model/<YOUR MODEL NAME>
```
## 6. Generate audio from structured prompts
Prepare the prompt file:
`data/gen_sound/prompts.json`
Expected format:
```json
[
  {
    "description": "...",
    "general_mood": "...",
    "genre_tags": ["...", "..."],
    "lead_instrument": "...",
    "accompaniment": "...",
    "tempo_and_rhythm": "...",
    "vocal_presence": "...",
    "production_quality": "..."
  },
  ...
]
```
Generate audio:
### Windows
```ps1
python scripts/finetune/generate_audio.py `
  --model_path hf_model/<YOUR MODEL NAME> `
  --prompts_json data/gen_sound/prompts.json `
  --out_dir data/gen_sound `
  --prefix "test" ` # example: "testprompt_1.wav"
  --top_k 250 `
  --cfg_coef 3.0 `
  --top_p 0.0 `
  --temp 1.0 ` # temperature
  --duration 10 `
  --num_variants 1
```
### Linux
```bash
python scripts/finetune/generate_audio.py \
  --model_path hf_model/<YOUR MODEL NAME> \
  --prompts_json data/gen_sound/prompts.json \
  --out_dir data/gen_sound \
  --prefix "test" \ # example: "testprompt_1.wav"
  --top_k 250 \ 
  --cfg_coef 3.0 \
  --top_p 0.0 \
  --temp 1.0 \ # temperature
  --duration 10 \
  --num_variants 1
```
Expected outputs:
```text
data/gen_sound/prompt_1.wav
...
data/gen_sound/prompt_n.wav
```
---
## 7. Recommended run order
If you start from scratch:
```text
1. scripts/finetune/download_musiccaps.py
2. scripts/finetune/musiccaps_openrouter.py
3. scripts/finetune/build_split.py
4. python -m audiocraft.data.audio_dataset ...
5. dora run ...
6. scripts/finetune/train_log_cometml.py
7. scripts/finetune/export_hf.py
8. scripts/finetune/generate_audio.py
```
If you wanna download dataset from link:
```text
1. Download/extract archive into data/
2. scripts/finetune/build_split.py
3. python -m audiocraft.data.audio_dataset ...
4. dora run ...
5. scripts/finetune/train_log_cometml.py
6. scripts/finetune/export_hf.py
7. scripts/finetune/generate_audio.py
```