# PixCell & ZoomLDM: Virtual Staining & Histopathology Image Generation

## Virtual Staining — Conceptual Framework

**This section defines how PixCell's virtual staining works. All code,
evaluation, and documentation must be consistent with this model.**

PixCell generates histopathology images conditioned on UNI2-h foundation model
embeddings. Virtual staining exploits this: given an H&E patch, translate its
UNI2-h embedding into the target stain's embedding space, then generate the
target stain image via conditional diffusion.

### Why H&E → IHC/IF works

H&E and IHC/IF stains reveal the **same underlying tissue morphology** with
different contrast. Because the tissue structure is shared, UNI2-h embeddings
of matched H&E and IHC patches are similar — the embedding translation is a
small, learnable transformation.

### Two-component adapter system

1. **Flow-matching MLP** (ResMLP): Translates UNI2-h embeddings from H&E space
   to target stain space. This is lightweight (trained in minutes) and captures
   the systematic shift between stain domains.

2. **LoRA adapter** (cross-attention): Fine-tunes PixCell's cross-attention
   layers to generate stain-specific visual features (chromogen patterns,
   staining intensity, background characteristics). Rank is configurable;
   targets the 8 `attn2` projection matrices.

Together: MLP translates the conditioning signal, LoRA adjusts the generation.

### What doesn't work

Targets that are **visually unrelated to H&E** (probability maps, segmentation
masks, categorical labels) break the conditioning mechanism. The UNI2-h
embeddings of H&E and such targets are far apart (cosine similarity ~0.27),
and LoRA on cross-attention alone cannot bridge that gap.

See `docs/archive/til-experiment/README.md` for the full analysis of a failed
H&E → TIL probability map experiment that established this boundary.

### Landscape

**Our models**:
- **PixCell** (arXiv 2506.05127): Pan-cancer diffusion foundation model, virtual staining
- **ZoomLDM** (CVPR 2025): Multi-scale generation, super-resolution, large-image synthesis

**External reference points**:
- **GigaTIME** (Microsoft, Cell 2025): H&E → 21-channel virtual multiplex IF,
  14,256 patients, 24 cancer types. The current competitive benchmark for
  virtual multiplexing at scale.
- **GenPercept** (ICLR 2025): Definitive ablation showing full fine-tuning >>
  LoRA for diffusion-based dense prediction. Relevant if extending beyond
  stain-to-stain translation.
- **Marigold** (CVPR 2024): Full U-Net fine-tuning for monocular depth; works
  because depth maps share statistics with natural images.

---

## Project Overview

PixCell is the first generative foundation model for digital histopathology
(arXiv 2506.05127, Stony Brook CVLab). It generates high-fidelity pathology
images conditioned on UNI2-h embeddings, enabling:

- **Virtual staining**: H&E → IHC translation (HER2, ER, PR, Ki67 demonstrated)
- **Data augmentation**: Targeted synthetic patch generation for rare phenotypes
- **Large-image generation**: 4K×4K coherent tissue images via sliding window

The model is trained progressively (256→512→1024) on ~70,000 WSIs from TCGA,
CPTAC, GTEx, and other sources.

**ZoomLDM** (CVPR 2025, same group) extends this to multi-scale generation.
It synthesizes coherent histopathology images across different magnification
levels using a magnification-aware conditioning mechanism with SSL embeddings.
Key capabilities:
- **Multi-scale generation**: Coherent patches at any zoom level with shared weights
- **Large-image synthesis**: Up to 4096×4096 via joint multi-scale sampling (8 min/image)
- **Super-resolution**: 4x enhancement via condition inversion with multi-scale enforcement
- **Data-scarce thumbnails**: Weight sharing across scales boosts quality where training data is limited

**Current virtual staining support**: Four IHC stains from the MIST dataset
(HER2, ER, PR, Ki67) plus HER2Match, with pre-trained LoRA + flow MLP weights
on HuggingFace. New stains can be trained with paired H&E/target data using
the existing pipeline.

## Key Repositories

- **PixCell**: This repo — diffusion model, training, sampling, virtual staining
- **ZoomLDM**: https://github.com/cvlab-stonybrook/ZoomLDM — multi-scale generation, super-resolution (CVPR 2025, arXiv 2411.16969)
- **HuggingFace models**:
  - [PixCell-256](https://huggingface.co/StonyBrook-CVLab/PixCell-256) — Diffusers checkpoint
  - [PixCell-1024](https://huggingface.co/StonyBrook-CVLab/PixCell-1024) — Diffusers checkpoint
  - [Original weights](https://huggingface.co/StonyBrook-CVLab/PixCell-original-weights) — Native checkpoints
  - [Virtual staining weights](https://huggingface.co/StonyBrook-CVLab/pixcell-virtual-staining) — LoRA + MLP for MIST/HER2Match
- **Datasets**:
  - [MIST](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE) — Paired H&E/IHC (HER2, ER, PR, Ki67)
  - [HER2Match](https://zenodo.org/records/15797050) — Paired H&E/HER2 IHC
- **UNI2-h**: https://huggingface.co/MahmoodLab/UNI2-h — ViT-Giant foundation model (gated, request access)
- **DSA**: https://github.com/DigitalSlideArchive/digital_slide_archive — slide management + HistomicsUI viewer

## Tech Stack

- **Diffusion models**: PixCell (DiT/PixArt-Sigma, 28-layer transformer, 16 heads, cross-attn dim 1152) + ZoomLDM (magnification-aware conditioning, shared weights across scales)
- **VAE**: Stable Diffusion 3.5 VAE (16 latent channels)
- **Foundation model**: UNI2-h (ViT-Giant, 1536-dim CLS token, encoded as 16 caption tokens per tile)
- **Virtual staining**: LoRA (PEFT, targets `attn2` cross-attention, 8 projection matrices) + flow-matching MLP (ResMLP)
- **Framework**: PyTorch >= 2.0.1, diffusers, accelerate, PEFT, transformers, timm
- **Viewer**: DSA/HistomicsUI on port 8081 (Girder REST API, MongoDB, FUSE mounts)
- **GPU**: MPS (local dev on Apple Silicon), CUDA (DGX Spark / cloud)

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `diffusion/` | Core diffusion model: transformer architecture (`model/`), data loaders (`data/`), DPM solver, IDDPM, SA sampler |
| `virtual_staining/` | Virtual staining pipeline: LoRA training (`train_lora.py`), flow MLP training (`train_flow_mlp.py`), datasets (`mist_dataset.py`, `her2match_dataset.py`), inference notebook (`virtual_staining.ipynb`) |
| `configs/` | Training configurations (`pan_cancer/` for progressive stages) |
| `tools/` | Sampling scripts (`sample_256.py`, `sample_4k.py`), feature extraction (`extract_features.py`) |
| `train_scripts/` | Base model training (`train_pixcell.py`) |
| `docker/` | DSA Docker Compose stack (5 services, port 8081) |
| `docs/archive/` | Archived experiments with lessons learned |
| `data/` | Slides, features, training data (gitignored) |
| `asset/` | Documentation assets (overview figure) |

### Virtual Staining Scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | LoRA adapter training for new IHC stains (accelerate-compatible) |
| `train_flow_mlp.py` | Flow-matching MLP training for embedding translation |
| `mist_dataset.py` | MIST dataset loader (HER2, ER, PR, Ki67) |
| `her2match_dataset.py` | HER2Match dataset loader |
| `pixcell_transformer_2d_lora.py` | LoRA-wrapped PixCell transformer |
| `resmlp.py` | ResMLP architecture for flow matching |
| `virtual_staining.ipynb` | End-to-end inference notebook (downloads weights from HF) |
| `extract_uni_embeddings.py` | Pre-extract UNI2-h embeddings for faster training |

### TIL Experiment Scripts (archived, for reference)

| Script | Purpose |
|--------|---------|
| `til_dataset.py` | TILOverlayDataset — paired H&E → probmap dataset (abandoned) |
| `train_til_lora.py` | LoRA training for H&E → TIL overlay (abandoned) |
| `train_til_flow_mlp.py` | Flow MLP for TIL embedding translation (abandoned) |
| `train_til_flow_mlp_cached.py` | Cached variant with pre-extracted embeddings (abandoned) |
| `eval_til_lora.py` | Inference + comparison grid generation (abandoned) |
| `infer_whole_slide.py` | Whole-slide PixCell inference + stitching (abandoned) |

## Model Zoo

| Model | Resolution | Checkpoint | Notes |
|-------|------------|------------|-------|
| PixCell-256 | 256×256 | [HF Diffusers](https://huggingface.co/StonyBrook-CVLab/PixCell-256) / [Original](https://huggingface.co/StonyBrook-CVLab/PixCell-original-weights) | Base generation model |
| PixCell-1024 | 1024×1024 | [HF Diffusers](https://huggingface.co/StonyBrook-CVLab/PixCell-1024) / [Original](https://huggingface.co/StonyBrook-CVLab/PixCell-original-weights) | High-res generation model |
| Virtual staining (MIST) | 1024×1024 | [HF](https://huggingface.co/StonyBrook-CVLab/pixcell-virtual-staining) | LoRA + MLP for HER2, ER, PR, Ki67 |
| Virtual staining (HER2Match) | 1024×1024 | [HF](https://huggingface.co/StonyBrook-CVLab/pixcell-virtual-staining) | LoRA + MLP for HER2 |
| ZoomLDM | Multi-scale | [HF](https://huggingface.co/StonyBrook-CVLab) (auto-downloaded) | Multi-scale generation + super-resolution |

## Commands

```bash
# === Environment ===
conda create -n pixcell python=3.9
conda activate pixcell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

# === Base Model Training (progressive) ===
# Stage 1: 256×256
accelerate launch train_scripts/train_pixcell.py configs/pan_cancer/pixart_20x_256.py \
    --work-dir /path/to/output_dir
# Stage 2: 512×512 (init from 256)
accelerate launch train_scripts/train_pixcell.py configs/pan_cancer/pixart_20x_512.py \
    --work-dir /path/to/output_dir --resume-from /path/to/pixcell_256.ckpt
# Stage 3: 1024×1024 (init from 512)
accelerate launch train_scripts/train_pixcell.py configs/pan_cancer/pixart_20x_1024.py \
    --work-dir /path/to/output_dir --resume-from /path/to/pixcell_512.ckpt

# === Sampling ===
# 256×256 generation
python tools/sample_256.py \
    --workdir /path/to/workdir --checkpoint checkpoints/last_ema.ckpt \
    --out_dir samples_256 --n_images 5000 --sampling_steps 20 \
    --guidance_strength 2 --sampling_algo dpm-solver

# 4K×4K large image generation
python tools/sample_4k.py \
    --output_dir samples_4k --num_samples 100 --num_timesteps 20 \
    --guidance_scale 2 --sliding_window_size 64 --gpu_id 0

# === Virtual Staining ===
# LoRA training (new IHC stain)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    virtual_staining/train_lora.py \
    --dataset MIST --root_dir /path/to/data/ --split train \
    --stain HER2 --train_batch_size 4 --num_epochs 10 \
    --gradient_accumulation_steps 2

# Flow MLP training
python virtual_staining/train_flow_mlp.py \
    --dataset MIST --root_dir /path/to/data/ --split train \
    --stain HER2 --device cuda --train_batch_size 4 \
    --num_epochs 100 --save_every 25

# Feature extraction (for faster training)
python tools/extract_features.py --dataset_name tcga_diagnostic --size 256

# === DSA ===
cd docker && docker compose up -d    # start DSA (port 8081)
cd docker && docker compose down     # stop DSA
# DSA UI: http://localhost:8081/histomics (admin/password)
```

## Archived Experiments

- **`docs/archive/til-experiment/`** — H&E → TIL probability map attempt
  (March 2026). Three formulations tried (blend overlay, probmap, GTT encoding),
  all failed due to conditioning mismatch between H&E and probability maps in
  latent space. Key findings: VAE can encode/decode targets perfectly (not the
  bottleneck), UNI2-h can distinguish tumor from TIL (information exists), but
  LoRA on cross-attention cannot redirect diffusion trajectories across the
  large domain gap. See `docs/archive/til-experiment/README.md` for full analysis.

## Documentation Rules — Hard Rules

1. **Lab notebook is append-only**: Always add new entries to `docs/lab-notebook.md`.
   Never delete or modify existing entries. Format: `## YYYY-MM-DD [thread-tag] Title`.
2. **Figures are append-only**: Never delete files from `figures/`. Use datetime prefix:
   `YYYYMMDD-HHMMSS-descriptive-name.ext`. Reference figures by filename, not local
   numbering (e.g., "Fig 1"). Filenames are globally unique; local numbering is ambiguous.
3. **Archive before reorganizing**: Before moving or restructuring any docs or figures,
   copy the originals to `docs/archive/` first. Nothing is ever destroyed.
4. **Thread documents**: Living research narratives in `docs/threads/<topic>.md`.
   When updating a thread, add a corresponding lab notebook entry.

## Conventions

- Functions over classes unless domain requires objects
- Minimal dependencies; prefer stdlib when reasonable
- No over-engineering — solve current problems only
- GPU backend auto-detected: MPS on Mac, CUDA on Linux/NVIDIA
- Slides are always volume-mounted, never copied into containers

## Infrastructure

| Path | Purpose |
|------|---------|
| `docker/` | DSA Docker Compose stack (5 services, port 8081 — independent from WSInfer on 8080) |
| `environment.yml` | Conda environment specification |
| `requirements.txt` | Pip requirements |

## Active Plans

- `~/.claude/plans/woolly-strolling-kahan.md` — CLAUDE.md creation plan
