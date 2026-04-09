# Session: PixCell LoRA Fine-Tuning for H&E → TIL Overlay
- **Goal**: Fine-tune PixCell with LoRA to generate synthetic H&E images with tumor/TIL overlays baked in
- **Date started**: 2026-03-23
- **Status**: Abandoned — pivoted to discriminative UNI2-h linear probes (2026-03-29)
- **Current step**: Stopped at step 3 of 8

## Steps
1. [x] Download 75 random TCGA-BRCA slides from GDC — 90 slides on disk (80GB)
2. [x] Run WSInfer tumor + lymphocyte inference on all new slides — 88 tumor + 88 lymphocyte CSVs complete
3. [~] Extract paired training patches — RUNNING on Mac (1024×1024, 88 slides, max 500/slide)
4. [x] Set up PixCell conda environment — NOW ON DGX SPARK: miniforge + conda env "ml", PyTorch 2.11.0+cu128, 130.7 GB GPU
5. [x] Adapt PixCell LoRA training scripts — Created til_dataset.py, train_til_flow_mlp.py, train_til_lora.py
6. [ ] Train blend adapter — will run on DGX Spark
7. [ ] Train classmap adapter — will run on DGX Spark
8. [ ] Inference pipeline + grid assembly + DSA integration

## Context & Decisions
- **Strategy**: PixCell first (has existing LoRA pipeline), port to ZoomLDM later for IP reasons
- **PixCell repo**: cloned to `/Users/aitestbed/projects/PixCell/`, rsynced to `spark:~/projects/PixCell/`
- **Plan file**: `~/.claude/plans/wondrous-swinging-lighthouse.md`
- **Training data**: 88 slides with both tumor + lymphocyte results
- **Patch extraction**: 1024×1024, max 500/slide, 10% val hold-out, min 10% tumor+lymph content
- **Output format**: PixCell paired directory convention (trainA/trainB matched by filename)
- **LoRA config**: rank=4, alpha=4, targets attn2 cross-attention only (8 projection matrices)
- **Two adapters**: lora_blend (H&E + semi-transparent wash) and lora_classmap (solid yellow/red/gray)
- **PixCell conditioning**: UNI2-h embeddings (1536-dim) from 4×4 grid of 256×256 patches → working at 1024×1024
- **DGX Spark**: GPU training moved from MPS (Mac) to DGX Spark (NVIDIA GB10, 130.7 GB, CUDA 12.8)
  - SSH: `ssh spark` (alias configured)
  - Conda env: `ml` with PyTorch 2.11.0+cu128, diffusers, PEFT, accelerate, transformers, timm
  - PixCell repo rsynced to spark:~/projects/PixCell/

## Key Files
- `WSInfer/scripts/extract_patches.py` — paired patch extraction (running now)
- `PixCell/virtual_staining/til_dataset.py` — TILOverlayDataset
- `PixCell/virtual_staining/train_til_flow_mlp.py` — Flow MLP training
- `PixCell/virtual_staining/train_til_lora.py` — LoRA training

## Outcome
(to be filled when complete)
