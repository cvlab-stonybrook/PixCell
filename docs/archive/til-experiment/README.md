# Archived Experiment: PixCell LoRA for H&E → TIL Probability Maps

**Dates**: March 23–29, 2026
**Status**: Abandoned — pivoted to discriminative UNI2-h linear probes
**Hardware**: NVIDIA DGX Spark (GB10, 130.7 GB, CUDA 12.8)

## What Was Tried

We attempted to repurpose PixCell's virtual staining pipeline (LoRA + flow-matching MLP) to generate tumor/TIL probability maps from H&E patches, using 88 TCGA-BRCA slides (~10,800 patches at 1024x1024).

Three target formulations were trained:

1. **Blend overlay** (60% H&E + 40% color wash): Model learned to reproduce the H&E component, not the overlay colors. Training signal too subtle.
2. **Probmap** (R=tumor, G=TIL, B=tissue mask): Architecturally correct but LoRA couldn't bridge the latent space gap. TIL channel showed 98.9% false positive rate in pure tumor regions.
3. **GTT** (R=grayscale H&E, G=tumor, B=TIL): Elegant 3-channel encoding proposed by Joel Saltz. After 1000 steps, tumor correlation still near zero.

## Why It Failed

PixCell's virtual staining works for H&E → IHC because both stains share underlying tissue morphology — their UNI2-h embeddings are similar. Probability maps are visually and semantically unrelated to H&E:

- **Latent space gap**: H&E vs probmap cosine similarity in VAE latent space = 0.274 (far apart)
- **LoRA capacity**: Cross-attention LoRA cannot redirect diffusion trajectories across such a large domain gap
- **Conditioning mismatch**: PixCell conditions on UNI2-h(target) during training. This works when target ≈ source in embedding space (IHC), not when they're fundamentally different (probmaps)

The VAE itself was not the bottleneck (per-channel reconstruction correlation >0.99), and UNI2-h can distinguish tumor from TIL (cosine distance 0.355). The failure is in the diffusion conditioning architecture, not in the feature representations.

## Key Lesson for Future Virtual Staining Work

**H&E → IHC/IF virtual staining works when the target shares morphological structure with H&E.** The UNI2-h conditioning mechanism relies on this similarity. Applications where the target is visually unrelated to H&E (probability maps, segmentation masks, classification outputs) require a different approach — e.g., discriminative models, or full fine-tuning rather than LoRA (see GenPercept, ICLR 2025).

## Relevance to IHC/IF Collaboration

This experiment's failure mode does NOT apply to genuine H&E → IHC or H&E → IF translation, where:
- Target stains reveal the same underlying tissue with different contrast
- UNI2-h embeddings of source and target will be similar
- PixCell's existing LoRA + flow MLP pipeline is designed for exactly this

The existing HER2/ER/PR/Ki67 virtual staining demonstrates this works. New IHC/IF stains should follow the same recipe with paired training data.

## Literature Context

The `literature-review.md` file contains a deep research report covering:
- **GenPercept (ICLR 2025)**: Full fine-tuning >> LoRA for diffusion-based dense prediction
- **Marigold (CVPR 2024)**: Requires full U-Net fine-tuning; works because depth maps share statistics with natural images
- **GigaTIME (Microsoft, Cell 2025)**: H&E → 21-channel virtual multiplex IF across 24 cancer types — the competitive benchmark
- **PathSegDiff (April 2025)**: Uses PathLDM as frozen feature extractor, not generator

## Files in This Archive

| File | Contents |
|------|----------|
| `session.md` | Original 8-step experiment plan (stopped at step 3) |
| `results.md` | Final experimental outcomes (spatial error maps, embedding separability) |
| `strategic-brainstorm.md` | Root cause analysis with Joel Saltz; VAE diagnostics; strategic goals |
| `hypotheses.md` | Three scientific hypotheses tested |
| `diagnostics-plan.md` | Diagnostic methodology for understanding failures |
| `literature-review.md` | Deep research report on diffusion for dense prediction |
| `figures/` | 34 evaluation images (blend/probmap/GTT evals, spatial error maps, VAE tests) |

## Related Code (in this repo)

The training scripts created for this experiment remain in `virtual_staining/`:
- `til_dataset.py` — TILOverlayDataset for paired H&E → probmap training
- `train_til_lora.py` — LoRA training script
- `train_til_flow_mlp.py` / `train_til_flow_mlp_cached.py` — Flow MLP training
- `eval_til_lora.py` — Inference + comparison grids
- `infer_whole_slide.py` — Whole-slide inference + stitching
- `extract_uni_embeddings.py` — Pre-extract UNI2-h embeddings
