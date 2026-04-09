# Diagnostics & Next Experiments for PixCell Probmap Training

## Current Status (Step 2250, Epoch 1)
- Tumor channel: Good spatial correspondence with ground truth
- TIL channel: Systematic false positives in pure tumor regions
- MSE oscillating 6K-10K, trending down overall

## Key Finding: Tumor/TIL Confusion
The model generates TIL signal (green) wherever there is tumor (red). It's learning
"high cellularity → both channels" rather than distinguishing tumor from lymphocyte
morphology. The WSInfer ground truth is clean (zero lymph in pure tumor regions),
so this is a model learning problem, not a data problem.

## Diagnostics to Gather During Remaining Training

### 1. Per-checkpoint error decomposition
For each eval checkpoint, compute and log:
- Mean absolute error per channel (tumor and TIL separately)
- TIL false positive rate: mean generated G where GT G < 5
- TIL false negative rate: mean GT G where generated G < 5
- Spatial correlation per channel: Pearson(generated, GT) for R and G separately
- Track these over training steps to see if TIL FP rate decreases with more training

### 2. UNI2-h embedding analysis
- Extract UNI2-h embeddings for our 17 eval patches
- Visualize with t-SNE/UMAP colored by tumor_frac and til_frac
- Question: Can UNI2-h embeddings distinguish tumor-only from tumor+TIL patches?
  If not, the LoRA fundamentally can't learn this distinction from UNI conditioning alone

### 3. Spatial error maps
- For each eval patch: compute |generated - GT| per pixel, per channel
- Visualize as heatmaps: where EXACTLY is the TIL false positive?
- Is it uniform across the tumor region, or concentrated at tumor boundaries/stroma interface?

### 4. Per-stratum performance
- Does the model perform better on stratum A (immune-hot tumor) than stratum B (immune-cold)?
- Stratum B is where TIL FP is most damaging — model should output zero G but doesn't

## Possible Next Experiment Directions

### A. Scale-aware conditioning
Instead of single-scale UNI2-h (256×256), provide multi-scale embeddings:
- 256×256 patches for architectural context (tumor detection)
- 64×64 or 128×128 sub-patches for cellular detail (TIL detection)
This mirrors WSInfer's multi-scale approach

### B. Separate LoRA models per channel
- One LoRA for tumor probability (R channel only)
- One LoRA for TIL probability (G channel only)
- Each can learn independently without cross-channel confusion
- Doubles training time but eliminates the coupling problem

### C. Channel-weighted loss
- Weight the loss more heavily on the G (TIL) channel in pure-tumor regions
- Or add a penalty term for G > 0 where GT_G ≈ 0
- Keeps single model but biases learning toward the harder task

### D. Conditional architecture
- Add tumor detection as an explicit input condition
- First predict tumor, then predict TIL conditioned on knowing where tumor is
- Mirrors the clinical workflow (pathologist identifies tumor first, then assesses immune infiltration)

## Files to Create
- `PixCell/virtual_staining/analyze_checkpoint.py` — computes all diagnostics above for any checkpoint
- Run after each eval alongside the comparison grid
