# Evaluating the Tumor/TIL Confusion: Hypotheses, Tests, and Next Design

## Context

The PixCell LoRA probmap model (step 2250, epoch 1) predicts tumor well but generates
false positive TIL signal in pure tumor regions. We hypothesized this was a multi-scale /
resolution problem. Analysis reveals it's more nuanced than that.

## Scale Analysis — The Resolution is NOT the Problem

Our slides are **40x** (0.248 µm/px). At this magnification:

| Component | Physical scale | Notes |
|-----------|---------------|-------|
| UNI2-h ViT token | 3.5 µm | Sub-cellular — can resolve nuclear morphology |
| Lymphocyte | 7 µm = 28 px = 2 ViT tokens | Resolvable in UNI2-h |
| Tumor cell | 15 µm = 60 px = 4.3 ViT tokens | Clearly resolvable in UNI2-h |
| UNI2-h tile | 63 µm = 256 px | Contains ~50-100 cells |
| WSInfer tumor FOV | 88 µm | Similar scale to UNI tile |
| WSInfer lymph FOV | 50 µm at 10x (downsampled) | Coarser than UNI |

**UNI2-h has BETTER resolution than the WSInfer lymphocyte model.**
The problem is not that UNI2-h can't see the difference — it's about how
the information is encoded and used.

## Three Hypotheses to Test

### Hypothesis 1: CLS Token Pooling Bottleneck
UNI2-h's 324 spatial tokens (per 256x256 tile) may encode tumor vs TIL differently,
but the CLS token (global pooling) averages this away. The PixCell LoRA only sees the
CLS token (1536 dims), not the spatial tokens.

**Test**: Extract FULL spatial token maps (324 × 1536) for eval patches. Compute
cosine similarity between tokens at tumor cell locations vs lymphocyte locations.
If spatial tokens differ but CLS tokens are similar → pooling is the bottleneck.

**What to build**: `analyze_uni_embeddings.py` that:
1. Takes an eval patch + its ground truth probmap
2. Extracts all 324 UNI2-h spatial tokens (not just CLS)
3. Groups tokens by ground truth label (tumor pixel, TIL pixel, stroma pixel)
4. Computes within-group vs between-group similarity
5. Visualizes with t-SNE/UMAP colored by tissue type

### Hypothesis 2: DINOv2 Feature Space Similarity
UNI2-h was trained with DINOv2 (self-supervised visual similarity). Dense tumor and
dense TIL aggregates are visually similar (both: dark, cellular, high nuclear-to-cytoplasm
ratio). DINOv2's objective is to match augmented views — dark cellular regions may look
similar under augmentation regardless of nuclear morphology.

**Test**: For the same spatial tokens analyzed in H1, measure:
- Mean embedding of "pure tumor tokens" vs "pure TIL tokens"
- Cosine distance between these means
- Compare to tumor-vs-stroma distance (which should be large)
- If tumor-TIL distance << tumor-stroma distance → DINOv2 conflates them

**This can be done with the same script as H1.**

### Hypothesis 3: Training Duration / Capacity
The model has only completed ~1.5 epochs. The LoRA has 1M parameters learning
a mapping from 16 × 1536-dim embeddings → 1024×1024 × 3-channel image. It may
simply need more training to learn the subtle tumor/TIL distinction.

**Test**: Track per-channel metrics across checkpoints:
- Tumor channel MAE (should be decreasing)
- TIL false positive rate in pure-tumor patches (critical metric)
- TIL channel MAE in patches WITH actual TIL
- If TIL FP rate is still decreasing at epoch 3-5 → continue training
- If TIL FP rate plateaus → the LoRA can't learn this from CLS tokens alone

**What to build**: `track_channel_metrics.py` that processes all existing checkpoints
and plots the trend.

## Experiments to Gather Data (Without Stopping Training)

All of these can run on the Mac (CPU) using saved checkpoints + eval patches.
Training continues uninterrupted on the Spark.

### Experiment A: UNI2-h Embedding Separability (H1 + H2)
**Time**: ~30 min on Mac (no GPU needed for analysis)
**Input**: 17 eval patches + their ground truth probmaps
**Process**:
1. Load UNI2-h on Mac (MPS or CPU)
2. For each eval patch, extract all 324 spatial tokens per 256x256 sub-tile (16 tiles × 324 tokens = 5,184 tokens per patch)
3. For each token, look up the ground truth label at that spatial location (tumor prob, TIL prob)
4. Categorize tokens: high-tumor (>0.5), high-TIL (>0.3), stroma (<0.2 both), mixed
5. Compute pairwise cosine distances between categories
6. Visualize with UMAP, colored by category
7. Save results + visualization

**Key question answered**: Can UNI2-h spatial tokens distinguish tumor from TIL at the sub-tile level?

### Experiment B: Checkpoint Progression Analysis (H3)
**Time**: ~2 hours on Spark (or Mac with GPU)
**Input**: All saved checkpoints (s250, s500, s750, s1000, s1250, s1500, s1750, s2000, s2250, s2500)
**Process**:
1. For each checkpoint, run inference on the 17 eval patches
2. Compute per-channel metrics:
   - Tumor MAE, TIL MAE (overall)
   - TIL false positive rate: mean(generated_G) where GT_G < 5, for pure-tumor patches
   - Spatial correlation: pearsonr(generated, GT) per channel
3. Plot metrics vs training step
4. Identify if TIL FP rate is still decreasing or has plateaued

**Key question answered**: Is more training helping, or has the model converged to a suboptimal solution?

### Experiment C: Spatial Error Maps (diagnostic)
**Time**: ~5 min on Mac
**Input**: Step 2250 eval grid (already downloaded)
**Process**:
1. For each eval patch, compute per-pixel |generated - GT| for R and G channels
2. Overlay error maps on the H&E image
3. Key question: Is the TIL false positive UNIFORM across tumor regions, or concentrated at specific tissue structures (tumor boundaries, stroma-tumor interface)?

**Key question answered**: WHERE does the confusion happen? This tells us whether the model is making a global error ("all cellularity → TIL") or a localized error ("tumor boundaries look like TIL").

## Decision Framework: Continue vs Redesign

After running experiments A-C:

| If... | Then... |
|-------|---------|
| UNI spatial tokens CAN separate tumor/TIL (A) + TIL FP rate still decreasing (B) | Continue training. The information is there, model needs more epochs. |
| UNI spatial tokens CAN separate, but TIL FP rate plateaued (B) | Redesign: Use spatial tokens (not CLS) as conditioning. The info exists but CLS pooling loses it. |
| UNI spatial tokens CANNOT separate tumor/TIL (A) | Redesign: UNI2-h can't do this. Consider separate LoRA per channel, or use a different encoder for TIL conditioning. |
| TIL error is localized to boundaries (C) | Potentially a resolution/alignment issue. Consider finer-grained spatial conditioning. |

## Recommended Execution Order

1. **Experiment C** first (5 min, Mac, immediate insight)
2. **Experiment A** next (30 min, Mac, answers the fundamental question)
3. **Experiment B** in background on Spark after current training or with saved checkpoints
4. Use results to make continue/redesign decision

## Next Design Directions (pending experiment results)

### If spatial tokens work but CLS pooling is the bottleneck:
- Modify PixCell conditioning to use spatial token maps instead of CLS
- Each 256x256 region gets its own 324-token spatial context
- The diffusion model can attend to fine-grained features per-location
- This is architecturally similar to what GigaPath does with LongNet

### If UNI2-h fundamentally can't separate tumor from TIL:
- Use a cell-level encoder (e.g., CellViT, HoVer-Net) for TIL detection
- Or train tumor LoRA and TIL LoRA as separate models with separate encoders
- Or add the WSInfer tumor prediction as an explicit input channel

### If more training is the answer:
- Continue current run through epochs 5-10
- Consider reducing learning rate at epoch 3-5 (cosine schedule)
- Monitor TIL FP rate at each checkpoint

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/analyze_uni_embeddings.py` | Experiment A: UNI2-h token separability analysis |
| `scripts/track_channel_metrics.py` | Experiment B: per-channel metric trends across checkpoints |
| `scripts/spatial_error_maps.py` | Experiment C: WHERE does the TIL confusion happen? |
