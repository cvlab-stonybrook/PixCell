# Experiment Results Summary (March 28, 2026)

## Experiment C: Spatial Error Maps
**Question**: WHERE does the TIL confusion happen?

**Result**: The TIL false positive is **UNIFORM across tumor regions**, not concentrated at boundaries.
- Pure tumor patch (A0RH): 98.9% of pixels are TIL false positives, 69% inside tumor regions
- Immune-cold tumor (stratum B): 96.8% FP, 88.5% inside tumor
- Near-100% tumor: 54.5% FP, 100% inside tumor

**Implication**: This is a global error ("all cellularity → TIL"), not an edge effect. The model treats any dense cellular region as having TIL signal.

**Image**: `figures/09_spatial_error_maps.jpg`

---

## Experiment A: UNI2-h Embedding Separability
**Question**: Can UNI2-h embeddings distinguish tumor from TIL?

**Result**: **YES — UNI2-h clearly separates tumor from TIL.**

Cosine distances between category centroids:

| Comparison | Spatial Tokens | CLS Tokens | Interpretation |
|-----------|---------------|------------|---------------|
| tumor vs TIL | **0.355** | **0.355** | LARGE — they ARE different |
| tumor vs stroma | 0.330 | 0.364 | LARGE — expected |
| TIL vs stroma | 0.195 | 0.291 | moderate |
| TIL vs both | 0.088 | 0.197 | small — expected (both contains TIL) |

**Key finding**: Tumor-vs-TIL distance (0.355) is as large as tumor-vs-stroma distance (0.330-0.364). UNI2-h encodes enough information to distinguish them, both at the individual spatial token level AND at the pooled CLS level.

**This falsifies Hypothesis 2** (DINOv2 conflates tumor and TIL). The information IS in the embeddings.

**Implication**: The TIL confusion is a **training problem**, not an encoding problem. The LoRA has the information it needs but hasn't learned to use it yet after ~1.5 epochs. Continue training.

**Image**: `figures/10_embedding_separability.png`

---

## Scale Analysis Revision
**Original hypothesis**: UNI2-h can't distinguish tumor from TIL because of insufficient resolution.

**Revised finding**: Slides are 40x (0.248 µm/px). At this magnification:
- UNI2-h ViT token = 3.5 µm (sub-cellular resolution)
- Lymphocyte (7 µm) spans 2 ViT tokens — easily resolvable
- Tumor cell (15 µm) spans 4.3 ViT tokens — clearly resolvable
- UNI2-h actually has BETTER resolution than the WSInfer lymphocyte model

**Resolution is not the problem.**

---

## Decision: Continue Training

Based on experiments A and C:
- The UNI2-h embeddings CAN separate tumor from TIL (Experiment A)
- The TIL FP is a global training error, not an architectural limitation (Experiment C)
- The model is only ~1.5 epochs in, with ~1M LoRA parameters
- MSE trend shows the model is still actively learning

**Recommendation**: Continue training through epochs 3-5. Monitor TIL FP rate at each checkpoint. If TIL FP rate stops decreasing after epoch 5, consider channel-weighted loss or separate LoRA models.

---

## Training Status at Time of Analysis
- **Step**: 2,644 (epoch 2, 2% complete)
- **GPU**: 96%, 79C
- **Training continues uninterrupted on DGX Spark**
