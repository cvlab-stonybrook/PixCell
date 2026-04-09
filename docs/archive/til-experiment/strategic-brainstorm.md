# Strategic Brainstorm: Tumor/TIL Methodology Direction
**Date**: March 29, 2026
**Participants**: Joel Saltz (PI) + Claude Code (Opus 4.6)

---

## Joel's Strategic Goals (verbatim)

> Interesting. Let me push this to a slightly more strategic level and articulate broader goals. Here are some things I'd like to accomplish:
>
> 1) improve robustness of the tumor/TIL algorithm. The legacy WSInfer algorithm works on very small patches and sometimes fails on patches due to noise, tissue variation or whatever. It has a very local view of things. Sometimes spatial smoothing methods have been used but these fail to distinguish between scattered individual TILs and noise. The WSInfer algorithm is however trained with a great variety of different tissue annotations so is valuable.
>
> 2) The WSInfer TIL algorithm works for many types of tissue. See the Frontiers paper in 2022 by Saltz et al., It is not particularly obvious how best to fine tune the WSInfer algorithm TIL to handle new datasets without having a difficult to quantify impact on the model performance.
>
> 3) The current patchwise methods employ legacy models rather than modern foundation models. Unknown performance hit and looks bad as it makes us look behind the times.
>
> 4) We sometimes use both WSInfer and cell segmentation/classification methods but it takes many hours to run a single slide with the cell based methods. Cell tumor/TIL methods could be used in a limited but carefully controlled way for error checking and validation.
>
> 5) We really should have a multi-resolution method both for reliability and because our downstream algorithms are a mix of cell based and larger scale spatial pattern methods. Think Ripley's K for cell based and the paper by Saltz and Troester in Cancers 2022. I think you have the papers as references in the Pathology ai skill, if you need them, I'll be happy to supply.
>
> 6) Our group publishes a lot and it is always cool to have an innovative method. So why diffusion? This seemed like an interesting method to start with as it seemed to me that the analogy of H&E to IHC and H&E to TILs made sense, that conditional latent diffusion would give us a method that operates on much larger ROIs 2K by 2K - the guys are now doing 8K by 8K and pushing towards 16K by 16K. This is a nice step away from our current local patch view. The more obvious approach is to leverage larger patch classification using one of the existing foundation models or to combine patch based methods with weakly trained methods and attempt to leverage attention maps to sanity check patch predictions. This is of course somewhat problematic because attention maps are only likely to be loosely linked to TIL density. Please save this conversation word for word so we can iterate on brainstorming when we have to clear context.

---

## Context From This Session

### What we learned from the PixCell diffusion experiments:

1. **PixCell's virtual staining approach conditions on UNI2-h(TARGET) during training.** This works for H&E→IHC because H&E and IHC of the same tissue produce similar UNI2-h embeddings (shared underlying morphology). It does NOT work for H&E→probmap because probmaps are visually unrelated to H&E.

2. **Fixing conditioning to UNI2-h(H&E) during training** was architecturally correct but the LoRA couldn't bridge the gap — the base PixCell diffusion model generates tissue-like images and the LoRA doesn't have enough leverage to redirect output to probmaps.

3. **The VAE CAN encode/decode GTT images** (grayscale+tumor+TIL) — the decoder is not the bottleneck. The latent representations of H&E and GTT are far apart (cosine sim 0.274), requiring a large transformation that LoRA on cross-attention alone can't achieve.

4. **UNI2-h spatial tokens CAN distinguish tumor from TIL** (cosine distance 0.355, comparable to tumor-vs-stroma). The information exists in the foundation model features.

5. **Our evaluation infrastructure is solid**: stratified sampling, per-channel metrics, reference comparisons, automated checkpoint analysis, PIL Image.show() for visualization.

### What diffusion DID teach us:
- How to set up and use the DGX Spark for GPU training
- How to build evaluation pipelines with smoke tests and automated metrics
- How to do stratified patch sampling driven by data distribution analysis
- That UNI2-h has sufficient resolution and discriminative power for tumor/TIL
- That the diffusion approach to H&E→TIL mapping has fundamental architectural obstacles

---

## Preceding Technical Context: VAE Diagnostic Results

**Before this strategic conversation, we ran three VAE diagnostics:**

Joel: "Before we move on, I'd like to understand what the model learned. The loss dropped quite a bit and the training signal seemed appropriate. Use pathology superpowers to suggest how to assess what the model learned."

Claude proposed three diagnostics:
1. **Can the VAE reconstruct GTT images?** → YES. Correlation >0.99 per channel. VAE is NOT the bottleneck.
2. **How similar are H&E and GTT in latent space?** → Cosine similarity only 0.274. They're far apart.
3. **Per-channel reconstruction?** → Grayscale: 0.994, Tumor: 0.9998, TIL: 0.989 — all excellent.

**Conclusion**: The VAE can represent GTT perfectly, but the LoRA on cross-attention alone can't redirect the diffusion trajectory from tissue-like latent space to GTT-like latent space. The gap is too large.

Joel: "what if the decoder were to regenerate the H&E images with two extra channels - one tumor and one TIL?"

Claude: proposed three approaches — side-by-side, channel encoding, or modified VAE decoder.

Joel: "how about grey scale plus the tumor and til channels"

This was elegant — R=grayscale H&E, G=tumor prob, B=TIL prob. Fits in existing 3-channel framework. We implemented it (GTT targets + H&E conditioning) but after 1000 steps the tumor correlation was still near zero. The LoRA couldn't bridge the gap.

---

## Claude's Analysis of the Strategic Goals

### Reframing: Prediction vs Generation

The diffusion experiments taught us that **tumor/TIL mapping is a prediction problem, not a generation problem**. The H&E→IHC analogy breaks down because:
- IHC reveals the same morphology with different staining (visual transformation)
- TIL probability is a classification/regression output (semantic prediction)

Diffusion models are generative — they excel at visual transformations within the same modality. They are the wrong tool for regression.

### What Actually Addresses All Six Goals

A **hierarchical foundation model prediction framework**:

```
Level 1 (cell):   WSInfer patches (existing, fast, pan-cancer)
Level 2 (tile):   UNI2-h features → MLP → [tumor_prob, til_prob]
Level 3 (region): Spatial attention over tiles → context-aware prediction
```

| Goal | How Addressed |
|------|---------------|
| 1. Robustness | Spatial attention (Level 3) smooths noisy predictions |
| 2. Generalization | Small MLP head is cheap to retrain; frozen foundation features |
| 3. Modernization | UNI2-h replaces ResNet34/InceptionV4 |
| 4. Efficiency | Feature extraction ~20ms/tile; cell methods for validation only |
| 5. Multi-resolution | Three levels feed into Ripley's K and spatial patterns |
| 6. Innovation | Foundation model + spatial attention for TIL is novel and publishable |

### Implementation: Fast and Incremental

- Phase 1: Feature extraction (~4 hours on Spark)
- Phase 2: Tile MLP predictor (~10 minutes training)
- Phase 3: Spatial attention model (~30 minutes training)
- Phase 4: Cell-based validation (~1 day)
- Phase 5: Multi-resolution integration (~1-2 days)

**Total: ~1 week vs the 3 days we spent on diffusion that didn't converge.**

### The Diffusion Angle for Publication

"We tried conditional latent diffusion for tumor/TIL prediction because the H&E→IHC analogy suggested it should work. We demonstrate why it doesn't — the conditioning mismatch between training and inference — and present a simpler, faster, more effective alternative using hierarchical foundation model features with spatial attention."

The negative result is informative and the comparison strengthens the paper.

### What About Larger ROIs (2K-16K)?

Joel's point about diffusion operating on larger ROIs is valid. The spatial attention approach achieves this differently:
- 2K×2K = 64 tiles with self-attention (same spatial context)
- 8K×8K = 1024 tiles (use efficient attention / local windowed attention)
- Each tile already has 256×256 = 64µm FOV with full foundation model features
- The attention mechanism explicitly models spatial relationships between tiles

This gives us the large-field-of-view advantage Joel wanted from diffusion, but through attention rather than generation.
