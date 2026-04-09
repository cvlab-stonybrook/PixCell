# Briefing: Virtual Staining Collaboration Opportunity

**Date**: 2026-04-09
**For**: Stony Brook CS collaborators (CVLab)
**From**: Joel Saltz

---

## Summary

We have an opportunity to build a high-impact collaboration with UNC Chapel
Hill that would create new research directions for both PixCell and ZoomLDM.
The collaboration involves three UNC faculty: Melissa Troester (Epidemiology,
PI of the Carolina Breast Cancer Study), Daiwei Zhang (Biostatistics/Genetics,
first author of iStar — Nature Biotechnology 2024), and Steve Marron
(Statistics, spatial data analysis). The core idea is combining our generative
models with their spatial gene expression prediction methods, validated on a
uniquely rich clinical dataset.

---

## The Technical Opportunity

### What we bring

**PixCell** generates high-fidelity histopathology images conditioned on UNI2-h
embeddings. Virtual staining (H&E → IHC) already works for HER2, ER, PR, Ki67
via LoRA + flow-matching MLP. Single-scale, patch-level generation.

**ZoomLDM** generates coherent multi-scale histopathology images across 8
magnification levels with a single shared-weight model. Joint multi-scale
sampling enforces cross-scale coherence at inference. No existing virtual
staining method operates across magnification levels — this is an open problem.

### What they bring

**iStar** (Zhang, Nature Biotechnology 2024) takes H&E images + sparse spatial
transcriptomics data (Visium spots) and predicts gene expression at
super-resolution (128x enhancement) using a hierarchical vision transformer.
9 minutes end-to-end. His follow-up **iSCALE** (Nature Methods 2025) extends
this to tissues larger than standard capture areas.

**CBCS dataset** (Troester): ~1,500 breast cancer patients with H&E whole-slide
images, tissue microarrays (~4 cores/patient), PAM50 gene expression scores,
multiplex immunofluorescence (CD8/FoxP3/cytokeratin on 1,467 patients), a
44-antibody spatial protein panel, and clinical IHC (ER/PR/HER2/Ki67). 52%
Black enrollment, 10+ year follow-up. Everything digitized.

**Spatial statistics** (Marron): distance-based morphometry for evaluating
spatial relationships between cell types — already published on this cohort.

### What nobody has done

1. **Multi-scale virtual staining**: generating virtual IHC/IF that is coherent
   across zoom levels. ZoomLDM has the multi-scale infrastructure; virtual
   staining has not been attempted in this framework.

2. **Cross-modal validation of virtual staining**: comparing generated virtual
   stains against both real immunostaining AND independently predicted spatial
   gene expression. Current virtual staining papers validate only against
   paired staining data.

3. **Gene expression-conditioned image generation**: using predicted spatial
   gene expression as a conditioning signal for diffusion models, enabling
   virtual staining for markers where no paired training data exists.

---

## Research Questions (CS/ML framing)

### Near-term (publishable now)

**Q1: Can PixCell virtual staining be validated against spatial gene expression
predictions from an independent model?**
- PixCell generates virtual HER2 IHC from H&E
- iStar predicts ERBB2 spatial expression from H&E + PAM50
- Do they agree? Where do they disagree? What does disagreement reveal about
  each method's failure modes?
- This is a novel evaluation framework for generative models in medical imaging

**Q2: Can ZoomLDM be extended to virtual staining?**
- Train ZoomLDM on paired H&E/IHC at multiple magnifications
- Demonstrate consistent virtual staining across zoom levels
- Key ML challenge: the conditioning mechanism needs to handle stain-specific
  features at each scale — tissue-level patterns at low magnification,
  chromogen granularity at high magnification

### Medium-term (high-impact paper)

**Q3: Can spatial gene expression serve as a conditioning signal for
multi-scale image generation?**
- This is the hard ML problem: gene expression values are not images. Our
  archived experiment (docs/archive/til-experiment/) showed that PixCell's
  UNI2-h cross-attention conditioning fails when the target is visually
  unrelated to H&E (cosine similarity ~0.27 in latent space)
- Possible approaches to explore:
  - Spatial conditioning: inject expression as additional latent channels
    rather than cross-attention tokens
  - Per-marker LoRA modulation: expression values scale LoRA weights
  - Hybrid conditioning: UNI2-h for morphology + expression encoder for
    molecular signal
  - Classifier-free guidance with expression as the class signal
- ZoomLDM's magnification-aware conditioning (`EmbeddingViT2_5`) already
  handles heterogeneous inputs (SSL features + mag embedding) — extending
  to include expression may be more natural than in PixCell's architecture

**Q4: Can we do whole-slide virtual multiplex IF?**
- Generate multiple virtual fluorescence channels at arbitrary magnification
  from a single H&E slide
- Directly competes with GigaTIME (Microsoft, Cell 2025) but with multi-scale
  coherence that they don't have
- ZoomLDM's joint multi-scale sampling + iStar's spatial expression = the
  technical pipeline

---

## Staged Approach

**Stage 1 — Cross-modal validation** (all groups contribute immediately):
PixCell virtual staining + iStar gene expression prediction + real IHC ground
truth on CBCS. Novel evaluation framework, publishable standalone.

**Stage 2 — Multi-scale virtual staining**: extend ZoomLDM with virtual
staining capability. Core CS contribution, publishable standalone.

**Stage 3 — Expression-conditioned generation**: the hard ML problem.
Gene expression as conditioning signal for ZoomLDM. High-impact if solved.

---

## Competitive Landscape

| Method | Group | Venue | What it does | What it doesn't do |
|--------|-------|-------|-------------|-------------------|
| PixCell | Us | arXiv 2025 | Virtual staining (4 IHC markers) | Multi-scale, gene expression conditioning |
| ZoomLDM | Us | CVPR 2025 | Multi-scale generation | Virtual staining |
| iStar | Zhang | Nat Biotech 2024 | Spatial gene expression prediction | Image generation |
| GigaTIME | Microsoft | Cell 2025 | Virtual multiplex IF (21 channels) | Multi-scale coherence, spatial transcriptomics |
| GenPercept | ICLR 2025 | Diffusion for dense prediction | Full fine-tuning >> LoRA | Not applied to virtual staining |

**Our unique position**: We are the only group with both a multi-scale
generative model (ZoomLDM) and a virtual staining pipeline (PixCell). Adding
spatial gene expression conditioning from iStar would create a capability
no other group has.

---

## What I Need From You

1. **Assessment**: Is extending ZoomLDM for virtual staining technically
   feasible? The conditioning mechanism handles SSL features — can it handle
   stain-specific targets?

2. **Architecture input**: For gene expression conditioning (Q3), what's the
   most promising approach given ZoomLDM's architecture? The `EmbeddingViT2_5`
   conditioner already processes heterogeneous inputs.

3. **Compute estimate**: What would Stage 2 (ZoomLDM virtual staining
   training) require? The original ZoomLDM trained on 3x H100.

4. **Publication strategy**: Stage 1 could be a methods paper (novel
   evaluation framework). Stage 2 could be a standalone CVPR/ICCV submission.
   Stage 3 is the Nature Methods target. How do we sequence these to maximize
   output?
