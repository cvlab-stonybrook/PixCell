# Spike: UNC Collaboration — Daiwei Zhang / iStar

**Date**: 2026-04-09
**Status**: Brainstorming
**Contact**: Daiwei Zhang (UNC, formerly UPenn)

## Background

Daiwei Zhang is interested in generating IHC and IF from H&E slides. His
published work is in a related but distinct domain — predicting spatial gene
expression from histology, not generating stained images. Zhang is an academic;
the prospect of a high-impact publication will be central to his interest.

## Key Papers

### iStar (Nature Biotechnology, September 2024)

**Paper**: Zhang et al., "Inferring super-resolution tissue architecture by
integrating spatial transcriptomics with histology"
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/38168986/)
- [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11260191/)

**What it does**: Predicts spatial gene expression at superpixel resolution
(up to 128x enhancement) from H&E images + spot-level spatial transcriptomics
data (Visium, etc.).

**How it works**:
1. Hierarchical vision transformer (DINO-pretrained on TCGA) extracts features
   at two scales: 16x16 px (cellular) and 256x256 px (tissue structure)
2. Feed-forward neural network predicts superpixel gene expression via weakly
   supervised learning (spot measurements = sums of superpixel expressions)
3. K-means clustering on expression embeddings → tissue segmentation + cell-type
   annotation via marker genes

**Key specs**:
- Inputs: H&E images + spatial transcriptomics spot data
- Outputs: Superpixel gene expression, tissue segmentation, cell-type maps
- Speed: 9 minutes end-to-end (vs. 32 hours for XFuse)
- Validated on: breast, prostate, colorectal, kidney cancer + mouse brain/kidney
- Notably detected tertiary lymphoid structures (TLS) — overlap with WSInfer work

**What it is NOT**: iStar is a prediction/regression model, not a generative
model. It predicts gene expression values, not stained images.

### ZoomLDM (CVPR 2025) — Our Model

**Paper**: Yellapragada, Graikos, Triaridis, Prasanna, Gupta, Saltz, Samaras
- [arXiv 2411.16969](https://arxiv.org/abs/2411.16969)
- [GitHub](https://github.com/cvlab-stonybrook/ZoomLDM)

**What it does**: Multi-scale latent diffusion model for coherent histopathology
image generation across magnification levels (20x down to thumbnail).

**Architecture**:
- OpenAI UNet backbone (LDM-style), VQ-f4 autoencoder (3-channel latent)
- Single shared model across all 8 magnification levels
- Magnification-aware conditioning via learned mag embedding + UNI SSL features
- Conditioner: `EmbeddingViT2_5` — 12-layer ViT encoder producing 65×512 tokens
  fed via cross-attention to UNet
- UNI embeddings (1024-dim) from spatial grid, not UNI2-h (1536-dim)

**Key capabilities**:
- Joint multi-scale sampling: enforces coherence between scales via constrained
  diffusion (gradient updates at inference, not architectural coupling during training)
- Super-resolution via condition inversion: optimize SSL embeddings to match
  low-res input, then generate high-res output
- Large image synthesis: 4096×4096 in ~8 minutes
- MIL features: multi-scale ZoomLDM features outperform UNI for cancer subtyping
  (94.91 AUC) and HRD prediction (88.03 AUC)

**Differences from PixCell**:

| Aspect | ZoomLDM | PixCell |
|--------|---------|---------|
| Backbone | OpenAI UNet (LDM) | DiT/PixArt-Sigma transformer |
| VAE | VQ-f4 (3-channel latent) | SD3.5 VAE (16-channel latent) |
| SSL model | UNI (ViT-L, 1024-dim) | UNI2-h (ViT-Giant, 1536-dim) |
| Training framework | PyTorch Lightning | Accelerate |
| Image sizes | 256×256 at all scales | 256 → 512 → 1024 progressive |
| Multi-scale | Yes (8 magnifications, shared weights) | No (single scale per model) |
| Virtual staining | No | Yes (LoRA + flow MLP) |

**Critical insight**: ZoomLDM and PixCell are sibling models from the same group
but share **no code at the model level**. They use the same conceptual approach
(SSL embeddings as conditioning) but different architectures.

---

## Brainstorming: Publication Concepts

### Concept A: Multi-Scale Virtual Staining via ZoomLDM

**Idea**: Extend ZoomLDM with virtual staining capability. Currently PixCell does
virtual staining at a single scale (1024×1024). ZoomLDM's multi-scale architecture
could generate virtual IHC/IF that is coherent across magnification levels — 
something no current method does.

**Why this is novel**:
- Current virtual staining operates at fixed patch size. Pathologists zoom in
  and out. A method that produces consistent staining across zoom levels would
  be genuinely new.
- ZoomLDM already has the multi-scale infrastructure; adding staining is an
  extension, not a rebuild.

**How it would work**:
1. Train ZoomLDM on paired H&E/IHC data at multiple magnifications (the same
   tissue at 20x, 10x, 5x etc.)
2. The conditioning would be UNI embeddings from the H&E at each scale
3. At inference: given an H&E WSI, extract multi-scale UNI features, condition
   ZoomLDM to generate IHC at each scale, with joint sampling ensuring coherence

**Where iStar fits**: iStar's super-resolution gene expression predictions could
provide additional conditioning signal at fine scales where the spatial
transcriptomics spots are too coarse. iStar resolves expression to near-single-cell;
ZoomLDM resolves imaging to arbitrary magnification. Together: gene-expression-
informed virtual staining at any zoom level.

**Zhang's contribution**: iStar conditioning, spatial transcriptomics expertise,
UNC tissue datasets
**Saltz group contribution**: ZoomLDM architecture, virtual staining pipeline,
GPU infrastructure, PixCell pretrained weights

**Publication strength**: High — first multi-scale virtual staining with
spatial transcriptomics conditioning. Nature Methods or Nature Biotechnology tier.

---

### Concept B: Spatial Transcriptomics-Guided Virtual Multiplexing

**Idea**: Use iStar's gene expression predictions to guide generation of
**multiple virtual stains simultaneously** from a single H&E image. Instead of
training one LoRA per stain (the PixCell approach), use predicted expression
of each marker gene to condition a single model.

**Why this is novel**:
- PixCell virtual staining requires paired H&E/IHC training data for each stain.
  This limits it to stains where paired datasets exist.
- If iStar can predict ERBB2 expression from H&E + Visium data, that prediction
  could replace the need for a real HER2 IHC slide in the training loop.
- Scales to many markers without requiring separate training for each.

**How it would work**:
1. Run iStar on H&E + spatial transcriptomics → per-marker gene expression maps
2. Use expression maps as conditioning signal (alongside or replacing UNI
   embeddings) to generate virtual IHC/IF
3. Validate against real IHC/IF where available

**Challenge — the TIL experiment lesson**: Our archived experiment showed that
conditioning on signals visually unrelated to H&E (probability maps) breaks
PixCell's UNI2-h conditioning mechanism. Gene expression maps are similarly
abstract. This would need a **different conditioning approach** — perhaps
injecting expression values as an additional channel or modifying the
cross-attention mechanism rather than relying on UNI2-h embedding translation.

**Zhang's contribution**: iStar, spatial transcriptomics data, gene expression
expertise
**Saltz group contribution**: Generative model, virtual staining evaluation

**Publication strength**: Very high if it works — directly competes with
GigaTIME but from H&E rather than requiring specialized assays. Risk: the
conditioning problem may be hard to solve.

---

### Concept C: Cross-Modal Validation Framework

**Idea**: Use iStar and PixCell/ZoomLDM as complementary validation for each
other. If PixCell generates virtual HER2 IHC, does iStar's predicted ERBB2
expression correlate spatially? If iStar predicts high CD8 expression, does
virtual IF show CD8+ T-cells in the same regions?

**Why this is valuable**:
- Virtual staining lacks gold-standard validation beyond paired datasets.
  Spatial transcriptomics provides an independent molecular ground truth.
- Neither method references the other during training, so agreement is
  genuinely informative.

**How it would work**:
1. Select tissues with H&E + Visium + real IHC/IF (three-way matched)
2. Run iStar → gene expression predictions
3. Run PixCell → virtual IHC/IF
4. Compare both against real IHC/IF and against each other
5. Analyze where they agree, disagree, and which is more accurate

**Zhang's contribution**: iStar, spatial transcriptomics analysis
**Saltz group contribution**: PixCell virtual staining, DSA visualization

**Publication strength**: Moderate as standalone. Better as a validation
component of Concept A or B.

---

### Concept D: iStar + ZoomLDM for Whole-Slide Virtual Multiplexing

**Idea**: Combine iStar's speed (9 min for gene expression) with ZoomLDM's
large-image synthesis (8 min for 4K) to produce whole-slide virtual multiplex
IF at interactive speed. This directly targets GigaTIME's territory but with
a fundamentally different approach.

**How it would work**:
1. iStar processes H&E WSI + spatial transcriptomics → super-resolution gene
   expression for N markers
2. Gene expression maps are used as additional conditioning for ZoomLDM
3. ZoomLDM generates coherent multi-scale virtual IF for each marker
4. Results displayed in DSA as togglable layers (like fluorescence channels)

**Why ZoomLDM not PixCell**: ZoomLDM's multi-scale coherence is essential for
whole-slide virtual staining. A pathologist zooming from 1.25x overview to 20x
detail needs consistent staining across scales. PixCell operates at fixed
patch size and would produce tiling artifacts at slide level.

**Zhang's contribution**: iStar, spatial transcriptomics, multi-marker expertise
**Saltz group contribution**: ZoomLDM, DSA integration, large-image pipeline

**Publication strength**: Very high — "virtual multiplex IF from H&E at any
magnification" is a direct competitor to GigaTIME with the added advantage of
multi-scale coherence. Nature Methods or Cell Systems tier.

---

## Assessment

**Concept D (whole-slide virtual multiplexing) is the strongest publication
target**, combining both groups' unique strengths in a way neither could achieve
alone. It builds on:
- ZoomLDM's proven multi-scale generation (CVPR 2025)
- iStar's proven gene expression prediction (Nature Biotechnology 2024)
- PixCell's proven virtual staining (existing HER2/ER/PR/Ki67)

**Concept A (multi-scale virtual staining) is the most achievable** as a first
step — it doesn't require solving the gene expression conditioning problem and
stays within ZoomLDM's existing architecture.

**Recommended approach**: Start with Concept A as a concrete deliverable, with
Concept D as the stretch goal that makes the paper high-impact.

## Key Technical Risk

The conditioning problem from the archived TIL experiment applies to Concepts
B and D: gene expression maps are not visually similar to H&E, so the standard
UNI2-h → cross-attention conditioning may not work. Solutions to explore:
- Inject gene expression as spatial conditioning (concatenate with latent,
  not via cross-attention)
- Use expression values to modulate LoRA weights per-marker
- Train a separate encoder for expression → conditioning space
- Use expression only for validation (Concept C), not conditioning

## Lessons from Archived TIL Experiment

See `docs/archive/til-experiment/README.md`. Key constraint: conditioning on
signals visually unrelated to H&E breaks the UNI2-h cross-attention mechanism.
Any design using gene expression as conditioning must address this.

## ZoomLDM (CVPR 2025) — Our Model

**Paper**: Yellapragada, Graikos, Triaridis, Prasanna, Gupta, Saltz, Samaras
- [arXiv 2411.16969](https://arxiv.org/abs/2411.16969)
- [GitHub](https://github.com/cvlab-stonybrook/ZoomLDM)

**What it does**: Multi-scale latent diffusion model — single model generates
coherent histopathology images at 8 magnification levels (20x down to thumbnail).

**Architecture**: OpenAI UNet (LDM-style), VQ-f4 autoencoder, conditioned via
`EmbeddingViT2_5` (12-layer ViT encoding UNI SSL features + learned magnification
embedding → 65×512 tokens via cross-attention). Shared weights across all scales.

**Key capabilities**:
- Joint multi-scale sampling enforces cross-scale coherence at inference
- Super-resolution via condition inversion (no separate SR model needed)
- 4096×4096 synthesis in ~8 min
- MIL features outperform UNI for cancer subtyping (94.91 AUC)

**Differs from PixCell**: UNet vs DiT backbone, VQ-f4 vs SD3.5 VAE, UNI vs
UNI2-h, but same conceptual approach (SSL embeddings as conditioning). No
shared code at the model level.

---

## Zhang's Publication Track

| Paper | Venue | Year | Method | Key capability |
|-------|-------|------|--------|----------------|
| TESLA | Cell Systems | 2023 | Super-resolution gene expression + tissue annotation | Pixel-level tumor/TME cell type annotation on histology, TLS detection |
| iStar | Nature Biotechnology | 2024 | Super-resolution spatial transcriptomics | 128x enhancement, 9 min runtime, hierarchical ViT features |
| iSCALE | Nature Methods | 2025 | Large-tissue spatial transcriptomics | Extends iStar beyond capture area limits, MS brain lesions |

Zhang's trajectory: TESLA → iStar → iSCALE shows a clear progression toward
**multi-scale, large-tissue spatial gene expression prediction from histology**.
This aligns directly with ZoomLDM's multi-scale generation capability.

---

## Data Landscape

### Datasets Zhang likely has access to

**From his own publications**:
- 10x Xenium breast cancer (313 genes, 2 sections) — used in iStar validation
- HER2+ breast cancer (HER2ST, Spatial Transcriptomics platform) — used in iStar
- 10x Xenium mouse brain (248 genes) — used in iStar
- Multiple Visium datasets (prostate, colorectal, kidney cancer, mouse kidney)
- MS brain samples (used in iSCALE)
- IDC, CSCC, melanoma samples (used in TESLA)

**From UNC/Troester group** (Zhang is in Genetics/Biostatistics, Troester is in
Epidemiology — both UNC, Lineberger Cancer Center. Troester is CBCS PI.
Everything Troester does is digitized. Steve Marron is a frequent co-author
on spatial/statistical work.):

- **Carolina Breast Cancer Study (CBCS) Phase 3** (2008–2013):
  - 1,339 patients used in MAKO study with H&E WSIs
  - ~1,000 patients with TMAs (~4 cores/patient ≈ 4,000 cores)
  - Population-based, oversampled young + Black women (52% Black)
  - **Ubiquitous PAM50/ROR-P scores** for all patients
  - 10+ years recurrence follow-up
  - PAM50 subtypes: Luminal A (38.8%), Luminal B (20.5%),
    HER2-enriched (11.1%), Basal (29.6%)
  - IRB: Letter of Intent process via CBCS website for data sharing;
    Troester as PI and collaborator should make access straightforward

- **Troester TMA marker panels** (confirmed from publications):

  **Multiplex IF on TMAs** (Walker et al., PLOS Medicine 2025):
  - 1,467 patients after QC, up to 4 cores per patient
  - Triple IF: **CD8, FoxP3, Cytokeratin** + Hoechst nuclear counterstain
  - Bond Rx automated staining, Aperio Versa 200 at 20x
  - Cell segmentation via QuPath
  - Spatial metrics: proximity (CD8→tumor distance), consistency
  - Authors: Walker, Gao, Wang, De la Cruz, Li, Perou, **Saltz**,
    **Marron**, Hoadley, **Troester**
  - [Paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1004647)

  **GeoMx Digital Spatial Profiling** (Troester et al., Lab Investigation 2021):
  - 44-antibody panel on CBCS3 TMAs (75 patients: 6 whole slides + 69 TMA)
  - **Full panel**: 4-1BB, ARG1, B7-H3, B2M, CD3, **CD4**, **CD8**, CD11c,
    CD14, **CD20**, **CD25**, CD27, CD34, CD40, CD44, CD45, CD45RO, CD56,
    CD66b, CD68, CD80, CD127, CD163, CTLA4, **FoxP3**, GITR, GZMB, HLA-DR,
    IDO1, ICOS, LAG3, OX40L, PD-1, PD-L1, PD-L2, SMA, Tim-3, VISTA,
    FAPalpha, Fibronectin, **Ki-67**, Pan-Cytokeratin, STING
  - ER/PR/HER2 validated (correlation r=0.978/0.934/0.993)
  - [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8140991/)

  **DSP TMA study** (Cancers 2024, DOI 10.3390/cancers17233797):
  - 37 proteins after QC, cross-benchmarked against chromogenic IHC and
    OPAL multiplex IF
  - 368 benign + 204 breast cancer + 110 BC-associated TDLU cores
  - **Paired H&E + IHC + IF on TMAs**

- **MAKO paper** (Saltz & Troester, npj Digital Medicine 2026):
  Kaczmarzyk, Van Alsten, Cozzo, Gupta, Koo, Troester, Hoadley, Saltz.
  Benchmarked 12 pathology foundation models for predicting PAM50-based
  ROR-P scores from H&E WSIs via ABMIL. Best: CONCH (AUC 0.809 CBCS,
  0.852 TCGA). Used HIPPO perturbation experiments to identify tissue
  regions driving recurrence risk predictions. **This establishes that
  H&E morphology encodes PAM50/ROR-P information.**
  [Paper](https://www.nature.com/articles/s41746-025-02334-2)

### Public datasets with paired H&E + spatial transcriptomics + IHC/IF

**HEST-1k** (Mahmood Lab, NeurIPS 2024):
- 1,229 spatial transcriptomic profiles paired with H&E WSIs
- 26 organs, 367 cancer samples, 25 cancer types
- 1.5 million spots, 60 million cells
- Freely available: https://github.com/mahmoodlab/hest
- Includes benchmark for gene expression prediction from histology
- Could serve as large-scale training/validation resource

**MIST** (already used by PixCell):
- Paired H&E/IHC for HER2, ER, PR, Ki67
- Used to train existing PixCell virtual staining adapters

**HER2Match** (Zenodo):
- Paired H&E/HER2 IHC
- Used for PixCell virtual staining

### What we likely need to generate/acquire

For Stage 1 (cross-modal validation):
- Tissue with **all three modalities**: H&E + spatial transcriptomics + real IHC/IF
- The UNC DSP study (IHC + IF on TMAs) is promising if Visium can be run on
  adjacent sections from the same TMA blocks
- Alternatively, HEST-1k samples where matching IHC/IF exists in TCGA or
  other archives

For Stage 2+ (multi-scale virtual staining):
- Paired H&E/IHC at multiple magnifications — straightforward to extract from
  existing paired WSI datasets at different zoom levels

---

### The PAM50 angle

Troester has ubiquitous PAM50 results across CBCS. PAM50 is bulk RNA (no
spatial resolution), but iStar is specifically designed to spatialize bulk/spot
gene expression using histology. The chain:

1. **Troester has PAM50 expression** on ~1,339 CBCS tumors (bulk, per-patient)
2. **iStar spatializes** those 50 genes across H&E sections — predicting where
   each gene is highly vs. lowly expressed at superpixel resolution
3. **PixCell/ZoomLDM generates virtual IHC** for protein products of key PAM50
   genes: ESR1→ER, ERBB2→HER2, MKI67→Ki67, PGR→PR
4. **Cross-validate**: does the virtual staining agree with iStar's spatialized
   gene expression? Does it agree with real IHC?

**Why this is powerful**:
- Troester doesn't need to generate new spatial transcriptomics — she already
  has PAM50 values, and iStar spatializes them using existing H&E
- MAKO already showed that H&E encodes enough PAM50/ROR-P information for
  foundation models to predict it (CONCH AUC 0.809)
- 4 of the 50 PAM50 genes have direct IHC equivalents (ER, PR, HER2, Ki67)
  that PixCell already does virtual staining for — immediate validation
- The ~1,000 CBCS patients with TMAs (~4 cores each) likely have ER/PR/HER2/Ki67
  IHC already done (standard clinical workup) — this IS the ground truth

**Catch**: iStar needs spatial transcriptomics training data to learn the
histology→expression mapping. Zhang's Xenium breast cancer data (313 genes)
almost certainly covers the PAM50 panel and could serve as the training set.
iStar then generalizes to CBCS sections where only H&E + bulk PAM50 exists.

**Questions for Troester about PAM50 and TMAs**:
1. How many CBCS patients have PAM50 results? (Likely all 1,339 from MAKO)
2. Do the ~1,000 TMA patients overlap with the MAKO WSI cohort?
3. What IHC has been done on CBCS TMAs? ER/PR/HER2/Ki67 at minimum (clinical)?
   Any immune markers (CD3, CD8, CD20)?
4. Is there any multiplex IF (OPAL or similar) on CBCS tissue?
5. Are the TMA H&E scans digitized and accessible?
6. Are there adjacent unstained sections from TMA blocks available for
   potential Visium profiling on a subset?
7. What's the IRB situation for computational methods research on CBCS tissue?

---

## Staged Approach

**Goal**: Whole-slide virtual multiplex IF from H&E — generating multi-channel
immunofluorescence at any magnification, conditioned on spatial transcriptomics-
derived gene expression. Positions directly against GigaTIME with two advantages:
multi-scale coherence and spatial transcriptomics as molecular conditioning.

**Stage 1 — Cross-modal validation (both groups contribute immediately)**:
PixCell generates virtual IHC stains from H&E; iStar predicts corresponding
gene expression from H&E + Visium; compare both against real IHC/IF ground
truth and against each other. This requires Zhang to bring spatial
transcriptomics data for tissues where IHC/IF ground truth also exists. Both
groups produce real work, generate a joint result, and build the relationship.

**Stage 2 — Multi-scale virtual staining (ZoomLDM extension)**:
Train ZoomLDM on paired H&E/IHC at multiple magnifications. Demonstrate
consistent virtual staining across zoom levels — a capability no current
method has. This validates the core infrastructure for whole-slide generation.

**Stage 3 — Transcriptomics-conditioned generation (the full vision)**:
Introduce iStar's spatial transcriptomics predictions as an additional
conditioning channel for ZoomLDM. First for validation enhancement, then as
direct input enabling virtual staining for markers where no paired IHC exists.

## Open Questions for Daiwei

1. Does he have tissue where **all three modalities** exist: H&E + spatial
   transcriptomics (Visium/Xenium) + real IHC or IF? This is the critical
   requirement for Stage 1.
2. The UNC DSP study has IHC + OPAL IF on TMAs — could Visium be run on
   adjacent sections from the same blocks? Would give us the triple-modality
   data we need.
3. Is there any connection to the CBCS TMA data (5,907 cores, 1,655 patients)?
   H&E exists; if IHC/IF or spatial transcriptomics could be added to a subset,
   this is a massive resource.
4. What cancer types and markers matter most to him? His published work spans
   breast, prostate, colorectal, kidney, melanoma, brain.
5. What compute does he have? ZoomLDM training used 3x H100.
6. Is iSCALE (the large-tissue extension) relevant here, or is iStar sufficient
   for the tissues we'd target?
7. Would the staged approach (validation → multi-scale → transcriptomics
   conditioning) work for his publication timeline?

## Next Steps

- [ ] Initial conversation with Daiwei to present staged approach and gauge interest
- [ ] Identify triple-modality datasets (H&E + ST + IHC/IF) — prioritize UNC
      DSP study TMAs and HEST-1k public data
- [ ] Ask about CBCS TMA blocks — potential for Visium on adjacent sections
- [ ] Stage 1 prototype: run PixCell virtual staining + iStar gene expression
      on overlapping tissue, compare against real IHC/IF
- [ ] Stage 2: train ZoomLDM on paired H&E/IHC at multiple scales
- [ ] Stage 3: design conditioning mechanism for gene expression input
