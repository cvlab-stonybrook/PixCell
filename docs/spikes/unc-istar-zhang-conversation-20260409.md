# Brainstorming Transcript: UNC Virtual Staining Collaboration
**Date**: 2026-04-09
**Participants**: Joel Saltz + Claude Code (Opus 4.6)

---

## Session Overview

This conversation started with project setup for PixCell and evolved into a
deep brainstorming session about a potential four-way collaboration (Saltz,
Troester, Marron, Zhang) on virtual staining validated with spatial gene
expression and the CBCS cohort.

---

## 1. Project Setup

### PixCell repo status
- PixCell cloned at `~/projects/PixCell/` from `cvlab-stonybrook/PixCell`
- Forked to `joelhsaltz/PixCell` for project-specific work
- Remotes: `origin` → joelhsaltz, `upstream` → cvlab-stonybrook

### Archived TIL experiment
- Moved archived H&E → TIL probability map experiment from WSInfer to
  `docs/archive/til-experiment/` (8 docs + 34 figures)
- This experiment (March 2026) established that PixCell's conditioning
  breaks down when target is visually unrelated to H&E
- Key lesson: virtual staining works for IHC/IF (shared morphology),
  not for probability maps

### Environment setup
- Created `.claude/settings.local.json` with permissions adapted from WSInfer
- Created independent DSA Docker stack on port 8081 (WSInfer uses 8080)
- Created CLAUDE.md for the project

### ZoomLDM added to core project
- Joel decided ZoomLDM should be core project technology, not spike-specific
- Added to CLAUDE.md: title, overview, key repos, tech stack, model zoo
- ZoomLDM (CVPR 2025): multi-scale diffusion, same group as PixCell,
  different architecture (UNet vs DiT, VQ-f4 vs SD3.5 VAE, UNI vs UNI2-h)

---

## 2. Collaboration Framing

### Key decisions by Joel
- **Not framed around any specific collaborator** — PixCell virtual staining
  is a standalone research project. Collaborations are treated as spikes.
- **Daiwei Zhang spike** — potential UNC collaborator interested in generating
  IHC and IF from H&E. Zhang is an academic; publication is key motivation.
- **Four-way collaboration** emerging: Saltz, Troester, Marron, Zhang

### Zhang's background
- Assistant Professor (tenure-track) at UNC, Depts of Biostatistics + Genetics
- Joined August 2024, formerly UPenn
- Publications: TESLA (Cell Systems 2023), iStar (Nature Biotechnology 2024),
  iSCALE (Nature Methods 2025)
- Trajectory: progressively larger-scale spatial gene expression from histology
- iStar predicts spatial gene expression at 128x enhancement from H&E + Visium

---

## 3. Publication Concepts Brainstormed

### Four concepts developed
- **Concept A**: Multi-scale virtual staining via ZoomLDM
- **Concept B**: Spatial transcriptomics-guided virtual multiplexing
- **Concept C**: Cross-modal validation framework
- **Concept D**: Whole-slide virtual multiplex IF (iStar + ZoomLDM)

### Joel's strategic framing
> "I like the notion of framing the goal as Concept D and targeting
> Concept A as an initial step."

Claude proposed a staged approach. Joel corrected: Concept A (multi-scale
virtual staining) doesn't involve Zhang, so it's not a collaboration.

### Revised staging agreed upon
- **Stage 1**: Cross-modal validation (both groups contribute immediately).
  PixCell generates virtual IHC; iStar predicts gene expression; compare
  both against real IHC/IF ground truth.
- **Stage 2**: Multi-scale virtual staining (ZoomLDM extension)
- **Stage 3**: Transcriptomics-conditioned generation (full Concept D vision)

---

## 4. Data Landscape Discovery

### Zhang's data
- 10x Xenium breast cancer (313 genes) — likely covers PAM50 panel
- HER2+ breast cancer (HER2ST, Visium)
- Multiple cancer types from TESLA/iStar/iSCALE papers

### Troester's CBCS data (the major asset)
Joel: "Troester has ubiquitous PAM50 results" and "TMAs for around 1000
patients with roughly 4 cores per patient." "Everything Troester does
is digitized."

Research confirmed:
- **CBCS Phase 3**: 1,339 patients (MAKO), ~1,000 with TMAs (~4 cores each)
- **PAM50/ROR-P scores**: ubiquitous across cohort
- **52% Black enrollment** — major differentiator from TCGA (15.9%)
- **10+ years recurrence follow-up**

### Troester TMA marker panels (confirmed from publications)
1. **Multiplex IF** (PLOS Medicine 2025, Walker et al. — Saltz is co-author):
   - 1,467 patients, CD8/FoxP3/Cytokeratin, Aperio Versa 200 at 20x
   - Spatial metrics: proximity, consistency (CD8→tumor distance)
   - Co-authors include Saltz and Marron

2. **44-antibody GeoMx DSP** (Lab Investigation 2021, Troester et al.):
   - 75 patients (6 whole slides + 69 TMA)
   - Full panel: CD3, CD4, CD8, CD20, CD25, CD68, FoxP3, PD-1, PD-L1,
     Ki-67, CTLA4, LAG3, Tim-3, VISTA, and 30+ more
   - ER/PR/HER2 validated at r>0.93

3. **DSP cross-benchmarked** (Cancers 2024): 37 proteins vs chromogenic IHC
   vs OPAL multiplex IF

### MAKO paper (Saltz & Troester, npj Digital Medicine 2026)
- Benchmarked 12 foundation models for PAM50/ROR-P prediction from H&E
- CONCH best (AUC 0.809 CBCS, 0.852 TCGA)
- **Establishes that H&E morphology encodes PAM50 information**
- Externally validated on TCGA BRCA (N=1,050)

### Public datasets
- **HEST-1k** (Mahmood Lab, NeurIPS 2024): 1,229 spatial transcriptomic
  profiles + H&E WSIs, 26 organs — for development/training
- **MIST / HER2Match**: existing PixCell virtual staining training data

---

## 5. The PAM50 Insight

Joel raised that Troester has ubiquitous PAM50 results and asked whether
they could be adapted. Claude identified the key chain:

1. Troester has PAM50 (bulk, no spatial)
2. iStar spatializes PAM50 genes onto H&E sections
3. PixCell generates virtual IHC for PAM50 protein products (ER, HER2, Ki67, PR)
4. Cross-validate against real IHC

**Why this works**: No new wet-lab data needed. iStar trained on Zhang's
Xenium breast data (313 genes, covers PAM50). CBCS patients almost certainly
have clinical ER/PR/HER2/Ki67 IHC as standard workup.

Joel: "I can answer some of this. CBCS is a research study and Troester is
the PI. She needs to ask for permission for each study but as long as she is
a collaborator, it should be fine."

Joel: "There is an extensive set of markers used in the TMA work... I know
that there is a lot of interest in CD8, FoxP3, I think they do CD4."

---

## 6. Collaboration Structure

Joel identified this as a **four-way collaboration**:
- **Troester**: CBCS data (H&E, TMAs, PAM50, IHC, multiplex IF), IRB, cohort
- **Marron**: Spatial statistics (proximity/consistency metrics), methodology
- **Zhang**: iStar/iSCALE spatial gene expression prediction, transcriptomics
- **Saltz**: PixCell/ZoomLDM virtual staining, generative models, DSA, compute

Joel: "Zhang is an academic so very likely the prospect of a high impact
publication will interest him. Key is to nail down what the publication
would consist of."

Joel meeting with all three later today (2026-04-09).

---

## 7. Draft Proposal Generated

Three-paragraph strawman proposal saved to
`docs/spikes/unc-istar-zhang-proposal-draft.md`.

**Title**: "Virtual Staining and Spatial Gene Expression: A Multi-Modal
Approach to Breast Cancer Tissue Characterization"

**Target venues discussed**: Nature Methods, Nature Biotechnology, Cell Systems

**Competitive positioning**: vs. GigaTIME (Microsoft, Cell 2025) — our
advantages are multi-scale coherence and spatial transcriptomics conditioning.

---

## 8. Open Items After This Session

- [ ] Joel meeting with Troester, Marron, Zhang (2026-04-09)
- [ ] Confirm what IHC exists on CBCS TMAs (ER/PR/HER2/Ki67 at minimum)
- [ ] Confirm TMA H&E scans are accessible
- [ ] Assess whether Zhang's Xenium breast data covers PAM50 genes
- [ ] Determine if adjacent unstained TMA sections exist for potential Visium
- [ ] Gauge Zhang's interest in the staged approach and publication timeline
- [ ] Gauge Marron's interest in providing spatial statistics methodology
- [ ] IRB: Troester to confirm computational methods research is covered
