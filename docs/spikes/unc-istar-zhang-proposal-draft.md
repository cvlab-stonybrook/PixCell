# Draft Proposal: Virtual Staining and Spatial Gene Expression

**Date**: 2026-04-09
**Status**: Strawman for discussion with Troester, Marron, Zhang
**Author**: Joel Saltz (with Claude Code)

---

## Virtual Staining and Spatial Gene Expression: A Multi-Modal Approach to Breast Cancer Tissue Characterization

We propose a collaboration combining generative AI for virtual immunostaining with spatial transcriptomics-based gene expression prediction, validated on the Carolina Breast Cancer Study. The goal is to demonstrate that realistic virtual IHC and IF images can be generated from H&E histology alone, and that these virtual stains are consistent with independently predicted spatial gene expression patterns. The CBCS offers a unique validation resource: ~1,500 patients with digitized H&E TMAs, ubiquitous PAM50 scores, multiplex IF for CD8/FoxP3/cytokeratin, a 44-marker GeoMx protein panel, and clinical IHC for ER/PR/HER2 — all from a diverse, population-based cohort with 52% Black enrollment and 10+ years of recurrence follow-up. No other dataset combines this breadth of molecular, protein-level, and morphological data at this scale and diversity.

The initial phase focuses on cross-modal validation: the Saltz group generates virtual IHC (ER, PR, HER2, Ki67) from H&E using PixCell, a published diffusion foundation model already demonstrated on these markers; Zhang spatializes bulk PAM50 gene expression onto the same H&E sections using iStar, predicting where ESR1, ERBB2, PGR, and MKI67 are expressed at super-resolution; and both predictions are validated against Troester's existing real IHC and multiplex IF ground truth. Marron's spatial statistical framework — the proximity and consistency metrics already developed for CD8+ T cell analysis in this cohort — provides a rigorous quantitative comparison methodology. This phase requires no new wet-lab data; it uses existing CBCS assets and produces a publishable result: the first three-way comparison of virtual staining, spatial gene expression prediction, and real immunostaining on a large diverse cohort.

The longer-term vision extends to multi-scale virtual multiplexing using ZoomLDM, a CVPR 2025 multi-scale diffusion model from the Saltz group, which generates coherent images across magnification levels. Combined with iStar's spatial gene expression predictions as a conditioning signal, this could enable whole-slide virtual multiplex IF — generating multiple virtual fluorescence channels at any zoom level from a single H&E slide. This would position directly against Microsoft's GigaTIME (Cell 2025) but with two distinctive advantages: multi-scale coherence for seamless pathologist navigation, and spatial transcriptomics as an independent molecular conditioning signal. The CBCS cohort, with its diversity, longitudinal follow-up, and rich multi-modal annotations, would be the validation platform throughout.
