# Strategic Research Directions for Saltz Group Computational Pathology

## From Diffusion Models to Segmentation-Free Spatial Immune Profiling

Joel Saltz's group sits at a pivotal inflection point: the PixCell diffusion experiment revealed a fundamental truth — **tumor/TIL mapping is regression, not generation** — while simultaneously confirming that UNI2-h spatial tokens discriminate tumor from TIL at sub-cellular resolution (cosine distance 0.355). The most impactful path forward is not salvaging diffusion models but leveraging those discriminative foundation model features for **segmentation-free spatial immune profiling** across 23 cancer types. This report synthesizes the 2023–2026 state of the art across six strategic questions, drawing on over 100 recent papers, to chart an actionable research agenda that exploits SBU-BMI's unique assets — WSInfer, pan-cancer TIL expertise, TCGA data, and the DGX Spark.

---

## 1. Diffusion Models Can Be Repurposed, But Not the Way It Was Attempted

The Saltz group's conclusion was correct: LoRA on cross-attention alone cannot bridge the **latent space gap** between H&E images and probability maps (cosine similarity 0.274). However, the broader literature reveals that diffusion models *can* perform dense semantic prediction — through fundamentally different architectural strategies than what was tried.

**GenPercept** (Xu et al., ICLR 2025) is the definitive ablation study on repurposing diffusion models for prediction. Its findings directly explain the PixCell failure: (1) the stochastic nature of diffusion has a **slightly negative impact** on deterministic prediction tasks; (2) multi-step generation can be collapsed to **one-step fine-tuning** without performance loss; (3) **full U-Net fine-tuning** significantly outperforms LoRA; and (4) task-specific pixel-level supervision matters more than latent space losses. GenPercept treats segmentation targets as color-coded maps encoded through the same VAE — a workaround for exactly the latent space mismatch problem. Marigold (Ke et al., CVPR 2024 Oral) similarly requires **full U-Net fine-tuning** with the VAE frozen, and works because depth maps share statistical properties with natural images when encoded — a property that probability maps do not share.

The **DMP** framework (Lee et al., CVPR 2024) offers the most theoretically elegant solution: replace stochastic noise with **deterministic interpolation** between input and output (y_t = sqrt(alpha_t) * y + sqrt(1-alpha_t) * x), converting diffusion into progressive morphing. DMP uses LoRA successfully, but only because the deterministic reformulation handles most of the domain gap and outputs remain image-like. **MedSegDiff** (Wu et al., MIDL 2023; V2 at AAAI 2024) takes a different route entirely, training conditional diffusion from scratch where noise is applied to the **ground truth segmentation mask**, not the input image. The model learns to denoise masks conditioned on images — an approach requiring **10+ ensemble predictions** merged via STAPLE.

The paper most directly relevant to PixCell is **PathSegDiff** (Danisetty et al., April 2025), which uses the PathLDM model (from Saltz's own group) as a **frozen feature extractor** rather than a generator. Intermediate U-Net features at specific timesteps (t~50) feed a lightweight fully convolutional network for segmentation. This approach avoids the latent space gap entirely — the diffusion model never generates probability maps. It outperforms VGG16-FCN8 and UNet-ResNet18 on BCSS. The critical structural insight from **Kosmala et al. (2024)** seals the case against LoRA for large domain shifts: LoRA introduces "intruder dimensions" — new high-ranking singular vectors with **near-zero cosine similarity to all pretrained singular vectors** — meaning it fundamentally cannot express the same transformations as full fine-tuning for domain-shifted tasks.

### Ranked Options for H&E to Tumor/TIL Probability Maps

**(A)** Use PixCell as a frozen feature extractor with a discriminative head (PathSegDiff approach) — most promising and preserves pretrained knowledge.

**(B)** Full fine-tuning with deterministic diffusion reformulation (GenPercept/DMP style), though this requires solving the VAE encoding problem for probability maps.

**(C)** Dual-VAE latent diffusion with a separate probability map VAE (MedSegLatDiff approach).

**(D)** Training from scratch (MedSegDiff style).

**What will not work:** LoRA on cross-attention, standard stochastic diffusion for prediction, or treating probability maps as images through the pretrained VAE.

---

## 2. Foundation Model Spatial Tokens Are the Underexplored Frontier

The most striking finding across the entire landscape is this: **nobody is systematically publishing on using ViT spatial token grids for sub-patch prediction in pathology**. Every major foundation model benchmark — UNI2-h, Virchow2, CONCH, Prov-GigaPath, H-Optimus-0 — evaluates tile-level tasks using the **CLS token** or CLS + mean patch token concatenation. The Mahmood Lab's own evaluation code explicitly states: "For all assessments, all models are evaluated using the global representation (e.g. CLS token)." Saltz's finding that individual spatial tokens at **3.5 micron resolution** can discriminate tumor from TIL is genuinely novel.

**CellViT++** (Horst et al., CMPB 2026) comes closest: it extracts deep cell embeddings from transformer spatial tokens during the forward pass at no extra computational cost, averaging all token embeddings in which a detected nucleus is located. With a Virchow encoder, it achieves best standalone cell segmentation performance. **CellVTA** (arXiv April 2025) addresses a key limitation: standard ViT tokenization downsamples by 16x, yielding patches comparable to individual cells. It injects high-resolution spatial information via a CNN adapter through cross-attention, achieving **0.538 mPQ** on CoNIC — state of the art. A 2025 benchmarking study (arXiv 2602.18747) found that standalone foundation models like UNI, Phikon, and Virchow show **poor performance on fine-grained segmentation** when using self-attention maps directly; concatenating attention maps from multiple FMs improved results, but CellViT's architecture-optimized approach still won.

### Foundation Model Landscape (2025-2026)

The foundation model landscape itself has matured considerably. **Campanella et al. (Nature Communications, April 2025)** benchmarked seven FMs and found that **dataset size does not strongly correlate with downstream performance** — model architecture and training strategy matter more. The Atlas/Midnight benchmark (kaiko.ai, MICCAI 2025) showed that **4x fewer WSIs** can achieve state of the art with careful curation, surpassing UNI2-h. **Virchow2** (Paige AI) achieves tile-level F1 of **0.966** across standard benchmarks; Virchow2G (ViT-Giant) reaches 0.971. For TIL classification specifically, multiple FMs achieve **92-93% balanced accuracy** via linear probe on the TCGA TIL dataset.

### Knowledge Distillation

On knowledge distillation, all published work flows in the **wrong direction** for Saltz's needs: from large FMs to smaller FMs (GPFM in Nature BME 2025, H0-mini from Bioptimus, Pathryoshka multi-teacher distillation, XMAG cross-magnification transfer). Nobody has distilled from legacy classifiers (ResNet34/InceptionV4) into FM feature extractors. However, one finding provides an important counterpoint: **Bilal et al. (2025)** showed that a fine-tuned ResNet34 achieved **0.836 balanced accuracy** on MSI prediction in small cohorts, outperforming frozen FM features (CONCH: 0.778) — suggesting legacy architectures with task-specific fine-tuning remain competitive in some settings.

The publication landscape is clear: "FM features + linear probe" alone is the **standard evaluation paradigm** (published routinely in Nature Medicine), but a paper doing *only* this without novelty would be incremental. The publishable angle requires novel spatial token analysis, clinical-scale validation, or architectural innovation.

---

## 3. Attention Maps Fail Spatial Localization, But Spatially-Calibrated Alternatives Exist

A critical finding for anyone building TIL spatial analysis on MIL attention: **attention heatmaps are unfaithful to model strategy in almost all cases**. The **xMIL** framework (Hense et al., NeurIPS 2024), through large-scale experiments on 10 histopathology datasets, demonstrated that the best explanation method depends on task and architecture, with **xMIL-LRP** (Layer-wise Relevance Propagation adapted for MIL) consistently outperforming attention-based explanations. LRP distinguishes positive from negative evidence and is context-sensitive — attention maps cannot do either.

### SMMILe: The Key Paper for Spatial Quantification

The most important paper for Saltz's spatial quantification needs is **SMMILe** (Gao et al., Nature Cancer, December 2025). It proves mathematically that instance-based MIL can achieve superior spatial quantification without compromising WSI-level performance. Representation-based attention methods (CLAM, TransMIL, DSMIL, DTFD-MIL) produce **highly skewed attention maps** focusing on a limited subset of discriminative regions, leading to decreased recall. SMMILe uses superpatch aggregation, instance-level detection and classification, consistency constraints, and MRF-based refinement. Benchmarked across **8 datasets, 6 cancer types, 3,850 WSIs against 9 methods**, it matches or exceeds state of the art at WSI classification while significantly outperforming all methods at spatial quantification — even when baselines use CONCH as encoder.

Notably, **SI-MIL** (Kapse, Pati et al., CVPR 2024) — from Saltz's own group — introduced interpretable-by-design MIL that co-learns from deep features and handcrafted PathExpert features (morphometric and spatial descriptors). It explicitly showed that patches with high attention in deep feature space "may not be optimal in PathExpert feature space."

### Graph Neural Networks for Spatial Context

Graph neural networks provide the most natural framework for spatial context. **Patch-GCN** (Chen et al., Mahmood Lab, MICCAI 2021) treats WSIs as 2D point clouds, hierarchically aggregating features to model local and global topological structures. The **IGT** framework (2024) combining GCN with global attention found that GCN alone outperforms global attention alone — **spatial information is essential**. A comprehensive review by Brussee et al. (Medical Image Analysis, 2025) documents **200+ GNN publications** in histopathology by 2024, with hierarchical GNNs, adaptive graph structure learning, and multimodal GNNs as emerging trends.

For TIL-specific spatial analysis, the pan-GI cancer study (ESMO Open, 2025) found that **spatial relationships between TILs and nearest cancer nuclei** were among the top 9 prognostic features selected by LASSO Cox across 1,700+ patients and 5 GI cancer types. The Saltz group's approach of WSInfer patch classification followed by Ripley's K spatial statistics remains well-suited precisely because it generates genuinely spatially-calibrated per-patch predictions — but upgrading the feature backbone would significantly improve the base predictions.

---

## 4. Multi-Resolution Architectures Bridge Cells to Slides But Need the Right Hierarchy

The multi-scale challenge for TIL analysis is specific: cell-level features feed Ripley's K spatial statistics, while region-level features capture broader immune phenotypes (inflamed, desert, excluded). **HIPT** (Chen et al., CVPR 2022 Oral) remains the foundational architecture for this, processing WSIs as nested sequences across three stages — **16x16 cell tokens, 256x256 patches, 4096x4096 regions, slide**. Each level retains spatial position and produces interpretable attention. For Saltz's needs, HIPT-style hierarchies are preferable to long-context approaches (GigaPath/LongNet) because they produce explicit features at defined scales that map directly to the multi-scale spatial analysis workflow.

### Key Successors to HIPT

**HIGT** (2023) introduces **bidirectional interaction** between resolution levels — cell features inform region predictions and vice versa — solving HIPT's unidirectional information flow. **MS-GCN** (2024) builds pyramidal graph connections from low to high magnification, inherently separating spatial and magnification information. **SPAN** (Wu et al., 2024) introduces sparse pyramid attention that preserves exact spatial relationships through progressive multi-scale representations. **LongMIL** (Li et al., ICLR 2025) provides theoretical grounding for why local attention outperforms global: the **low-rank nature** of long-sequence attention matrices constrains representation, and local attention masks improve rank.

### Cell-Level Analysis State of the Art

For cell-level analysis, **CellViT** (Horst et al., Medical Image Analysis 2024) and **CellViT++** (CMPB 2026) represent the state of the art. CellViT uses a ViT encoder with UNETR-style decoders achieving best nuclei instance segmentation on PanNuke. CellViT++ decouples segmentation from classification, using frozen foundation models as encoders and training only lightweight classifiers — enabling rapid adaptation to new cell type taxonomies. **HistoPLUS** (2025) achieves state of the art across all nuclei types with 5x fewer parameters than CellViT SAM-H, using the H0-mini backbone.

The **HEST-1k** dataset (Jaume et al., Mahmood Lab, NeurIPS 2024 Spotlight) provides the most comprehensive multi-scale benchmark: **1,229 paired spatial transcriptomic profiles with H&E WSIs**, 2.1M expression-morphology pairs, and 76M+ CellViT-segmented nuclei — connecting histology features at patch level with gene expression and cell-level segmentation.

### DGX Spark Compute Considerations

For the DGX Spark (GB10, 128GB), the two-stage pipeline is ideal. UNI2-h feature extraction can process tiles in small batches with features stored in unified memory. A typical WSI at 20x has ~50K-100K tiles; at 1536-dim FP16, this requires only **150-300 MB** — trivially fitting in 128 GB. The memory bandwidth limitation (~273 GB/s vs. ~1 TB/s for RTX 5090) makes inference 3-5x slower but research-practical. Training lightweight classifiers and spatial statistics computation on the ARM CPU cores can proceed in parallel with GPU feature extraction. Full foundation model fine-tuning is impractical on this hardware.

---

## 5. Continual Learning for Pan-Cancer TIL Analysis Is an Open Research Gap

No paper directly addresses continual learning for pan-cancer TIL analysis — this is a genuine open opportunity. The closest work spans several relevant directions.

### Generative Replay Approaches

**GLRCL** (Kumari et al., 2024) proposes Generative Latent Replay using Gaussian Mixture Models to capture previous data distributions without storing original samples — critical for privacy-preserving pathology. Evaluated on three domain-shift scenarios (across stains, across organs, heterogeneous shifts), it **significantly outperforms buffer-free CL methods** and matches rehearsal-based approaches. Its extension **AGLR-CL** handles WSI-level MIL classification, preserving knowledge across organs, diseases, and institutions.

### Adapter-Based Continual Learning

For adapter-based continual learning, **InfLoRA** (Liang & Li, CVPR 2024) adds LoRA branches per task where the dimensionality reduction matrix is designed to **mathematically eliminate interference** with old tasks — guaranteeing new task updates don't degrade old performance. **SD-LoRA** (Wu et al., ICLR 2025) surpasses InfLoRA by 7.68% accuracy on 20-task scenarios by decoupling LoRA magnitude and direction, with scalable rank adjustment. **FM-LoRA** (CVPR Workshops 2025) factorizes task-specific updates into shared and task-dependent components, with advantages growing as task count increases — ideal for a 23-cancer-type model expanding to new types.

### Domain Adaptation Challenges

Domain adaptation work shows that stain normalization alone is **insufficient**. **FLEX** (Nature Communications, 2025) demonstrates that standard normalization fails to mitigate deeper biases from tissue preparation and scanner artifacts, requiring knowledge-guided adaptation that suppresses site-specific patterns. The **de Jong et al. (January 2025)** finding is devastating: all current foundation models encode **medical center signatures more strongly than biological signals** (Robustness Index <1 for 9/10 models). For test-time adaptation, **UAD-FM** (npj Digital Medicine, 2025) introduces causal test-time adaptation with uncertainty decomposition for colorectal cancer pathology — deferring uncertain cases to human experts.

### Recommended Strategy Stack

The recommended strategy stack for Saltz's 23-cancer-type TIL model:

1. Use foundation model features as a domain-agnostic base
2. Deploy InfLoRA or SD-LoRA with cancer-type-specific adapters keeping the base frozen
3. Use GLRCL for privacy-preserving replay across institutions
4. Implement test-time stain adaptation
5. Consider MoE routing (as in ConSurv's MS-MoE) to capture shared versus cancer-type-specific knowledge

---

## 6. The Highest-Impact Paper Leverages What No Other Group Has

### Publication Landscape Assessment

The computational pathology publication landscape has shifted dramatically. **Foundation model fatigue is real**: 20+ pathology FMs now exist, with explicit critiques emerging. Tizhoosh (October 2025) published "Why Foundation Models in Pathology Are Failing," arguing fundamental conceptual mismatches. The Nature BME benchmark (Liang et al., 2025) showed no single FM dominates and bigger is not better. "Yet another pathology foundation model" is firmly oversaturated.

### What Is Hot

**Spatial tumor microenvironment modeling from H&E.** GigaTIME (Microsoft, Cell 2025) translates H&E to 21-channel virtual multiplex immunofluorescence across **14,256 patients and 24 cancer types** — validating that spatial immune prediction from morphology is now a Cell/Nature-level direction. Multimodal integration (pathology + spatial transcriptomics + genomics) is surging with HEST, STPath, and Nicheformer. Foundation model robustness and failure mode analysis has strong demand. Cell-level analysis with FM features (CellViT++) and lightweight/efficient models (FEATHER, ICML 2025) round out the emerging topics.

### Evaluation Requirements

For evaluation, top-tier publications now require: **multi-site validation** (TCGA alone is insufficient), pan-cancer evaluation (>=5 types), comparison against CellViT and current cell segmentation state of the art, clinical outcome correlation, and open-source code with model weights. **Patho-Bench** (Mahmood Lab, 2025) with 95 tasks across 33 datasets is becoming the standard benchmark.

### Ranked Strategic Opportunities

Given Saltz's unique assets, the strategic opportunities rank as follows:

**Highest impact — "Segmentation-free spatial immune profiling via foundation model tokens."** Use UNI2-h 3.5 micron spatial tokens for dense TIL mapping across 23 cancer types without cell segmentation. This directly competes with GigaTIME but from a different angle: direct feature space analysis without cross-modal translation. The **23 cancer types, sub-cellular resolution, spatial statistics expertise, and WSInfer ecosystem** are assets no other group possesses in combination. Target: Nature Biomedical Engineering or Nature Medicine with clinical outcome validation.

**Strong tool contribution — "WSInfer 2.0 with foundation model backends."** Integrating UNI2/CONCH/Virchow2 as backbone options while maintaining QuPath integration (a unique value proposition pathologists care about) would be a high-community-impact contribution. Mahmood Lab's TRIDENT supports 25+ FMs but lacks the pathologist-friendly QuPath integration. Target: Nature Methods or npj Precision Oncology.

**Analytical contribution — "When generation meets discrimination: failure modes of diffusion for spatial prediction."** The PixCell experience, combined with controlled experiments comparing diffusion versus discriminative heads on matched backbone features, plus theoretical grounding from GenPercept and Lotus-2, makes a genuine contribution. Position not as "it didn't work" but as systematic analysis with ablations. Target: Medical Image Analysis or MICCAI.

**The negative result alone is publishable** but gains most value as supplementary analysis within the primary spatial profiling paper. The combination — "here is why the generative approach fails (diffusion latent space gap), and here is why the discriminative approach succeeds (FM spatial tokens encode cell-level morphology)" — creates a compelling narrative arc that top venues want.

---

## Recommended Pipeline Architecture and Immediate Next Steps

The technical path forward integrates findings across all six questions into a concrete pipeline feasible on the DGX Spark:

1. **Feature extraction**: Run UNI2-h on every 224x224 tile to extract the 16x16x1536 spatial token grid. Each token covers ~3.5 microns at 40x — sub-cellular resolution. Store features in unified memory (~150-300 MB per WSI).

2. **Token-level classification**: Train a lightweight MLP (or even linear probe) on spatial tokens for per-token tumor/TIL/stroma classification. This creates a **16x resolution boost** over current WSInfer patch classification without changing the tiling scheme.

3. **Spatial analysis**: From token-level predictions, compute Ripley's K function, spatial entropy, and immune phenotype classification (inflamed/desert/excluded) at multiple distance scales. Compare directly against cell-segmentation-based approaches (CellViT++ pipeline).

4. **Continual learning**: Implement SD-LoRA or InfLoRA adapters per cancer type on the lightweight classification head. Each new cancer type adds a small adapter while preserving performance on existing types.

5. **Validation**: Evaluate across 23 TCGA cancer types, validate externally on CPTAC, correlate with survival and treatment response. Compare against WSInfer ResNet34/InceptionV4 baseline, CellViT++ cell-segmentation pipeline, and GigaTIME virtual mIF approach.

This architecture exploits every existing asset — the 23-cancer WSInfer labels as training data, UNI2-h features already showing discrimination, the DGX Spark's large unified memory for whole-slide feature storage, and decades of spatial analysis expertise that no competing group can match. The key insight driving everything: **the field is moving from "detect cells then analyze" to "analyze spatial features directly"** — and Saltz's group is uniquely positioned to lead that transition.

---

## Key References

### Diffusion for Dense Prediction
- GenPercept (Xu et al., ICLR 2025) — arxiv.org/abs/2403.06090
- Marigold (Ke et al., CVPR 2024 Oral)
- DMP (Lee et al., CVPR 2024)
- MedSegDiff (Wu et al., MIDL 2023) — proceedings.mlr.press/v227/wu24a
- PathSegDiff (Danisetty et al., 2025) — arxiv.org/abs/2504.06950
- LoRA vs Full Fine-tuning (Kosmala et al., 2024) — arxiv.org/abs/2410.21228

### Foundation Models in Pathology
- UNI2-h (Mahmood Lab) — huggingface.co/MahmoodLab/UNI2-h
- CellViT++ (Horst et al., CMPB 2026) — arxiv.org/abs/2501.05269
- CellVTA (2025) — arxiv.org/abs/2504.00784
- FM Benchmarking (Campanella et al., Nat Comm 2025) — arxiv.org/abs/2504.05186
- Virchow2 (Paige AI) — arxiv.org/abs/2408.00738
- Patho-Bench (Mahmood Lab) — github.com/mahmoodlab/Patho-Bench

### Spatial Analysis and MIL
- xMIL (Hense et al., NeurIPS 2024)
- SMMILe (Gao et al., Nature Cancer 2025) — medrxiv.org/content/10.1101/2024.04.25.24306364
- SI-MIL (Kapse et al., CVPR 2024)
- Patch-GCN (Chen et al., MICCAI 2021) — arxiv.org/abs/2107.13048
- GNN Review (Brussee et al., MedIA 2025)

### Multi-Resolution Architectures
- HIPT (Chen et al., CVPR 2022 Oral) — arxiv.org/abs/2206.02647
- HIGT (2023) — arxiv.org/abs/2309.07400
- SPAN (Wu et al., 2024) — arxiv.org/abs/2406.09333
- LongMIL (Li et al., ICLR 2025)
- HEST-1k (Jaume et al., NeurIPS 2024)

### Continual Learning and Robustness
- SD-LoRA (Wu et al., ICLR 2025)
- FLEX (Nature Communications 2025)
- FM Center Robustness (de Jong et al., 2025) — arxiv.org/abs/2501.18055
- FM Survey (2025) — arxiv.org/abs/2504.04045

### Publication Strategy
- GigaTIME (Microsoft, Cell 2025)
- Pan-GI TIL Spatial (ESMO Open 2025)
- "Why FMs Are Failing" (Tizhoosh 2025) — arxiv.org/abs/2510.23807
- FEATHER (Mahmood Lab, ICML 2025) — github.com/mahmoodlab/MIL-Lab
- WSInfer — bmi.stonybrookmedicine.edu
