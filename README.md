# BiFuse-CervNet

**A Bidirectional Attention-Gated CNN–ViT Fusion Technique for Cervical Histopathology Grading**

BiFuse-CervNet investigates how local tissue features and neighbouring-patch context can be combined for cervical histopathology analysis. The framework combines a frozen ResNet-50 encoder, a Spatial-Aware Neighbour-Contextual Vision Transformer (SN-ViT), and a learnable feedback gate, with separate heads for histological grade prediction and HSV-saturation proxy classification.

> **Research use only.** The HSV-saturation task distinguishes high- and low-saturation tissue patches. Its results do not measure cancer detection, normal-versus-abnormal classification, or clinically validated tumour localisation.

## Overview

Conventional patch-based analysis can overlook surrounding tissue morphology. BiFuse-CervNet processes each target patch together with its eight spatial neighbours to study whether contextual feature fusion improves prediction.

The manuscript describes two tasks:

- **HSV-saturation proxy classification:** binary prediction of high versus low patch saturation, using automatically derived labels without pathologist validation.
- **Histological grade prediction:** three-class prediction of G1, G2, and G3. Patches inherit their parent patient's grade; GX cases are excluded from grade supervision.

## Architecture

| Component | Description |
| --- | --- |
| Input | A target patch and eight neighbouring patches, each 256 × 256 pixels |
| Backbone | Frozen ResNet-50 encoding all nine patches |
| SN-ViT | Four Transformer layers and eight attention heads; a 10-token sequence comprising CLS, target, and eight neighbours |
| Feedback gate | A learned sigmoid of the target-token output norm, uniformly scaling the target CNN feature map |
| Fusion | Concatenation of a 512-dimensional gated CNN representation and a 256-dimensional context embedding, followed by projection, batch normalisation, ReLU, and dropout |
| Outputs | Binary saturation-proxy head and three-class grade head |

The gate provides a Transformer-to-CNN feedback connection before fusion. It is a scalar modulation, not a spatial attention mask.

The joint training objective is:

$$
\mathcal{L} = 0.6\mathcal{L}_{\mathrm{BCE}} + 0.4\mathcal{L}_{\mathrm{CE}}.
$$

## Dataset and preprocessing

The manuscript uses **190 patients from TCGA-CESC**, with H&E-stained whole-slide images and clinical grade information obtained through the NCI Genomic Data Commons (GDC).

- Patient-level, grade-stratified train/validation/test split: 70% / 15% / 15%.
- HSV-based tissue filtering with a minimum 70% tissue fraction.
- Macenko stain normalisation.
- Non-overlapping 256 × 256 patches at 20× equivalent magnification.
- Stored grid coordinates for retrieving spatial neighbours.
- Proxy labels based on mean patch saturation relative to the median of non-zero patch means across the corpus.

**Label interpretation:** TCGA-CESC is a cancer-confirmed cohort. High/low saturation labels are colour-derived proxies, not expert annotations of tumour, stroma, normal tissue, or abnormal tissue. G1/G2/G3 are tumour differentiation grades, not CIN stages.

Source images and clinical metadata should be obtained from their original provider under the applicable access and use conditions. Dataset acquisition and preprocessing commands still need to be documented against the released code.

## Training protocol described in the manuscript

| Setting | Principal run |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Batch size | 128 |
| Epoch sampling | 100 patches per patient, using deterministic epoch-indexed seeds |
| Gradient clipping | Maximum norm 1.0 |
| Precision | Automatic mixed precision |
| Maximum epochs | 50 |
| Model selection | Validation proxy-task AUROC |
| Early stopping | Patience of 10 epochs |
| Reported hardware | NVIDIA Quadro GV100, 32 GB VRAM |

The principal run evaluates complete validation and test patch sets. The six-variant ablation uses a separate 10-epoch budget and subsampled validation, so its results should be interpreted separately.

## Reported results

These are manuscript-reported results, not a fresh execution of this repository.

### Principal run: HSV-saturation proxy task

Evaluated on 272,289 held-out test patches; threshold-dependent metrics use the default threshold of 0.5.

| Metric | Value |
| --- | ---: |
| AUROC | 0.9986 |
| Sensitivity | 0.9786 |
| Specificity | 0.9801 |
| Macro F1 | 0.9774 |

### Shared short-budget comparison

| Model | Proxy AUROC | Grade weighted F1 | Grade quadratic weighted kappa |
| --- | ---: | ---: | ---: |
| M3: Unidirectional fusion | 0.9971 | 0.5315 | 0.147 |
| M4-Full: Bidirectional fusion | 0.9975 | 0.5342 | 0.165 |

Grade metrics exclude GX cases. Absolute grading performance remains limited. The bidirectional model does not lead every grading metric across all ablations: the CNN-only model reports grade QWK of 0.170.

The six ablation variants are CNN only, ViT only, unidirectional fusion, no neighbours, no grade supervision, and the full model.

## Installation and usage

**Reproducibility documentation is pending verification against the project files.** Exact dependency versions, script names, folder paths, and execution commands have not yet been specified here.

The workflow to document when the implementation is uploaded is:

1. Create the project environment and install the verified dependencies.
2. Obtain TCGA-CESC slides and grade metadata.
3. Prepare the patient-level split and preprocessing configuration.
4. Extract patches, neighbourhood coordinates, and proxy labels.
5. Train the selected model variant using the documented configuration.
6. Evaluate the selected checkpoint and export task-specific metrics.

Before a reproducible release, add the actual environment specification, dataset preparation commands, training and evaluation entry points, split manifests, random seeds, and checkpoint information. No model weights or runnable commands are promised by this README.

## Limitations

- Evaluation is limited to TCGA-CESC, without external validation in the reported study.
- Proxy labels are recoverable from their defining colour statistic and lack expert validation; high proxy-task performance does not demonstrate diagnostic accuracy.
- The reported proxy threshold uses the entire corpus. This should be disclosed when interpreting the evaluation; a training-only threshold is a useful stricter follow-up protocol.
- Patient grades are inherited by patches rather than independently annotated at patch level.
- Short-budget ablations are exploratory and do not establish fully converged performance.
- Post-hoc analysis found gate values near unity (approximately 0.99988), with little variation between patches. Adaptive context-dependent modulation has therefore not been established.
- Grad-CAM visualisations are exploratory and do not establish clinical validity.

## Manuscript and authors

**BiFuse-CervNet: A Bidirectional Attention-Gated CNN-ViT Fusion Technique for Cervical Histopathology Grading**

Nafisat Abdulkadir, Taiwo Kolajo, Emeka Ogbuju, and Francisca Onaolapo Oladipo.

Affiliations listed in the manuscript:

- Federal University Lokoja, Nigeria.
- Confluence University of Science and Technology, Osara, Nigeria (Nafisat Abdulkadir).

Publication link, DOI, and final citation details will be added when available.

## License

A software license has not yet been specified in this README. Add the chosen `LICENSE` file before distributing the project under a particular license. Dataset terms remain separate from the code license.
