# Dataset & Model Licenses — BioIntelligence Platform

This document provides full licensing, citation, restriction, and download information for every dataset and model used by the BioIntelligence Platform. **You must read and comply with each license before ingesting any data or deploying any model.**

---

## Datasets

### 1. 1000 Genomes Project — Phase 3

| Field | Value |
|---|---|
| **Full name** | 1000 Genomes Project Phase 3 |
| **Source URL** | https://www.internationalgenome.org/data/ |
| **FTP root** | ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ |
| **License** | Public Domain (no restrictions) |
| **Commercial use** | Permitted without restriction |
| **Subjects** | 2,504 individuals from 26 populations |
| **Data types** | WGS VCF (GRCh37 and GRCh38 liftovers), BAM alignments, sample metadata |

**Citation:**

```
1000 Genomes Project Consortium et al. (2015). A global reference for human
genetic variation. Nature, 526(7571), 68–74.
https://doi.org/10.1038/nature15393
```

**Restrictions:** None. Data is released to the public domain under the Fort Lauderdale principles. The platform uses only the Phase 3 VCF release files (chr*.vcf.gz) and sample metadata (integrated_call_samples_v3.20130502.ALL.panel).

**Download instructions:**

```bash
# Example: download chr22 VCF
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.tbi

# Sample metadata
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
```

The Airflow DAG `genomics_pipeline` handles this download automatically when `INGEST_1000G=true` is set in `.env`.

---

### 2. IXI Dataset

| Field | Value |
|---|---|
| **Full name** | Information eXtraction from Images (IXI) |
| **Source URL** | https://brain-development.org/ixi-dataset/ |
| **Direct download** | https://brain-development.org/ixi-dataset/ (HTTP, no registration required) |
| **License** | Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) |
| **Commercial use** | Permitted with attribution and ShareAlike conditions |
| **Subjects** | ~600 healthy subjects from three London hospitals |
| **Data types** | T1, T2, PD-weighted MRI; DTI; MRA; demographic metadata |

**Citation:**

```
Hammers, A. et al. (2003). Three-dimensional maximum probability atlas of
the human brain, with particular reference to the temporal lobe.
Human Brain Mapping, 19(4), 224–247. https://doi.org/10.1002/hbm.10123

Faillenot, I. et al. (2017). Macroanatomy and 3D probabilistic atlas of
the human insula. NeuroImage, 150, 88–98.
https://doi.org/10.1016/j.neuroimage.2017.01.073
```

**Restrictions:** Derivative works must be released under the same CC BY-SA 3.0 license. Attribution to the IXI project must appear in any publication using this data.

**Download instructions:**

```bash
# T1 images (Guys hospital site, ~4 GB compressed)
wget https://brain-development.org/downloads/IXI-T1.tar -P data/raw/ixi/

# T2 images
wget https://brain-development.org/downloads/IXI-T2.tar -P data/raw/ixi/

# Demographic data
wget https://brain-development.org/downloads/IXI.xls -P data/raw/ixi/
```

The Airflow DAG `neuroimaging_pipeline` handles this download automatically when `INGEST_IXI=true` is set in `.env`.

---

### 3. OASIS-3

| Field | Value |
|---|---|
| **Full name** | Open Access Series of Imaging Studies — Longitudinal (OASIS-3) |
| **Source URL** | https://www.oasis-brains.org/ |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Commercial use** | Permitted with attribution |
| **Subjects** | 1,098 participants (aged 42–95); longitudinal across multiple sessions |
| **Data types** | Longitudinal T1w, T2w, FLAIR MRI; FreeSurfer outputs; cognitive assessments; clinical metadata |

> **⚠️ Data Use Agreement Required.**
> OASIS-3 requires acceptance of a Data Use Agreement (DUA) before downloading. Register and accept the DUA at https://www.oasis-brains.org/#access. After acceptance, you will receive credentials for the XNAT Central repository.
>
> The platform checks for the existence of `data_use_agreements/oasis3_dua_accepted.marker` before triggering the `neuroimaging_pipeline` DAG. Create this file manually after accepting the DUA:
> ```bash
> mkdir -p data_use_agreements
> touch data_use_agreements/oasis3_dua_accepted.marker
> ```

**Citation:**

```
LaMontagne, P.J. et al. (2019). OASIS-3: Longitudinal Neuroimaging, Clinical,
and Cognitive Dataset for Normal Aging and Alzheimer Disease.
medRxiv, https://doi.org/10.1101/2019.12.13.19014902
```

**Download instructions:**

```bash
# After DUA acceptance — use XNAT CLI or the oasis-scripts helper
pip install oasis-scripts
oasis-download --project OASIS3 --session OAS30001_MR_d0129 \
  --output data/raw/oasis3/ --user YOUR_XNAT_USERNAME
```

---

### 4. ClinVar

| Field | Value |
|---|---|
| **Full name** | NCBI ClinVar |
| **Source URL** | https://www.ncbi.nlm.nih.gov/clinvar/ |
| **FTP root** | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ |
| **License** | Public Domain (NCBI/NIH; US Government works) |
| **Commercial use** | Permitted without restriction |
| **Data volume** | ~2.2 million variants (as of 2026); updated weekly |
| **Data types** | VCF (GRCh37 and GRCh38), XML, TSV |

**Citation:**

```
Landrum, M.J. et al. (2018). ClinVar: Improving access to variant
interpretations and supporting evidence. Nucleic Acids Research, 46(D1),
D1062–D1067. https://doi.org/10.1093/nar/gkx1153
```

**Restrictions:** None. ClinVar is a U.S. Government work and is not subject to copyright protection in the United States. International users should verify their local copyright obligations, though the data is commonly considered public domain worldwide.

**Download instructions:**

```bash
# GRCh38 VCF (weekly release)
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi

# Variant summary (TSV, useful for ClinSig lookups)
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
```

The Airflow DAG `genomics_pipeline` handles this download automatically when `INGEST_CLINVAR=true` is set in `.env`.

---

### 5. gnomAD v4.1

| Field | Value |
|---|---|
| **Full name** | Genome Aggregation Database v4.1 |
| **Source URL** | https://gnomad.broadinstitute.org/ |
| **Downloads** | https://gnomad.broadinstitute.org/downloads |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Commercial use** | Permitted with attribution |
| **Subjects** | 125,748 exomes + 76,215 WGS (v4.1) |
| **Data types** | VCF (per-chromosome), Hail MatrixTable, constraint tables |

**Citation:**

```
Chen, S. et al. (2024). A genomic mutational constraint map using variation
in 76,156 human genomes. Nature, 625, 92–100.
https://doi.org/10.1038/s41586-023-06045-0
```

**Restrictions:** Attribution required. The Broad Institute asks that any publication using gnomAD data cite the gnomAD paper and acknowledge the project. Data must not be used to attempt re-identification of individuals.

**Download instructions:**

```bash
# gnomAD v4.1 exomes — chromosome 22 VCF (as example)
gsutil cp gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz .
gsutil cp gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz.tbi .

# Or via AWS S3
aws s3 cp s3://gnomad-public-us-east-1/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz . --no-sign-request
```

> **Note on file sizes:** gnomAD v4.1 whole-exome VCFs total approximately 700 GB uncompressed across all chromosomes. The platform downloads only the chromosomes listed in the `GNOMAD_CHROMS` Airflow Variable (default: `1,2,3,4,5`). Set `GNOMAD_CHROMS=all` to download the full dataset.

---

## Datasets Requiring Free Registration (Not Included by Default)

The following datasets are **not ingested by default** because they require institutional registration. Support for them can be enabled by setting the appropriate Airflow Variable.

### ADNI — Alzheimer's Disease Neuroimaging Initiative

| Field | Value |
|---|---|
| **Registration** | Free; requires PI institutional account at adni.loni.usc.edu |
| **URL** | https://adni.loni.usc.edu/ |
| **License** | ADNI Data Use Agreement (academic research only) |
| **Enable** | Set `INGEST_ADNI=true` in `.env` and create `data_use_agreements/adni_dua_accepted.marker` |

### HCP — Human Connectome Project

| Field | Value |
|---|---|
| **Registration** | Free; requires ConnectomeDB account at db.humanconnectome.org |
| **URL** | https://www.humanconnectome.org/study/hcp-young-adult |
| **License** | HCP Open Access Data Use Terms |
| **Enable** | Set `INGEST_HCP=true` in `.env` and create `data_use_agreements/hcp_dua_accepted.marker` |

---

## Models

### 1. DNABERT-2

| Field | Value |
|---|---|
| **Hugging Face ID** | `zhihan1996/DNABERT-2-117M` |
| **Paper DOI** | https://doi.org/10.48550/arXiv.2306.15006 |
| **License** | Apache License 2.0 |
| **Commercial use** | Permitted |
| **Parameters** | 117 M (BERT-base scale) |
| **Architecture** | Transformer encoder; trained on multi-species DNA with BPE tokenisation |

**Citation:**

```
Zhou, Z. et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark
For Multi-Species Genome. arXiv:2306.15006.
https://doi.org/10.48550/arXiv.2306.15006
```

---

### 2. HyenaDNA

| Field | Value |
|---|---|
| **Hugging Face ID** | `LongSafari/hyenadna-medium-450k-seqlen-hf` |
| **Paper DOI** | https://doi.org/10.48550/arXiv.2306.15794 |
| **License** | Apache License 2.0 |
| **Commercial use** | Permitted |
| **Parameters** | ~6.5 M (subquadratic Hyena operator) |
| **Architecture** | Hyena operator-based SSM; supports context lengths up to 450k nucleotides |

**Citation:**

```
Nguyen, E. et al. (2023). HyenaDNA: Long-Range Genomic Sequence Model at
Single Nucleotide Resolution. arXiv:2306.15794.
https://doi.org/10.48550/arXiv.2306.15794
```

---

### 3. BioMistral-7B

| Field | Value |
|---|---|
| **Hugging Face ID** | `BioMistral/BioMistral-7B` |
| **Paper DOI** | https://doi.org/10.48550/arXiv.2402.10373 |
| **License** | Apache License 2.0 |
| **Commercial use** | Permitted |
| **Parameters** | 7 B |
| **Architecture** | Mistral-7B fine-tuned on PubMed Central open-access corpus |

**Citation:**

```
Labrak, Y. et al. (2024). BioMistral: A Collection of Open-Source Pretrained
Large Language Models for Medical Domains. arXiv:2402.10373.
https://doi.org/10.48550/arXiv.2402.10373
```

---

### 4. Llama-3.1-8B-Instruct

| Field | Value |
|---|---|
| **Hugging Face ID** | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| **Paper DOI** | https://doi.org/10.48550/arXiv.2407.21783 |
| **License** | Meta Llama 3.1 Community License |
| **Commercial use** | Permitted for products with <700 M monthly active users |
| **Parameters** | 8 B |
| **Architecture** | Llama 3.1 transformer; GQA; 128k context window |

> **⚠️ License acceptance required.** You must visit https://ai.meta.com/llama/ and accept Meta's Community License before downloading. After acceptance, generate a Hugging Face access token and set `HF_TOKEN=<your_token>` in `.env`.

**Citation:**

```
Dubey, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
https://doi.org/10.48550/arXiv.2407.21783
```

---

### 5. Qwen3-8B

| Field | Value |
|---|---|
| **Hugging Face ID** | `Qwen/Qwen3-8B` |
| **Paper DOI** | https://doi.org/10.48550/arXiv.2505.09388 |
| **License** | Apache License 2.0 |
| **Commercial use** | Permitted |
| **Parameters** | 8 B |
| **Architecture** | Qwen3 transformer; supports thinking mode / non-thinking mode; 128k context |

**Citation:**

```
Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388.
https://doi.org/10.48550/arXiv.2505.09388
```

---

### 6. FastSurfer

| Field | Value |
|---|---|
| **Docker Hub** | `deepmi/fastsurfer` |
| **GitHub** | https://github.com/Deep-MI/FastSurfer |
| **Paper DOI** | https://doi.org/10.1016/j.neuroimage.2020.117012 |
| **License** | Apache License 2.0 |
| **Commercial use** | Permitted |
| **Architecture** | 3D CNN segmentation network; produces 95 brain parcellations equivalent to FreeSurfer |

> **Note:** FastSurfer produces FreeSurfer-compatible outputs. Downstream analysis using FreeSurfer atlases (Desikan-Killiany, Destrieux) may be subject to FreeSurfer's own license terms. FreeSurfer is free for non-commercial use; see https://surfer.nmr.mgh.harvard.edu/fswiki/License.

**Citation:**

```
Henschel, L. et al. (2020). FastSurfer — A fast and accurate deep learning
based neuroimaging pipeline. NeuroImage, 219, 117012.
https://doi.org/10.1016/j.neuroimage.2020.117012
```

---

## License Compatibility Summary

| Dataset / Model | License | Commercial OK | Attribution | ShareAlike | DUA |
|---|---|---|---|---|---|
| 1000 Genomes | Public Domain | Yes | No | No | No |
| IXI | CC BY-SA 3.0 | Yes | **Required** | **Required** | No |
| OASIS-3 | CC BY 4.0 | Yes | **Required** | No | **Yes** |
| ClinVar | Public Domain | Yes | No | No | No |
| gnomAD | CC BY 4.0 | Yes | **Required** | No | No |
| ADNI | ADNI DUA | Research only | **Required** | No | **Yes** |
| HCP | HCP OA Terms | Conditional | **Required** | No | **Yes** |
| DNABERT-2 | Apache 2.0 | Yes | **Required** | No | No |
| HyenaDNA | Apache 2.0 | Yes | **Required** | No | No |
| BioMistral-7B | Apache 2.0 | Yes | **Required** | No | No |
| Llama-3.1-8B | Llama 3.1 CL | Yes (<700M MAU) | **Required** | No | **Yes** |
| Qwen3-8B | Apache 2.0 | Yes | **Required** | No | No |
| FastSurfer | Apache 2.0 | Yes | **Required** | No | No |
