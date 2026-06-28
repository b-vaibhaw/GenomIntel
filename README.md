# 🧬 BioIntelligence Platform

Welcome to the **BioIntelligence Platform** — an interactive digital laboratory designed to analyze genetic information (DNA) and brain scans (MRI) in a single, unified workspace. 

This platform allows researchers, analysts, and testers to run DNA annotations, explore genetic ancestries, predict estimated brain age, and run custom testing profiles.

---

## 🌟 What This Platform Does

The platform is divided into two primary types of health analysis:

### 1. 🧬 Genetic & DNA Analysis
* **Ancestry Visualization**: Plots genetic profiles to show how individual genetic backgrounds group together, helping users visualize ancestral backgrounds.
* **Variant Discovery & Lookup**: Allows users to search and filter genetic variants (changes in DNA sequences). It displays population frequency (how common a variant is in the general population) and annotations from global clinical archives indicating whether a variant is classified as pathogenic (linked to a disease) or benign (harmless).
* **Pathogenicity Prediction (Custom Deep Learning)**: The platform features a high-accuracy, custom-built genomic model that scans DNA sequence segments and scores the likelihood that a particular genetic mutation is disease-causing (pathogenic) or harmless (benign).

### 2. 🧠 Brain Scan (MRI) Analysis
* **Brain Structure Volumes**: Analyzes volume measurements from key brain regions (such as the hippocampus, entorhinal cortex, and temporal lobes) that are critical for cognitive function and memory.
* **Estimated Brain Age**: Compares your biological brain structure volumes against a reference population to calculate your **Estimated Brain Age**. If the estimated brain age is lower than or equal to chronological age, it suggests a healthy, youthful brain structure; if it is significantly higher, it identifies a "brain age delta" indicating accelerated structural aging.
* **Brain-Region Correlations**: Explores how different genetic ancestral dimensions overlap with structural brain sizes, identifying patterns in brain anatomy.

---

## 🔬 The Genomic Model — How It Works

The core of the DNA analysis is the **GenomicAttentionClassifier**, a custom Hybrid CNN-Transformer architecture trained end-to-end on real clinical variant data.

### Architecture

| Component | Detail |
|---|---|
| **Embedding** | Learned nucleotide embeddings (`vocab=5`: A, C, G, T, N) + Learned positional embeddings (512 positions) |
| **CNN Block** | Two 1D dilated convolutional layers (`kernel=5`, `dilation=1,2`) with GELU activation and a **residual skip connection** for local motif detection (e.g., promoters, splice sites) |
| **Attention Block** | Multi-Head Self-Attention (`4 heads, dim=64`) with LayerNorm and residual connection for long-range sequence context |
| **Classifier** | Residual Gated Linear MLP (`hidden=128`) → Sigmoid output |
| **Regularization** | Dropout (0.2), AdamW weight decay (1e-3), Cosine Annealing LR schedule |

### Training Strategy

> [!IMPORTANT]
> The model uses two critical techniques that make it robust for real-world inference:

1.  **Balanced Data (50/50 Pathogenic/Benign)**: The training set is explicitly balanced to prevent the model from developing a trivial bias toward the majority class. The `scale_dataset.py` script enforces an equal split of Pathogenic and Benign variants from the NCBI ClinVar archive.

2.  **Spatial Jitter (±20 bp)**: During training, the variant's position within the 512-nucleotide context window is randomly shifted by up to ±20 base pairs. This prevents the model from memorizing that "the answer is always in the center" (the **Center-Bias Problem**) and forces it to learn the actual mutation *signature* (e.g., premature stop codons `TAG/TAA/TGA` for pathogenic, normal codons for benign).

### Why This Is Different

Most simple genomic classifiers overfit to the *position* of a variant inside their fixed-length window. When deployed in the real world — where mutations can appear at any offset — they fail silently. Our spatial jitter strategy is inspired by **data augmentation** techniques widely used in computer vision (random crops, flips) and ensures the model generalizes to variants at arbitrary positions.

### Validated Results

| Metric | Value |
|---|---|
| **Training Accuracy** | 99.93% (30 epochs, 10,000 variants) |
| **Generalization Test** | ✅ Correctly classifies off-center nonsense mutations (`TAA` at position 236) as Pathogenic and normal codons at the same position as Benign |
| **GPU** | NVIDIA RTX 3050 Ti (local) or T4 (Google Colab) |

---

## 🚀 How to Use the App

Here is a step-by-step walkthrough of how you can interact with the platform:

### 🔑 Step 1: Create an Account & Log In
1. **Invite Code**: To prevent unauthorized access, creating an account requires a valid **Invite Code**. (If you are a administrator, you can check or edit your codes in the backend config).
2. **Credits System**: Each user account is created with a set number of **Credits** (e.g., 5 credits). This allows testers to upload and process custom genetic sequences or MRI dimensions, while preventing system abuse.
3. **Password Recovery**: If you forget your password, you can recover it securely using two custom security questions set up during registration.

### 🧪 Step 2: The Interactive Lab (Run Your Own Tests)
Once logged in, navigate to the **Interactive Lab** page:
* **Option A: Estimate Brain Age**: Enter a subject's age, gender, and the volume measurements of 5 critical brain regions. Click **Estimate Brain Age** to get a detailed aging report, visualization of where they sit on the population distribution, and a scientific disclaimer.
* **Option B: DNA Variant Search & Inference**: Paste standard gene sequence strings or enter coordinate details. The custom genomic model will analyze the sequence, score its pathogenicity, and project it visually in a 3D genetic similarity space.
* **Option C: Upload a VCF File**: Upload a patient's VCF file (standard gene sequence format) to parse, clean, and retrieve a full report on all detected clinical variants and their health classifications.

> [!NOTE]
> Running custom tests inside the Interactive Lab consumes **1 Credit** per action. If your credits run out, you can request an administrator to issue a new invite code or top up your account.

### 🕒 Step 3: Analysis History
* Review all your past sessions and tests in the **Analysis History** log.
* All data stored in this log (including emails, session details, and test outputs) is encrypted at the database level so that hackers or database administrators cannot read your private test details.

### 📊 Step 4: Comparison View
* Compare two different analysis sessions side-by-side to track progress over time (e.g., tracking a patient's brain volume changes or comparing two genetic sequences).

---

## 💻 Quick Start (Running Locally)

For non-programmers, the entire platform is pre-packaged to run with a single command. 

### 1. Prerequisites
Make sure you have Python installed on your computer. Then, install the required packages by running this command in your terminal:
```bash
pip install -r requirements.txt
```

### 2. Launch the Platform
To initialize the database, prepare the reference datasets, train the custom DNA model, and launch the web interface, simply run:
```bash
python run_project.py
```
This will automatically open the platform in your web browser at `http://localhost:8501`.

### 3. Cloud-Scale Training (Optional)
To train the model on the full 2.5 GB ClinVar dataset using a free Google Colab GPU, see the [Cloud Training Guide](docs/CLOUD_TRAINING_GUIDE.md).

---

## 📂 Project Structure

```
try_new/
├── app.py                          # Streamlit web application
├── run_project.py                  # One-click launcher
├── requirements.txt                # Python dependencies
├── biointel.db                     # SQLite database (auto-generated)
├── models/
│   └── genomic_attention.pt        # Trained model weights
├── scripts/
│   ├── genomic_model.py            # Model architecture + training loop
│   ├── scale_dataset.py            # ClinVar data ingestion (balanced 50/50)
│   ├── mri_analysis.py             # Brain MRI pipeline + correlations
│   ├── colab_training.py           # Google Colab automation script
│   └── run_pipelines.py            # Idempotent pipeline orchestrator
├── docs/
│   ├── architecture.md             # Full system architecture
│   ├── CLOUD_TRAINING_GUIDE.md     # Step-by-step Colab GPU guide
│   └── dataset_licenses.md         # Data source licenses & citations
└── db/
    └── init_sqlite.py              # Database schema initialization
```

---

## 📂 Reference Datasets & Credits

The reference data and population distributions shown in the app are derived from these global repositories:
* **1000 Genomes Project**: For ancestral genetic profiles.
* **IXI & OASIS-3 Datasets**: For reference MRI brain structural volumes.
* **ClinVar & gnomAD**: For clinical gene sequence annotations and population frequencies.

See [docs/dataset_licenses.md](docs/dataset_licenses.md) for full licensing, citation, and download details.

---

## ⚖️ Disclaimer

> [!IMPORTANT]
> All predictions, estimated brain ages, and pathogenicity labels generated by this platform are for **RESEARCH and EDUCATIONAL USE ONLY**. They do not constitute clinical diagnoses or professional medical advice.
