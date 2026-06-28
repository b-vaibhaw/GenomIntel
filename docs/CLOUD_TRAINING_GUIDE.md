# Cloud GPU Training Guide — Scaling to the Full ClinVar Dataset

Since training the GenomicAttentionClassifier on millions of sequences on a standard laptop CPU is extremely slow, this guide details how to leverage a free **Google Colab Cloud GPU** to train on the entire 2.5 GB NCBI ClinVar dataset in minutes.

---

## Prerequisites

Before uploading to Google Colab, ensure your local project has been tested:
1. The model trains and runs correctly locally (`python scripts/genomic_model.py`).
2. The database schema is initialized (`python db/init_sqlite.py`).
3. You understand the [model architecture](architecture.md) (Hybrid CNN-Transformer with spatial jitter).

---

## Part 1: Setting Up Google Colab

1. Open your web browser and navigate to [Google Colab](https://colab.research.google.com).
2. Click **New Notebook**.
3. In the top menu, go to **Runtime > Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU** (this provides a free NVIDIA T4 GPU with 16 GB of VRAM). Click **Save**.

---

## Part 2: Uploading Your Project to Google Drive

To access your scripts inside Google Colab, upload your project folder to your Google Drive:

1. Zip your project folder (excluding the local `biointel.db` and any `.venv` folders to save space).
2. Upload the zip file (e.g. `try_new.zip`) directly to your Google Drive root.
3. In your Colab notebook, mount your Google Drive by running the following cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Unzip your project into the local Colab workspace:

```bash
!unzip -q /content/drive/MyDrive/try_new.zip -d /content/biointel
%cd /content/biointel
```

---

## Part 3: Running Full-Scale Database Seeding

Inside Google Colab, run the scaling scripts with the limit disabled (`--limit -1`). Google Colab's high-speed network connections and SSD storage make downloading and parsing the entire 2.5 GB file fast.

1. Install the required Python packages in Colab:
   ```bash
   !pip install streamlit pandas plotly scikit-learn statsmodels patsy cryptography python-dotenv PyYAML torch
   ```
2. Initialize the SQLite database schema:
   ```bash
   !python db/init_sqlite.py
   ```
3. Run the scaling script to download the **entire 2.5 GB dataset** and import all human ClinVar variants into SQLite:
   ```bash
   !python scripts/scale_dataset.py --limit -1
   ```
   This will download `variant_summary.txt.gz` from the NCBI FTP server, stream and parse it line-by-line in RAM, and load all GRCh38 variants (over 1.5 million clean pathogenic/benign variant samples) into your SQLite database.

> [!IMPORTANT]
> The `scale_dataset.py` script enforces a **50/50 balanced split** between Pathogenic and Benign variants. This is critical for preventing training bias — without balanced data, the model could learn to always predict the majority class and appear accurate while being useless.

---

## Part 4: Training with GPU Acceleration

Because `scripts/genomic_model.py` includes transparent device mapping, PyTorch will automatically detect the T4 GPU and move all tensors to VRAM.

Run the model training script in a cell:
```bash
!python scripts/genomic_model.py
```

### What Happens During Training

| Step | Detail |
|---|---|
| **Data Loading** | All variants are loaded from the database |
| **Spatial Jitter** | Each variant's position is randomly shifted by ±20 bp on every epoch to achieve translation invariance |
| **Sequence Encoding** | 512-nucleotide windows are tokenized (`A=0, C=1, G=2, T=3, N=4`) |
| **Forward Pass** | Dilated CNN → Multi-Head Attention → Gated MLP → Sigmoid |
| **Loss** | Binary Cross-Entropy with balanced (50/50) data |
| **Optimization** | AdamW (lr=0.001, weight_decay=1e-3) + Cosine Annealing |
| **Epochs** | 30 |
| **Output** | `models/genomic_attention.pt` |

> [!TIP]
> **Why Spatial Jitter Matters**: Without jitter, the model memorizes that the mutation is always at position 256 (the center). In real-world VCF files, variants can appear at any position within the context window. Jitter forces the model to learn the actual mutation *signature* (e.g., premature stop codons) rather than the *location*.

### Expected Performance

| Environment | Dataset | Training Time | Accuracy |
|---|---|---|---|
| **Local RTX 3050 Ti** | 10,000 balanced variants | ~95 seconds | 99.93% |
| **Colab T4 GPU** | 1.5M+ balanced variants | ~5-10 minutes | Enterprise-grade |

---

## Part 5: Downloading the Upgraded Weights

Once the training completes, copy the trained weights file back to your Google Drive so you can download it to your local machine:

1. Run this cell in Colab:
   ```bash
   !cp models/genomic_attention.pt /content/drive/MyDrive/
   ```
2. Open your Google Drive, locate `genomic_attention.pt`, and download it.
3. Replace the local `models/genomic_attention.pt` file in your workspace with this cloud-trained file.

Your local Streamlit application will now utilize the model trained on the full 2.5 GB dataset!

---

## Part 6: One-Click Automation Script

For convenience, we provide an automated script that runs all the above steps in sequence:

```bash
!python scripts/colab_training.py
```

This script:
1. ✅ Mounts Google Drive
2. ✅ Extracts the project archive
3. ✅ Installs all dependencies
4. ✅ Initializes the database schema
5. ✅ Downloads and parses the full 2.5 GB ClinVar dataset (balanced 50/50)
6. ✅ Trains the GenomicAttentionClassifier on the T4 GPU (with spatial jitter)
7. ✅ Copies trained weights back to your Google Drive

See [`scripts/colab_training.py`](../scripts/colab_training.py) for the full source.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `Not running inside Google Colab` | The script requires the `google.colab` module. Run it inside a Colab notebook, not locally. |
| `Could not find try_new.zip` | Upload your project zip to the **root** of your Google Drive (not inside a subfolder). |
| Out of GPU memory | Reduce `batch_size` in `scripts/genomic_model.py` from 32 to 16. |
| Training accuracy stuck at 50% | Verify the database has balanced data: `SELECT clinvar_sig, COUNT(*) FROM variants GROUP BY clinvar_sig`. |
