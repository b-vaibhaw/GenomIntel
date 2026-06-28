#!/usr/bin/env python3
import os
import sys
import gzip
import urllib.request
import sqlite3
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("scale_dataset")

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scripts_dir)
DB_FILE = os.path.join(project_dir, "biointel.db")
TEMP_GZ = os.path.join(project_dir, "data", "variant_summary.txt.gz")

def download_clinvar():
    """Download ClinVar summary dataset if not present."""
    os.makedirs(os.path.dirname(TEMP_GZ), exist_ok=True)
    if os.path.exists(TEMP_GZ):
        log.info(f"ClinVar dataset already exists locally at: {TEMP_GZ}")
        return True

    log.info(f"Downloading official NCBI ClinVar dataset (~160MB compressed)...")
    log.info(f"Source: {CLINVAR_URL}")
    
    try:
        # Download with simple progress updates
        with urllib.request.urlopen(CLINVAR_URL) as response, open(TEMP_GZ, 'wb') as out_file:
            meta = response.info()
            total_size = int(meta.get("Content-Length", 0))
            chunk_size = 1024 * 1024 # 1MB chunks
            downloaded = 0
            
            while True:
                buffer = response.read(chunk_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    log.info(f"Download Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                else:
                    log.info(f"Downloaded: {downloaded / (1024*1024):.1f} MB")
        log.info("✔ Download completed successfully.")
        return True
    except Exception as e:
        log.error(f"Failed to download ClinVar dataset: {e}")
        return False

def parse_and_insert(limit=10000):
    """Stream raw gz ClinVar dataset, parse GRCh38 human variants, and bulk insert into SQLite."""
    if not os.path.exists(TEMP_GZ):
        log.error("ClinVar source file not found. Download aborted.")
        return False

    log.info("Streaming and parsing ClinVar dataset...")
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Clear previous ClinVar records to ensure clean balanced set
    log.info("Clearing previous ClinVar variants from database...")
    cur.execute("DELETE FROM variants WHERE dataset_source = 'ClinVar'")
    conn.commit()
    
    # Track metrics
    inserted = 0
    skipped_label = 0
    skipped_assembly = 0
    
    pathogenic_count = 0
    benign_count = 0
    target_per_class = limit // 2 if limit != -1 else float('inf')
    
    # Batch collection
    batch = []
    batch_size = 5000
    
    col_map = {}
    
    # Open gzip file and read line-by-line
    with gzip.open(TEMP_GZ, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#') or line_num == 1:
                # Parse header
                header = line.strip().lstrip('#').lower().split('\t')
                col_map = {col: idx for idx, col in enumerate(header)}
                log.info(f"Parsed ClinVar header columns: {list(col_map.keys())[:10]}...")
                continue
                
            cols = line.strip().split('\t')
            if len(cols) <= max(col_map.values(), default=0):
                continue
                
            # Filter Assembly
            assembly_idx = col_map.get('assembly')
            if assembly_idx is not None and cols[assembly_idx] not in ('GRCh38', 'NCBI38'):
                skipped_assembly += 1
                continue
                
            # Filter Significance labels
            sig_idx = col_map.get('clinicalsignificance')
            if sig_idx is None:
                continue
            sig = cols[sig_idx]
            
            # Map significance labels to primary categories
            norm_sig = None
            if sig in ("Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"):
                norm_sig = "Pathogenic"
            elif sig in ("Benign", "Likely benign", "Benign/Likely benign"):
                norm_sig = "Benign"
                
            if not norm_sig:
                skipped_label += 1
                continue
                
            # Balance checks
            if norm_sig == "Pathogenic":
                if pathogenic_count >= target_per_class:
                    continue
                pathogenic_count += 1
            elif norm_sig == "Benign":
                if benign_count >= target_per_class:
                    continue
                benign_count += 1
                
            # Retrieve variant variables
            chrom_idx = col_map.get('chromosome')
            pos_idx = col_map.get('positionvcf')
            ref_idx = col_map.get('referenceallelevcf')
            alt_idx = col_map.get('alternateallelevcf')
            gene_idx = col_map.get('genesymbol')
            
            # Check dbSNP rsID key
            rs_idx = None
            for key in col_map:
                if 'dbsnp' in key or 'rs#' in key:
                    rs_idx = col_map[key]
                    break
                    
            if None in (chrom_idx, pos_idx, ref_idx, alt_idx):
                continue
                
            chrom = cols[chrom_idx]
            pos_val = cols[pos_idx]
            ref = cols[ref_idx]
            alt = cols[alt_idx]
            
            # Ignore placeholder values
            if pos_val == '-1' or ref == '-' or alt == '-':
                continue
                
            try:
                pos = int(pos_val)
            except ValueError:
                continue
                
            gene = cols[gene_idx] if gene_idx is not None else "Unknown"
            rsid = f"rs{cols[rs_idx]}" if (rs_idx is not None and cols[rs_idx] != '-1') else None
            
            # Make variant ID matching standard: chrom:pos:ref:alt
            variant_id = f"{chrom}:{pos}:{ref}:{alt}"
            
            batch.append((
                variant_id, chrom, pos, ref, alt, rsid, gene, "Variant annotation", norm_sig, 0.01, "ClinVar"
            ))
            
            if len(batch) >= batch_size:
                try:
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO variants (
                            variant_id, chrom, pos, ref, alt, rsid, gene_symbol, consequence, clinvar_sig, gnomad_af, dataset_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch
                    )
                    conn.commit()
                    inserted += len(batch)
                except Exception as e:
                    log.error(f"Database insertion failed: {e}")
                batch = []
                
            # Exit if both categories have reached their target
            if pathogenic_count >= target_per_class and benign_count >= target_per_class:
                break
                    
        # Flush remaining batch
        if batch:
            try:
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO variants (
                        variant_id, chrom, pos, ref, alt, rsid, gene_symbol, consequence, clinvar_sig, gnomad_af, dataset_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch
                )
                conn.commit()
                inserted += len(batch)
            except Exception as e:
                log.error(f"Database insertion failed: {e}")
                
    conn.close()
    
    log.info(f"✔ Parsing and seeding completed.")
    log.info(f"   - Added/Updated variants: {inserted} (Pathogenic: {pathogenic_count}, Benign: {benign_count})")
    log.info(f"   - Skipped non-human genomes: {skipped_assembly}")
    log.info(f"   - Skipped other labels: {skipped_label}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Scale SQLite variants database using NCBI ClinVar summary.")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum variants to import (default: 10000. Set to -1 for all).")
    args = parser.parse_args()
    
    if download_clinvar():
        parse_and_insert(limit=args.limit)

if __name__ == "__main__":
    main()
