# Transcriptomic Foundation Models

Unified interface for generating embeddings from single-cell foundation models. Supports local execution and HPC deployment via Apptainer containers.

## Quick Start

### Installation

```bash
# Create environment and install base dependencies
make create_environment
make requirements

# Install model-specific dependencies (optional, for local use)
make install-model MODEL=scgpt
# or
uv sync --extra scgpt
```

### List Available Models

```bash
make list-models
```

### Pre-Embedding Check

Validate that an `.h5ad` file or directory is ready for the bundled embedding models without
installing model-specific packages or checkpoints:

```bash
make pre-embedding-check INPUT=data/test.h5ad
```

By default this checks all bundled vocabularies under `transcriptomic_fms/models/vocabs`, including
Geneformer's GC30M token dictionary and TranscriptFormer's human gene vocabulary. For TranscriptFormer
it also warns when `obs['assay']` is missing or contains labels outside the checkpoint's assay vocabulary
(those cells fall back to the `unknown` assay token).

### Generate Embeddings

**Locally:**
```bash
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad
```

**On HPC (interactive):**
```bash
# Get interactive GPU node first
salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G \
    --account=def-jagillis --gres=gpu:1

# Run embedding
make hpc-embed-interactive MODEL=scgpt \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda"
```

**On HPC (batch job):**
```bash
make hpc-embed MODEL=scgpt \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda"
```

**Note:** 
- Output files are automatically prefixed with the model name (e.g., `scgpt_embeddings.h5ad`)
- Output is a barebones AnnData object with embeddings in `X` and `obs` preserved for cell mapping

### Sensitivity analysis

Compute input gradients (Jacobians) and sensitivity metrics per cell in a memory-efficient, chunked pipeline:

```bash
# Single run: output file or directory
make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sensitivity.h5ad

# Chunked run: OUTPUT is a directory; chunk_0_100.h5ad, chunk_100_200.h5ad, ...
make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sens CHUNK_SIZE=100

# Optional cap for quick runs
make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sens N_CELLS=50
```

On HPC (after building the model container), use `make hpc-sensitivity-analysis-interactive` (interactive node) or `make hpc-sensitivity-analysis` (SLURM batch) with the same `MODEL`, `INPUT`, `OUTPUT`, and optional `CHUNK_SIZE`, `N_CELLS`, `MODEL_ARGS`.

**Supported models:** Geneformer, scFoundation (cell embedding only), SCimilarity, scGPT. scConcept and TranscriptFormer are not yet implemented and will report a clear error.

**Output layout (per chunk or single file):** AnnData with `obs` pass-through from input plus `seq_length`, `obsm['X_baseline']` (n_cells, d_emb) cell embeddings, `obsm['jacobian_U']` (n_cells, d_emb, 50) float16 left singular vectors of the Jacobian, and `obsm['jacobian_S']` (n_cells, 50) float32 singular values. The full Jacobian is never stored; it is computed per cell, reshaped to (d_emb, seq_len*d_token), truncated SVD (top 50) is taken, then the Jacobian is discarded. Use `--chunk-size` to process in chunks and manage memory.

## Available Models

### PCA

Baseline PCA embedding with optional HVG selection.

```bash
make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--n-components 100 --use-hvg --n-hvg 2000"
```

**Arguments:**
- `--n-components <int>`: Number of PCA components (default: 50)
- `--use-hvg`: Enable highly variable gene selection
- `--n-hvg <int>`: Number of HVGs to select (default: 2000)

### scGPT

scGPT foundation model for single-cell embeddings.

**Installation:**
```bash
make install-model MODEL=scgpt
```

**Arguments:**
- `--model-dir <path>`: Path to scGPT model directory (auto-downloads if not provided)
- `--n-hvg <int>`: Number of HVGs (default: 1200, use 0 for all genes)
- `--hvg-list <path>`: File with pre-computed HVG list (one gene per line)
- `--n-bins <int>`: Binning resolution (default: 51)
- `--batch-size <int>`: Batch size (default: 64)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)

**Examples:**
```bash
# Auto-download model
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad

# Use existing model
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--model-dir /path/to/scGPT_human --batch-size 32"

# Use all genes (no HVG filtering)
make embed MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--n-hvg 0"
```

### SCimilarity

SCimilarity model for cell state similarity embeddings.

**Installation:**
```bash
make install-model MODEL=scimilarity
```

**Arguments:**
- `--model-path <path>`: Path to SCimilarity model directory (auto-downloads if not provided)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)
- `--use-gpu <bool>`: Explicitly enable/disable GPU

**Examples:**
```bash
# Auto-download model (~30GB, may take time)
make embed MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad

# Use existing model
make embed MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--model-path /path/to/scimilarity/model_v1.1"
```

**Note:** SCimilarity expects raw counts (not log-normalized). The model will automatically log-normalize during preprocessing.

### Geneformer

Geneformer foundation transformer model for single-cell embeddings (V1-10M, cell embeddings).

**Installation:**
```bash
make install-model MODEL=geneformer
```

**Arguments:**
- `--model-path <path>`: Path to Geneformer model directory (auto-downloads if not provided)
- `--batch-size <int>`: Batch size for forward pass (default: 100)
- `--device <str>`: 'cuda' or 'cpu' (auto-detects if not specified)

**Examples:**
```bash
# Auto-download model (requires git-lfs)
make embed MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad

# Use existing model with custom batch size
make embed MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--model-path /path/to/Geneformer --batch-size 50"

# On HPC with custom batch size
make hpc-embed-interactive MODEL=geneformer \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--batch-size 50 --device cuda"
```

**Note:** 
- Geneformer requires Ensembl IDs (not gene symbols) in `var.index` or `var['ensembl_id']` column
- Data should be raw counts with `n_counts` attribute per cell (computed automatically if missing)
- Tokenization creates intermediate .dataset files automatically
- Model auto-download requires git-lfs to be installed

### scConcept

scConcept: contrastive pre-training for technology-agnostic single-cell representations beyond reconstruction ([GitHub](https://github.com/theislab/scConcept)).

**Installation (local, optional):**
```bash
make install-model MODEL=scconcept
```

**Arguments:**
- `--pretrained-model-name <str>`: Name of the pretrained scConcept model to load (default: `Corpus-30M`)
- `--cache-dir <path>`: Cache directory for scConcept/lamin artifacts
- `--device <str>`: `cuda` or `cpu` (auto-detects if not specified)
- `--gene-id-column <str>`: Gene ID column in `adata.var` (default: `gene_id`)

**Examples (local):**
```bash
# Use default pretrained model (Corpus-30M) with auto-download via scConcept
make embed MODEL=scconcept INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad

# Explicitly set cache directory and device
make embed MODEL=scconcept INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--cache-dir ./cache --device cuda"
```

**HPC container:**
```bash
module load apptainer
make build-container MODEL=scconcept

make hpc-embed-interactive MODEL=scconcept \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda"
```

**Notes:**
- scConcept expects Ensembl-style gene IDs via `var['gene_id']` (or `var['ensembl_id']` / index, which are normalized during preprocessing).
- FlashAttention (`flash-attn==2.7.*`) is installed and compiled inside the scConcept container against CUDA 12.1 (via `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`), matching the CUDA runtime exposed on the HPC cluster through `apptainer --nv`.

### scFoundation

scFoundation: foundation model for single-cell transcriptomics ([GitHub](https://github.com/biomap-research/scFoundation)).

**Installation (local, optional):**
```bash
make install-model MODEL=scfoundation
# or
uv sync --extra scfoundation
```

**Prerequisites:**
1. **Model Checkpoint**: Download the pre-trained model checkpoint manually from SharePoint:
   - URL: https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk
   - Place at: `models/scfoundation/models/models.ckpt` (or specify path via `--model-path`)

**Arguments:**
- `--model-path <path>`: Path to model checkpoint file (default: `models/scfoundation/models/models.ckpt`)
- `--ckpt-name <str>`: Checkpoint name (default: `01B-resolution`)
- `--gene-index-path <path>`: Path to gene index file `OS_scRNA_gene_index.19264.tsv` (auto-downloads if not provided)
- `--device <str>`: `cuda` or `cpu` (auto-detects if not specified)
- `--output-type <str>`: Output type - `cell`, `gene`, or `gene_batch` (default: `cell`)
- `--pool-type <str>`: Pooling type for cell embeddings - `all` or `max` (default: `all`, only valid when `output-type=cell`)
- `--tgthighres <str>`: Target high resolution token - `t<number>`, `f<number>`, or `a<number>` (default: `t4`)
  - `t<number>`: targeted high resolution (T=number)
  - `f<number>`: fold change (T/S=number)
  - `a<number>`: addition (T=S+number)
- `--pre-normalized <str>`: Whether input is pre-normalized - `F`, `T`, or `A` (default: `F`)
- `--version <str>`: Model version - `ce` (cell embedding) or `rde` (read depth enhancement) (default: `ce`, only valid when `output-type=cell`)
- `--auto-download <bool>`: Auto-download gene index file if not found (default: `true`)
- `--download-dir <path>`: Directory for downloads (default: `models/scfoundation`)

**Examples (local):**
```bash
# Auto-download gene index, use default model path
make embed MODEL=scfoundation INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda"

# Specify custom model checkpoint and gene index paths
make embed MODEL=scfoundation INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda --model-path /path/to/models.ckpt --gene-index-path /path/to/gene_index.tsv"

# Generate gene embeddings instead of cell embeddings
make embed MODEL=scfoundation INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--output-type gene --device cuda"
```

**HPC container:**
```bash
# Build container (clones scFoundation repo and installs dependencies)
module load apptainer
make build-container MODEL=scfoundation

# Interactive test
make hpc-embed-interactive MODEL=scfoundation \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda --model-path models/scfoundation/models/models.ckpt"

# Batch job
make hpc-embed MODEL=scfoundation \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda --model-path models/scfoundation/models/models.ckpt"
```

### TranscriptFormer

TranscriptFormer: generative cross-species single-cell foundation model from CZI ([GitHub](https://github.com/czi-ai/transcriptformer), [Science](https://www.science.org/doi/10.1126/science.aec8514)). Produces 2048-dimensional mean-pooled cell embeddings.

**Checkpoints** (public S3, extracted to `models/transcriptformer/<variant>/`):

| Variant | Training data | Download |
|---------|---------------|----------|
| `tf-sapiens` (default) | 57M human cells | 1.8 GB |
| `tf-exemplar` | human + mouse, zebrafish, fly, C. elegans | 4.1 GB |
| `tf-metazoa` | 12 species | 6.3 GB |
| `all-embeddings` | ESM-2 gene embeddings for out-of-distribution species | 4.8 GB |

```bash
# Download on a node with internet access (HPC compute nodes usually have none)
make download-transcriptformer VARIANT=tf-sapiens
```

If the checkpoint is missing at run time it is downloaded automatically unless `--no-auto-download` is passed.

**Installation (local, optional):**
```bash
make install-model MODEL=transcriptformer
```
TranscriptFormer pins `torch==2.5.1` and `numpy==2.2.6`, so this extra cannot be installed together with `scconcept`; the container is the recommended way to run it.

**Arguments:**
- `--variant <str>`: `tf-sapiens`, `tf-exemplar` or `tf-metazoa` (default: `tf-sapiens`)
- `--checkpoint-dir <path>`: Directory containing extracted checkpoints (default: `models/transcriptformer`)
- `--checkpoint-path <path>`: Explicit checkpoint directory (overrides `--variant` / `--checkpoint-dir`)
- `--no-auto-download`: Fail instead of downloading a missing checkpoint
- `--batch-size <int>`: Inference batch size (default: 8; use 1-4 on 16 GB GPUs)
- `--device <str>`: `cuda` or `cpu` (auto-detects if not specified)
- `--precision <str>`: `16-mixed` or `32` (default: `16-mixed` on CUDA, `32` on CPU)
- `--assay <str>`: Assay label applied to all cells (e.g. `"10x 3' v3"`)
- `--assay-column <str>`: `obs` column holding assay labels (default: `assay`)
- `--clip-counts <int>`: Per-gene count clip (default: 30, as in training)
- `--normalize-to-scale <float>`: Scale total counts per cell before clipping (default: 0, disabled)
- `--pretrained-embedding <path>`: Comma-separated ESM-2 gene embedding `.h5` file(s) for species outside the checkpoint's training set (from `all-embeddings`)
- `--remove-duplicate-genes`: Keep the first of duplicated Ensembl IDs instead of failing
- `--disable-compile-block-mask`: Disable FlexAttention block-mask compilation (always disabled on CPU)

**Examples:**
```bash
# Default human checkpoint
make embed MODEL=transcriptformer INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad

# Large dataset: TranscriptFormer densifies its input, so process in chunks
make embed MODEL=transcriptformer INPUT=data/test.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--chunk-size 20000 --batch-size 16 --variant tf-metazoa"

# Mouse cells with the human-only checkpoint (out-of-distribution species)
make embed MODEL=transcriptformer INPUT=data/mouse.h5ad OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--pretrained-embedding models/transcriptformer/all_embeddings/mus_musculus_gene.h5"
```

**HPC container:**
```bash
module load apptainer
make build-container MODEL=transcriptformer
make download-transcriptformer VARIANT=tf-sapiens

make hpc-embed-interactive MODEL=transcriptformer \
    INPUT=data/test.h5ad \
    OUTPUT=output/embeddings.h5ad \
    MODEL_ARGS="--device cuda"
```

**Notes:**
- Input must be raw counts in `adata.X` (`adata.raw` is ignored) with Ensembl IDs in `var['ensembl_id']` (or `var['gene_id']` / `var.index`); version suffixes are stripped and genes outside the vocabulary are dropped.
- Checkpoints condition on an assay token read from `obs['assay']` using CELLxGENE assay labels (e.g. `10x 3' v3`, `Smart-seq2`). Missing or unrecognized labels map to `unknown`.
- Up to 2047 expressed genes per cell are used (non-zero genes in `var` order).
- Inference runs on a single GPU; A100 40 GB recommended. The container sets writable Triton / TorchInductor cache directories because block masks are JIT-compiled.

## HPC Deployment

### Building Containers

```bash
module load apptainer
make build-container MODEL=scgpt
make build-container MODEL=scimilarity
make build-container MODEL=geneformer
make build-container MODEL=scfoundation
make build-container MODEL=scconcept
make build-container MODEL=transcriptformer
```

### Container Updates

Containers use editable installs, so code changes are picked up automatically without rebuilding. However, if you add new dependencies or change container configuration, rebuild the container.

### SLURM Job Options

Default options in `run_job.sh`:
- `--time=4:00:00`
- `--mem=64G`
- `--cpus-per-task=8`
- `--gres=gpu:1`

Override by passing options to `sbatch`:
```bash
sbatch --time=8:00:00 --mem=128G transcriptomic_fms/hpc/run_job.sh embed ...
sbatch --time=8:00:00 --mem=128G transcriptomic_fms/hpc/run_job.sh sensitivity-analysis --model scimilarity --input data/test.h5ad --output output/sens.h5ad --n-cells 10
```

## Data Requirements

All models require AnnData objects (`.h5ad` files) with gene identifiers:
- **Most models**: Gene symbols in `var['gene_symbol']` (preferred, singular), `var['gene_symbols']` (plural), `var.index`, `var['feature_name']`, or `var['gene_name']`
- **Geneformer**: Requires Ensembl IDs in `var.index` or `var['ensembl_id']` column
- **TranscriptFormer**: Requires raw counts in `X`, Ensembl IDs in `var['ensembl_id']` (or `var['gene_id']` / `var.index`), and ideally CELLxGENE assay labels in `obs['assay']`

## Architecture

Models inherit from `BaseEmbeddingModel` and implement:
- `preprocess()`: Model-specific preprocessing
- `embed()`: Generate embeddings
- `decode()`: (Optional) Decode embeddings back to expression
- `compute_sensitivity()`: (Optional) Input gradients / Jacobians per cell (Geneformer, scFoundation)

Models are auto-registered via the `@register_model` decorator.

## CLI Reference

```bash
# Embed command
python -m transcriptomic_fms.cli.main embed \
    --model <model_name> \
    --input <path/to/input.h5ad> \
    --output <path/to/output.h5ad> \
    [--chunk-size N] \
    [--model-arg-name <value>]

# Sensitivity analysis (Geneformer, scFoundation)
python -m transcriptomic_fms.cli.main sensitivity-analysis \
    --model <model_name> \
    --input <path/to/input.h5ad> \
    --output <path/to/output.h5ad_or_dir> \
    [--chunk-size N] [--n-cells N] [--batch-size N] \
    [--model-arg-name <value>]

# List models
python -m transcriptomic_fms.cli.main list
```

Model-specific arguments are passed as `--key=value`, `--key value`, or `--flag`.

## Authors

- Ahmed Sallam
