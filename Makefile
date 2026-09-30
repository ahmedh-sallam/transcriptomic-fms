## Variables
PYTHON := python
UV := uv

## Default target
.DEFAULT_GOAL := help

## Help target - shows all available commands
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/^## //' | sed 's/^##//'

## Create Python environment using uv
.PHONY: create_environment
create_environment:
	$(UV) venv

## Install project dependencies
.PHONY: requirements
requirements:
	$(UV) pip install -e .

## List available models
.PHONY: list-models
list-models:
	$(UV) run python -m transcriptomic_fms.cli.main list

## Validate h5ad embeddability without loading model packages
## Usage: make pre-embedding-check INPUT=<path/to/file-or-dir.h5ad> [PRECHECK_ARGS="..."]
## Example: make pre-embedding-check INPUT=data/test.h5ad
.PHONY: pre-embedding-check
pre-embedding-check:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make pre-embedding-check INPUT=<path/to/file-or-dir.h5ad> [PRECHECK_ARGS=\"...\"]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make pre-embedding-check INPUT=data/test.h5ad"; \
		exit 1; \
	fi
	$(UV) run python -m transcriptomic_fms.cli.main pre-embedding-check \
		--input $(INPUT) \
		$(PRECHECK_ARGS)

## Generate embeddings locally
## Usage: make embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
## Example: make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS="--n-components 100 --use-hvg"
.PHONY: embed
embed:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Available models:"; \
		uv run python -m transcriptomic_fms.cli.main list; \
		echo ""; \
		echo "Examples:"; \
		echo "  make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy"; \
		echo "  make embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--n-components 100 --use-hvg\""; \
		exit 1; \
	fi
	uv run python -m transcriptomic_fms.cli.main embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Run sensitivity analysis (input gradients / Jacobians per cell)
## Usage: make sensitivity-analysis MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS="..."]
## Example: make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sensitivity
.PHONY: sensitivity-analysis
sensitivity-analysis:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make sensitivity-analysis MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS=\"...\"]"; \
		echo ""; \
		echo "Available models:"; \
		uv run python -m transcriptomic_fms.cli.main list; \
		echo ""; \
		echo "Examples:"; \
		echo "  make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sensitivity"; \
		echo "  make sensitivity-analysis MODEL=geneformer INPUT=data/test.h5ad OUTPUT=output/sens CHUNK_SIZE=100"; \
		exit 1; \
	fi
	$(UV) run python -m transcriptomic_fms.cli.main sensitivity-analysis \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) \
		$(if $(N_CELLS),--n-cells $(N_CELLS),) \
		$(MODEL_ARGS)

## Build model container for HPC
## Usage: make build-container MODEL=<model_name>
## Example: make build-container MODEL=scgpt
.PHONY: build-container
build-container:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make build-container MODEL=<model_name>"; \
		echo ""; \
		echo "Available models with containers:"; \
		ls -d transcriptomic_fms/models/containers/*/ 2>/dev/null | sed 's|transcriptomic_fms/models/containers/||; s|/||' || echo "  (none found)"; \
		exit 1; \
	fi
	@if [ ! -f "transcriptomic_fms/models/containers/$(MODEL)/Singularity.def" ]; then \
		echo "Error: Container definition not found for model $(MODEL)"; \
		echo "Expected: transcriptomic_fms/models/containers/$(MODEL)/Singularity.def"; \
		exit 1; \
	fi
	@if ! command -v apptainer &> /dev/null; then \
		echo "Error: Apptainer is not available. Please load the Apptainer module:"; \
		echo "  module load apptainer"; \
		exit 1; \
	fi
	@echo "Building container for model $(MODEL)..."; \
	CONTAINER_NAME="transcriptomic-fms-$(MODEL).sif"; \
	CONTAINER_DEF="transcriptomic_fms/models/containers/$(MODEL)/Singularity.def"; \
	if [ -f "$$CONTAINER_NAME" ]; then \
		echo "Warning: $$CONTAINER_NAME already exists. Removing it..."; \
		rm -f "$$CONTAINER_NAME"; \
	fi; \
	echo "Building from project root (file paths in Singularity.def are relative to build context)"; \
	echo "Container definition: $$CONTAINER_DEF"; \
	echo "Output container: $$CONTAINER_NAME"; \
	apptainer build "$$CONTAINER_NAME" "$$CONTAINER_DEF" && \
	echo "Container built successfully: $$CONTAINER_NAME"

## Download TranscriptFormer checkpoint(s) (run on a node with internet, e.g. HPC login node)
## Usage: make download-transcriptformer [VARIANT=tf-sapiens|tf-exemplar|tf-metazoa|all-embeddings] [CHECKPOINT_DIR=models/transcriptformer]
## Example: make download-transcriptformer VARIANT=tf-sapiens
TF_VARIANT := $(or $(VARIANT),tf-sapiens)
TF_CHECKPOINT_DIR := $(or $(CHECKPOINT_DIR),models/transcriptformer)
.PHONY: download-transcriptformer
download-transcriptformer:
	@case "$(TF_VARIANT)" in \
		tf-sapiens|tf-exemplar|tf-metazoa|all-embeddings) ;; \
		*) echo "Error: unknown VARIANT=$(TF_VARIANT) (tf-sapiens, tf-exemplar, tf-metazoa, all-embeddings)"; exit 1 ;; \
	esac
	@NAME="$(subst -,_,$(TF_VARIANT))"; \
	if [ -f "$(TF_CHECKPOINT_DIR)/$$NAME/model_weights.pt" ] || [ "$$NAME" = "all_embeddings" -a -d "$(TF_CHECKPOINT_DIR)/$$NAME" ]; then \
		echo "$(TF_CHECKPOINT_DIR)/$$NAME already exists; skipping."; \
		exit 0; \
	fi; \
	mkdir -p "$(TF_CHECKPOINT_DIR)"; \
	echo "Downloading $$NAME to $(TF_CHECKPOINT_DIR)/$$NAME ..."; \
	curl -fL "https://czi-transcriptformer.s3.amazonaws.com/weights/$$NAME.tar.gz" | tar -xz -C "$(TF_CHECKPOINT_DIR)" && \
	echo "Done: $(TF_CHECKPOINT_DIR)/$$NAME"

## Run embedding interactively on HPC (requires interactive node via salloc)
## Usage: make hpc-embed-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
## Note: Run this after getting an interactive node with: salloc --gres=gpu:1 ...
.PHONY: hpc-embed-interactive
hpc-embed-interactive:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-embed-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  # First, get an interactive GPU node:"; \
		echo "  salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G --account=def-jagillis --gres=gpu:1"; \
		echo ""; \
		echo "  # Then run:"; \
		echo "  make hpc-embed-interactive MODEL=scgpt INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--device cuda\""; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	@export PYTHONNOUSERSITE=1; \
	export APPTAINER_USE_GPU=1; \
	apptainer exec --nv \
		--bind "$(shell pwd)/data:/transcriptomic-fms/data" \
		--bind "$(shell pwd)/output:/transcriptomic-fms/output" \
		--bind "$(shell pwd)/models:/transcriptomic-fms/models" \
		"transcriptomic-fms-$(MODEL).sif" \
		python -m transcriptomic_fms.cli.main embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Check GPU compatibility and availability
## Usage: make check-gpu MODEL=<model_name>
## This checks GPU access both on host and inside container
.PHONY: check-gpu
check-gpu:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make check-gpu MODEL=<model_name>"; \
		echo ""; \
		echo "Example: make check-gpu MODEL=scgpt"; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	@echo "========================================="; \
	echo "GPU Compatibility Diagnostics"; \
	echo "========================================="; \
	echo ""; \
	echo "1. HOST SYSTEM GPU CHECK:"; \
	echo "------------------------"; \
	if command -v nvidia-smi &> /dev/null; then \
		echo "✓ nvidia-smi available"; \
		echo ""; \
		echo "GPU Information:"; \
		nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || echo "  (nvidia-smi query failed)"; \
		echo ""; \
		echo "Full GPU status:"; \
		nvidia-smi 2>/dev/null | head -15 || echo "  (nvidia-smi failed)"; \
		echo ""; \
		echo "CUDA_VISIBLE_DEVICES: $${CUDA_VISIBLE_DEVICES:-not set}"; \
		echo "SLURM_GPUS_ON_NODE: $${SLURM_GPUS_ON_NODE:-not set}"; \
		echo "SLURM_GPUS: $${SLURM_GPUS:-not set}"; \
	else \
		echo "✗ nvidia-smi not found (GPU drivers may not be available)"; \
	fi; \
	echo ""; \
	echo "2. CUDA VERSION COMPATIBILITY:"; \
	echo "-----------------------------"; \
	echo "Container CUDA version: 11.7.1"; \
	echo "Container PyTorch: 1.13.1+cu117"; \
	if command -v nvidia-smi &> /dev/null; then \
		echo "Host CUDA version (from driver): 13.0"; \
		echo ""; \
		echo "Compatibility: CUDA 13.0 driver is forward-compatible with CUDA 11.7"; \
		echo "This should work, but MIG devices may need special handling."; \
	fi; \
	echo ""; \
	echo "========================================="

## Run embedding job on HPC (requires SLURM)
## Usage: make hpc-embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS="--arg1 value1 --arg2"]
.PHONY: hpc-embed
hpc-embed:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-embed MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output.npy> [MODEL_ARGS=\"--arg1 value1 --arg2\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make hpc-embed MODEL=pca INPUT=data/test.h5ad OUTPUT=output/embeddings.npy MODEL_ARGS=\"--n-components 100\""; \
		exit 1; \
	fi
	@if [ ! -f "transcriptomic_fms/hpc/run_job.sh" ]; then \
		echo "Error: run_job.sh not found."; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	mkdir -p run_logs
	sbatch transcriptomic_fms/hpc/run_job.sh embed \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(MODEL_ARGS)

## Run sensitivity analysis interactively on HPC (requires interactive node via salloc)
## Usage: make hpc-sensitivity-analysis-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS="..."]
## Note: Run this after getting an interactive node with: salloc --gres=gpu:1 ...
.PHONY: hpc-sensitivity-analysis-interactive
hpc-sensitivity-analysis-interactive:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-sensitivity-analysis-interactive MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS=\"...\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  salloc --time=4:00:00 --nodes=1 --cpus-per-task=8 --mem=64G --account=def-jagillis --gres=gpu:1"; \
		echo "  make hpc-sensitivity-analysis-interactive MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/sensitivity.h5ad N_CELLS=1"; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	@export PYTHONNOUSERSITE=1; \
	export APPTAINER_USE_GPU=1; \
	apptainer exec --nv \
		--bind "$(shell pwd)/data:/transcriptomic-fms/data" \
		--bind "$(shell pwd)/output:/transcriptomic-fms/output" \
		--bind "$(shell pwd)/models:/transcriptomic-fms/models" \
		"transcriptomic-fms-$(MODEL).sif" \
		python -m transcriptomic_fms.cli.main sensitivity-analysis \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) \
		$(if $(N_CELLS),--n-cells $(N_CELLS),) \
		$(MODEL_ARGS)

## Run sensitivity analysis job on HPC (requires SLURM)
## Usage: make hpc-sensitivity-analysis MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS="..."]
.PHONY: hpc-sensitivity-analysis
hpc-sensitivity-analysis:
	@if [ -z "$(MODEL)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make hpc-sensitivity-analysis MODEL=<model_name> INPUT=<path/to/input.h5ad> OUTPUT=<path/to/output> [CHUNK_SIZE=N] [N_CELLS=N] [MODEL_ARGS=\"...\"]"; \
		echo ""; \
		echo "Example:"; \
		echo "  make hpc-sensitivity-analysis MODEL=scimilarity INPUT=data/test.h5ad OUTPUT=output/sensitivity.h5ad N_CELLS=10"; \
		exit 1; \
	fi
	@if [ ! -f "transcriptomic_fms/hpc/run_job.sh" ]; then \
		echo "Error: run_job.sh not found."; \
		exit 1; \
	fi
	@MODEL_NAME="$(MODEL)"; \
	CONTAINER="transcriptomic-fms-$$MODEL_NAME.sif"; \
	if [ ! -f "$$CONTAINER" ]; then \
		echo "Error: Container not found: $$CONTAINER"; \
		echo "Build it with: make build-container MODEL=$$MODEL_NAME"; \
		exit 1; \
	fi
	mkdir -p run_logs
	sbatch transcriptomic_fms/hpc/run_job.sh sensitivity-analysis \
		--model $(MODEL) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		$(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) \
		$(if $(N_CELLS),--n-cells $(N_CELLS),) \
		$(MODEL_ARGS)

## Install model-specific dependencies locally
## Usage: make install-model MODEL=<model_name>
## Example: make install-model MODEL=scgpt
.PHONY: install-model
install-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make install-model MODEL=<model_name>"; \
		echo ""; \
		echo "Available models:"; \
		uv run python -m transcriptomic_fms.cli.main list; \
		exit 1; \
	fi
	@echo "Checking dependencies for $(MODEL)..."
	@DEP_GROUP=$$(uv run python -m transcriptomic_fms.models.get_dep_group $(MODEL) 2>/dev/null); \
	if [ -z "$$DEP_GROUP" ]; then \
		echo "Model $(MODEL) has no special dependencies."; \
	else \
		if [ "$$DEP_GROUP" = "scgpt" ]; then \
			echo ""; \
			echo "Note: Installing scgpt dependencies (includes flash-attn which requires CUDA/nvcc)."; \
			echo "On HPC clusters, ensure CUDA module is loaded:"; \
			echo "  module load cuda/11.7  # or appropriate CUDA version"; \
			echo "  module avail cuda     # to see available versions"; \
			echo ""; \
		fi; \
		echo "Installing dependencies for $(MODEL) (group: $$DEP_GROUP)..."; \
		uv sync --extra $$DEP_GROUP; \
	fi

## Format code
.PHONY: format
format:
	@$(UV) run --preview-features extra-build-dependencies ruff format transcriptomic_fms 2>&1 | grep -v "warning.*extra-build-dependencies.*experimental" || true
	@$(UV) run --preview-features extra-build-dependencies ruff format pyproject.toml 2>&1 | grep -v "warning.*extra-build-dependencies.*experimental" || true

## Lint code
.PHONY: lint
lint:
	@$(UV) run --preview-features extra-build-dependencies ruff check transcriptomic_fms 2>&1 | grep -v "warning.*extra-build-dependencies.*experimental" || true

## Run both format and lint
.PHONY: check
check: format lint
