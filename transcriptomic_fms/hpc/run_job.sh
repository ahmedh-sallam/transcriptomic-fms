#!/bin/bash
#SBATCH --job-name=transcriptomic-fms
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=def-jagillis
#SBATCH --output=run_logs/%x_%j.out
#SBATCH --error=run_logs/%x_%j.err
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1

# Load required modules
# Temporarily suspend strict error handling to prevent exit on profile warnings
set +e
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
set -e

# Load required modules
module load apptainer

# CUDA version handling
# Set CUDA_VERSION environment variable to load a specific CUDA module version
# Example: CUDA_VERSION=12.9 sbatch transcriptomic_fms/hpc/run_job.sh ...
# Or check available versions with: module avail cuda
if [ -n "${CUDA_VERSION:-}" ]; then
    echo "Loading CUDA module version: $CUDA_VERSION"
    module load cuda/$CUDA_VERSION || {
        echo "Warning: Failed to load cuda/$CUDA_VERSION"
        echo "Available CUDA versions (check with: module avail cuda):"
        module avail cuda 2>&1 | head -20 || echo "Could not list available CUDA modules"
    }
else
    # Try to detect if CUDA is needed and load default if available
    # This is optional - CUDA is usually in the container
    if command -v nvidia-smi &> /dev/null; then
        CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1,2)
        echo "Detected NVIDIA driver version: $CUDA_DRIVER_VERSION"
        echo "Note: CUDA is available in the container. Host CUDA module may not be needed."
    fi
fi

# Get project root directory (parent of transcriptomic_fms/hpc directory)
# Use SLURM_SUBMIT_DIR if available and valid (directory where sbatch was run), otherwise use script location
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
    # We're in a SLURM job, use the submit directory
    PROJ_ROOT="$SLURM_SUBMIT_DIR"
    cd "$PROJ_ROOT" || {
        echo "Warning: Could not cd to SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR, using script location instead"
        HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        PROJ_ROOT=$(cd "$HPC_DIR/../.." && pwd)
        cd "$PROJ_ROOT"
    }
else
    # Not in SLURM or SLURM_SUBMIT_DIR is invalid, use script location
    HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PROJ_ROOT=$(cd "$HPC_DIR/../.." && pwd)
    cd "$PROJ_ROOT"
fi

# Detect model name from command arguments to select appropriate container
MODEL_NAME=""
PREV_ARG=""
for arg in "$@"; do
    if [ "$PREV_ARG" = "--model" ] || [ "$PREV_ARG" = "-m" ]; then
        MODEL_NAME="$arg"
        break
    elif [[ "$arg" == --model=* ]]; then
        MODEL_NAME="${arg#--model=}"
        break
    fi
    PREV_ARG="$arg"
done

# Determine which container to use
if [ -n "$MODEL_NAME" ]; then
    MODEL_CONTAINER="$PROJ_ROOT/transcriptomic-fms-${MODEL_NAME}.sif"
    if [ -f "$MODEL_CONTAINER" ]; then
        export APPTAINER_IMAGE="$MODEL_CONTAINER"
        echo "Using container: $APPTAINER_IMAGE"
    else
        echo "Error: Container not found: $MODEL_CONTAINER"
        export APPTAINER_IMAGE=""
    fi
else
    echo "Error: Model name required but not found in arguments"
    export APPTAINER_IMAGE=""
fi

export DATA_DIR="${DATA_DIR:-$PROJ_ROOT/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/output}"

# Note: SLURM log files are written to the current working directory
# (as specified in #SBATCH directives above)
# The job output/error files will be in the directory where sbatch was run

# Check if Apptainer image exists
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found at $APPTAINER_IMAGE"
    echo ""
    echo "Current working directory: $(pwd)"
    echo "Project root: $PROJ_ROOT"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-not set}"
    if [ -n "$MODEL_NAME" ]; then
        echo "Model: $MODEL_NAME"
        echo ""
        echo "To build container:"
        echo "  make build-container MODEL=$MODEL_NAME"
    fi
    echo ""
    echo "Please ensure:"
    if [ -n "$MODEL_NAME" ]; then
        echo "  1. You've run 'make build-container MODEL=$MODEL_NAME' to build the container"
    fi
    echo "  2. The container image exists at: $APPTAINER_IMAGE"
    echo "  4. You're submitting the job from the project root directory"
    echo ""
    echo "If the image is in a different location, set APPTAINER_IMAGE environment variable:"
    echo "  export APPTAINER_IMAGE=/path/to/transcriptomic-fms.sif"
    echo "  sbatch transcriptomic_fms/hpc/run_job.sh ..."
    exit 1
fi

# Check if run_apptainer.sh exists
if [ ! -f "$PROJ_ROOT/run_apptainer.sh" ]; then
    echo "Error: run_apptainer.sh not found. Please run ./transcriptomic_fms/hpc/setup_hpc.sh first"
    exit 1
fi

# Detect if GPU is requested
USE_GPU=0

# Check SLURM GPU allocation (Compute Canada sets CUDA_VISIBLE_DEVICES when --gres=gpu is used)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]; then
    USE_GPU=1
    echo "GPU detected via CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Check SLURM_GPUS_ON_NODE (some SLURM configurations)
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE:-0}" -gt 0 ]; then
    USE_GPU=1
    echo "GPU detected via SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
# Check SLURM_GPUS (alternative SLURM variable)
elif [ -n "${SLURM_GPUS:-}" ] && [ "${SLURM_GPUS:-0}" -gt 0 ]; then
    USE_GPU=1
    echo "GPU detected via SLURM_GPUS: $SLURM_GPUS"
# Check if --device cuda is in command arguments
elif echo "$@" | grep -q -- "--device cuda"; then
    USE_GPU=1
    echo "GPU requested via --device cuda flag"
fi

# Set environment variable for GPU access
export APPTAINER_USE_GPU=$USE_GPU

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
if [ "$USE_GPU" = "1" ]; then
    echo "GPU: Enabled (--nv flag will be used)"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    fi
else
    echo "GPU: Disabled"
fi
echo "========================================="
echo "Command: $@"
echo "========================================="

# Run the command using the wrapper script
# All arguments after the script name are passed to the CLI
"$PROJ_ROOT/run_apptainer.sh" python -m transcriptomic_fms.cli.main "$@"

# Print completion information
echo "========================================="
echo "End Time: $(date)"
echo "Job completed successfully"
echo "========================================="

