#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cat > .env <<EOL
# Environment setup for the HypeFlow project
export PROJECT_ROOT="${SCRIPT_DIR}"
export CONDA_ENV_NAME="HypeFlow"
export WANDB_DIR="${SCRIPT_DIR}"
EOL