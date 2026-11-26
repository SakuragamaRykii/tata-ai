#!/usr/bin/env bash
#SBATCH --job-name=emotion-transformer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=amd_student

set -uo pipefail

# toolchain: gcc for the linker, rustup-installed cargo/rustc
module purge 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true
source "$HOME/.cargo/env"
# if you installed rustup to scratch, instead do:
#   export RUSTUP_HOME=/iridisfs/scratch/$USER/.rustup
#   export CARGO_HOME=/iridisfs/scratch/$USER/.cargo
#   source "$CARGO_HOME/env"

command -v cargo >/dev/null || { echo "FATAL: cargo not on PATH — did rustup install?" >&2; exit 1; }
echo "cargo=$(cargo --version)"
echo "rustc=$(rustc --version)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR" >&2; exit 1; }

export CARGO_TARGET_DIR="$PROJECT_DIR/target"
cargo build --release --offline || { echo "FATAL: build failed" >&2; exit 1; }

BIN="$CARGO_TARGET_DIR/release/tata-ai"       # <-- match [package].name in Cargo.toml
[[ -x "$BIN" ]] || { echo "FATAL: binary not found: $BIN" >&2; ls -la "$CARGO_TARGET_DIR/release" >&2; exit 1; }

echo "running $BIN"
"$BIN"