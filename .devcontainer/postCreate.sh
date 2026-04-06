#!/bin/bash
# Post-create setup script for Dev Container
set -e

echo "=== Setting up AVVA Dev Container ==="

# Install Node/Bun dependencies
echo "Installing Node/Bun dependencies..."
bun install

# Install Python dependencies with local MediaPipe wheel
echo "Installing Python dependencies..."
pip install --break-system-packages ./apps/hand-detector/.wheels/mediapipe-0.10.35-cp312-cp312-linux_aarch64.whl
pip install --break-system-packages -r apps/hand-detector/requirements.txt

# Build Rust/Tauri dependencies (optional, can be slow)
echo "Setting up Rust/Tauri..."
cd apps/desktop/src-tauri
cargo fetch
cd ../../..

echo "=== Setup complete! ==="
echo ""
echo "Available commands:"
echo "  bun run dev       - Start all apps in dev mode"
echo "  bun run build     - Build all apps"
echo "  cd apps/hand-detector && python src/app.py  - Run hand detector"
echo "  cd apps/desktop && bun run tauri dev        - Run desktop app"
