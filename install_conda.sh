#!/bin/bash

# Conda Installation Script
# This script downloads and installs Miniconda3

set -e  # Exit on any error

echo "🐍 Starting Conda Installation..."
echo "=================================="

# Define variables
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_FILE="Miniconda3-latest-Linux-x86_64.sh"
INSTALL_DIR="$HOME/miniconda3"

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "✅ Conda is already installed!"
    conda --version
    exit 0
fi

# Create temporary directory for download
cd /tmp

echo "📥 Downloading Miniconda installer..."
if wget -q --show-progress "$MINICONDA_URL" -O "$INSTALLER_FILE"; then
    echo "✅ Download completed successfully!"
else
    echo "❌ Failed to download Miniconda installer"
    exit 1
fi

# Verify the installer was downloaded
if [ ! -f "$INSTALLER_FILE" ]; then
    echo "❌ Installer file not found"
    exit 1
fi

echo "🔧 Installing Miniconda..."
echo "This will install Miniconda to: $INSTALL_DIR"

# Run the installer in batch mode (non-interactive)
bash "$INSTALLER_FILE" -b -p "$INSTALL_DIR"

# Initialize conda for bash shell
echo "🔧 Initializing conda for bash shell..."
"$INSTALL_DIR/bin/conda" init bash

# Add conda to PATH for current session
export PATH="$INSTALL_DIR/bin:$PATH"

# Clean up installer
rm "$INSTALLER_FILE"

echo ""
echo "✅ Conda installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Verify installation: conda --version"
echo "3. Create environment from your project: conda env create --file environment.yml"
echo ""
echo "🎉 Installation complete! Please restart your terminal to use conda."
