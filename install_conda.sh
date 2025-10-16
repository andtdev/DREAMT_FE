#!/bin/bash

# Script to install and activate Miniconda
set -e

echo "======================================"
echo "Installing Miniconda..."
echo "======================================"

# Download Miniconda installer
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"

echo "Downloading Miniconda installer..."
wget -q $MINICONDA_URL -O /tmp/$MINICONDA_INSTALLER

# Install Miniconda
echo "Installing Miniconda to $HOME/miniconda3..."
bash /tmp/$MINICONDA_INSTALLER -b -p $HOME/miniconda3

# Clean up installer
rm /tmp/$MINICONDA_INSTALLER

# Initialize conda for bash
echo "Initializing conda for bash..."
$HOME/miniconda3/bin/conda init bash

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"
echo ""
echo "To activate conda, please run:"
echo "  source ~/.bashrc"
echo ""
echo "Or close and reopen your terminal."
echo ""
echo "After that, you can verify the installation with:"
echo "  conda --version"
echo ""
