#!/bin/bash
# Script to fix permissions for Docker volumes
# Run this before starting docker-compose if you encounter permission errors

set -e

echo "Fixing permissions for Docker volumes..."

# Create directories if they don't exist
mkdir -p outputs/logs outputs/plots outputs/reports

# Fix ownership to match container user (UID 1000)
if [ "$EUID" -eq 0 ]; then
    echo "Running as root - setting ownership to UID 1000..."
    chown -R 1000:1000 outputs
    echo "Permissions fixed!"
else
    echo "Not running as root. Attempting to fix permissions..."
    if chmod -R 777 outputs 2>/dev/null; then
        echo "Made outputs directory world-writable (less secure but works)"
    else
        echo "ERROR: Cannot fix permissions. Please run with sudo:"
        echo "  sudo chown -R 1000:1000 outputs"
        echo "  OR"
        echo "  sudo chmod -R 777 outputs"
        exit 1
    fi
fi

echo "Done! You can now start docker-compose."

