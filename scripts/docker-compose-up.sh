#!/bin/bash
# Wrapper script to run docker-compose with automatic permission fixes
# Usage: ./scripts/docker-compose-up.sh [docker-compose arguments]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Docker Compose Setup"
echo "=========================================="

# Check if outputs directory exists
if [ ! -d "$PROJECT_ROOT/outputs" ]; then
    echo "Creating outputs directory structure..."
    mkdir -p "$PROJECT_ROOT/outputs/logs" "$PROJECT_ROOT/outputs/plots" "$PROJECT_ROOT/outputs/reports"
fi

# Check permissions and fix if needed
echo "Checking outputs directory permissions..."
if [ ! -w "$PROJECT_ROOT/outputs" ] || [ ! -w "$PROJECT_ROOT/outputs/plots" ] || [ ! -w "$PROJECT_ROOT/outputs/reports" ]; then
    echo "Permission issues detected. Attempting to fix..."
    "$SCRIPT_DIR/fix-permissions.sh"
else
    echo "Permissions look good!"
fi

echo ""
echo "Starting docker-compose..."
echo ""

# Change to project root and run docker-compose
cd "$PROJECT_ROOT"
exec docker-compose "$@"

