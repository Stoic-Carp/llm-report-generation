#!/bin/bash
set -e

# Ensure required directories exist with proper permissions
# This handles cases where volumes are mounted but subdirectories don't exist
# Note: If volumes are mounted, ensure the host directory is writable by UID 1000
mkdir -p /app/outputs/logs /app/outputs/plots /app/outputs/reports 2>/dev/null || true
mkdir -p /tmp/uploads 2>/dev/null || true

# Try to set permissions (may fail if volume is mounted with different ownership)
# This is non-critical - the app will work even if this fails
chmod -R 755 /app/outputs 2>/dev/null || true

# Test if we can write to outputs directory and subdirectories
test_dirs=("/app/outputs" "/app/outputs/logs" "/app/outputs/plots" "/app/outputs/reports")
can_write=true

for test_dir in "${test_dirs[@]}"; do
    if ! touch "${test_dir}/.write_test" 2>/dev/null; then
        echo "WARNING: Cannot write to ${test_dir}"
        can_write=false
    else
        rm -f "${test_dir}/.write_test" 2>/dev/null || true
    fi
done

if [ "$can_write" = false ]; then
    echo ""
    echo "=========================================="
    echo "PERMISSION ERROR DETECTED"
    echo "=========================================="
    echo "The outputs directory is not writable by the container user (UID 1000)."
    echo ""
    echo "To fix this, run one of the following commands on your host:"
    echo "  sudo chown -R 1000:1000 ./outputs"
    echo "  OR"
    echo "  sudo chmod -R 777 ./outputs"
    echo ""
    echo "Or use the provided script:"
    echo "  ./scripts/fix-permissions.sh"
    echo "=========================================="
    echo ""
fi

# Execute the command passed as arguments
exec "$@"

