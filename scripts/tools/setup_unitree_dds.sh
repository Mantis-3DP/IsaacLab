#!/bin/bash
# =============================================================================
# Setup Unitree DDS dependencies in IsaacLab container
# =============================================================================
#
# Usage (run inside container):
#   bash /workspace/isaaclab/scripts/tools/setup_unitree_dds.sh
#
# Or from host:
#   ./docker/container.py enter
#   bash /workspace/isaaclab/scripts/tools/setup_unitree_dds.sh
#
# This script:
#   1. Installs build dependencies (cmake, build-essential)
#   2. Compiles CycloneDDS C library from source (version 0.10.x)
#   3. Sets CYCLONEDDS_HOME environment variable
#   4. Installs unitree_sdk2py Python bindings
#   5. Installs additional dependencies (pyzmq, msgpack, etc.)
#
# =============================================================================

set -e

# Detect the correct Python/pip path for IsaacLab container
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
PYTHON_CMD="${ISAACLAB_PATH}/_isaac_sim/python.sh"
PIP_CMD="${PYTHON_CMD} -m pip"

# Fallback if not in IsaacLab container
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo ""
echo "=============================================="
echo "  Installing Unitree DDS Dependencies"
echo "=============================================="
echo ""
echo "Using Python: $PYTHON_CMD"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install build dependencies
# -----------------------------------------------------------------------------
echo "[1/5] Installing build dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git

# -----------------------------------------------------------------------------
# Step 2: Build CycloneDDS from source (version 0.10.x required)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Building CycloneDDS from source..."

if [ -d "/cyclonedds/install" ]; then
    echo "  CycloneDDS already built at /cyclonedds/install, skipping..."
else
    echo "  Cloning CycloneDDS releases/0.10.x branch..."
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds

    echo "  Compiling CycloneDDS..."
    cd /cyclonedds
    mkdir -p build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    cmake --build . --target install

    echo "  CycloneDDS built successfully!"
fi

# -----------------------------------------------------------------------------
# Step 3: Set environment variable
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Setting CYCLONEDDS_HOME environment variable..."
export CYCLONEDDS_HOME=/cyclonedds/install

# Add to bashrc for persistence
if ! grep -q "CYCLONEDDS_HOME" ~/.bashrc 2>/dev/null; then
    echo 'export CYCLONEDDS_HOME=/cyclonedds/install' >> ~/.bashrc
    echo "  Added CYCLONEDDS_HOME to ~/.bashrc"
fi

# -----------------------------------------------------------------------------
# Step 4: Install unitree_sdk2py
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Installing unitree_sdk2py..."

if [ -d "/workspace/unitree/unitree_sdk2_python" ]; then
    cd /workspace/unitree/unitree_sdk2_python
    # IMPORTANT: Use --no-deps to avoid pulling numpy 2.x which breaks IsaacLab
    # IsaacLab requires numpy<2, but unitree_sdk2py has unpinned numpy dependency
    $PIP_CMD install --no-deps -e .
    echo "  unitree_sdk2py installed (--no-deps to preserve numpy<2)"
else
    echo "  ERROR: /workspace/unitree/unitree_sdk2_python not found!"
    echo ""
    echo "  The unitree volume is not mounted. You need to restart the container with:"
    echo "    ./docker/container.py start \\"
    echo "        --files docker-compose.cloudxr-runtime.patch.yaml \\"
    echo "        --files docker-compose.unitree.patch.yaml \\"
    echo "        --env-file .env.cloudxr-runtime"
    echo ""
    echo "  Make sure docker-compose.unitree.patch.yaml contains:"
    echo "    services:"
    echo "      isaac-lab-base:"
    echo "        volumes:"
    echo "          - /home/mats/Bot/unitree:/workspace/unitree"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 5: Install additional dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Installing additional Python dependencies..."
$PIP_CMD install pyzmq msgpack logging_mp rerun-sdk

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="
echo ""

echo "Testing CycloneDDS import..."
$PYTHON_CMD -c "from cyclonedds.domain import DomainParticipant; print('  ✓ CycloneDDS OK')"

echo "Testing unitree_sdk2py import..."
$PYTHON_CMD -c "from unitree_sdk2py.core.channel import ChannelPublisher; print('  ✓ unitree_sdk2py OK')"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: The CYCLONEDDS_HOME variable is set for this session."
echo "For new terminal sessions, it will be loaded from ~/.bashrc"
echo ""
echo "You can now run:"
echo "  ./isaaclab.sh -p scripts/tools/run_dds.py \\"
echo "      --task Isaac-Stack-RgyBlock-G129-Inspire-Joint \\"
echo "      --robot_type g129 --enable_inspire_dds"
echo ""
