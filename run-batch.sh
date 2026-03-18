#!/bin/bash
# Batch Poster Generation Orchestrator
#
# Orchestrates two-phase poster generation when the text model (vLLM)
# and image model (InvokeAI) share the same GPU.
#
# Phase 1: Start vLLM → generate image prompts → stop vLLM (free GPU)
# Phase 2: Start InvokeAI → generate images from prompts → stop all
#
# Usage:
#   ./run-batch.sh                    # Run both phases
#   ./run-batch.sh --prompts-only     # Only generate prompts (phase 1)
#   ./run-batch.sh --images-only      # Only generate images (phase 2)
#   ./run-batch.sh --build            # Rebuild batch container before running

set -e

COMPOSE_FILE="docker-compose.batch.yml"
COMPOSE="docker compose -f ${COMPOSE_FILE}"
BUILD_FLAG=""
RUN_PROMPTS=true
RUN_IMAGES=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompts-only)
            RUN_IMAGES=false
            shift
            ;;
        --images-only)
            RUN_PROMPTS=false
            shift
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--prompts-only] [--images-only] [--build]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Batch Poster Generation Orchestrator"
echo "============================================"
echo ""

# ── Phase 1: Generate image prompts with vLLM ──────────────────────────
if [ "$RUN_PROMPTS" = true ]; then
    echo "=== Phase 1: Generating image prompts (vLLM) ==="
    echo ""

    # Start vLLM + batch-prompts, wait for batch-prompts to finish
    ${COMPOSE} up ${BUILD_FLAG} batch-prompts --abort-on-container-exit
    prompts_exit=$?

    # Stop vLLM to free the GPU
    echo ""
    echo "=== Stopping vLLM to free GPU ==="
    ${COMPOSE} stop vllm
    ${COMPOSE} rm -f vllm

    if [ $prompts_exit -ne 0 ]; then
        echo ""
        echo "[ERROR] Prompts phase failed (exit code: ${prompts_exit})"
        ${COMPOSE} down
        exit $prompts_exit
    fi

    echo ""
    echo "=== Phase 1 complete ==="
    echo ""
fi

# ── Phase 2: Generate images with InvokeAI ──────────────────────────────
if [ "$RUN_IMAGES" = true ]; then
    echo "=== Phase 2: Generating images (InvokeAI) ==="
    echo ""

    # Start InvokeAI + batch-images, wait for batch-images to finish
    ${COMPOSE} up ${BUILD_FLAG} batch-images --abort-on-container-exit
    images_exit=$?

    if [ $images_exit -ne 0 ]; then
        echo ""
        echo "[ERROR] Images phase failed (exit code: ${images_exit})"
        ${COMPOSE} down
        exit $images_exit
    fi

    echo ""
    echo "=== Phase 2 complete ==="
    echo ""
fi

# ── Cleanup ─────────────────────────────────────────────────────────────
echo "=== Cleaning up ==="
${COMPOSE} down

echo ""
echo "============================================"
echo "  Batch poster generation complete!"
echo "============================================"
