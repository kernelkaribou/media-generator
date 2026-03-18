#!/bin/bash
# Batch poster generation entrypoint
#
# Supports two modes via BATCH_PHASE environment variable:
#   BATCH_PHASE=prompts  - Generate image prompts using AI text model
#   BATCH_PHASE=images   - Generate images using InvokeAI
#   BATCH_PHASE=all      - Both phases in one pass (original behavior, requires both services)

set -e

PHASE="${BATCH_PHASE:-all}"
echo "=== Batch Poster Generation (phase: ${PHASE}) ==="

if [ "$PHASE" = "prompts" ]; then
    # Prompts-only phase: no InvokeAI needed, just the text model
    python batch_poster_generate.py \
        --phase prompts \
        --media-api "${MEDIA_API_URL:-http://localhost:8000}" \
        --prompts-file "${PROMPTS_FILE:-/data/batch-prompts.json}" \
        --api-key "${API_KEY}" \
        ${VERBOSE:+--verbose}
elif [ "$PHASE" = "images" ]; then
    # Images-only phase: needs InvokeAI, reads prompts from file
    ./batch-init-models.sh
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model initialization failed"
        exit 1
    fi

    python batch_poster_generate.py \
        --phase images \
        --media-api "${MEDIA_API_URL:-http://localhost:8000}" \
        --invokeai "${INVOKEAI_URL:-http://invokeai:9090}" \
        --prompts-file "${PROMPTS_FILE:-/data/batch-prompts.json}" \
        --api-key "${API_KEY}" \
        ${VERBOSE:+--verbose}
else
    # Combined phase (original behavior)
    ./batch-init-models.sh
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model initialization failed"
        exit 1
    fi

    echo ""
    echo "=== Starting poster generation ==="
    python batch_poster_generate.py \
        --media-api "${MEDIA_API_URL:-http://localhost:8000}" \
        --invokeai "${INVOKEAI_URL:-http://invokeai:9090}" \
        --api-key "${API_KEY}" \
        ${VERBOSE:+--verbose}
fi

exit_code=$?

echo ""
echo "=== Batch complete (exit code: ${exit_code}) ==="
exit $exit_code
