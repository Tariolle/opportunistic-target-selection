#!/bin/bash
# Watch benchmark progress: formats new CSV lines as they arrive.
# Usage:
#   bash slurm/monitor.sh                          # standard benchmark (4500 runs)
#   bash slurm/monitor.sh results/other.csv 1600   # custom CSV + total

CSV="${1:-results/benchmark_standard.csv}"
TOTAL="${2:-4500}"

if [ ! -f "$CSV" ]; then
    echo "Waiting for $CSV to appear..."
    while [ ! -f "$CSV" ]; do sleep 1; done
fi

DONE=$(($(wc -l < "$CSV") - 1))
echo "Progress: ${DONE}/${TOTAL} runs completed"
echo "========================================"

TOTAL_ITERS=0
START=$(date +%s)

tail -n 0 -f "$CSV" | while IFS=, read -r model method epsilon seed image mode iterations success rest; do
    DONE=$(($(wc -l < "$CSV") - 1))
    TOTAL_ITERS=$((TOTAL_ITERS + iterations))
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [ $ELAPSED -gt 0 ]; then
        IPS=$(echo "scale=1; $TOTAL_ITERS / $ELAPSED" | bc)
    else
        IPS="--"
    fi
    [ "$success" = "True" ] && status="OK" || status="FAIL"
    printf "[%d/%d] %s | %s | %s | %s | %s iters | %s | %s iter/s\n" \
        "$DONE" "$TOTAL" "$model" "$method" "$mode" "$image" "$iterations" "$status" "$IPS"
done
