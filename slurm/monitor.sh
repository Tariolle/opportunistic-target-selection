#!/bin/bash
# Watch ablation progress: formats new CSV lines as they arrive.
# Usage: bash slurm/monitor.sh

CSV="results/benchmark_ablation_naive.csv"
TOTAL=1600

if [ ! -f "$CSV" ]; then
    echo "Waiting for $CSV to appear..."
    while [ ! -f "$CSV" ]; do sleep 1; done
fi

DONE=$(($(wc -l < "$CSV") - 1))
echo "Progress: ${DONE}/${TOTAL} runs completed"
echo "========================================"

TOTAL_ITERS=0
START=$(date +%s)

tail -n 0 -f "$CSV" | while IFS=, read -r method t_value image true_label iterations success adv_class switch_iter locked_class timestamp; do
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
    extra=""
    [ -n "$switch_iter" ] && extra=" (switch@${switch_iter}, locked=${locked_class})"
    printf "[%d/%d] %s T=%-3s | %s | %s iters | %s%s | %s iter/s\n" \
        "$DONE" "$TOTAL" "$method" "$t_value" "$image" "$iterations" "$status" "$extra" \
        "$IPS"
done
