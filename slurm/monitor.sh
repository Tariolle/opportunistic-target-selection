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

COUNT=0
START=$(date +%s)

tail -n 0 -f "$CSV" | while IFS=, read -r method t_value image true_label iterations success adv_class switch_iter locked_class timestamp; do
    DONE=$(($(wc -l < "$CSV") - 1))
    COUNT=$((COUNT + 1))
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [ $COUNT -gt 0 ] && [ $ELAPSED -gt 0 ]; then
        AVG=$(echo "scale=1; $ELAPSED / $COUNT" | bc)
        RATE=$(echo "scale=1; $COUNT * 60 / $ELAPSED" | bc)
        REMAINING=$(echo "scale=0; ($TOTAL - $DONE) * $ELAPSED / $COUNT" | bc)
        REMAINING_MIN=$(echo "scale=1; $REMAINING / 60" | bc)
    else
        AVG="--"
        RATE="--"
        REMAINING_MIN="--"
    fi
    [ "$success" = "True" ] && status="OK" || status="FAIL"
    extra=""
    [ -n "$switch_iter" ] && extra=" (switch@${switch_iter}, locked=${locked_class})"
    printf "[%d/%d] %s T=%-3s | %s | %s iters | %s%s | avg %.1fs/run, %.1f runs/min, ETA %smin\n" \
        "$DONE" "$TOTAL" "$method" "$t_value" "$image" "$iterations" "$status" "$extra" \
        "$AVG" "$RATE" "$REMAINING_MIN"
done
