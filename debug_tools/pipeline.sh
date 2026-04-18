#!/bin/bash
# End-to-end BVH -> CSV -> PKL -> filtered PKL pipeline for HU_D04
set -e

STAGE=${1:-all}  # all | retarget | convert | filter

BVH_ROOT=/workspace/bones-seed/soma_uniform/bvh
CSV_ROOT=/workspace/hu_d04_motions/csv
PKL_ROOT=/workspace/hu_d04_motions/robot
FILTERED_ROOT=/workspace/hu_d04_motions/robot_filtered
LOGDIR=/workspace/logs/retarget

SOMA_ENV=/workspace/soma_env
GR00T=/workspace/GR00T-WholeBodyControl

mkdir -p "$CSV_ROOT" "$PKL_ROOT" "$FILTERED_ROOT" "$LOGDIR"

# -------- Stage 1: BVH -> CSV (6-way GPU-sharded) --------
run_retarget() {
    echo "[pipeline] $(date -Is) STAGE 1: BVH -> CSV"
    local total_bvhs=$(find "$BVH_ROOT" -name '*.bvh' | wc -l)
    echo "[pipeline] total BVHs: $total_bvhs"

    source "$SOMA_ENV/bin/activate"
    export PYTHONUNBUFFERED=1

    local PIDS=()
    local SHARDS=${SHARDS:-12}
    local GPUS=${GPUS:-6}
    local BATCH=${BATCH:-30}
    for i in $(seq 0 $((SHARDS - 1))); do
        local gpu=$((i % GPUS))
        local logf="$LOGDIR/shard_${i}.log"
        : > "$logf"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            python /workspace/logs/retarget_shard.py \
                --input "$BVH_ROOT" --output "$CSV_ROOT" \
                --shard-idx $i --shard-count $SHARDS \
                --batch-size $BATCH --target-fps 30 --device cuda:0 \
                --log "$logf" \
                > "$logf" 2>&1
        ) &
        PIDS+=($!)
        echo "[pipeline] shard $i gpu=$gpu pid=${PIDS[$i]} -> $logf"
    done

    # Periodic summary
    (
        while true; do
            sleep 300  # every 5 min
            local cnt=$(find "$CSV_ROOT" -name '*.csv' 2>/dev/null | wc -l)
            echo "[pipeline $(date -Is)] CSVs so far: $cnt / $total_bvhs"
        done
    ) &
    local REPORTER=$!

    local FAIL=0
    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        if wait $pid; then
            echo "[pipeline] shard $i (pid=$pid) OK"
        else
            echo "[pipeline] shard $i (pid=$pid) FAILED rc=$?"
            FAIL=$((FAIL+1))
        fi
    done
    kill $REPORTER 2>/dev/null

    local final_csv=$(find "$CSV_ROOT" -name '*.csv' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 1 DONE. CSVs: $final_csv. Failed shards: $FAIL"
}

# -------- Stage 2: CSV -> PKL (multi-process CPU) --------
run_convert() {
    echo "[pipeline] $(date -Is) STAGE 2: CSV -> PKL"
    source "$SOMA_ENV/bin/activate"
    local NWORKERS=$(nproc)
    NWORKERS=$((NWORKERS > 32 ? 32 : NWORKERS))
    echo "[pipeline] using $NWORKERS workers"

    cd "$GR00T"
    # v2 retargeter decimates BVH to 30 fps before IK, so CSVs are already 30 fps.
    python gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py \
        --input "$CSV_ROOT" \
        --output "$PKL_ROOT" \
        --fps 30 --fps_source 30 --individual --num_workers $NWORKERS \
        2>&1 | tee -a "$LOGDIR/convert.log"

    local final_pkl=$(find "$PKL_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 2 DONE. PKLs: $final_pkl"
}

# -------- Stage 3: filter motion set --------
run_filter() {
    echo "[pipeline] $(date -Is) STAGE 3: filter"
    source "$SOMA_ENV/bin/activate"
    local NWORKERS=$(nproc)
    NWORKERS=$((NWORKERS > 16 ? 16 : NWORKERS))

    cd "$GR00T"
    python gear_sonic/data_process/filter_and_copy_bones_data.py \
        --source "$PKL_ROOT" \
        --dest "$FILTERED_ROOT" \
        --workers $NWORKERS \
        2>&1 | tee -a "$LOGDIR/filter.log"

    local final=$(find "$FILTERED_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    local src=$(find "$PKL_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 3 DONE. Kept $final / $src PKLs in $FILTERED_ROOT"
}

case "$STAGE" in
    retarget) run_retarget ;;
    convert)  run_convert ;;
    filter)   run_filter ;;
    all)
        run_retarget
        run_convert
        run_filter
        echo "[pipeline] ALL STAGES COMPLETE at $(date -Is)"
        echo "[pipeline] Final output: $FILTERED_ROOT"
        ;;
    *)
        echo "Usage: $0 {all|retarget|convert|filter}"; exit 1 ;;
esac
