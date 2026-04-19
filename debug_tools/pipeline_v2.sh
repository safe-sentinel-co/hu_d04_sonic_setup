#!/bin/bash
# v2 pipeline: 3-segment scaler config (legs=0.90, torso=1.10, arms=1.00)
# Output to /root/hu_d04_motions_v2/, then upload to S3.
set -e

STAGE=${1:-all}  # all | retarget | convert | filter | upload

BVH_ROOT=/workspace/bones-seed/soma_uniform/bvh
OUT_ROOT=/root/hu_d04_motions_v2
CSV_ROOT=$OUT_ROOT/csv
PKL_ROOT=$OUT_ROOT/robot
FILTERED_ROOT=$OUT_ROOT/robot_filtered
LOGDIR=/workspace/logs/retarget_v2

SOMA_ENV=/workspace/soma_env
GR00T=/workspace/GR00T-WholeBodyControl

S3_BUCKET=safesentinel-inc
S3_KEY=hu_d04_motions/robot_filtered.tar.gz
TAR_PATH=$OUT_ROOT/robot_filtered.tar.gz

mkdir -p "$CSV_ROOT" "$PKL_ROOT" "$FILTERED_ROOT" "$LOGDIR"

run_retarget() {
    echo "[pipeline] $(date -Is) STAGE 1: BVH -> CSV (3-seg scaler)"
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
        echo "[pipeline] shard $i gpu=$gpu pid=${PIDS[$i]}"
    done
    (
        while true; do
            sleep 300
            local cnt=$(find "$CSV_ROOT" -name '*.csv' 2>/dev/null | wc -l)
            echo "[pipeline $(date -Is)] CSVs so far: $cnt / $total_bvhs"
        done
    ) &
    local REPORTER=$!
    local FAIL=0
    for i in "${!PIDS[@]}"; do
        if wait ${PIDS[$i]}; then echo "[pipeline] shard $i OK"; else echo "[pipeline] shard $i FAILED"; FAIL=$((FAIL+1)); fi
    done
    kill $REPORTER 2>/dev/null
    local final_csv=$(find "$CSV_ROOT" -name '*.csv' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 1 DONE. CSVs: $final_csv. Failed shards: $FAIL"
}

run_convert() {
    echo "[pipeline] $(date -Is) STAGE 2: CSV -> PKL"
    source "$SOMA_ENV/bin/activate"
    local NWORKERS=$(nproc); NWORKERS=$((NWORKERS > 32 ? 32 : NWORKERS))
    cd "$GR00T"
    python gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py \
        --input "$CSV_ROOT" --output "$PKL_ROOT" \
        --fps 30 --fps_source 30 --individual --num_workers $NWORKERS \
        2>&1 | tee -a "$LOGDIR/convert.log"
    local final_pkl=$(find "$PKL_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 2 DONE. PKLs: $final_pkl"
}

run_filter() {
    echo "[pipeline] $(date -Is) STAGE 3: filter"
    source "$SOMA_ENV/bin/activate"
    local NWORKERS=$(nproc); NWORKERS=$((NWORKERS > 16 ? 16 : NWORKERS))
    cd "$GR00T"
    python gear_sonic/data_process/filter_and_copy_bones_data.py \
        --source "$PKL_ROOT" --dest "$FILTERED_ROOT" --workers $NWORKERS \
        2>&1 | tee -a "$LOGDIR/filter.log"
    local final=$(find "$FILTERED_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    local src=$(find "$PKL_ROOT" -name '*.pkl' 2>/dev/null | wc -l)
    echo "[pipeline] $(date -Is) STAGE 3 DONE. Kept $final / $src PKLs in $FILTERED_ROOT"
}

run_upload() {
    echo "[pipeline] $(date -Is) STAGE 4: tar + upload to s3://$S3_BUCKET/$S3_KEY"
    cd "$OUT_ROOT"
    tar czf "$TAR_PATH" robot_filtered/
    local sz=$(du -h "$TAR_PATH" | cut -f1)
    echo "[pipeline] tar size: $sz"
    : "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID before running upload}"
    : "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY before running upload}"
    pip install -q awscli 2>/dev/null || true
    aws s3 cp "$TAR_PATH" "s3://$S3_BUCKET/$S3_KEY"
    echo "[pipeline] $(date -Is) STAGE 4 DONE."
}

case "$STAGE" in
    retarget) run_retarget ;;
    convert)  run_convert ;;
    filter)   run_filter ;;
    upload)   run_upload ;;
    all)
        run_retarget
        run_convert
        run_filter
        run_upload
        echo "[pipeline] ALL STAGES COMPLETE at $(date -Is)"
        echo "[pipeline] Final output: $FILTERED_ROOT"
        echo "[pipeline] S3:           s3://$S3_BUCKET/$S3_KEY"
        ;;
    *)
        echo "Usage: $0 {all|retarget|convert|filter|upload}"; exit 1 ;;
esac
