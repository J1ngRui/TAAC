#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: one token per semantic group + current-time context ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --use_time_context \
    --time_context_tz_offset_hours 8 \
    --d_model 84 \
    --rank_mixer_mode full \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --loss_type weighted_bce \
    --linear_reweight_start_loss 1.2 \
    --linear_reweight_end_loss 2 \
    --linear_reweight_min_weight 0.2 \
    --linear_reweight_start_epoch 3 \
    --dropout_rate 0.01 \
    "$@"
