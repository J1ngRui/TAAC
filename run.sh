#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: one token per semantic group + current-time context + target-cate match ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --use_time_context \
    --time_context_tz_offset_hours 8 \
    --use_target_hist_match \
    --target_hist_match_target_cate_item_fid 13 \
    --target_hist_match_cate_seq_fids seq_a:46,seq_b:68,seq_c:32,seq_d:25 \
    --d_model 88 \
    --rank_mixer_mode full \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --loss_type weighted_bce \
    --tail_neg_p_start 0.95 \
    --tail_neg_p_end 0.99 \
    --tail_neg_end_weight 0.05 \
    --tail_neg_min_weight 0.01 \
    --tail_neg_gamma 1.0 \
    --tail_neg_start_epoch 3 \
    --dropout_rate 0.01 \
    "$@"
