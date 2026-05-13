#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: group + time context v1 + target-cate history match + BCE ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --use_time_context \
    --time_context_tz_offset_hours 8 \
    --use_target_hist_match \
    --target_hist_match_target_cate_item_fid 13 \
    --target_hist_match_cate_seq_fids seq_a:46,seq_b:68,seq_c:32,seq_d:25 \
    --use_recent_activity \
    --recent_activity_mode global \
    --recent_activity_feature_fids 210001,210002,210003,210004 \
    --d_model 88 \
    --rank_mixer_mode full \
    --use_rope \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --loss_type bce \
    --dropout_rate 0.01 \
    "$@"
