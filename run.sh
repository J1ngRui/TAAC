#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: final_pair + user-side full-time features ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --time_context_tz_offset_hours 8 \
    --use_full_time_user_features \
    --d_model 84 \
    --rank_mixer_mode full \
    --use_final_pair \
    --final_pair_json "${SCRIPT_DIR}/final_pair.json" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --loss_type bce \
    "$@"
