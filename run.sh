#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: main_gapReg = main_base + AMP + strong generalization regularizers ----
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
    --patience 3 \
    --use_amp \
    --amp_dtype fp16 \
    --loss_type bce \
    --dropout_rate 0.05 \
    --token_dropout_rate 0.08 \
    --rdrop_alpha 0.5 \
    "$@"
