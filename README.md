# TAAC 2026 PCVR Experiments

This repository contains the current TAAC 2026 PCVR modeling code and the
remaining clean experiment baselines.

## Data Summary

| Feature group | Count | Type | Notes |
| --- | ---: | --- | --- |
| ID and label | 5 | `int64` / `int32` | Core identifiers, label, and timestamp. |
| User int features | 46 | `int64` / `list<int64>` | User profile and multi-value preference features. |
| User dense features | 10 | `list<float>` | User embeddings and dense values aligned with selected int fids. |
| Item int features | 14 | `int64` / `list<int64>` | Item side information, category, and multi-label features. |
| Domain sequences | 45 | `list<int64>` | Four historical behavior domains. |

Important observations:

- Dataset row order is effectively shuffled and should not be treated as time order.
- User overlap between train and valid/test is nearly zero; user history is already represented through sequence features.
- Item overlap is limited, so the model should not rely on memorizing item ids.
- Stronger signals came from user history, time features, user int/dense alignment, and sequence-hour modeling.

## Current Mainline

The main usable baseline is:

```text
 group NS tokenizer
 final_pair int-dense residual interaction
 full-time user features
 relative-time FiLM modulation
 d_model = 84
 RankMixer full mode
```

Current working branch:

```text
final_pair_fulltime2
```

Reference branch:

```text
final_pair
```

## Key Modules

### final_pair

`FidPairResidualGate` performs fid-aligned int/dense interaction before group
token construction. It does not add new pair tokens and injects residual deltas
back into the existing int and dense fid embeddings.

Current pair config is stored in:

```text
final_pair.json
```

### full-time user features

Time is represented on the user side rather than as a standalone legacy
`time_context_token`.

The full-time path adds user-side discrete time features and dense cycle
features, then uses lightweight interaction to improve the user time token.

### seqhour / UE-wide rescue notes

Some late experiments used sequence-hour features and a UE-wide logits branch.
Those branches were cleaned from the active local branch list, but the final
submitted inference package may still need the matching `model.py` for any
checkpoint trained with `ue_wide_*` parameters.

For that case, keep the inference files together as a self-contained package:

```text
dataset.py
infer.py
model.py
ns_groups.json
```

The `infer.py` must import the colocated `dataset.py` and `model.py`; do not
mix a checkpoint trained with `ue_wide_*` against a pure model definition.

## Training

Use `run.sh` as the single entrypoint for the current branch:

```bash
bash run.sh
```

The script expects the standard platform environment variables:

```text
TRAIN_DATA_PATH
TRAIN_CKPT_PATH
TRAIN_LOG_PATH
```

## Inference Package

The platform imports:

```python
from infer import main
```

Therefore the submitted inference directory must place `infer.py` at the
package root, alongside the exact `model.py`, `dataset.py`, and `ns_groups.json`
that match the checkpoint.

## Branch Hygiene

Keep only active baselines and final references. Old A/B branches for focal,
fulldata, DIN variants, miss/ref/gact, and intermediate time-context attempts
should not be used as final submission bases unless explicitly restored for an
A/B run.
