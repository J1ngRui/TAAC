# TAAC 2026 PCVR 实验记录

本仓库记录 TAAC 2026 PCVR 任务的建模代码、关键实验路线和最终保留分支。

## 比赛结果

![Team Performance](assets/team-performance.png)

| 赛道 | 排名 | Best Score | 最佳提交时间 |
| --- | ---: | ---: | --- |
| Academic Track | 334 | 0.827799 | 2026-05-23 12:23:21 |

## 当前保留分支

最终保留的核心分支：

```text
best_seqhour
```

参考分支：

```text
final_pair
final_pair_fulltime2
```

其中 `best_seqhour` 是当前保留的最佳版本；`final_pair` 和
`final_pair_fulltime2` 作为结构参考和回溯用基座保留。

## 当前主线结构

`best_seqhour` 主要包含：

```text
group NS tokenizer
final_pair int-dense residual interaction
full-time user features
relative-time FiLM modulation
sequence-hour residual embedding
d_model = 84
RankMixer full mode
```

## 关键模块

### final_pair

`FidPairResidualGate` 在 fid-level 阶段对齐 user int 和 user dense 表征。

它的特点是：

- 不新增 pair token。
- 不改变 group token 数量。
- 在 int embedding 与 dense embedding 之间做轻量 residual 交互。
- 增强后的 int/dense fid embedding 继续走原来的 group token 流程。

pair 配置文件：

```text
final_pair.json
```

### full-time user features

时间特征从独立 `time_context_token` 转移到 user 侧表达中。

核心思路：

- 绝对时间作为 user int 侧离散特征。
- 周期 sin/cos 作为 user dense 侧连续特征。
- 通过轻量交互让离散时间与周期时间互相补充。

### seqhour

`seqhour` 是后期保留的有效增强之一，用于补充历史序列中的小时级时间信息。

相比单独的全局 time context，seqhour 更贴近历史行为发生时刻，因此更适合和 sequence encoder 结合。

### UE-wide / semi-local 说明

部分后期提交曾使用：

```text
UE-wide logits fusion
semi-local sequence encoder
```

这类 checkpoint 必须使用与训练时完全一致的 `model.py` 推理，否则会因为
`ue_wide_*` 参数不匹配导致 strict load 失败。

如果要提交这类 checkpoint，请保证推理包根目录至少包含：

```text
dataset.py
infer.py
model.py
ns_groups.json
```

并且 `infer.py` 需要加载同目录下的 `dataset.py` 和 `model.py`。

## 训练

当前分支使用：

```bash
bash run.sh
```

平台环境变量：

```text
TRAIN_DATA_PATH
TRAIN_CKPT_PATH
TRAIN_LOG_PATH
```

## 推理

平台推理入口固定为：

```python
from infer import main
```

因此提交推理包时，`infer.py` 必须位于包根目录，并且要和对应 checkpoint 的
`model.py`、`dataset.py`、`ns_groups.json` 保持一致。

## 分支清理原则

旧的 focal、fulldata、DIN、miss/ref/gact、time-context 中间实验分支已清理。

后续如果需要复现某个 A/B，应从明确的保留分支重新拉出新分支，避免继续在旧实验分支上叠变量。
