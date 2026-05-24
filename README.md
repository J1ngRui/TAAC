# TAAC 2026 PCVR

## 比赛结果

![Team Performance](assets/team-performance.png)

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

### full-time user features

时间特征从独立 `time_context_token` 转移到 user 侧表达中。

核心思路：

- 绝对时间作为 user int 侧离散特征。
- 周期 sin/cos 作为 user dense 侧连续特征。
- 通过轻量交互让离散时间与周期时间互相补充。

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

### seqhour

`seqhour` 是后期保留的有效增强之一，用于补充历史序列中的小时级时间信息。

相比单独的全局 time context，seqhour 更贴近历史行为发生时刻，因此更适合和 sequence encoder 结合。

### UE-wide / semi-local 说明

部分后期提交曾使

```text
UE-wide logits fusion
semi-local sequence encoder
```

## 训练

当前分支使用：

```bash
bash run.sh
```

<details>
<summary>实验记录</summary>

### base: time context v1

```text
Epoch 1 Validation | AUC: 0.858992, LogLoss: 0.227252
Epoch 2 Validation | AUC: 0.863453, LogLoss: 0.223940
Epoch 3 Validation | AUC: 0.864340, LogLoss: 0.223234
Epoch 4 Validation | AUC: 0.864564, LogLoss: 0.223251
Epoch 5 Validation | AUC: 0.865527, LogLoss: 0.222461
Epoch 6 Validation | AUC: 0.865100, LogLoss: 0.222718
Test AUC: 0.820503
```

### final: time context v1.5

```text
Epoch 1 Validation | AUC: 0.859006, LogLoss: 0.227242
Epoch 2 Validation | AUC: 0.863474, LogLoss: 0.223943
Epoch 3 Validation | AUC: 0.864767, LogLoss: 0.222967
Epoch 4 Validation | AUC: 0.864326, LogLoss: 0.223255
Epoch 5 Validation | AUC: 0.865205, LogLoss: 0.222598
Epoch 6 Validation | AUC: 0.864897, LogLoss: 0.222708
Test AUC: 0.821334
```

### time context v1.5 + pair

```text
Epoch 1 Validation | AUC: 0.862605, LogLoss: 0.224580
Epoch 2 Validation | AUC: 0.864252, LogLoss: 0.223110
Epoch 3 Validation | AUC: 0.865933, LogLoss: 0.222011
Epoch 4 Validation | AUC: 0.866221, LogLoss: 0.221438
Epoch 5 Validation | AUC: 0.866106, LogLoss: 0.222873
Epoch 6 Validation | AUC: 0.866664, LogLoss: 0.221478
Test AUC: 0.822588
```

### user time-feat x pair

```text
Epoch 1 Validation | AUC: 0.862734, LogLoss: 0.227111
Epoch 2 Validation | AUC: 0.865353, LogLoss: 0.222581
Epoch 3 Validation | AUC: 0.867281, LogLoss: 0.220654
Epoch 4 Validation | AUC: 0.866842, LogLoss: 0.220870
Epoch 5 Validation | AUC: 0.867006, LogLoss: 0.222049
Test AUC: 0.824264
```

### time context v1.5 + pair + din

```text
Epoch 1 Validation | AUC: 0.862512, LogLoss: 0.225694
Epoch 2 Validation | AUC: 0.865116, LogLoss: 0.222954
Epoch 3 Validation | AUC: 0.866477, LogLoss: 0.221825
Epoch 4 Validation | AUC: 0.867315, LogLoss: 0.221283
Epoch 5 Validation | AUC: 0.866663, LogLoss: 0.221703
Epoch 6 Validation | AUC: 0.865863, LogLoss: 0.222247
Test AUC: 0.823751
```

### user FiLM-time-feat x pair -> ft2

```text
Epoch 1 Validation | AUC: 0.862310, LogLoss: 0.226019
Epoch 2 Validation | AUC: 0.865402, LogLoss: 0.222579
Epoch 3 Validation | AUC: 0.866596, LogLoss: 0.221473
Epoch 4 Validation | AUC: 0.865965, LogLoss: 0.221903
Epoch 5 Validation | AUC: 0.866744, LogLoss: 0.221368
Epoch 6 Validation | AUC: 0.866530, LogLoss: 0.222194
Test AUC: 0.826725
```

### ft2_seqhour

```text
Epoch 1 Validation | AUC: 0.862895, LogLoss: 0.224336
Epoch 2 Validation | AUC: 0.865056, LogLoss: 0.222731
Epoch 3 Validation | AUC: 0.865942, LogLoss: 0.222415
Epoch 4 Validation | AUC: 0.866997, LogLoss: 0.221081
Epoch 5 Validation | AUC: 0.866992, LogLoss: 0.221450
Epoch 6 Validation | AUC: 0.866849, LogLoss: 0.222424
Epoch 7 Validation | AUC: 0.866868, LogLoss: 0.221543
Test AUC: 0.827799
```

</details>
