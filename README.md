### 特征

| Category                 | Count | Dateset                 | Description                                                  |
| :----------------------- | :---- | :---------------------- | :----------------------------------------------------------- |
| ID & Label               | 5     | `int64` / `int32`       | Core identifiers, label, and timestamp.                      |
| User Int Features        | 46    | `int64` / `list<int64>` | Discrete user features, including both single-value scalar features (such as age, gender, etc.) and multi-value array features (like marital status, etc.), describing user basic attributes and preferences. |
| User Dense Features      | 10    | `list<float>`           | Continuous-valued user features, including embeddings and other aligned signals for some corresponding integer features. |
| Item Int Features        | 14    | `int64` / `list<int64>` | Discrete item features, including item categories, types, and other basic information, as well as multi-label information for items. |
| Domain Sequence Features | 45    | `list<int64>`           | Behavioral sequence features from 4 domains.                 |

### DataSet 数据分布分析

问题：val与online test 指标存在较大的gap

------

timestamp_order increase=50.08%, decrease=49.92%
当前读取顺序/index/RowGroup 顺序 **和 timestamp 没有关系**，基本是打散状态。

user_overlap = 0
每个 user 基本只出现一次，用户历史已经被提前聚合进 seq_a/b/c/d

item overlap 只有 29%左右，item 冷启动压力不小

```
item_overlap=4299
valid_item_seen_in_train=28.92%
train_item_seen_in_valid=31.95%
```

也就是说，valid 里只有约 29% 的 item 在 train 中出现过，**模型不能太依赖 item_id 记忆。**

如果 item embedding 过强，它在 train 上可能有效，但对 valid/test 的大量未见 item 帮助有限。

应该更重视：

```
item 类目特征
item 多标签特征
用户行为序列和 item 的匹配
domain sequence 表征
用户 dense 特征
```

## Baseline

Baseline 设计与 NS tokenizer 对比已拆到 [baseline.md](baseline.md)，README 只保留后续实验 timeline。



## TS / Time Context 优化 Timeline

> 当前主线：`group tokenizer + time context v1 + target_cate_hist + recent_activity + BCE`。当前 time context 只保留 day 内周期与 week 周期，已删除 month 周期特征；Weighted Loss / WLoss 已从代码中删除，不再作为训练入口。

| 序号 | 模型 / 实验名     | 背景 / 动机                                       | 结构 / 变更                                             | 结果                          | 增幅 / 降幅                                | 结论                                    |
| ---: | :---------------- | :------------------------------------------------ | :------------------------------------------------------ | :---------------------------- | :----------------------------------------- | :-------------------------------------- |
|    0 | Baseline          | 建立原始参照。                                    | 原始 baseline                                           | valid 0.86224 / test 0.8123   | -                                          | valid-test gap 明显，需要泛化信息。     |
|    1 | Timestamp split   | 怀疑 valid/test 时间分布不一致。                  | 按 timestamp 尝试重划分                                 | 分布更近，但 test 下降        | test 下降                                  | 不能靠切分解决，转向显式时间建模。      |
|    2 | TS1               | 让模型直接看到时间上下文。                        | group + hybrid + time context                           | valid ≈0.8648 / test 0.8213   | vs baseline: valid +0.00256 / test +0.0090 | time context 是核心增益。               |
|    3 | TS2               | 检查 hybrid compression 是否损失语义。            | 移除 hybrid，保留原始 group token                       | valid ≈0.8655 / test 0.820503 | vs TS1:    valid +0.0007 / test -0.000797  | 表达更强但更容易过拟合。                |
|    4 | TS3               | 尝试更细地表达时间周期。                          | 更细粒度 time context                                   | valid 下降                    | valid 下降                                 | 丢弃。                                  |
|    5 | TS2 Dropout       | valid 高但 test 不同步，怀疑时间特征过拟合。      | time context projection 加 dropout                      | 目标拉回 test                 | 待验证                                     | 轻量正则方向，后续被 cyclic time 替代。 |
|    6 | Time Shift        | valid-test gap 更像时间状态偏移问题。             | 加强 temporal shift 视角                                | anchor delta 负收益           | test  -0.000543                            | 保留“时间状态/偏移”这个建模思路。       |
|    7 | TS4               | 尝试补充 month 周期信号。                         | day/week 周期 + month 周期                              | 持平                          | 持平                                       | month 周期已删除，回到 v1 day/week 口径。 |
|    8 | WLoss             | 尝试缓解少量异常样本带来的 loss 震荡。            | BCE linear reweight / tail negative downweight          | logloss 有收益但 AUC 不稳     | test 明显下降                              | 已删除模块，当前只保留 BCE / focal 入口。 |
|    9 | Target Hist Match | 需要比 item_id 更泛化的 target/history 匹配信号。 | time context v1 + target/history semantic match + BCE   | 待重跑干净口径                | 待验证                                     | 当前 active 结构方向。                  |
|   10 | Domain Time Mild  | 时间是强特征，但需要温和正则和分 domain 表达。    | target_cate_hist + domain time buckets + time dropout 0.02 | 待训练                     | 待验证                                     | 已从当前主线回退。                      |
|   11 | Recent Activity   | 用户近期活跃度可能是强泛化信号。                  | 每路序列生成 last_delta / 1h / 1d / 7d count bucket，追加到最后一个 user NS group | 待训练 | 待验证 | 不新增 token，保持 `T=22,d_model=88`。 |



## A / B 实验对照

> 对照口径：`best valid` 按验证集 AUC 最高 epoch 统计。`相对 Best Test` = Test AUC - 当前已测试最优 Test AUC，当前 best = 0.821200。

| 实验 | 关键变更 | Best Valid AUC | Best Epoch | Test AUC | 相对 Best Test | 结论 |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| time context v1 + hybrid v1 | time context + hybrid 压缩 token | 0.864835 | 6 | **0.821200** | **+0.000000** | 当前已测试版本里 test 最好。 |
| time context v1 | 移除 hybrid，保留原始 group token | **0.865527** | 5 | 0.820503 | -0.000697 | valid 最高，但 test 低于 hybrid，可能存在过拟合或 hybrid 有信息增强/正则效果。 |
| time context v1 + anchor_time | 加入 anchor time 特征 | 0.864272 | 5 | 未测试 | - | valid 没超过 v1，不优先投入 test。 |
| time context v1 + anchor_time + WLoss | anchor time + weighted loss | 0.864433 | 4 | 0.817130 | -0.004070 | anchor_time / WLoss 组合置信度不高。 |
| time context v2 + WLoss | v2 time context + weighted loss | 0.864621 | 6 | 0.817673 | -0.003527 | valid 尚可，但 test 明显下降，排除。 |
| time context v2 + WLoss + target_cate_hist | v2 + WLoss + target/history 类目匹配 | 0.863951 | 5 | 0.820517 | -0.000683 | target_cate_hist 对 test 有恢复信号，但 WLoss 拖累仍明显。 |
| time context v2 | 纯 v2 time context | 0.864367 | 3 | 0.820265 | -0.000935 | 纯 v2 低于 v1 / v1 hybrid，暂不作为主线。 |
| time context v2 + target_cate_hist | v2 + target/history 类目匹配，标准 BCE | 0.863920 | 5 | 未测试 | - | v2 本体已弱于 v1，是否继续测取决于是否单独验证 target_cate_hist。 |

阶段判断：

1. 当前模型主线固定为 `group + time context v1 + target_cate_hist + recent_activity + BCE`。
2. `time context v1` 的 clean 口径保留 day 内周期与 week 周期，不再包含 month 周期特征。
3. WLoss 相关训练入口和实现已删除，后续 A/B 不再混入 loss reweight 变量。
4. 下一步重点是重跑当前 clean 主线，对照 `time context v1` 与 `time context v1 + hybrid v1` 的已知 test 结果。

<details>
<summary>逐 epoch 原始记录</summary>

### time context v1 + compressed hybrid v1

```text
Epoch 1 Validation | AUC: 0.858602, LogLoss: 0.227267
Epoch 2 Validation | AUC: 0.861969, LogLoss: 0.225463
Epoch 3 Validation | AUC: 0.862991, LogLoss: 0.223749
Epoch 4 Validation | AUC: 0.863880, LogLoss: 0.223123
Epoch 5 Validation | AUC: 0.864183, LogLoss: 0.222909
Epoch 6 Validation | AUC: 0.864835, LogLoss: 0.222290
Epoch 7 Validation | AUC: 0.864470, LogLoss: 0.222460
Test AUC: 0.821200
```

### time context v1

```text
Epoch 1 Validation | AUC: 0.858992, LogLoss: 0.227252
Epoch 2 Validation | AUC: 0.863453, LogLoss: 0.223940
Epoch 3 Validation | AUC: 0.864340, LogLoss: 0.223234
Epoch 4 Validation | AUC: 0.864564, LogLoss: 0.223251
Epoch 5 Validation | AUC: 0.865527, LogLoss: 0.222461
Epoch 6 Validation | AUC: 0.865100, LogLoss: 0.222718
Test AUC: 0.820503
```

### time context v1 + anchor_time

```text
Epoch 1 Validation | AUC: 0.859736, LogLoss: 0.226455
Epoch 2 Validation | AUC: 0.862898, LogLoss: 0.223629
Epoch 3 Validation | AUC: 0.863885, LogLoss: 0.223102
Epoch 4 Validation | AUC: 0.864269, LogLoss: 0.222635
Epoch 5 Validation | AUC: 0.864272, LogLoss: 0.222971
Epoch 6 Validation | AUC: 0.863825, LogLoss: 0.223036
Test AUC: 未测试
```

### time context v1 + anchor_time + WLoss

```text
Epoch 1 Validation | AUC: 0.859736, LogLoss: 0.226455
Epoch 2 Validation | AUC: 0.862898, LogLoss: 0.223629
Epoch 3 Validation | AUC: 0.863856, LogLoss: 0.223685
Epoch 4 Validation | AUC: 0.864433, LogLoss: 0.222764
Epoch 5 Validation | AUC: 0.864027, LogLoss: 0.223762
Test AUC: 0.817130
```

### time context v2 + WLoss

```text
Epoch 1 Validation | AUC: 0.860049, LogLoss: 0.226179
Epoch 2 Validation | AUC: 0.863377, LogLoss: 0.223471
Epoch 3 Validation | AUC: 0.864380, LogLoss: 0.222985
Epoch 4 Validation | AUC: 0.864342, LogLoss: 0.222643
Epoch 5 Validation | AUC: 0.864359, LogLoss: 0.222981
Epoch 6 Validation | AUC: 0.864621, LogLoss: 0.222876
Epoch 7 Validation | AUC: 0.863733, LogLoss: 0.223559
Test AUC: 0.817673
```

### time context v2 + WLoss + target_cate_hist

```text
Epoch 1 Validation | AUC: 0.859629, LogLoss: 0.226133
Epoch 2 Validation | AUC: 0.863386, LogLoss: 0.223394
Epoch 3 Validation | AUC: 0.863715, LogLoss: 0.223392
Epoch 4 Validation | AUC: 0.863032, LogLoss: 0.223757
Epoch 5 Validation | AUC: 0.863951, LogLoss: 0.223181
Epoch 6 Validation | AUC: 0.863165, LogLoss: 0.225804
Test AUC: 0.820517
```

### time context v2

```text
Epoch 1 Validation | AUC: 0.860049, LogLoss: 0.226179
Epoch 2 Validation | AUC: 0.863377, LogLoss: 0.223471
Epoch 3 Validation | AUC: 0.864367, LogLoss: 0.222941
Epoch 4 Validation | AUC: 0.864267, LogLoss: 0.222647
Epoch 5 Validation | AUC: 0.864346, LogLoss: 0.222892
Test AUC: 0.820265
```

### time context v2 + target_cate_hist

```text
Epoch 1 Validation | AUC: 0.859629, LogLoss: 0.226133
Epoch 2 Validation | AUC: 0.863386, LogLoss: 0.223394
Epoch 3 Validation | AUC: 0.863712, LogLoss: 0.223384
Epoch 4 Validation | AUC: 0.863016, LogLoss: 0.223749
Epoch 5 Validation | AUC: 0.863920, LogLoss: 0.223018
Epoch 6 Validation | AUC: 0.863553, LogLoss: 0.225041
Test AUC: 未测试，因为v2不如v1
```

</details>

<details>
<summary>Main epoch 记录</summary>

### main_base (time context v1 + target_cate_hist)


```text
Epoch 1 Validation | AUC: 0.858521, LogLoss: 0.227749
Epoch 2 Validation | AUC: 0.862416, LogLoss: 0.223996
Epoch 3 Validation | AUC: 0.863655, LogLoss: 0.223146
Epoch 4 Validation | AUC: 0.863852, LogLoss: 0.223166
Epoch 5 Validation | AUC: 0.864069, LogLoss: 0.223746
Epoch 6 Validation | AUC: 0.863666, LogLoss: 0.222999
Test AUC：0.8222
```

### main_hybrid_ns_self

```text
Epoch 1 Validation | AUC: 0.859000, LogLoss: 0.227512
Epoch 2 Validation | AUC: 0.862230, LogLoss: 0.227064
Epoch 3 Validation | AUC: 0.863340, LogLoss: 0.224158
Epoch 4 Validation | AUC: 0.863180, LogLoss: 0.223397
Epoch 5 Validation | AUC: 0.864470, LogLoss: 0.222976
Epoch 6 Validation | AUC: 0.864040, LogLoss: 0.223229
Test AUC：
```

### main_hybrid_ns_learnQ

```text
Epoch 1 Validation | AUC: 0.859421, LogLoss: 0.226930
Epoch 2 Validation | AUC: 0.861414, LogLoss: 0.224316
Epoch 3 Validation | AUC: 0.862625, LogLoss: 0.224154
Epoch 4 Validation | AUC: 0.863670, LogLoss: 0.223041
Epoch 5 Validation | AUC: 0.864177, LogLoss: 0.222886
Epoch 6 Validation | AUC: 0.864850, LogLoss: 0.222479
Test AUC：
```

### main_v2 (domain-only emb)

```text
Epoch 1 Validation | AUC: 0.8599513096535375, LogLoss: 0.22652412950992584
Epoch 2 Validation | AUC: 0.8627960718404153, LogLoss: 0.22386665642261505
Epoch 3 Validation | AUC: 0.8631914155663172, LogLoss: 0.2236473262310028
Epoch 4 Validation | AUC: 0.8641966468145745, LogLoss: 0.22266629338264465
Epoch 5 Validation | AUC: 0.8649472459418838, LogLoss: 0.22258137166500092
Epoch 6 Validation | AUC: 0.8635087410949094, LogLoss: 0.22377458214759827
Test AUC：
```

### main_v2_Act

```text
Epoch 1 Validation | AUC: 0.861214, LogLoss: 0.225494
Epoch 2 Validation | AUC: 0.863329, LogLoss: 0.223689
Epoch 3 Validation | AUC: 0.863973, LogLoss: 0.222719
Epoch 4 Validation | AUC: 0.864003, LogLoss: 0.223151
Epoch 5 Validation | AUC: 0.863648, LogLoss: 0.223132
Epoch 6 Validation | AUC: 0.863579, LogLoss: 0.223120
Test AUC：auc: 0.813019
```



### main_base (time context v1 + target_cate_hist)

```text
Epoch 1 Validation | AUC: 0.858521, LogLoss: 0.227749
Epoch 2 Validation | AUC: 0.862416, LogLoss: 0.223996
Epoch 3 Validation | AUC: 0.863655, LogLoss: 0.223146
Epoch 4 Validation | AUC: 0.863852, LogLoss: 0.223166
Epoch 5 Validation | AUC: 0.864069, LogLoss: 0.223746
Epoch 6 Validation | AUC: 0.863666, LogLoss: 0.222999
Test AUC：0.8222
```



</details>