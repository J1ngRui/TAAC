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

## 实验 Timeline / A-B 总表

> 当前 clean main_base：`group tokenizer + time context v1 + target_cate_hist + BCE`。当前 time context 只保留 day 内周期与 week 周期，已删除 month 周期特征；Weighted Loss / WLoss 已从代码中删除，不再作为训练入口。
>
> 对照口径：`Best Valid AUC` 按验证集 AUC 最高 epoch 统计。`相对 main_base Test` = Test AUC - clean main_base Test AUC，main_base Test AUC = 0.822200。

| 序号 | 实验 | 背景 / 动机 | 关键变更 | Best Valid AUC | Best Epoch | Test AUC | 相对 main_base Test | 结论 |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | Baseline | 建立原始参照。 | 原始 baseline | 0.862240 | - | 0.812300 | -0.009900 | valid-test gap 明显，需要泛化信息。 |
| 1 | Timestamp split | 怀疑 valid/test 时间分布不一致。 | 按 timestamp 尝试重划分 | - | - | 下降 | - | 不能靠切分解决，转向显式时间建模。 |
| 2 | time context v1 + hybrid v1 | 让模型直接看到时间上下文。 | group + hybrid + time context | 0.864835 | 6 | 0.821200 | -0.001000 | time context 是核心增益，但 hybrid 低于 clean main_base。 |
| 3 | time context v1 | 检查 hybrid compression 是否损失语义。 | 移除 hybrid，保留原始 group token | **0.865527** | 5 | 0.820503 | -0.001697 | valid 最高，但 test 低于 hybrid/main_base，存在过拟合风险。 |
| 4 | TS3 | 尝试更细地表达时间周期。 | 更细粒度 time context | 下降 | - | 未测试 | - | 丢弃。 |
| 5 | TS2 Dropout | valid 高但 test 不同步，怀疑时间特征过拟合。 | time context projection 加 dropout | 待验证 | - | 未测试 | - | 轻量正则方向，后续被 cyclic time 替代。 |
| 6 | Time Shift | valid-test gap 更像时间状态偏移问题。 | 加强 temporal shift 视角 | - | - | 负收益 | -0.000543 | 保留“时间状态/偏移”这个建模思路。 |
| 7 | TS4 | 尝试补充 month 周期信号。 | day/week 周期 + month 周期 | 持平 | - | 持平 | - | month 周期已删除，回到 v1 day/week 口径。 |
| 8 | WLoss | 尝试缓解少量异常样本带来的 loss 震荡。 | BCE linear reweight / tail negative downweight | 不稳 | - | 下降 | - | 已删除模块，当前只保留 BCE / focal 入口。 |
| 9 | time context v1 + anchor_time | 验证 anchor time 是否能补充时间上下文。 | 加入 anchor time 特征 | 0.864272 | 5 | 未测试 | - | valid 没超过 v1，不优先投入 test。 |
| 10 | time context v1 + anchor_time + WLoss | 验证 anchor time 与 weighted loss 组合。 | anchor time + weighted loss | 0.864433 | 4 | 0.817130 | -0.005070 | anchor_time / WLoss 组合置信度不高。 |
| 11 | time context v2 + WLoss | 验证 v2 time context 与 weighted loss。 | v2 time context + weighted loss | 0.864621 | 6 | 0.817673 | -0.004527 | valid 尚可，但 test 明显下降，排除。 |
| 12 | time context v2 + WLoss + target_cate_hist | 检查 target_cate_hist 能否修复 v2/WLoss 的 test 损失。 | v2 + WLoss + target/history 类目匹配 | 0.863951 | 5 | 0.820517 | -0.001683 | target_cate_hist 对 test 有恢复信号，但 WLoss 拖累仍明显。 |
| 13 | time context v2 | 单独验证 v2 time context。 | 纯 v2 time context | 0.864367 | 3 | 0.820265 | -0.001935 | 纯 v2 低于 v1 / v1 hybrid / main_base，暂不作为主线。 |
| 14 | time context v2 + target_cate_hist | 验证 v2 口径下 target/history 类目匹配。 | v2 + target/history 类目匹配，标准 BCE | 0.863920 | 5 | 未测试 | - | v2 本体已弱于 v1，不优先继续。 |
| 15 | main_base | 需要比 item_id 更泛化的 target/history 匹配信号。 | time context v1 + target_cate_hist + BCE | 0.864069 | 5 | **0.822200** | **+0.000000** | 当前 clean anchor。 |
| 16 | main_v2 domain-time | 时间是强特征，但需要温和正则和分 domain 表达。 | target_cate_hist + domain time buckets + time dropout 0.02 | 0.864947 | 5 | 未测试 | - | valid 小幅提升，但缺少 test 证明，暂不进主线。 |
| 17 | main_gact | 用户近期活跃度可能是强泛化信号。 | main_base 上叠加 global recent_activity | 0.865337 | 4 | 0.821239 | -0.000961 | valid 提升但 test 低于 main_base，不能作为独立收益证明。 |
| 18 | main_v2_Act | 验证 domain-time 口径叠加 recent_activity。 | domain-time + recent_activity | 0.864003 | 4 | 0.813019 | -0.009181 | 该口径 test 明显差，排除。 |
| 19 | Global Target Timewise | target/history 需要表达近期趋势，但避免 per-domain 稀疏。 | target_cate_hist 后追加全局 last_position / recent_ratio / recent_trend bucket | 待补充 | - | 待补充 | - | 比 per-domain timewise 更低风险，但需 clean A/B 后再进主线。 |

阶段判断：

1. 当前 clean 主线固定为 `group + time context v1 + target_cate_hist + BCE`。
2. `time context v1` 的 clean 口径保留 day 内周期与 week 周期，不再包含 month 周期特征。
3. `domain_time_buckets` / `time_context_dropout` 属于结构容量改动，缺少强 test 收益证明，已从主线回退。
4. WLoss 相关训练入口和实现已删除，后续 A/B 不再混入 loss reweight 变量。
5. 当前 A/B 重点是以 `main_base` 为 anchor，单独验证 `+act` / `+global_target_timewise` 的独立和边际收益。

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

### main_prev （time context v1）


```
Epoch 1 Validation | AUC: 0.8589916953859507, LogLoss: 0.22725152969360352
Epoch 2 Validation | AUC: 0.8634527978728386, LogLoss: 0.22394008934497833
Epoch 3 Validation | AUC: 0.8643401958003973, LogLoss: 0.22323431074619293
Epoch 4 Validation | AUC: 0.864563507016832, LogLoss: 0.22325144708156586
Epoch 5 Validation | AUC: 0.8655272198441499, LogLoss: 0.22246094048023224
Epoch 6 Validation | AUC: 0.8651004818893825, LogLoss: 0.22271838784217834
Test AUC：0.8212
```

### main_base （time context v1 + target_cate_hist）


```text
Epoch 1 Validation | AUC: 0.858521, LogLoss: 0.227749
Epoch 2 Validation | AUC: 0.862416, LogLoss: 0.223996
Epoch 3 Validation | AUC: 0.863655, LogLoss: 0.223146
Epoch 4 Validation | AUC: 0.863852, LogLoss: 0.223166
Epoch 5 Validation | AUC: 0.864069, LogLoss: 0.223746
Epoch 6 Validation | AUC: 0.863666, LogLoss: 0.222999
Test AUC：0.8222
```

### main_gact

```text
Epoch 1 Validation | AUC: 0.8610430630945604, LogLoss: 0.22593118250370026
Epoch 2 Validation | AUC: 0.8641009270322928, LogLoss: 0.22306373715400696
Epoch 3 Validation | AUC: 0.8642881203021332, LogLoss: 0.22289903461933136
Epoch 4 Validation | AUC: 0.8653365021171011, LogLoss: 0.22231131792068481
Epoch 5 Validation | AUC: 0.8641097150917216, LogLoss: 0.22321653366088867
Test AUC：0.821239
```

### main_ref

```text
Epoch 1 Validation | AUC: 0.8602357429639006, LogLoss: 0.22603169083595276
```

### main_miss

```text
Epoch 1 Validation | AUC: 0.8589506321191401, LogLoss: 0.22670724987983704
Epoch 2 Validation | AUC: 0.8633029522261406, LogLoss: 0.22371038794517517
```

### main_trendc

```text

```

</details>
