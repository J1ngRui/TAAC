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
5. `main_gact` / `main_ref` / `main_miss` 的共同现象是 valid AUC 或 LogLoss 继续变好，但 Test AUC 全部低于 `main_base`；不能再按 valid 单点提升推进主线。
6. 当前 token 数是 `T=22`，来自 `2*4` 个 Q token 加 `14` 个 NS token；旧 full RankMixer 为了满足 `d_model % T == 0` 使用 `d_model=88`。
7. 因为当前使用的是 `ns_tokenizer_type=group`，风险点不是 `RankMixerNSTokenizer` 的长向量切块，而是旧 `RankMixerBlock(mode=full)` 的无参数 token mixing 会把新增 NS token 的通道切片重排进所有 Q/NS token。
8. 主线改为 UniMixer-lite 风格的 `rank_mixer_mode=learned`，用可学习 soft token mixing 替代固定切片重排，并把 `d_model` 回归 64；mixer 后使用 shared SwiGLU FFN，只有 legacy `full` 模式仍要求 `d_model % T == 0`。

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

### time context v1 + target_cate_hist（main）

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
Epoch 2 Validation | AUC: 0.8643096070531222, LogLoss: 0.22333790361881256
Epoch 3 Validation | AUC: 0.8656528005616155, LogLoss: 0.22246608138084412
Epoch 4 Validation | AUC: 0.8651367361649579, LogLoss: 0.22213102877140045
Test AUC：0.820324
```

### main_miss

```text
Epoch 1 Validation | AUC: 0.8589506321191401, LogLoss: 0.22670724987983704
Epoch 2 Validation | AUC: 0.8633029522261406, LogLoss: 0.22371038794517517
Epoch 3 Validation | AUC: 0.864102765572538, LogLoss: 0.22351820766925812
Epoch 4 Validation | AUC: 0.8653373722327503, LogLoss: 0.22177353501319885
Epoch 5 Validation | AUC: 0.8645241416161386, LogLoss: 0.22299005091190338
Test AUC: 0.820701
```

### main_timeV15

```
Epoch 1 Validation | AUC: 0.8587096057391499, LogLoss: 0.22766029834747314
Epoch 2 Validation | AUC: 0.8625916652714761, LogLoss: 0.223570838570594
Epoch 3 Validation | AUC: 0.8632375230054341, LogLoss: 0.22510819137096405
Epoch 4 Validation | AUC: 0.8634255060057077, LogLoss: 0.2238648384809494
Epoch 5 Validation | AUC: 0.8644639678505021, LogLoss: 0.22247228026390076
```

### main_userPair_v1

```
Epoch 1 Validation | AUC: 0.8587585587062858, LogLoss: 0.22734025120735168
Epoch 2 Validation | AUC: 0.8611411895450327, LogLoss: 0.22540171444416046
Epoch 3 Validation | AUC: 0.8636768982765632, LogLoss: 0.22318686544895172
Epoch 4 Validation | AUC: 0.8638127107284859, LogLoss: 0.22348766028881073
Epoch 5 Validation | AUC: 0.8643185857558675, LogLoss: 0.2228502333164215
```

### main_userPair_v2

```
Epoch 1 Validation | AUC: 0.8599211103400455, LogLoss: 0.22627413272857666
Epoch 2 Validation | AUC: 0.8625859812763018, LogLoss: 0.2242143452167511
Epoch 3 Validation | AUC: 0.8636834708542607, LogLoss: 0.22450315952301025

```

### main_twc

```
Epoch 1 Validation | AUC: 0.8603345820184504, LogLoss: 0.22630852460861206
Epoch 2 Validation | AUC: 0.8631763531148908, LogLoss: 0.2242639660835266
Epoch 3 Validation | AUC: 0.8643155452393851, LogLoss: 0.22268152236938477
Epoch 4 Validation | AUC: 0.8643509163638711, LogLoss: 0.22328625619411469
Test AUC：早早拟合，感觉不对
```

### main_gapGag

```
Epoch 1 Validation | AUC: 0.8576212415126411, LogLoss: 0.2272794097661972
Epoch 2 Validation | AUC: 0.8611784372297984, LogLoss: 0.22493386268615723
Epoch 3 Validation | AUC: 0.863180056266262, LogLoss: 0.2235773205757141
Epoch 4 Validation | AUC: 0.8632286746570932, LogLoss: 0.2235700488090515
Epoch 5 Validation | AUC: 0.8633181178748037, LogLoss: 0.22385279834270477
Epoch 6 Validation | AUC: 0.8638972759757558, LogLoss: 0.22319595515727997
Epoch 7 Validation | AUC: 0.8639215734935893, LogLoss: 0.22353345155715942
```

对以下移除了tar-hist的得分点

tc_userPair

tc_timeV15

tc_miss

tc_gapGeg

tc_userPair_timeV15

</details>