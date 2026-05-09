### 特征分析

| Category                 | Count | Dateset                 | Description                                                  |
| :----------------------- | :---- | :---------------------- | :----------------------------------------------------------- |
| ID & Label               | 5     | `int64` / `int32`       | Core identifiers, label, and timestamp.                      |
| User Int Features        | 46    | `int64` / `list<int64>` | Discrete user features, including both single-value scalar features (such as age, gender, etc.) and multi-value array features (like marital status, etc.), describing user basic attributes and preferences. |
| User Dense Features      | 10    | `list<float>`           | Continuous-valued user features, including embeddings and other aligned signals for some corresponding integer features. |
| Item Int Features        | 14    | `int64` / `list<int64>` | Discrete item features, including item categories, types, and other basic information, as well as multi-label information for items. |
| Domain Sequence Features | 45    | `list<int64>`           | Behavioral sequence features from 4 domains.                 |

| Column    | user_id | item_id | label_type | label_time   | timestamp  |
| :-------- | ------- | ------- | ---------- | ------------ | ---------- |
| Date Type | `int64` | `int64` | `int32`    | `int64`      | `int64`    |
|           |         |         | 标签类型   | 时间监督信号 | 样本时间戳 |



### DataSet数据分布

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





## TS / Time Context 优化 Timeline

```text
Stage0｜Baseline
结构：原始 baseline

结果：
valid best = 0.86224 | test = 0.8123

结论：
baseline 有明显 valid-test gap，说明 test 分布上缺少某些泛化信息。
```



```text
Stage1｜Timestamp split 尝试
结构：尝试用 timestamp 做数据划分，让 valid/test 分布更接近。

结果：分布更接近，但 test 反而下降。

结论：
timestamp split 不是主方向，不能靠切分方式解决泛化问题。
需要让模型显式学习时间上下文。
```



```text
Stage2｜Group + Hybrid + Time Context（TS1模型）
结构：
从 timestamp 中提取周期时间特征：
day / hour / week 等 sin-cos 特征。
baseline + group + hybrid + time context token。
建模方式：
sin-cos 不作为离散 ID；
而是作为连续 dense time feature：
sin/cos -> Linear -> time embedding。

结果：         valid ≈ 0.8648 | test = 0.8213
相对 baseline：valid +0.00256 | test +0.0090

结论：
时间特征应该作为 context 信息进入模型，而不是简单拼进原始 int feature。
time context 是核心有效增益。
valid 只小涨，但 test 大涨，说明时间上下文对 test 分布特别重要。
TS1模型 成为当前真实 best。
```



```text
Stage3｜TS2：Remove Hybrid，保留原始 Group Token
结构：
baseline + group + time context
移除 hybrid compression。

动机：
保留原始 semantic group ns token，避免 hybrid 压缩损失信息。

结果：
valid ≈ 0.8655
test = 0.820503

现象：
valid 比 TS1 高：0.8648 -> 0.8655
但 test 比 TS1 低：0.8213 -> 0.820503

结论：
remove hybrid 让表达能力更强，valid 更高；
但也去掉了 hybrid 的隐式正则，导致更容易过拟合。
```



```text
Stage4｜TS3：更细粒度 Time Context 提取与融合
结构：
在 TS2 / time context 主线上继续优化时间上下文。

核心变化：
不再只是粗粒度 timestamp context；
而是更细致拆分 day / week 等周期信息。

结论：
TS3 的目标是让 time context 表达更细，不只告诉模型“现在是什么时间”，而是同时表达日内周期、周内周期、长期周期等上下文。valid降低，丢弃。
```



```text
Stage5｜ts2_dropout1：TS2 + Time Context Dropout
结构：
baseline + group + time context + remove hybrid
在 TS2 基础上增加 time context dropout。

动机：
TS2 相比 TS1：
valid 更高：0.8648 -> 0.8655
test 更低：0.8213 -> 0.820503
说明 remove hybrid 后，模型表达能力增强；
但 hybrid 原本可能提供了一种隐式正则：
1. 压缩 token 数量；
2. 降低模型自由度；
3. 减少对局部 group/time pattern 的过拟合。
移除 hybrid 后，原始 group token 保留更完整，
valid 上涨说明信息确实更充分；
但 test 小降说明模型可能过拟合同分布 valid。

新增：
time context projection:
Linear -> LayerNorm -> GELU -> Dropout

原因：
time context 是当前增益核心，也是最可能导致过拟合的强特征；
先对 time token 做轻量正则，风险最低。

结论：
ts2_dropout1 是当前正在跑的主线。
目标不是让 valid 继续变高，而是让 test 回到甚至超过 TS1 的 0.8213
```



```text
Stage6｜Time Shift Generalization：时间偏移视角下的泛化增强
结构：
baseline + group + time context
在 ts2_dropout1 的基础上，继续围绕时间偏移/泛化能力做增强。

核心假设：
valid/test gap 不一定只是传统意义上的数据分布问题；
更可能是 valid 中可被模型拟合的信息，和 test 真正需要的泛化信息之间存在 gap。

也就是说：
valid 可能包含更多同窗口、同分布、局部 pattern 或短期记忆信号；
模型容量变强后，这些信号会抬高 valid；
但这些信息到 test 上不稳定，导致 test 不同步上涨。

对 timestamp 的重新理解：
timestamp / time context 的价值，不只是告诉模型“样本发生在什么时间”；
更重要的是引入一个时间偏移视角，让模型意识到 train -> valid/test 之间存在时间状态变化。

因此 time context 可能起到两层作用：
1. 提供周期性上下文，例如日内、周内行为差异；
2. 作为 temporal shift indicator，帮助模型减少对局部记忆信号的依赖，提升跨时间泛化。

下一步实验：
在保留 ts2_dropout1 的 dropout 设定下，增强 time context 的表达，
重点不是继续增加模型容量，而是更明确地表达时间偏移和时间状态。

候选方向：
- 加入绝对/相对时间趋势特征，弥补当前 sin-cos 只表达周期、不表达整体时间漂移的问题；
- 保留 day/week sin-cos 周期特征；
- 可尝试加入 coarse time bucket / normalized timestamp rank / weekend flag 等低维特征；
- 暂时不要同时给所有其他 token generator 加 dropout，避免变量混在一起。

目标：
验证 gap 缩小是否来自 time context 对 temporal shift 的建模，
而不仅是普通正则或数据切分变化。
```



```text
Stage7｜TS4：Fixed Early Anchor Time Delta
结构：
baseline + group + enhanced time context
在原有 time context 的周期特征基础上，加入一个相对于训练早期锚点的时间推进特征。

核心动机：
前面的 norm_time / after_train 思路存在一个问题：
如果 test 整体都晚于 train，模型在训练阶段可能没有足够监督来学习“越界之后应该怎么处理”。

Fixed Early Anchor 的做法不是使用 test 统计，也不是构造不可训练的未来越界特征；
而是在训练集中选一个固定早期时间锚点 anchor_ts，
让所有样本都能得到一个从训练早期开始累计推进的连续时间距离。

anchor_ts 定义：
不直接使用训练集最小 timestamp，避免极端 outlier。
使用训练集 timestamp 的低分位点，例如 p01：

anchor_ts = percentile(train_timestamps, 1)

anchor_delta 定义：
delta_t = max(timestamp - anchor_ts, 0)
anchor_delta = log1p(delta_t / 86400)

其中 anchor_delta 表示当前样本相对于训练早期固定锚点的天级推进距离。
log1p 用于压缩秒级 timestamp 带来的大尺度差异，使该特征更平滑、更容易训练。

建模方式：
原有 time context 仍保留 4 个周期特征：
day_sin, day_cos, week_sin, week_cos

这 4 个周期特征先经过原来的 time projection 转换；
anchor_delta 单独经过一个小的 embedding/projection 得到 anchor_delta embedding；
然后将二者 concat：

cyclic_time_feat = proj([day_sin, day_cos, week_sin, week_cos])
anchor_delta_emb = anchor_delta_proj(anchor_delta)
time_feat = GELU(Linear(concat(cyclic_time_feat, anchor_delta_emb)))

最终 time_feat 仍作为 1 个 time context NS token 拼入 user/item/dense NS tokens 后面。

预期作用：
1. 周期特征负责表达日内/周内行为周期；
2. anchor_delta 负责表达样本距离训练早期锚点的时间推进程度；
3. 模型可以自行学习：当样本时间推进更远时，哪些局部 pattern / group token / item 信号应该降低依赖。

优势：
- 只依赖 train timestamp 统计，不使用 test 统计；
- 训练集中所有样本都有 anchor_delta，不像 after_train 那样大部分为 0；
- 相比绝对 timestamp，anchor_delta 更稳定、尺度更合理；
- 相比单纯周期 sin-cos，能表达随时间移动的动态趋势。

当前定位：
TS4 是 ts2_dropout1 之后的 enhanced time 主线。
如果 ts2_dropout1 证明 dropout 能缩小 valid/test gap，
下一步用 Fixed Early Anchor Time Delta 验证 gap 是否还能通过时间趋势建模进一步缩小。

实现开关：
只保留一个实验开关：
--time_context_use_anchor_delta

关闭时：保持当前 ts2_dropout1 的 cyclic time context，不改变原有 time token 生成逻辑。
开启时：训练阶段只用 train timestamp 抽样估计 anchor_ts，并启用 anchor_delta residual 分支。

anchor_ts 不是手动实验开关，而是训练时统计出来的运行参数：
sample 5% train Row Groups -> timestamp p01 -> 写入 train_config.json。
推理时必须复用 train_config.json 中保存的 anchor_ts，不能重新用 valid/test 统计。

残差结构：
cyclic_tok = time_context_proj([day_sin, day_cos, week_sin, week_cos])
anchor_tok = time_anchor_delta_proj(anchor_delta)
anchor_alpha = learnable scalar, initialized to 0.1
time_tok = cyclic_tok + anchor_alpha * anchor_tok

这样原来的周期 time feature 不会被二次 concat/fuse 压缩；
anchor_delta 只作为趋势残差注入。
初始阶段 anchor 分支影响较小，模型先沿用 TS2 的周期 time context，
再通过训练自行决定 anchor_delta 应该加多强。
```



```text
Stage8｜TS5：Conflict-Downweighted BCE
结构：baseline + group + time context + fixed early anchor delta
在 TS4 的时间上下文结构基础上，不继续加大全局 dropout，
而是在 loss 层对极端 label/prediction 冲突样本做局部降权。

动机：
当前训练中 logloss 和 loss 一直存在震荡。
继续依赖 dropout 会全局削弱模型表达，尤其会同时影响正常样本和有效模式；
但震荡更可能来自少量极端冲突样本对 BCE 梯度的放大。

TS5 的目标不是让模型整体变弱，
而是只降低“当前预测极度自信但与 label 相反”的样本对单步更新的影响，
让主体样本继续按 BCE 正常学习。

loss 定义：
先计算逐样本 BCEWithLogits：

loss_raw = BCEWithLogits(logits, label, reduction="none")

再根据当前预测概率识别极端冲突样本：

pos_conflict = (label == 1) & (sigmoid(logits) < 0.05)
neg_conflict = (label == 0) & (sigmoid(logits) > 0.95)

这些样本的 loss weight 从 1.0 降到 0.2：

loss = sum(loss_raw * weight) / sum(weight)

实验开关：
--loss_type conflict_bce
--conflict_pos_prob_threshold 0.05
--conflict_neg_prob_threshold 0.95
--conflict_weight 0.2

当前定位：
TS5 是 TS4 之后的 loss-stabilization 主线。
它保留 TS4 的 time context + anchor_delta residual，
只改变训练目标的样本加权方式，不改变模型结构和推理路径。

评估注意：
validation logloss 仍然使用标准 BCEWithLogits 计算，
因此 TS5 的 valid logloss 可以和之前实验直接比较。
如果 TS5 有效，预期现象应该是训练 loss/logloss 震荡减弱，
同时 valid/test gap 不因继续增大 dropout 而被动扩大。
```









