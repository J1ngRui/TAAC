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
Stage5｜TS2 + Regularization：回到 TS2 做泛化增强
结构：
baseline + group + time context + remove hybrid
在 TS2 基础上增加正则项。

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
TS2_regularized 是当前最值得继续跑的主线。
目标不是让 valid 继续变高，而是让 test 回到甚至超过 TS1 的 0.8213
```









