#### --time_context

`time context` 是**当前样本曝光时间**的全局上下文 token
它做的是把当前样本的 `timestamp` 转成周期特征：

```text
当前本地一天中的位置：
sin(hour/minute/second)
cos(hour/minute/second)

当前一周中的位置：
sin(day_of_week)
cos(day_of_week)
```

然后构造 4 维特征：

```text
sin(day_angle)
cos(day_angle)
sin(week_angle)
cos(week_angle)
```

再过一层投影MLP：变成一个 NS token，time_context_tok
它表达的是，作为独立的上下文token：

```text
当前曝光发生在一天/一周中的什么时间位置
```



#### --target_hist_match

###### `main` base 原版的 `target-hist` 是这样接的：

```text
原始 item_int schema
+ 追加 4 个 target_hist_match 动态 item_int 特征
+ 这 4 个特征组成一个新增 item NS group
```

这 4 个 fid 默认是：

```text
200001,200002,200003,200004
```

对应语义是：

```text
cate_in_hist:
  target category 是否在历史 category 里出现过

cate_count_bucket:
  target category 在有效历史 category 中出现次数的分桶

cate_ratio_bucket:
  cate_count / hist_cate_len 的比例分桶

cate_last_delta_bucket:
  最近一次 target category 命中距离当前曝光时间有多久
```

然后训练侧会把它们作为一个 `match_group` append 到 `item_ns_groups`：

```text
Added I5_target_hist_match item NS group
```



#### --use_recent_activity

当前配置：

```bash
--use_recent_activity
--recent_activity_mode global
--recent_activity_feature_fids 210001,210002,210003,210004
```

所以实际只生成 4 个 user_int 动态特征：

```text
210001: global_last_delta_bucket
210002: global_count_1h_bucket
210003: global_count_1d_bucket
210004: global_count_7d_bucket
```

逻辑是：

```text
把 seq_a / seq_b / seq_c / seq_d 的历史 timestamp 合并
过滤：
  hist_ts > 0
  hist_ts < 当前样本 timestamp
  position < seq_len

然后计算：
  最近一次任意历史行为距离当前多久
  过去 1h 任意历史行为数
  过去 1d 任意历史行为数
  过去 7d 任意历史行为数
```

然后在 [train.py](D:/code/project/TAAC/train.py:419) 被追加到最后一个 user NS group 里，不单独新增一个 NS token



#### --s_token_timeRefine

`time bucket refine` 是这次加的 **seq 侧历史时间增强**
它不替代原来的 time bucket，而是在原来的基础上做一个温和修正。
refine 是 **“多久以前这件事发生在什么周期位置，会不会改变这个多久以前的含义”。**

原主干是：

```text
time_bucket_emb = Embedding(当前曝光 timestamp - 历史行为 timestamp)
seq_tok = seq_base_tok + time_bucket_emb
```

现在变成：

```text
hist_hour_emb = Embedding(历史行为发生在几点)
hist_day_emb  = Embedding(历史行为发生在周几)

delta = PeriodRefiner(time_bucket_emb, hist_hour_emb, hist_day_emb)
gate  = sigmoid(Gate(time_bucket_emb, hist_hour_emb, hist_day_emb))

refined_time_emb = time_bucket_emb + gate * delta

seq_tok = seq_base_tok + refined_time_emb
```

它表达的是：

```text
历史行为距离当前多久 + 在这个时间差下，这个历史行为发生在几点/周几是否改变它的时间语义
```

它们的 recency bucket 可能相近，但周期语义不同。refine 就是让模型学这种差异。

所以训练刚开始时近似：

```text
refined_time_emb ≈ time_bucket_emb
```

不会一上来破坏原来已经有效的 time bucket 主干。训练过程中才逐渐学：

```text
time_bucket_emb + 小幅周期修正
```



#### --trend_context_current

这次的 `trend context` 是一个单独新增的 NS token。
它不改旧的 `--time_context`、`--target_hist_match`、`--use_recent_activity` 说明；这里只记录当前 active config 里这次合并出来的 `trend_context_token`。

当前 `run.sh` 打开的相关参数是：

```bash
--use_target_hist_match
--target_hist_match_target_cate_item_fid 13
--target_hist_match_cate_seq_fids seq_a:46,seq_b:68,seq_c:32,seq_d:25
--target_hist_match_feature_fids 200001,200002,200003,200004,200005

--use_recent_activity
--recent_activity_mode global
--recent_activity_feature_fids 210001,210002,210003
--recent_activity_recent_window_seconds 604800
--recent_activity_recent_k 64
```

它用到的原始信号是：

```text
target item category:
  item_int fid 13

history category:
  seq_a fid 46
  seq_b fid 68
  seq_c fid 32
  seq_d fid 25

history validity / recency:
  当前样本 timestamp
  各 domain 历史 timestamp
  各 domain seq_len
```

最终喂给 `trend_context_token` 的是 8 个 synthetic bucket 特征：

```text
200001: target_in_hist
200002: target_count_bucket
200003: target_ratio_bucket
200004: target_last_position_delta_bucket
200005: target_recent_trend_bucket

210001: user_activity_level
210002: user_activity_trend
210003: user_activity_conf
```

组 token 的方式是：

```text
trend_context_token =
Emb(210001,210002,210003) + Emb(200001,200002,200003,200004,200005)
-> Linear + LayerNorm + SiLU
```

