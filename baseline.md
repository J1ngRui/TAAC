# Baseline

Baseline 的核心目标是建立一个可解释、可复现的原始参照，用来判断后续 time context、target-hist match、loss 调整等改动到底带来了什么收益。

当前 baseline 的重点不是堆模型容量，而是先把异质特征组织成稳定的 token 表达。这里主要比较两种 NS tokenizer：`GroupNSTokenizer` 和 `RankMixerNSTokenizer`。



## 一、Tokenizer

NS tokenizer 的作用是把原始离散特征 embedding 转成模型后续模块可以处理的 token 序列。

对于 single-value fid，直接查对应 embedding table。对于 multi-value fid，先对多个 value 的 embedding 做 mask mean pooling，得到一个固定维度的 fid embedding。之后不同 tokenizer 会用不同方式把这些 fid embedding 组合成 NS token。

## Group NS Tokenizer

`GroupNSTokenizer` 按语义 group 生成 token，一个 group 对应一个 NS token。

处理方式：

1. 遍历每个离散特征 fid。
2. 对每个 fid 查 embedding，multi-value fid 先做 mask mean pooling。
3. 将同一个 group 内的 fid embedding concat 到一起。
4. 通过 `Linear + LayerNorm + SiLU` 投影成一个 `d_model` 维 token。
5. 最终输出形状为 `(B, num_groups, d_model)`。

```
nn.Linear(len(group) * emb_dim, d_model),
nn.LayerNorm(d_model)
cat_emb = torch.cat(fid_embs, dim=-1)  
tokens.append(F.silu(proj(cat_emb)).unsqueeze(1)) 
```

这种方式的好处是语义边界清楚。用户属性、item 属性、时间上下文、行为序列等不同特征块不会被随意打散，不同 group 也可以使用独立 projection，更适合当前这种异质特征较多的 PCVR 任务。

代价是 token 数量由 group 数决定。如果 group 划分过细，后续 RankMixer / Attention 的计算压力会变大。另外，multi-value 特征当前使用 mean pooling，可能会损失 value 内部的重要性差异。

## RankMixer NS Tokenizer

`RankMixerNSTokenizer` 不保留严格的语义 group 边界，而是把所有 fid embedding 拼成一个长向量，再切成固定数量的 NS token。

处理方式：

1. 遍历所有 group 内的 fid。
2. 对每个 fid 查 embedding，multi-value fid 先做 mask mean pooling。
3. 按 group 顺序 concat 成一个长向量。
4. 如果总维度不能被 `num_ns_tokens` 整除，则在末尾 padding。
5. 将长向量均匀 split 成 `num_ns_tokens` 个 chunk。
6. 每个 chunk 通过独立的 `Linear + LayerNorm + SiLU` 投影成一个 `d_model` 维 token。
7. 最终输出形状为 `(B, num_ns_tokens, d_model)`。

这种方式的优势是 token 数量可控，方便限制模型复杂度和计算量，也更接近 RankMixer 原始的 token 化思路。

主要问题是语义边界可能被打散。一个 token 可能同时包含某个 group 的后半部分和另一个 group 的前半部分；当特征异质性很强时，这种均匀切块可能造成语义混合和信息损失。

## 对比

| 维度 | GroupNSTokenizer | RankMixerNSTokenizer |
| :-- | :-- | :-- |
| token 来源 | 一个语义 group 一个 token | 全部 fid embedding 拼接后均匀切块 |
| token 数量 | 由 group 数决定 | 由 `num_ns_tokens` 显式控制 |
| 语义边界 | 清楚 | 可能被打散 |
| 计算控制 | 依赖 group 设计 | 更容易控制 |
| 更适合 | 异质特征多、需要保留语义结构 | 更关注 token 数和计算预算 |



## S Tokenizer / Sequence Tokenizer

S tokenizer 的作用是把原始历史行为序列转成模型后续模块可以处理的 S token 序列。NS token 表示非序列上下文，S token 表示历史行为侧的逐位置 token。

处理方式：

1. 每个 sequence domain 单独处理，例如 `seq_a / seq_b / seq_c / seq_d`。
2. 对某个 domain，输入 `seq_data[domain]` 的形状为 `(B, S, L)`。
3. 其中 `S` 是该 domain 的 side-info 特征数，`L` 是历史长度。
4. 对每个 side-info fid，取出该 fid 在整条历史序列上的 value 序列。
5. 每个 fid 使用自己对应的 embedding table 查表。
6. 同一个历史位置上的多个 side-info embedding 在最后一维 concat。
7. concat 后通过 `Linear + LayerNorm + GELU` 投影成 `d_model` 维。
8. 如果启用 time bucket，则再加上 `time_embedding(seq_time_buckets[domain])`。
9. 最终每个 domain 输出一串 S token。

输出形状：

```text
seq_a -> seq_a_tokens: (B, L_a, d_model)
seq_b -> seq_b_tokens: (B, L_b, d_model)
seq_c -> seq_c_tokens: (B, L_c, d_model)
seq_d -> seq_d_tokens: (B, L_d, d_model)
```

```text
fid   = 特征字段身份，表示“这是哪个 side-info 特征”
value = 当前样本、当前历史位置上该字段的具体取值
```

time bucket 处理：

sequence schema 里会有一个 timestamp fid。dataset 侧会把它从普通 side-info 里拆出来：

```text
sideinfo = [fid for fid in all_fids if fid != ts_fid]
```

**timestamp 不参与普通 side-info embedding concat**，而是被转换成 `seq_time_buckets[domain]`。模型生成基础 S token 后，再加上对应 time bucket embedding。

建模含义：

一个 S token 表示某个历史位置上的完整行为状态。它不是单个 fid 的 embedding，而是该位置上多个 side-info fid embedding 的融合表达，同时带有时间间隔/时间桶信息。

**注意点：如果某些高基数 fid 被 `emb_skip_threshold` 过滤，对应 embedding 会用零向量代替。**

当前 baseline 不再给 S token 额外接 hybrid/remixer 分支。S tokenizer 只负责生成基础逐位置 token，真正的序列内部交叉统一放到后面的 Encoder / `MultiSeqHyFormerBlock` 里，避免和已有序列建模重复。

后续流向：

```text
S token
-> MultiSeqQueryGenerator：用 NS token + 每一路 S token 的 mask mean pooling 生成初始 Q token
-> MultiSeqHyFormerBlock
   -> Sequence Encoder / Sequence Evolution：每一路 S token 先做序列内部建模
   -> Cross Attention / Query Decoding：Q token attend 编码后的对应 domain S token
   -> RankMixer / Query Boosting：所有 Q token 和 NS token 融合
```



## 二、Encoder

Encoder 的作用是对每一路 S token 做序列内部建模，
负责让同一路历史里的不同位置发生交互，得到 encoded S token。

它发生在 `MultiSeqHyFormerBlock` 内部：

```text
seq_tokens_list[i]
-> seq_encoders[i]
-> encoded_seq_i
```

```text
--seq_encoder_type transformer（默认）
--seq_encoder_type swiglu
--seq_encoder_type longer
```

RoPE 位置编码在代码参数里默认是关闭的：

```text
--use_rope default=False
```

但当前 active run 显式开启：

```text
--use_rope
```

原因是当前默认 encoder 是 `TransformerEncoder`，它本身做序列内部 self-attention；开启 RoPE 后，attention 的 Q/K 会带上位置旋转信息，模型可以更明确地区分历史行为的先后顺序。当前 `run.sh` 使用 `d_model=88`、默认 `num_heads=4`，对应 `head_dim=22`，满足 RoPE 使用条件。

## SwiGLUEncoder

`SwiGLUEncoder` 是轻量 attention-free encoder。
它不做 self-attention，只对每个 S token 做逐 token 的非线性增强。

处理方式：

```text
residual = x
        x = self.layerNorm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
```

每个历史位置的 token 表达会被增强，但不同历史位置之间不会通过 attention 显式交互。

缺点：

1. 不显式建模序列内部依赖。
2. 对行为顺序、位置关系的表达能力弱于 attention encoder。

## TransformerEncoder

使用标准 Pre-LN Transformer layer，对同一路历史序列内部的 S token 做 self-attention。

处理方式：

1. 输入 S token：`x = (B, L, d_model)`。
2. `LayerNorm` 后进入 multi-head self-attention。
3. attention 使用 `key_padding_mask` 避免 padding 位置参与建模。
4. 如果启用 RoPE，则在 attention 的 Q/K 上加入位置编码。
5. self-attention 输出和原输入做 residual add。
6. 再经过一个 Pre-LN FFN。
7. FFN 输出再做 residual add。
8. 输出形状仍然是 `(B, L, d_model)`。

公式近似为：

```text
# Self-Attention (Pre-LN) with RoPE
residual = x
x = self.norm1(x)
x = self.RopeMultiSelfAttn(x)
x = residual + x

# FFN (Pre-LN)
residual = x
x = self.norm2(x)
x = self.ffn(x)
x = residual + x

# 模块结构
self.self_attn = RoPEMultiheadAttention()
self.ffn = nn.Sequential(
    nn.Linear(d_model, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, d_model),
    nn.Dropout(dropout)
)
```

建模含义：

`TransformerEncoder` 让同一路历史中的不同行为位置互相交互。某个历史位置的表示，不只看自己这个位置的 side-info，还可以通过 attention 感知同一路历史里的其他行为。

## LongerEncoder

`LongerEncoder` 是面向长序列的压缩 encoder。它的目标是把很长的 S token 序列压缩成最近的 `top_k` 个 encoded token，降低后续模块的计算成本。

处理方式分两种情况。

当输入长度 `L > top_k`：

1. 从每个样本中取最近的 `top_k` 个有效 token 作为 query。
2. 原始完整序列作为 key/value。
3. query attend 全量历史 token。
4. 输出压缩后的 `(B, top_k, d_model)`。
5. 同时更新 padding mask 为 `(B, top_k)`。

当输入长度 `L <= top_k`：

1. 不再做 top-k 压缩。
2. 对当前 token 序列做 self-attention。
3. 如果 `--seq_causal` 开启，则 self-attention 使用 causal mask。
4. 输出长度保持不变。

建模含义：

`LongerEncoder` 用最近 `top_k` 个位置承载长序列信息。第一次遇到长序列时，它让最近的 token 通过 cross-attention 汇聚全量历史；后续 block 里序列长度已经变短，就在压缩后的 token 上继续 self-attention。

优点：

1. 适合长序列，能降低后续 cross attention 和 self-attention 成本。
2. 保留最近行为作为主要承载 token，符合很多推荐场景里近因行为更重要的假设。
3. 输出 mask 会同步更新，后续模块可以继续正确忽略 padding。

缺点：

1. 会丢掉全量历史位置级别的 token 数量，只保留 `top_k` 个承载位置。
2. 如果远期历史很重要，压缩可能造成信息损失。
3. 行为比 `transformer` 更复杂，调参时需要关注 `seq_top_k` 和 `seq_causal`。
