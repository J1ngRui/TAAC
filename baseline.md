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



## 三、Token Flow Overview

从 token 视角看，baseline 里有三类状态：

| token | 来源 | 主要职责 | 是否进入 RankMixer | 是否直接输出 |
| :-- | :-- | :-- | :-- | :-- |
| `NS token` | non-sequence 特征 tokenizer | 承载用户、item、dense、time、target-hist 等全局上下文 | 是 | 否 |
| `S token` | sequence tokenizer | 承载每个历史位置的行为表达 | 否 | 否 |
| `Q token` | Query Generator | 作为可学习读取槽位，从对应历史序列里读信息 | 是 | 是 |

最核心的信息流是：

```text
ns_tokens, seq_tokens
-> QueryGenerator(ns_tokens, masked_mean_pool(seq_tokens_i))
-> initial query_tokens_i

每层 MultiSeqHyFormerBlock:
  seq_tokens_i -> Encoder_i -> encoded_seq_tokens_i
  query_tokens_i cross-attn encoded_seq_tokens_i -> decoded_Q_i
  concat(all decoded_Q_i, ns_tokens) -> RankMixer + FFN
  -> updated query_tokens_i, updated ns_tokens

最终:
  concat(final query_tokens)
  -> output projection
  -> classifier
```

注意：`S token` 不走 RankMixer，也不直接进最终预测头。它只在每层先经过 sequence encoder，然后作为该层 cross attention 的 K/V，并把 encoder 后的结果传给下一层 block。



## 四、Query Generator

`MultiSeqQueryGenerator` 负责给每一路历史序列生成初始 `Q token`。这些 Q token 后续会在 `MultiSeqHyFormerBlock` 里 attend 对应 domain 的 encoded S token。

对第 `i` 路 sequence，输入是共享的 `ns_tokens` 和该路的 `seq_tokens_i`：

```text
ns_tokens:      (B, num_ns, D)
seq_tokens_i:   (B, L_i, D)
seq_mask_i:     (B, L_i)
```

处理流程：

1. 将 `ns_tokens` flatten 成 `ns_flat`，形状为 `(B, num_ns * D)`。
2. 对 `seq_tokens_i` 做 mask mean pooling，得到 `seq_pooled_i`，形状为 `(B, D)`。
3. 拼接 `global_info_i = concat(ns_flat, seq_pooled_i)`。
4. 对 `global_info_i` 做 LayerNorm。
5. 使用该路独立的 `num_queries` 个 MLP，生成 `num_queries` 个 query token。

形式上：

```text
global_info_i = concat(flatten(ns_tokens), masked_mean_pool(seq_tokens_i))

global_info_i -> MLP_i_1 -> query_i_1
global_info_i -> MLP_i_2 -> query_i_2

Q_i = stack([query_i_1, query_i_2])     # (B, num_queries, D)
```

因此 `num_queries=2` 表示每一路历史有两个可学习读取槽位。代码没有显式规定它们分别代表长期兴趣、短期兴趣或 target-aware 兴趣；它们是否分化出不同关注角度，取决于后续 cross attention 和 RankMixer 的训练信号。



## 五、MultiSeqHyFormerBlock

一个 `MultiSeqHyFormerBlock` 是一次完整的三类 token 更新过程，可以拆成三个子模块：

```text
1. Sequence Evolution
   S token -> Encoder -> encoded S token

2. Query Decoding
   Q token cross-attn encoded S token -> decoded Q token

3. Query / NS Fusion
   concat(decoded Q tokens, NS token) -> RankMixer + FFN
   -> updated Q token + updated NS token
```

输入形状：

```text
q_tokens_list:
  seq_a Q: (B, num_queries, D)
  seq_b Q: (B, num_queries, D)
  seq_c Q: (B, num_queries, D)
  seq_d Q: (B, num_queries, D)

ns_tokens:
  (B, num_ns, D)

seq_tokens_list:
  seq_a S: (B, L_a, D)
  seq_b S: (B, L_b, D)
  seq_c S: (B, L_c, D)
  seq_d S: (B, L_d, D)
```

### 5.1 Sequence Evolution

每一路 sequence domain 有自己独立的 `seq_encoder[i]`。第 `i` 路的处理是：

```text
encoded_seq_i, next_mask_i = seq_encoder_i(seq_tokens_i, seq_mask_i)
```

这里的 encoder 可以是 `transformer`、`swiglu` 或 `longer`：

- `transformer`：在同一路历史内部做 self-attention，再接 FFN。
- `swiglu`：不做位置间 attention，只做逐 token 非线性增强。
- `longer`：当序列较长时，用最近 `top_k` token 作为 query 压缩全量历史。

输出的 `encoded_seq_i` 有两个用途：

1. 作为当前层 Query Decoding 的 key/value。
2. 作为下一层 block 的 `seq_tokens_i`。

也就是说，`S token` 会在层间继续传递，但不会进入 RankMixer，也不会直接输出到 classifier。

### 5.2 Query Decoding / Cross Attention

Query Decoding 让每一路 `Q_i` 从对应 domain 的 encoded S token 中读取历史信息。

第 `i` 路的计算近似为：

```text
residual = Q_i
Q_i_norm = LayerNorm(Q_i)
S_i_norm = LayerNorm(encoded_seq_i)

attn_out = CrossAttention(
    query = Q_i_norm,
    key   = S_i_norm,
    value = S_i_norm,
    key_padding_mask = next_mask_i
)

decoded_Q_i = residual + attn_out        # (B, num_queries, D)
```

当前实现里 cross attention 是按 domain 独立做的：

```text
seq_a Q 只 attend seq_a encoded S
seq_b Q 只 attend seq_b encoded S
seq_c Q 只 attend seq_c encoded S
seq_d Q 只 attend seq_d encoded S
```

不同 sequence domain 不会在 cross attention 阶段直接互相 attend。跨 domain 的融合放在下一步 RankMixer 里完成。

### 5.3 Query / NS Fusion: RankMixer + FFN

Query Decoding 之后，模型把所有路的 `decoded_Q` 和当前层的 `ns_tokens` 拼起来：

```text
combined = concat(
    decoded_Q_seq_a,
    decoded_Q_seq_b,
    decoded_Q_seq_c,
    decoded_Q_seq_d,
    ns_tokens
)

combined: (B, num_queries * num_sequences + num_ns, D)
combined: (B, T, D)
```

当前 active 配置下：

```text
decoded Q tokens = 2 * 4 = 8
NS tokens        = 14
T                = 22
D                = 88
combined         = (B, 22, 88)
```

`RankMixerBlock` 接收的就是这个 `combined`。它支持三种模式：

```text
--rank_mixer_mode full      # token mixing + per-token FFN
--rank_mixer_mode ffn_only  # 只做 per-token FFN
--rank_mixer_mode none      # identity passthrough
```

`full` 模式下，RankMixer 先做一次无参数 token mixing。它要求 `D` 能被 token 总数 `T` 整除：

```text
d_sub = D / T

(B, token_T, D)
-> view(B, token_T, subspace_T, d_sub)
-> transpose token_T and subspace_T
-> view(B, T, D)
```

这一步不是 attention，也不是 learned projection，而是固定的 token/channel 轴重排。它让新的每个 token 都由所有原始 token 的同一个 channel 子空间拼接而成。

token mixing 后进入共享参数的 per-token FFN：

```text
Q     = combined
Q_hat = token_mixing(Q)          # full mode
# ffn_only mode: Q_hat = Q

x = LayerNorm(Q_hat)
x = Linear(D, D * hidden_mult)
x = GELU(x)
x = Dropout(x)
Q_e = Linear(D * hidden_mult, D)

boosted = LayerNorm(Q + Q_e)
```

这里 residual 从原始输入 `Q` 加到 FFN 输出 `Q_e` 上，而不是从 `Q_hat` 加回去。含义是：token mixing 负责生成增强分支里的交互信息，主路径仍保留原始 `combined` token 序列。

最后把 `boosted` split 回两类状态：

```text
next_q_list = boosted[:, :num_queries * num_sequences, :]
next_ns     = boosted[:, num_queries * num_sequences:, :]
```

所以 RankMixer 同时更新两类 token：

1. `Q token` 吸收 NS 上下文和其他 domain 的 decoded Q 信息。
2. `NS token` 也被 decoded Q 反向更新，进入下一层时携带历史交互后的全局上下文。



## 六、Block Stack / Final Output

`PCVRHyFormer` 会堆叠多个 `MultiSeqHyFormerBlock`。每一层的输出会作为下一层输入：

```text
curr_qs, curr_ns, curr_seqs, curr_masks = block(
    q_tokens_list = curr_qs,
    ns_tokens = curr_ns,
    seq_tokens_list = curr_seqs,
    seq_padding_masks = curr_masks
)
```

层间状态更新如下：

| 状态 | 下一层输入来自哪里 | 说明 |
| :-- | :-- | :-- |
| `Q tokens` | RankMixer 输出切出来的 Q 部分 | 先读历史，再和 NS/其他 Q 融合 |
| `NS tokens` | RankMixer 输出切出来的 NS 部分 | 每层都和 decoded Q 交互，但不直接输出 |
| `S tokens` | Sequence Encoder 输出 | 只经过 encoder 演化，不进入 RankMixer |
| `mask` | encoder 输出的 mask | `LongerEncoder` 可能改变长度和 mask |

所有 block 结束后，模型只取最终的 Q tokens 作为主输出：

```text
all_q = concat(curr_qs, dim=1)      # (B, num_queries * num_sequences, D)
all_q = all_q.view(B, -1)           # (B, num_queries * num_sequences * D)
output = output_proj(all_q)         # (B, D)
logits = classifier(output)         # (B, action_num)
```

当前 active 配置下：

```text
all_q:  (B, 8, 88)
flat:   (B, 704)
output: (B, 88)
logits: (B, 1)
```

最终预测不直接 concat `NS token` 或 `S token`：

- `Q token` 是最终输出 token，负责汇总每一路历史读出的信息。
- `NS token` 是中间全局上下文状态，每层通过 RankMixer 影响 Q。
- `S token` 是历史序列状态，每层通过 encoder 演化，并作为 Q cross attention 的 K/V。

压缩成一句话：

```text
NS + pooled S 生成 Q；
Q attend encoded S 得到 decoded Q；
decoded Q + NS 过 RankMixer + FFN；
S 不进 RankMixer，只把 encoder 后的 S 传给下一层；
最终只输出最后一层 Q。
```
