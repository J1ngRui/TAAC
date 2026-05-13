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



## 三、Query Generator

它的作用是给每一路历史序列生成固定数量的初始 Query token。后续 block 里，这些 Query token 会去 attend 对应 sequence domain 的 S token，从该路历史中读取信息。

**每一路序列单独生成自己的 Query token**。
每一路都共享同一份 NS 上下文，但只拼接自己这一路的 pooled S 表达。

对第 `i` 路 sequence，处理流程是：

1. 将 NS tokens 在 token 维度上 flatten。
2. 对第 `i` 路 S tokens 做 mask mean pooling，忽略 padding 位置。
3. 将 `ns_flat` 和 `seq_pooled_i` concat 成 `global_info_i`。
4. 对 `global_info_i` 做 LayerNorm。
5. 使用 `num_queries` 个独立 MLP 分别生成 `num_queries` 个 Query token。
6. 将这些 Query token stack 成该路的 `Q_i`。

需要注意的是，两个 Query token 来自同一个 `global_info_i`，但经过两个独立 MLP：

```text
global_info_i -> MLP_i_1 -> query_i_1
global_info_i -> MLP_i_2 -> query_i_2

Q_i = stack([query_i_1, query_i_2]) [num_query,d_model]

后面进入 RankMixer 前会把四路 decoded Q 拼起来：
all_Q: (B, 8, 88)
```

因此 baseline 里 `num_queries=2` 表示给每一路历史提供两个可学习的查询槽位。代码没有显式规定这两个 Query 分别代表长期兴趣、短期兴趣或 target-aware 兴趣；它们是否学出不同关注角度，主要由独立参数初始化和后续 cross attention / RankMixer 的训练信号决定。



## 四、MultiSeqHyFormerBlock

`MultiSeqHyFormerBlock` 是 Q token、S token、NS token 真正发生交互的地方。

整体流程可以拆成三步：

```text
每一路 S token
-> Sequence Evolution：先做序列内部建模，得到 encoded S token
-> Query Decoding：该路 Q token attend 该路 encoded S token

所有 decoded Q + NS token
-> RankMixer / Query Boosting：跨路 Q token 与 NS token 融合
-> split 回每一路 Q token 和共享 NS token
```

输入形状：

```text
q_tokens_list:
  seq_a Q: (B, num_queries, d_model)
  seq_b Q: (B, num_queries, d_model)
  seq_c Q: (B, num_queries, d_model)
  seq_d Q: (B, num_queries, d_model)

ns_tokens:
  (B, num_ns, d_model)

seq_tokens_list:
  seq_a S: (B, L_a, d_model)
  seq_b S: (B, L_b, d_model)
  seq_c S: (B, L_c, d_model)
  seq_d S: (B, L_d, d_model)
```

## Query Decoding / Cross Attention

Query Decoding 的作用是让每一路 Q token 从对应 domain 的历史 S token 中读取信息。

对第 `i` 路 sequence，处理流程是：

1. 先用对应的 `seq_encoder[i]` 对 `seq_tokens_list[i]` 做 Sequence Evolution。
2. 得到 `encoded_seq_i`，形状为 `(B, L_i', d_model)`。
3. 取该路的 `Q_i` 作为 attention query。
4. 取 `encoded_seq_i` 同时作为 key 和 value。
5. 使用 `seq_padding_mask_i` 屏蔽 padding 位置。
6. 如果启用 RoPE，则 RoPE 只作用在 sequence 的 K/V 侧。
7. attention 输出和原始 `Q_i` 做 residual add，得到 `decoded_Q_i`。

公式近似为：

```text
residual = Q_i
Q_i_norm = LayerNorm(Q_i)
S_i_norm = LayerNorm(encoded_seq_i)

attn_out = CrossAttention(
    query = Q_i_norm,
    key   = S_i_norm,
    value = S_i_norm,
    key_padding_mask = seq_mask_i
)

decoded_Q_i = residual + attn_out
```

输出形状：

```text
decoded_Q_i: (B, num_queries, d_model)
```

建模含义：

`Q_i` 是该路历史的可学习读取槽位。它不是简单地把整条历史池化成一个向量，而是通过 cross attention 在该 domain 的所有有效历史位置上分配权重，从而读取和当前样本上下文相关的信息。

需要注意的是，当前 cross attention 是 **每一路 domain 独立做的**：

```text
seq_a Q 只 attend seq_a encoded S
seq_b Q 只 attend seq_b encoded S
seq_c Q 只 attend seq_c encoded S
seq_d Q 只 attend seq_d encoded S
```

不同 domain 之间不会在 cross attention 阶段直接互相 attend。跨 domain 的信息融合被放到后面的 RankMixer 里。

## RankMixer / Query Boosting

RankMixer 的作用是把所有 domain 读出来的 `decoded_Q` 和全局 `NS token` 放到同一个 token 序列里做融合。

进入 RankMixer 前，模型会先拼接：

```text
combined = concat(
    decoded_Q_seq_a,
    decoded_Q_seq_b,
    decoded_Q_seq_c,
    decoded_Q_seq_d,
    ns_tokens
)

combined: (B, num_queries * num_sequences + num_ns, d_model)
combined: (B, T, D)
```

在当前 active 配置下：

```text
decoded Q tokens = 2 * 4 = 8
NS tokens        = 14
T                = 22
D                = 88
combined         = (B, 22, 88)
```

`RankMixerBlock` 接收的就是这个 `combined`：

```text
RankMixerBlock.forward(Q)
Q: (B, T, D)
```

`RankMixerBlock` 支持三种模式：

```text
--rank_mixer_mode full      # token mixing + per-token FFN
--rank_mixer_mode ffn_only  # 只做 per-token FFN
--rank_mixer_mode none      # identity passthrough
```

### Token Mixing

`full` 模式下，RankMixer 先做一次无参数的 token mixing。

它要求 `D` 能被 token 总数 `T` 整除，并定义：

```text
d_sub = D / T
```

处理方式：

1. 输入 `combined`，形状为 `(B, T, D)`，其中第一个 `T` 是 token 数。
2. 因为 `D = T * d_sub`，所以把每个 token 的通道维拆成 `T` 个子空间。这里第二个 `T` 是 channel 子空间数，数值上等于 token 数，但语义上是另一条轴：

```text
(B, token_T, D) -> (B, token_T, subspace_T, d_sub)
```

3. 交换 token 轴和子空间轴：

```text
(B, token_T, subspace_T, d_sub)
-> (B, subspace_T, token_T, d_sub)
```

4. 再 flatten 回 token 表达：

```text
(B, subspace_T, token_T, d_sub) -> (B, T, D)
```

这一步没有新增参数。本质上，原来的第 `j` 个 channel 子空间会变成新的第 `j` 个 token；新的每个 token 都由所有原始 token 的同一个子空间拼接而成。因此它不是 attention，也不是 learned projection，而是一次固定的 token/channel 轴重排。

### Per-token FFN + Residual

token mixing 之后，RankMixer 会把重排后的 `Q_hat` 送进一个共享参数的 per-token FFN。

FFN 的输入仍然是 token 序列：

```text
Q_hat: (B, T, D)
```

`nn.Linear` 作用在最后一维 `D` 上，所以它会对每个 token 的 `D` 维表示做同一套 MLP 变换：

```text
(B, T, D)
-> Linear(D, D * hidden_mult)
-> GELU
-> Dropout
-> Linear(D * hidden_mult, D)
-> (B, T, D)
```

它不会把 `T` 个 token flatten 成一个大向量，也不会为不同 token 创建不同 FFN 参数。不同 token 之间的交互主要来自前面的 token mixing；FFN 负责对混合后的每个 token 做非线性增强。

完整流程是：

```text
Q     = combined                         # (B, T, D)
Q_hat = token_mixing(Q)                  # (B, T, D), full mode
# 如果 mode=ffn_only，则 Q_hat = Q

x = LayerNorm(Q_hat)
x = Linear(D, D * hidden_mult)
x = GELU(x)
x = Dropout(x)
Q_e = Linear(D * hidden_mult, D)

boosted = Q + Q_e
boosted = LayerNorm(boosted)
```

输出形状保持不变：

```text
boosted: (B, T, D)
```

这里的 residual 是从原始输入 `Q` 加到 FFN 输出 `Q_e` 上，而不是从 `Q_hat` 加回去。这样做的含义是：token mixing 只负责产生增强分支里的交互信息，主路径仍然保留原始 `combined` token 序列。

建模含义：

RankMixer 负责让不同来源的 token 发生全局融合：

1. 不同 domain 的 decoded Q token 可以互相交换信息。
2. Q token 可以吸收 NS token 中的用户属性、item 属性、dense 特征、time context、target-hist match 等上下文。
3. NS token 也会被 Q token 更新，进入下一层 block 时携带更多历史交互后的信息。

也就是说，cross attention 阶段解决“每一路 Q 如何读自己的历史”，RankMixer 阶段解决“多路历史读出来的信息如何与全局上下文融合”。

## Block Stack / Final Output

`PCVRHyFormer` 会堆叠多个 `MultiSeqHyFormerBlock`：

```text
curr_qs, curr_ns, curr_seqs, curr_masks = block(
    q_tokens_list = curr_qs,
    ns_tokens = curr_ns,
    seq_tokens_list = curr_seqs,
    seq_padding_masks = curr_masks
)
```

每一层 block 的输出会作为下一层 block 的输入：

```text
Q tokens:  被 cross attention 和 RankMixer 更新
NS tokens: 被 RankMixer 更新
S tokens:  被 sequence encoder 更新
mask:      如果使用 LongerEncoder，可能随 top-k 压缩同步更新
```

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

这里最终不直接 concat NS token 输出，而是让 NS token 在每层 RankMixer 中参与更新 Q token。最后的预测由 Q token 汇总后的表示完成。

整体信息流可以概括为：

```text
NS tokens + S tokens
-> Query Generator 生成初始 Q
-> 每层 block:
   1. S token 做序列内部建模
   2. Q token attend 对应 domain 的 S token
   3. decoded Q 和 NS token 进入 RankMixer 全局融合
-> concat 最终 Q
-> output projection
-> classifier
```
