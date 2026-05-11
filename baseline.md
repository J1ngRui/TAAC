# Baseline

Baseline 的核心目标是建立一个可解释、可复现的原始参照，用来判断后续 time context、target-hist match、loss 调整等改动到底带来了什么收益。

当前 baseline 的重点不是堆模型容量，而是先把异质特征组织成稳定的 token 表达。这里主要比较两种 NS tokenizer：`GroupNSTokenizer` 和 `RankMixerNSTokenizer`。

## NS Tokenizer

NS tokenizer 的作用是把原始离散特征 embedding 转成模型后续模块可以处理的 token 序列。

对于 single-value fid，直接查对应 embedding table。对于 multi-value fid，先对多个 value 的 embedding 做 mask mean pooling，得到一个固定维度的 fid embedding。之后不同 tokenizer 会用不同方式把这些 fid embedding 组合成 NS token。

## GroupNSTokenizer

`GroupNSTokenizer` 按语义 group 生成 token，一个 group 对应一个 NS token。

处理方式：

1. 遍历每个离散特征 fid。
2. 对每个 fid 查 embedding，multi-value fid 先做 mask mean pooling。
3. 将同一个 group 内的 fid embedding concat 到一起。
4. 通过 `Linear + LayerNorm + SiLU` 投影成一个 `d_model` 维 token。
5. 最终输出形状为 `(B, num_groups, d_model)`。

这种方式的好处是语义边界清楚。用户属性、item 属性、时间上下文、行为序列等不同特征块不会被随意打散，不同 group 也可以使用独立 projection，更适合当前这种异质特征较多的 PCVR 任务。

代价是 token 数量由 group 数决定。如果 group 划分过细，后续 RankMixer / Attention 的计算压力会变大。另外，multi-value 特征当前使用 mean pooling，可能会损失 value 内部的重要性差异。

## RankMixerNSTokenizer

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

## 当前建议

当前项目优先使用 `GroupNSTokenizer` 作为 baseline 主线。

原因是 valid/test gap 和 item cold-start 压力都说明模型不能只依赖 item_id 记忆，需要更好地利用 item 类目、多标签、用户行为序列和 target/history 匹配等泛化信号。按语义 group 保留特征边界，更适合作为后续 time context 和 target-hist match 的稳定基础。

`RankMixerNSTokenizer` 可以作为控制计算量的备选方案，尤其适合 group 数过多、后续 mixer 压力过大的场景。
