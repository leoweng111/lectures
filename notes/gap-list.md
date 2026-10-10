# 缺口清单(gap-list)

> 学习/答疑中暴露的"前置知识缺口"都登记在这里,按 A/B/C 分类,AI 对谈时联动更新。
> 分类说明:A 必补(影响主线理解)/ B 延后(不紧急,可稍后)/ C 放弃或降级(与本目标无关)。
> 规则:单次补给 ≤ 1 小时;超时的深坑标记为 backlog,留到周末整块时间处理。

## A 必补
- 信息论最小集:熵 / 交叉熵 / KL 散度与困惑度。
  - 来源:Lecture 01 的 compression_ratio、vocab 与"adaptive chunks"背后都是"信息量/熵"的直觉;且 Assignment 1 的 loss(交叉熵)马上要用。
  - 建议资源:任意一节把"交叉熵为什么是语言模型损失"讲透的材料(可搭配 CS229 讲义 5 与花书 3.13),再用 tiktoken/自己 BPE 算几次压缩率找感觉。
- 线性注意力 / 状态空间模型(SSM)最小集:矩阵状态递推与“对偶性”。
  - 来源:Lecture 04 P4-P9(linear attention → RetNet → Mamba-2 → Gated DeltaNet)与 P12(DSA);这是理解 2024-2026 架构主线(为什么都在做“线性层 + 全注意力”混合)的前提。
  - 建议资源:Mamba-2(SSD,arXiv 2405.21060)第 2-3 节 + Gated DeltaNet(arXiv 2412.06464)第 2 节;能自己写出“并行/串行两种形式等价”的推导、并用 numpy 验证“线性注意力的 $Q(K^\top V)$ 与递推形式结果一致”即可。
- MoE 的系统与训练目标:稀疏路由不可微、负载均衡、专家并行。
  - 来源:Lecture 04 P27-P46(路由三派、top-k 与 softmax 的位置、平衡损失 / aux-loss-free 偏置、EP + all-to-all、块稀疏 GEMM / LatentMoE)。
  - 建议资源:Switch Transformer(arXiv 2101.03961)第 2 节 + DeepSeekMoE(2401.06066)/ DeepSeek-V3(2412.19437)的 MoE 章节;把“平衡损失的求导”与“aux-loss-free 偏置的更新”各手推一遍。

## B 延后
- 幂律拟合与对数空间回归(log-log)。
  - 来源:Lecture 01 scaling laws 小实验 → Assignment 3 拟合 Chinchilla 型幂律;到 P4/Assignment 3 阶段补即可。
  - 建议资源:简单回归复习(线性化 log-log 后最小二乘)+ 官方 Assignment 3 文档。
- GPU 执行模型(warp/occupancy/访存带宽 vs 算力)。
  - 来源:Lecture 01 kernels 单元只是术语表,Lecture 13/14 + Assignment 2(Triton)会系统展开。
  - 建议资源:随课补即可,可先翻 How to Scale Your Model 的硬件/带宽章节。
- hyperparameter transfer / μP(宽度缩放下 LR/初始化如何迁移)。
  - 来源:Lecture 01 提出"predictability & transfer"思想,μP 论文是深入工具;属前沿内容,放学习后期。
  - 建议资源:Yang et al. 2203.03466 + microsoft/mup 仓库。
- 浮点数的二进制表示与数值格式(规格化/下溢/分辨率)。
  - 来源:Lecture 02 fp32/fp16/bf16/nvfp4 单元——不需要背 IEEE 全表,但"指数位定范围、尾数位定分辨率、bias/规格化/下溢"要会推。
  - 建议资源:W"ikipedia fp16/bf16 词条 + NVIDIA fp8/nvfp4 primer;能徒手写一个小数的二进制分解即可。
- Einstein 求和记号与 einops 实操。
  - 来源:Lecture 02 einops 单元,Assignment 1/后续注意力实现大量使用。
  - 建议资源:einops.rocks 教程 + 用 trace 里的张量把"旧写法"改写成 einops 对拍 allclose。
- 旋转矩阵 / 复数与内积不变性(Lecture 03 RoPE 的数学底座)。
  - 来源:Lecture 03 P31-34;理解"旋转后点积只差位置角差"需要二维旋转、复数、内积几何直觉。
  - 建议资源:3B1B 线性代数系列"旋转与复数"两节 + RoPE 原文(arXiv 2104.09864)推导。
- 学习率调度与优化动力学(cosine/warmup + weight decay×LR 交互)。
  - 来源:Lecture 03 P50、讲义 2310.04415;理解 weight decay 为何与 schedule 耦合需要一点 SGD/自适应优化器的动力学直觉。
  - 建议资源:花书第 8 章(优化)相关节 + Andriushchenko 2310.04415 的图示;到 Assignment 1 调优时动手体会。
- 二分匹配 / 线性分配(匈牙利算法、Sinkhorn、线性规划)。
  - 来源:Lecture 04 P30 的 linear-assignment routing;也见于推荐系统里的“分配/匹配”问题。
  - 建议资源:运筹学教材的“指派问题”一节 + Sinkhorn 算法的入门推导;能说清“为什么它天然均衡、却不好塞进训练循环”即可。
- 在线学习 / 快速权重视角(把状态更新看成在线 SGD)。
  - 来源:Lecture 04 P9 的 Gated Delta Net(讲义点出它与 fast weight programming / test-time training 的关系)。
  - 建议资源:GDN 论文里的 online-learning 解释 + Schlag et al. 的 fast weight programmers;把它和“递归最小二乘”对上就通了。
- 稀疏注意力的“检索式选择”(indexer + top-k)。
  - 来源:Lecture 04 P12-P13(DSA / DeepSeek-V3.2),以及 NSA。
  - 建议资源:DeepSeek-V3.2(arXiv 2512.02556)第 2 节 + NSA 论文;重点吃透“indexer 仍是二次、主注意力被 k 截断”这条成本结构。
- MLA 的工程细节:潜变量压缩 + decoupled RoPE。
  - 来源:Lecture 04 P57-P58(Lecture 03 P62 也出现过);L03 笔记已覆盖一半。
  - 建议资源:DeepSeek-V2(arXiv 2405.04434)第 2.1 节;自己推一遍“为什么 $W^{UK}$ 能吸收进 $Q$、加了 RoPE 后为什么不能”。

## C 放弃 / 降级

(空)

## Backlog(待整块时间深挖)

| 日期 | 来源(讲座/问题) | 缺口描述 | 分类 | 建议资源 | 状态 |
|---|---|---|---|---|---|
|  |  |  |  |  | 未处理 |

## 已解决(留档备查)

| 日期 | 缺口描述 | 解决方式 | 关联笔记 |
|---|---|---|---|
|  |  |  |  |
