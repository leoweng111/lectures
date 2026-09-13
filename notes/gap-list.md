# 缺口清单(gap-list)

> 学习/答疑中暴露的"前置知识缺口"都登记在这里,按 A/B/C 分类,AI 对谈时联动更新。
> 分类说明:A 必补(影响主线理解)/ B 延后(不紧急,可稍后)/ C 放弃或降级(与本目标无关)。
> 规则:单次补给 ≤ 1 小时;超时的深坑标记为 backlog,留到周末整块时间处理。

## A 必补
- 信息论最小集:熵 / 交叉熵 / KL 散度与困惑度。
  - 来源:Lecture 01 的 compression_ratio、vocab 与"adaptive chunks"背后都是"信息量/熵"的直觉;且 Assignment 1 的 loss(交叉熵)马上要用。
  - 建议资源:任意一节把"交叉熵为什么是语言模型损失"讲透的材料(可搭配 CS229 讲义 5 与花书 3.13),再用 tiktoken/自己 BPE 算几次压缩率找感觉。

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
