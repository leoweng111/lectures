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
