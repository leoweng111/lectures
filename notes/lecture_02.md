# Lecture 02 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_02.py`(可执行讲义)
- 本地查看:前端打开 `http://localhost:5173/?trace=var/traces/lecture_02.json`(启动方法见 [README.md](README.md) 第 2 节)
- 状态:进行中(看完后在 [progress.md](progress.md) 更新)
- 用法:看讲时把内容写进 ①~③;然后把本文件路径填入 [README.md](README.md) 第 6 节的 Prompt 发给 AI,AI 会把回答直接写在问题下方并勾选 [x]

> 主题速览:本讲 = **资源账本(resource accounting)**——浮点数值格式、FLOPs/内存怎么数、roofline 判断瓶颈、以及两大省显存技巧。

---

## A. 数值格式(fp32 / fp16 / bf16 / nvfp4)

- [x] **Q1 · fp32 的三段布局:sign / exponent / fraction 分别是什么?各占几位、为何这样设计?**
  > 原问:fp32 的具体含义,其中的 sign、exponent 和 fraction 分别的含义和位数,以及为何这样设计,配合具体例子说明。分别是符号位、指数位和尾数位?
  - **一句话直觉**:就是二进制的科学计数法——`± 1.xxx × 2^yyy`,一段管正负、一段管"量级跨度"、一段管"精度细度"。
  - **位布局(共 32 bit)**:`sign` 1 位 + `exponent` 8 位 + `fraction(尾数)` 23 位;指数用**偏置编码**(bias=127,存的是 `e+127`),以便比较大小和表示负指数。
  - **数值含义**(normal 情形):$(-1)^s \times (1.\text{fraction})_2 \times 2^{e-127}$,其中 `1.fraction` 是隐式前导 1 加上 23 位小数。
  - **为何这样设计**:sign 处理负数;exponent 用较少位数换取巨大的**动态范围**(fp32 约 $10^{\pm38}$);fraction 决定同一量级内的**分辨率**(约 7 位十进制有效数字)。三者就是科学计数法的三要素。
  - **例子**:`0.5` = $1.0_2 \times 2^{-1}$ → sign=0, exponent 存 `127-1=126`,fraction 全 0;`-12.0` = $-1.5_2\times 2^3$ → sign=1, exponent 存 `127+3=130`,fraction 表示 `0.5`(即二进制 `.1`)。参考:https://en.wikipedia.org/wiki/Single-precision_floating-point_format

- [x] **Q2 · bit / 字节 / KB / MB 等单位的换算**
  > 原问:补充下 bit、字节、kb、mb 等等常用单位的换算关系。
  - 1 字节 = 8 bit;**存储/显存容量**按 2 的幂:1 KB=$2^{10}$=1024 B,1 MB=$2^{20}$≈1.05e6 B,1 GB=$2^{30}$≈1.07e9 B,1 TB=$2^{40}$≈1.10e12 B(硬盘厂商有时按 1000 进制标注,注意区分)。
  - 结合本讲:1 个 fp32 数 = 4 B,1 个 fp16/bf16 数 = 2 B;例:80 GB HBM = $80\times 2^{30}\approx 8.6\times10^{10}$ 字节,bf16 下能放约 43B 个数值。

- [x] **Q3 · 为什么 `1e-8` 在 fp16 下溢成 0?fp16 的表示范围是多少?**
  > 原问:为什么这里对于 1e-8 会出现下溢?是因为 float16 的表示范围有限吗?具体的表示范围是多少?
  - **一句话直觉**:fp16 的指数只有 5 位,最小能表示的非零数约 $6\times10^{-8}$;$1\times10^{-8}$ 比它还小,放不下 → 归零(下溢)。
  - **范围**:fp16 = 1 符号 + 5 指数 + 10 尾数,bias=15。最大值 $(2-2^{-10})\times2^{15}\approx 65504$;最小 normal $2^{-14}\approx6.1\times10^{-5}$;靠 subnormal 可延伸到约 $2^{-24}\approx5.96\times10^{-8}$。
  - **1e-8 为什么失败**:$10^{-8} < 5.96\times10^{-8}$,连 subnormal 都表示不了,硬件 flush 成 0(`assert x == 0` 成立)。
  - **训练中的危害**:梯度或 loss 里的极小量若被冲刷成 0,参数就"不动了",导致不稳定;这也是为什么优化器状态/主权重常用 fp32(见 Q14)。
  - 对比:fp32 指数 8 位,最小 normal 约 $1.2\times10^{-38}$,$1e\text{-}8$ 完全没问题;bf16 与 fp32 同范围,也没问题(见 Q4)。

- [x] **Q4 · bf16 "resolution is worse, but matters less for deep learning" 如何理解?**
  > 原问:bf16 的 The only catch is that the resolution is worse, but this matters less for deep learning.要如何理解?
  - **事实**:bf16 = 1 符号 + **8 指数**(与 fp32 相同)+ 7 尾数,所以**动态范围与 fp32 一致**(不会像 fp16 那样小树下溢/大树上溢),代价是分辨率粗:约 2~3 位十进制有效数字。例:在 $x=1$ 附近,fp16 的最小间隔约 $2^{-10}\approx0.001$,bf16 约 $2^{-7}\approx0.008$——bf16 的"刻度"粗约 8 倍。
  - **为何对 DL 影响小**(两层):① 深度学习对参数量化的"噪声"相当鲁棒,权重/激活的微小舍入误差会被训练中的随机性淹没、且在大量样本上平均掉;② 真正需要高精度的部分(梯度累积、loss、优化器状态)单独用 fp32 兜底(混合精度,见 Q5)。
  - **但"影响小"不等于"没影响"**:数值敏感的算子(exp、softmax、归一化)仍要回 fp32,这正是 Q5 的 AMP 策略。
  - 参考:https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

- [x] **Q5 · AMP "cast into bf16 when safe (matmuls, not exp)" 的原因**
  > 原问:如果计算产生的值可能溢出,那么需要更大的精度表示,对吗?
  - **你的猜测只对了一半**:问题通常**不是"溢出范围"**(bf16 范围比 fp16 还大),而是**"分辨率/数值敏感度"**。
  - 为什么 matmul 安全:矩阵乘法对低分辨率容忍度高——结果由大量乘积累加而来,舍入误差像独立噪声一样被平均掉,且 matmul 占大头,降 bf16 直接**省一半带宽、算力翻倍**。
  - 为什么 exp 等不安全:exp/log/softmax 是"输出对输入非常敏感 + 误差会被放大"的算子(尤其 softmax 要 log-sum-exp 保持数值稳定);bf16 只有约 3 位有效数字,会在这类算子上引入可见误差。
  - 严谨表述(AMP 的做法):把算子分成"可降精度"(matmul、conv 等 memory/compute 密集、误差可平均)与"必须 fp32"(exp、log、softmax、normalization 等),前者走 bf16、后者 fall through。参考 PyTorch AMP 文档(https://pytorch.org/docs/stable/amp.html)与混合精度论文(https://arxiv.org/pdf/1710.03740.pdf)。

- [x] **Q6 · dynamic range 与 resolution 的区别**
  > 原问:dynamic range 是指能表示的数值范围,而 resolution 是指在这个范围内能表示的最小间隔吗?
  - **对,你的理解正确**。两个概念分别由指数位与尾数位决定:
  - **dynamic range**(动态范围)= 能表示的最大与最小非零数之比,≈ 由**指数位数**决定(跨度能到多大);fp16 vs bf16 对比最直观:bf16 因为指数位同 fp32(8 位),范围与 fp32 相同。
  - **resolution**(分辨率)= 同一量级内相邻可表示值的间隔(常称 ulp / machine epsilon),≈ 由**尾数位数**决定;fp16 尾数 10 位比 bf16 的 7 位精细,所以"范围小但更细腻"。
  - 一句话记忆:**exponent 管"能到多远",mantissa 管"刻度多细"**;两者是正交的两个旋钮。

- [x] **Q7 · nvfp4 的可表示值 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} 是怎么来的?**
  > 原问:这里为啥对应的 value 是这些?怎么推导的?
  - **格式**:nvfp4 = **E2M1**(1 符号 + 2 指数 + 1 尾数),指数偏置 bias=1,不保留 inf/nan。
  - **逐位映射**(尾数 M∈{0,1},指数字段整数 E∈{0..3}):E=0 时值 = $(-1)^s \times M \times 0.5$;E≥1 时值 = $(-1)^s \times (1+0.5M)\times 2^{E-1}$:

    | 指数 E | 尾数 M=0 | 尾数 M=1 |
    |:---:|:---:|:---:|
    | 00 | 0 | 0.5 |
    | 01 | 1 | 1.5 |
    | 10 | 2 | 3 |
    | 11 | 4 | 6 |

  - 于是正侧 = {0, 0.5, 1, 1.5, 2, 3, 4, 6},负侧对称,共 16 个 bit pattern → 15 个不同实数值(+0 与 −0)。
  - **为什么这么少的裸值还能用**:nvfp4 不是直接用这 4 位当最终值,而是**每 16 个元素共享一个 fp8 缩放因子 s**(外加 per-tensor 的 fp32 二级缩放),真实值 $x=x_q\times s$——s 让整块数值的"有效范围"随内容伸缩,补回了动态范围;代价是**同一 block 内不能各自任意取值**(只能共享刻度)。Nemotron 3 Super 就是用 nvfp4 训练的。参考 NVIDIA 官方博客:https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

---

## B. 计算与内存账本(FLOPs / MFU / roofline / 训练开销)

- [x] **Q8 · 矩阵乘 $B\times D$ @ $D\times K$ 的 FLOPs = $2BDK$ 的推导**
  > 原问:能否理解为:每个元素需要 2*D 次运算,生成的矩阵有 B*K 个元素,所以总 FLOPs = B*K*D*2?
  - **你的理解完全正确**。设 $Y_{i,k}=\sum_{j=1}^{D}X_{i,j}W_{j,k}$:每个输出元素做 $D$ 次乘法 + $D$ 次加法(累加)= $2D$ 次浮点运算;输出共有 $B\times K$ 个元素 → 总 $2BDK$。
  - 补充:这是"一次乘加记 2 FLOPs"的计数约定(讲义 313-314 行同款),也是前面 $6ND$ 公式里"2"的来源之一。
  - 讲义(可用 trace 复跑):`y = x @ w` 计时 + `actual_num_flops = 2*B*D*K`。

- [x] **Q9 · MFU 的分子/分母含义,以及"promised"和"actual"各自怎么来**
  > 原问:promised_flop_per_sec 是指理论上硬件能达到的最大 FLOP/s 吗?actual 是实际测量到的吗?它们分别如何计算/测量?
  - **MFU = actual FLOP/s ÷ promised FLOP/s**,衡量"硬件算力被用到了几成"(忽略通信/开销)。
  - **promised(理论峰值)**:来自硬件规格表(datasheet)上**对应数据类型**的峰值——注意它强依赖 dtype!如 H100:bf16/fp16 **dense** = 1979/2 = 989.5 TFLOPS(1979 是开稀疏的数值,讲义取 dense 一半);fp32 则只有 67.5 TFLOPS(见 `get_promised_flop_per_sec`,A100/B200 同理查表)。
  - **actual(实测)**:选一个你知道确切 FLOPs 的操作(如 matmul = 2BDK),跑 `benchmark()`(带 `torch.cuda.synchronize()` 的计时)得到秒数,FLOPs÷秒即 actual FLOP/s。
  - **为什么 MFU 到不了 1**:roofline 视角下 MFU ≈ $\min(1,\ \text{arithmetic intensity}/\text{accelerator intensity})$——当算力吃不满时(如 memory-bound 或调度/占用率不足)就低于 1;≥0.5 已经算相当好(讲义原话)。
  - 参考 roofline 详解:https://jax-ml.github.io/scaling-book/roofline/

- [x] **Q10 · 通信与计算完美重叠时 $total = \max(\text{comm},\ \text{compute})$**
  > 原问:两者会同时运行吗?总时间就是较长的那个,对吗?
  - **对**。若通信(数据搬运)与计算能在不同硬件资源上**完全并行**(如 NVLink/拷贝引擎搬数据的同时 SM 在算),总墙钟时间 = 两者中较长者;谁长谁是瓶颈。
  - 直觉:水管注水(算)与搬桶(传)同时进行,总时长取决于"算完"和"传完"哪个更晚。
  - 严谨补一句:**现实里重叠通常不完美**,总时间落在 $\max$ 与 $sum$ 之间;公式 `max` 是"完美重叠"的理想上界/下界情形,讲义用它是为了先把概念讲清楚(后面并行单元会算真实重叠效率)。

- [x] **Q11 · ReLU 的 bytes 与 FLOPs 怎么数**(arithmetic_intensity_relu)
  > 原问:bytes = 读 x 2n + 写 y 2n = 4n,对吗?flops = n 次与 0 比较,对吗?bytes/flops 的含义分别是什么?
  - **bytes**:读入 $x$ 的 $n$ 个 bf16($2n$ 字节)+ 写出 $y$ 的 $n$ 个 bf16($2n$ 字节)= $4n$ 字节,**对**(讲义代码 `(2*n)+(2*n)`)。
  - **FLOPs**:每个元素做 1 次与 0 的比较 → $n$ 次,**对**。
  - 小修正(诚实声明):ReLU 的比较严格说不是"浮点运算",但资源账本把它按"每元素 1 次单位运算"计,讲义也如此近似($n$ 与 $2n,4n$ 比只是量级估计),不影响 memory-bound 结论。
  - **两概念**:bytes = 必须搬运的数据量(决定 memory 时间),flops = 必须执行的运算量(决定 compute 时间);二者比值 = arithmetic intensity(见 Q12)。
  - 结论:ReLU 的 arithmetic intensity ≈ $n/(4n)=0.25$ FLOP/B,远小于 H100 的 accelerator intensity(≈989.5e12/3.35e12≈295 FLOP/B)→ **memory-bound**。

- [x] **Q12 · bytes 与 flops 的语义确认**
  > 原问:bytes 是在计算过程中需要传输的数据量,flops 是需要进行的浮点运算次数?
  - **对**。资源账本的基本量:FLOPs(要做多少运算)→ 算力时间 = FLOPs ÷ (FLOP/s);bytes(要搬多少数据)→ 带宽时间 = bytes ÷ (bytes/s);两者之比 = arithmetic intensity(每搬 1 字节能干多少 FLOP),拿来和硬件 accelerator intensity 比,判断 memory-bound 还是 compute-bound。roofline 图就是把"每个算子"画成"强度 vs 速度"的点,看它落在带宽斜坡还是算力平台(讲义 roofline_plots + scaling-book 参考)。

- [x] **Q13 · backward 两个 einsum 的 FLOPs 推导 + "反向 2 倍于前向"**
  > 原问:详细解释 h1_grad / w2_grad 的 einsum 与 num_backward_flops,为什么 backward 是 forward 的 2 倍?
  - 记第 2 层 $h_2 = h_1 @ w_2$($h_1$: B×D,$w_2$: D×D)。前向一个 matmul = $2BDD$ FLOPs(讲义 529 行)。
  - **反向要算两类梯度**(讲义 537-543 行),各是一次同量级 matmul:
    - $h_1.\text{grad} = h_2.\text{grad}@w_2^{\top}$:einsum `"batch out, in out -> batch in"`,输出 B×D,每个输出对 out 求和 D 项 → $2BDD$;
    - $w_2.\text{grad} = h_1^{\top}@h_2.\text{grad}$:einsum `"batch out, batch in -> in out"`,输出 D×D,每个对 batch 求和 B 项 → $2BDD$;
    - 合计 $4BDD = 2\times$ 前向 ✓(讲义 543 行的 `(2*B*D*D)+(2*B*D*D)`)。
  - **澄清你的猜测**:反向 2 倍**不是因为"对 w 和 b 两个参数算梯度"**(bias 只是小项),而是因为反向必须同时产出**"给上一层回传的激活梯度"**和**"本层权重梯度"**——每层都要两个与 forward 同量级的矩阵乘。每层都如此 → 整网 backward ≈ 2×forward。
  - 延伸:这也是 $6ND$(前向 2 + 反向 4)公式里 4 的来源(见 Q15)。

- [x] **Q14 · 训练内存四件套:$total = params + activations + gradients + optimizer\ state$**
  > 原问:这个公式如何理解?
  - 训练一轮需要同时存活的四类张量(讲义 646 行):
    - **参数 parameters**:模型权重(bf16 下 2 B/参数);
    - **激活 activations**:前向各层的中间结果(反向算梯度要用),大小 ∝ batch × 序列长 × 隐藏维 × 层数,与 batch/seq 强相关;
    - **梯度 gradients**:反向产出的 $\partial L/\partial\theta$(bf16 下 2 B/参数);
    - **优化器状态 optimizer state**:Adam 要保存一阶矩 m + 二阶矩 v,惯例用 fp32 保稳 → 4+4 = 8 B/参数(AdaGrad 只存二阶矩 → 4 B/参数)。
  - 拿讲义 napkin 算一遍(79-83 行):bf16 训练 + AdamW,每参数 $2+2+8=12$ B → 8×80 GB ÷ 12 B/参数 ≈ 530 亿参数上限(还没算激活,故只是上界)。
  - 要点:参数量大了以后,**优化器状态常是内存大头**;激活则随 batch 线性涨——这直接引出 Q16/Q17 两个省显存技巧。

- [x] **Q15 · "前向 2、反向 4、合计 6 次运算/参数/数据点" 的推导**
  > 原问:backward 的 flops 是 forward 的 2 倍吗,是不是因为 backward 需要对 w 和 b 两个参数都计算梯度?
  - **是 2 倍,但不是因为 w/b 两个参数**(与 Q13 同因):每层反向要算"回传梯度 + 权重梯度"两个同量级 matmul → 每层 backward = 2×forward;全网络加总 → backward ≈ 2×forward。
  - 逐数据点逐参数看:forward 每个参数被乘一次并累加一次 ≈ 2 FLOPs;backward 每个参数平均 ≈ 4 FLOPs(2 用于算权重梯度、2 用于把误差往前传);合计 **6 FLOPs/参数/数据点**,即 $6\times B\times N$(训练一个 epoch/token 数版本就是 $6ND$)。
  - 这是对 MLP 的精确结果,对 Transformer(短上下文)是很好的近似;attention、logits 层、embedding 等二阶项在 Assignment 1 里会精确逐层数。
  - 参考:https://www.adamcasson.com/posts/transformer-flops(Transformer 版逐层 FLOPs)

- [x] **Q16 · gradient accumulation:是什么、为什么**
  > 原问:batch 太大显存不够,就拆 micro-batch 算梯度累加、再更新参数,对吗?目的是啥?
  - **机制对**(讲义 718-730 行):把大 batch 拆成 $K$ 个 micro-batch,逐个前向+反向得到梯度,`grad` **累加不清零**;每 $K$ 个 micro-batch 才做一次 `optimizer.step()` 并清零。
  - **目的**:① **用更小显存模拟更大的有效 batch**(激活内存 ∝ batch,拆小后激活峰值大幅下降,而梯度本身只占 2 B/参数,累加不贵);② 大 batch 训练更稳定(梯度估计的方差更小)、利于学习率调度与分布式扩展。
  - **严谨提醒**:它 ≠ 真正的大 batch 完全等价——优化器状态每 $K$ 步才更新一次,且 BatchNorm 等依赖 batch 统计量的模块行为略有不同;对 Adam/纯 SGD 而言,累加的平均梯度与大 batch 在数学上一致。分布式训练里它还能**减少通信频率**(每 K 步同步一次)。

- [x] **Q17 · activation checkpointing(梯度检查点/重物化):是什么、怎么实现、作用**
  > 原问(原在 ② 区):详细解释下这块。这个是什么技巧?是如何实现的?作用是什么?
  - **是什么**:训练中为省显存,前向**只保留部分层(checkpoint)的激活**,其余丢弃;反向算到缺激活的层时,**从最近 checkpoint 重放一次前向**把激活重算出来,再继续算梯度。同义词:gradient checkpointing / rematerialization(讲义 751-755 行)。
  - **如何实现**:PyTorch 一行 `torch.utils.checkpoint.checkpoint(layer, x)`(讲义 786 行),等价于"该层前向时不存中间量、反向前按需重算"。
  - **作用(用算力换显存)**:显存占用从 $O(L)$ 降到 $O(1)\sim O(\sqrt{L})$(取决于 checkpoint 间隔),代价是多算约一次前向的时间。
  - **间隔的选择是显存/时间的谱系**(讲义 770-773 行):
    - 每层都存:显存 $O(L)$,零重算;
    - 一层都不存:显存 $O(1)$,但每层反向都要从头重算 → 计算 $O(L^2)$;
    - **每隔 $\sqrt{L}$ 层存一个**:显存 $O(\sqrt L)$,重算 $O(L)$(工程上常用的甜点位)。
  - 注:推理不需要它(无反向、只需当前层激活);大模型训练(如 Megatron 系)标配此技巧。

---

## ② 用自己的话复述本讲核心思想

(看完后随手写。可以试着用 3~5 句话串起来:本讲一切围绕"给定资源最大化效率":① 存储上四类张量(params/activation/grad/optimizer state)各占多少;② 算力上每数据点每参数 ≈ 前向 2、反向 4、合计 6 FLOPs;③ 用 roofline(arithmetic intensity vs accelerator intensity)判断一个算子是 memory-bound 还是 compute-bound,为什么 matmul compute-bound、逐元素算子 memory-bound、推理(mat-vec)memory-bound;④ 两大省显存技巧 = gradient accumulation 与 activation checkpointing。)

---

## ③ 触发的新问题 · 问答

- [x] **Q18 · einops 的实操入门**
  > 原问:einops 的实操。
  - **一句话**:einops 给张量操作起"带名字的维度",用类似爱因斯坦求和记号的字符串描述形状变化,防手滑、可读性高(讲义 202-207 行:受 Einstein 1916 记法启发)。
  - **三个核心函数**(讲义 222-276 行):
    - `rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)`:拆分/合并/转置维度(把扁平的 `heads*hidden` 拆开);
    - `einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")`:带记号的矩阵乘,**没出现在输出里的维度自动求和**;
    - `reduce(x, "... hidden -> ...", "sum")`:对指定维度做聚合(等价 `x.sum(-1)`)。
  - **Transformer 里的典型用法**(建议在 trace 里逐步跑 lecture_02 对应段落):
    - 多头切分:`rearrange(x, "b s (h d) -> b h s d", h=n_heads)`;
    - 注意力打分:`einsum(q, k, "b h s d, b h t d -> b h s t")`;
    - 注意力输出:`einsum(attn, v, "b h s t, b h t d -> b h s d")`;
    - 头合并:`rearrange(x, "b h s d -> b s (h d)")`。
  - **实操建议**:完整教程在 https://einops.rocks/1-einops-basics/ ;练习方法——拿 lecture_02 trace 里的 `x = torch.ones(2,3,4)` 等张量,把每段"旧写法"改写成 einops,再对拍 `torch.allclose`;Assignment 1 里会大量用到(讲义也推荐用它写注意力)。

---

## AI 补充笔记(按日期追加)

### 2026-09-08 · Lecture 02 首次答疑

本次已将 ①/②/③ 的问题全部就地解答并重排为结构化 Q&A。速记要点:

1. **数值格式速查**:fp32=1+8+23(范围 ±1e38,7 位有效数字);fp16=1+5+10(最大 65504,最小约 6e-5/subnormal 6e-8,`1e-8` 会下溢);bf16=1+8+7(**范围同 fp32、分辨率粗**,DL 通常可容忍);nvfp4=E2M1,4 bit 仅 15 个裸值但靠 per-block fp8 scale 补范围;AMP 原则 = matmul 类降 bf16、exp/softmax 类保 fp32。
2. **账本核心公式**:matmul $=2BDK$;每参数每数据点 = 前向 2 / 反向 4 / 合计 6(反向贵在"回传梯度 + 权重梯度"两个 matmul);训练显存四件套 ≈ 2+2+8(+activation) B/参数(AdamW/bf16)。
3. **roofline 判断**:arithmetic intensity(FLOP/byte)vs accelerator intensity(≈989.5e12/3.35e12≈295 FLOP/B on H100):matmul compute-bound、逐元素与 mat-vec memory-bound → 解释了"训练快、单 token 推理慢"。
4. **两个省显存技巧**:gradient accumulation(小显存模拟大 batch,每 K 步一更新);activation checkpointing(显存 $O(\sqrt L)$、多算一次前向)。
5. **呼应上一讲**:讲义开头宣布 Marin 的 1e23 FLOPs 预注册预测已实跑吻合(https://x.com/WilliamBarrHeld/status/2039373983632814318)——正是 Lecture 01 讲的"predictability + 预注册"闭环的现实结果。

延伸阅读:
- NVIDIA nvfp4 官方博客:https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- einops 教程:https://einops.rocks/1-einops-basics/
- fp16 / bf16 / fp32(Wikipedia):single-precision、half-precision、bfloat16 词条
- 混合精度论文(Micikevicius et al.):https://arxiv.org/pdf/1710.03740.pdf ;PyTorch AMP:https://pytorch.org/docs/stable/amp.html
- Transformer 训练显存拆解:https://erees.dev/transformer-memory/ ;Transformer 逐层 FLOPs:https://www.adamcasson.com/posts/transformer-flops
- roofline 详解(scaling-book):https://jax-ml.github.io/scaling-book/roofline/

### 2026-09-08 · 追问:activation checkpointing 实例详解(核心认知:省的是峰值内存,不是总计算量)

**三个先决概念**

1. **activation 指什么?** 反向传播必需的那些层间中间张量——通常是每层的**输入与输出**(如线性层前的 $h_{i-1}$、ReLU 前的 $g_i$、ReLU 后的 $h_i$)。算 ReLU 的梯度需要激活前的 $g_i$,算权重梯度需要输入 $h_{i-1}$,所以往往输入/输出都要存;不只是一份"最终向量"。
2. **反向确实需要全部 activation——没错。** 但"需要用到"≠"所有 activation 同时驻留显存"。无 checkpointing 时,前向算完全部并一直留到反向用到那一刻 → 峰值 = 全部层激活之和;checkpointing 让大部分激活**在反向用前一瞬间才现算、用完即丢** → 峰值降下来。
3. **重算发生在训练的反向阶段**(PyTorch 在 backward 里重放前向),所以计算量确实增加——这就是讲义说的 **tradeoff memory for compute(用算力换显存)**。

**具体例子:L = 9 层,每层 activation 占 1 单位内存 A,每层前向 1 单位时间**

| 方案 | 峰值内存 | 前向总工作量 | 反向行为 |
|---|---|---|---|
| ① 全存(无 checkpoint) | 9A | 9 | 按 h₉→g₉→…直接取用,零重算 |
| ② 每 √L=3 层存 1 个 checkpoint(存 h₃、h₆ + 输入 x) | ≈ 2A(两个 checkpoint)+ ≤3A(当前段临时)≈ **5A** | 9 + 9(反向重放)≈ **18** | 分 3 段,每段从最近 checkpoint 重放前向→现算梯度→释放 |

逐步走一遍方案②:前向只留 h₃、h₆(x 恒在),其余算完即丢;反向时:
- 算 h₉~h₇ 的梯度 → 从 h₆ 重放 g₇h₇g₈h₈g₉h₉(临时 ≤3A),算完释放;
- 算 h₆~h₄ → 从 h₃ 重放,用完即弃;
- 算 h₃~h₁ → 从 x 重放。

推广到 L 层、每 $\sqrt L$ 层存一个:峰值 $O(\sqrt L)$,总重算 $O(L)$(每层至多重算一次),整体训练时间约多一次前向(~30-40%)。

**一句话总结**:反向需要的所有 activation **一个都不会少算**;checkpointing 改变的是它们**何时被物化、在显存里活多久**——从"前向后全部并存、等反向来用"变为"反向前一刻现算、用完即弃",把同时占用从 $O(L)$ 压到 $O(\sqrt L)$。

讲义注释对比(757-758 行):
```
Store all activations:    x g1 h1 g2 h2 g3 h3 g4 h4   (9 份并存)
Activation checkpointing: x    h1    h2    h3    h4   (只留层输出,反向现算 g_i)
```
PyTorch 一行实现:`torch.utils.checkpoint.checkpoint(layer, x)`(每调用一次 = 一个 checkpoint 边界,块内激活反向前重算)。
