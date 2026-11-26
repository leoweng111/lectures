# Lecture 11 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_11.pdf`(静态讲义,共 58 页;用 PDF 阅读器/浏览器打开)
- 状态:进行中(①「卡住的地方」2 条 `#TODO` 已于 2026-11-17 全部作答并勾选;看完后在 [progress.md](progress.md) 更新)
- 说明:本讲 PDF 只有 2 条批注(都在 μP 部分:p8 的缩放表、p10 的最优 LR),已全部作答;展开推导与延伸阅读写在文末「AI 补充笔记」的 2026-11-17 一节

> **主题速览**:Lecture 11 = **scaling 的实战细节(case study)**。
> ① 两条公开 recipe:**MiniCPM**(用 **μP** 稳住 init/LR + **WSD** schedule 让 Chinchilla 分析变便宜 + joint fit)、**DeepSeek**(不用 μP,直接在小规模上拟合 batch/LR + WSD + IsoFLOP)。
> ② 其他模型的做法:Qwen(LR/batch 拟合)、Kimi K2(稀疏 scaling)、Hunyuan(96:1 数据/激活参数)、LLaMA 3(39:1)、MiniMax(架构 scaling)。
> ③ 优化器:StepFun 的纯经验网格研究(loss 对 LR/batch 是凸的、batch 主要随数据量走)、**muon**(矩阵参数的正交化更新)。
> ④ μP 深入:从 A1/A2 两条假设推出 init 与 LR 的宽度缩放;以及它在 **RMSNorm gain / 强 weight decay / sign-based 优化器**上的失效。

---

## ① 卡住的地方(按讲义页码排序;2 条已全部作答)

### A. MiniCPM 用 μP 稳住 scaling(p8)

- [x] **Q1(p8)｜MiniCPM 这张 μP 表(init_std、lr、scale_emb、scale_depth)是不是"小模型调好、再按比例放大"?**
  > 原问(还原):这里是不是说:在小模型上调好一套超参数、拿到最佳效果,想在大模型上也取得最佳效果,就按这张表**按比例**调整大模型的超参数(初始化、logit 缩放、学习率等);小模型 model_dim 是 $d_{\text{base}}$,大模型是 $d_m$。这个理解对吗?
  - **先纠正一个记号**:$d_{\text{base}}$ **不是超参**,它是**基准宽度**(MiniCPM 取 256),只当换算的"参照点"用;真正在小模型(0.009B,约 400 次贝叶斯搜索)上调出来的是 `init_std=0.1`、`lr=0.01`、`scale_emb=12`、`scale_depth=1.4` 这组 **base 超参**。
  - **一句话**:基本对 —— 但要说清"**变的不是 base 超参,而是按宽度换算出来的有效值**"。
  - **`scale_emb` / `scale_depth` 是什么**(它们**不是** μP 的宽度缩放,而是"把各部分量级调平"的常数):
    - `scale_emb = 12`:把 **embedding 的输出**乘 12 ⇒ 让"输入信号"与残差流处在同一量级(embedding 初始 std 只有 0.1,输出偏小)。
    - `scale_depth = 1.4`:把**每层 block 的输出(残差增量)**乘 `1.4/√num_layers`(`1.4/√40 ≈ 0.22`)⇒ 因为 $L$ 个增量相加时标准差 $\propto\sqrt L$,除 $\sqrt L$ 才能让残差流的尺度不随**深度**膨胀;1.4 是调出来的常数。
    - 两者都与**宽度**无关,所以不属于 μP 的宽度缩放项。
  - **表里五项逐个看**(p8 的 Table 7):
    - **Embedding Output Scaling**:输出乘 `scale_emb`(=12),常数,与宽度无关。
    - **Residual Connection Scaling**:残差增量乘 `scale_depth/√num_layers`,随**深度**缩放,与宽度无关。
    - **Initialization of Tensors**:**二维(矩阵型)**参数的 init std $=\text{init\_std}/\sqrt{d_m/d_{\text{base}}}$,即 $\propto1/\sqrt d$;其他参数(bias/norm)恒为 0.1。
    - **Learning Rate Scaling of Tensors**:**二维参数**的 LR 乘 $1/(d_m/d_{\text{base}})=d_{\text{base}}/d_m$,即 $\propto1/d$;其余部分的 LR 不变。
    - **LM Head Scaling**:输出 logits 乘 $1/(d_m/d_{\text{base}})$,即 $\propto1/d$(否则宽度越大 logits 越大,等于偷偷放大了 LR)。
  - **代入具体数字(为什么说"有效值变了、base 值没变")**:MiniCPM-2B 的 $d_m=2304$、$d_{\text{base}}=256$ ⇒ $d_m/d_{\text{base}}=9$ ⇒ ① 二维参数 init std $=0.1/\sqrt9\approx0.033$;② 二维参数的有效 LR $=0.01/9\approx0.0011$;③ logits 乘 $0.11$ —— 而你要"调"的那四个数(0.1、0.01、12、1.4)**一个都没动**。
  - **和讲义 p46–p50 的推导对得上**:A1(初始化激活 $\Theta(1)$)⇒ init $\propto1/\sqrt{n}$;A2(一步更新后激活变化 $\Theta(1)$)⇒(Adam、矩阵参数)LR $\propto1/n$ —— 表里的 $1/\sqrt{d_m/d_{\text{base}}}$ 与 $1/(d_m/d_{\text{base}})$ 就是这两条。
  - **要补三点精确性**:① 只有**二维参数**参与 init/LR 缩放,别一刀切;② `scale_emb`/`scale_depth` 与宽度无关,不属于宽度缩放;③ 换算是从 base 模型出发的,base 模型要够宽(μP 是无限宽极限下的理论)。
  - **一句话总结**:在小模型上调出一组 **base 超参**;换到大模型时**不改 base 超参**,只按表换算"矩阵参数的 init / LR"与"logits 缩放" ⇒ 大模型的**最优点仍在同一组 base 超参上**(这就是 μTransfer 的做法)。

### B. 为什么"最优 LR"在 μP 下是稳定的(p10)

- [x] **Q2(p10)｜μP 说最优 LR 应稳定,但表里 LR 明明要缩放,矛盾吗?稳定是不是"任何规模都用同一个 LR"?**
  > 原问(还原):为啥最佳学习率是 stable 的?按前几页的表格,面对更大的模型,学习率不是也要做缩放吗?如果 LR 有一个 optimal stable,是不是意味着无论想训多大的模型都可以用这个学习率、省掉调 LR,这也是 muP 的目标?
  - **一句话**:**是** —— μP 就是"**一套从小模型的超参算出大模型超参的固定规则**";更准确地说:μP 是这套**参数化/换算规则**,μTransfer 是它的**用法**(在小模型上调一次,再用规则放大)。
  - **"稳定"到底指什么(不用比喻,直接说)**:指"**你手调的那组 base 超参**"在大模型上仍然是最优的;而"**实际生效的超参**"(逐张量 LR、init std、logit 缩放)**确实变了**,按 $d_{\text{base}}/d_m$ 之类的比例缩小。
  - **一张表说清谁变谁不变**:

    | 量 | 小模型($d_{\text{base}}$) | 大模型($d_m$) |
    | --- | --- | --- |
    | base 超参(你调的 lr) | 0.01 | **仍是 0.01** |
    | 二维参数的有效 LR | 0.01 | $0.01\times d_{\text{base}}/d_m$ |
    | 二维参数 init std | 0.1 | $0.1\times\sqrt{d_{\text{base}}/d_m}$ |

    ⇒ 所以"要缩放"(有效值)与"最优稳定"(base 值)可以同时成立。
  - **"是不是任何规模都用同一个 LR"**:用的是同一个 **base LR**;实际步长仍按规则缩小。
  - **μP 只针对学习率吗**:**不是** —— 它同时规定 ① 初始化(每层 std)、② 逐层学习率、③ 输出/embedding 的缩放(部分实现还包括 attention logits 按 $1/d$ 缩放);LR 只是其中最常被提到的一项。而且它**不做搜索**:搜索仍在小模型上做一次,μP 只负责"怎么换算"。
  - **为什么会稳定(讲义 p46–p50)**:A1 让初始化激活 $\Theta(1)$、A2 让"一步更新后的激活变化" $\Theta(1)$ ⇒ init $\propto1/\sqrt n$、LR $\propto1/n$ ⇒ 每步对激活的推动量与宽度无关 ⇒ 最优 base LR 不动;SP 下推动量随宽度增长,最优 base LR 只好变小(漂移)。
  - **p10 的图**:五个尺寸(0.04B / 0.1B / 0.3B / 0.5B / 2.1B)的最小值几乎落在同一 base LR(约 $10^{-2}$),图注即 "the learning rate shift becomes minimal"。
  - **三个折扣**:① 仍要在**一个**小模型上扫一次(然后按规则放大);② 只保证**宽度**方向,深度/batch/schedule 另说;③ 失效项:强 weight decay(p56)、可学习 RMSNorm gain(p54)、sign-based 优化器(p55) ⇒ 大模型上留小范围确认。
  - **一句话收尾**:μP 的价值是"**让小模型上调好的超参直接按规则迁移**",代价是接受它的参数化(改 init、改 LR、改 logit 缩放),而且保证只在宽度维度上成立。

---

## ② 用自己的话复述本讲核心思想

(看完后随手写,一句话也行。可参考线索:① "scaling 的 recipe"不是一条数学公式,而是"**参数化 + schedule + 拟合方式**"的组合;② μP 让超参沿**宽度**可迁移,但深度/batch/wd/优化器仍要单独处理;③ WSD 让 Chinchilla 分析从"每个点都要重训"变成"一次训练取多个 checkpoint",把成本从 $n^2$ 压回 $n$。)

## ③ 触发的新问题 / 想扩展的知识

- [ ] (示例)想深入:xxx 与贝叶斯/优化理论的关系
- (AI 提示:本讲引出的延伸问题已整理进 [gap-list.md](gap-list.md);如果之后要自己做 scaling 实验,建议先在最便宜的一档上把"WSD + checkpoint 复用"的流程跑通。)

## AI 补充笔记(按日期追加)
(AI 的展开讲解、推导、延伸阅读写在这里;不要删除历史记录)

### 2026-11-17 · Lecture 11 展开笔记

#### 补充 1:μP 的推导骨架(A1 / A2 推出两个指数)

考虑 L 层线性网 $h_l=W_lh_{l-1}$,$W_l\in\mathbb{R}^{n_l\times n_{l-1}}$(宽度 $n$)。

- **A1(初始化时激活 $\Theta(1)$)**:由矩阵集中,取 $\sigma_l^2\propto\dfrac{1}{n_{l-1}}\min\left(1,\dfrac{n_l}{n_{l-1}}\right)$,则 $\lVert h_l\rVert^2=\Theta(n_l)$,即"每个坐标 $\Theta(1)$、范数 $\Theta(\sqrt n)$"。对**方阵层**($n_l=n_{l-1}=n$)退化成 $\sigma\propto1/\sqrt n$,也就是 SP 的 He 初始化。
- **A2(一步更新后激活变化 $\Theta(1)$)**:SGD 的更新是秩一外积 $\Delta W_l=-\eta_l\,\nabla_{h_l}\ell\,h_{l-1}^{\top}$,于是 $\lVert\Delta W_l\rVert_*\lVert h_{l-1}\rVert=\Theta(\eta_l n_{l-1})$;要求它 $=\Theta(n_l)$,得 $\eta_l=\Theta\left(\dfrac{n_l}{n_{l-1}}\right)$。**Adam 下**则 $\lVert\Delta W_l\rVert_*\propto\eta_l$,于是 $\eta_l=\Theta(1/n_{l-1})$。
- **p50 的对照表**:SP 是 init $1/\sqrt{n_{l-1}}$、LR $\Theta(1)$;μP 是 init $\propto\frac{1}{\sqrt{n_{l-1}}}\min(1,n_l/n_{l-1})$、LR $=n_l/n_{l-1}$(Adam:$1/n_{l-1}$)。差异就在"**LR 随宽度缩小**"(以及 fan-out 小于 fan-in 时的 init 修正)。
- **一句话**:μP 不是"新算法",而是"**选一套让每层更新幅度都与宽度无关的参数化**",于是最优点自然不动。

#### 补充 2:"stable"到底稳的是什么(三种写法容易混)

| 写法 | 会随宽度变吗 | 对应讲义 |
| --- | --- | --- |
| **base LR**(你手调的那个数,如 0.01) | **不变** | p8 表里的 `lr`、p10 图的横轴 |
| **有效 LR**(真正乘到矩阵参数上) | **变**,$\propto1/d$ | p8 表的 "Learning Rate Scaling of Tensors" |
| **最优 base LR** | **近似不变**(μP) / 漂移(SP) | p10 的 Figure 3 |

- 这三行能同时成立,是因为:你调的是 **base 值**(不变),而训练框架按宽度把它换算成 **有效值**(变)。
- **SP 下为什么会漂移**(粗略直觉):SP 用同一个 LR 作用于所有宽度的矩阵;宽度变大后,同样一步对激活的"推动量"随宽度变大($\propto\eta\sqrt n$ 或 $\eta n$),为了不炸,最优 LR 只能调小;μP 把这个推动量钉成常数,所以最优点不动。
- **实操含义**:只需要在**一个**宽度上扫出 base LR(以及 init_std),把这张表固化到训练框架里;换宽度时**只改 $d_m/d_{\text{base}}$ 这个比例**。

#### 补充 3:μP 的失效清单(讲义 p52–p57)

- **会破坏 μP 的**:① **可学习的 RMSNorm gain**(p54:gain 会随宽度累积,去掉它们性能损失很小);② **强 weight decay(0.1)** —— 讲义说这是 "maybe the only significant μP failure"(p56);③ **sign-based / 奇异优化器**(p55:Lion 之类)未必迁移。
- **基本没事的**:SwiGLU / squared-ReLU 激活、大或小的 batch、zero-init attention、常规正则化(p53)。
- **总体结论**(p57):μP 下 SP 明显更不稳,μP 的参数化"更好调";但**不保证泛化更好**(独立复现发现"最优 LR 稳定"成立、泛化未必提升,见 [ETH 复现](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/726357/3636_A_thorough_reproduction_a.pdf?isAllowed=y&sequence=7))。
- **一句话**:μP 是"**让调参变省事**"的工具,不是"让模型变强"的魔法。

#### 补充 4:MiniCPM 与 DeepSeek 两条 recipe 的对照,以及 WSD 为什么让 Chinchilla 分析变便宜

| 维度 | MiniCPM(p6–p18) | DeepSeek(p19–p24) |
| --- | --- | --- |
| 超参迁移 | 用 **μP**(init/LR/logits 按宽度缩放) | **不用 μP**,直接在小规模上拟合最优 LR / batch |
| 学习率 schedule | **WSD**(warmup + stable + decay) | WSD(快速 warmup + 两段各 10% 的 decay) |
| Chinchilla 分析 | method 1(下包络)+ method 3(joint fit) | **method 2**(IsoFLOP) |
| 结论 | 数据/模型比很高(远高于 20) | 类似,并给出 LR/batch 的拟合式 |

- **WSD 为什么关键**(p13–p16):Chinchilla 式拟合要求"**从头训到目标长度**",不能"训一半 early stop";若每个 $(N,D)$ 点都要独立训,拟合成本从 $n$ 变成 $n^2$。WSD 的 stable 阶段可以把**同一个 checkpoint** 拿去做不同长度的 decay,一次训练产出多个 $(D,\text{loss})$ 点,成本回到 $n$。
- **一句话**:WSD 把"scaling 分析"从"贵的实验设计问题"变成"便宜的工程问题"。

#### 补充 5:延伸阅读

- MiniCPM:[arXiv:2404.06395](https://arxiv.org/abs/2404.06395)(μP 表格 + WSD + Chinchilla 分析都在这篇)
- DeepSeek LLM:[arXiv:2401.02954](https://arxiv.org/abs/2401.02954)(batch/LR 拟合 + IsoFLOP)
- μP:[Tensor Programs V(arXiv:2203.03466)](https://arxiv.org/abs/2203.03466)、[`mup` 实现](https://pypi.org/project/mup/)
- WSD schedule:"Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"(Hägele et al., 2024)
- muon:[Keller Jordan 的博客](https://kellerjordan.github.io/posts/muon/)(矩阵参数用 Newton–Schulz 近似正交化:B 步 $USV^{\top}\to UV^{\top}$,p42)
- 课程内:上一讲 L10 讲 scaling law 的拟合与失效,本讲是"**在真实模型上怎么落地**";再往后会把 scaling 用到 MoE / 长上下文。
