# Lecture 04 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_04.pdf`(静态讲义,共 60 页;用 PDF 阅读器/浏览器打开)
- 状态:进行中(①「卡住的地方」32 条 `#TODO` 已于 2026-10-10 全部作答并勾选;看完后在 [progress.md](progress.md) 更新)
- 说明:本讲 PDF 共 78 条批注,其中 `#TODO` 32 条。每条问题的答案就写在下方;较长推导、公式与延伸阅读统一放在文末「AI 补充笔记」的 2026-10-10 一节

> **主题速览**:Lecture 04 分两大块。
> ① **注意力的替代方案**:线性注意力(矩阵结合律重排 $QK^\top V=Q(K^\top V)$)→ RNN 递推形式与"对偶性"→ Mamba-2(数据相关标量衰减)→ Gated DeltaNet(门控 + delta 规则)→ 稀疏注意力 DSA;"线性层 + 全注意力"的混合(H1/H2、7:1、3:1)是目前最稳的落点。
> ② **混合专家(MoE)**:用稀疏激活把"参数量"和"算力"解耦。要点依次是路由三派(token/expert/global)与 top-k、softmax 放在 top-k 前还是后、共享专家与细粒度专家、负载均衡损失与 aux-loss-free 偏置、专家并行 / 分块稀疏 GEMM / LatentMoE、训练稳定性与微调、upcycling,最后是 DeepSeek MoE v1→v3 的演化(附 MLA / MTP)。

---

## ① 卡住的地方(按讲义页码排序;32 条已全部作答)

### A. 注意力替代方案与稀疏注意力(p2–p13)

- [x] **Q1(p2)｜"attention 与 feed-forward 各占多少比例"指的是什么?为何随序列长度变化?**
  > 原问(还原):这里说的 attention 和 feed-forward 所占比例指的是什么?推理的耗时吗?如何理解这两者所占比例随着 sequence length 的变化?
  - **一句话**:指的是**计算量(FLOPs)的占比**,不是墙钟耗时;序列越长,注意力占的比例越高,直到几乎吃掉全部算力。
  - **严谨表述**:每 token 每层的 FLOPs(一次乘加算 2):注意力 $\approx 4nd+8d^2$(打分与加权各约 $2nd$,QKVO 四个投影约 $8d^2$),FFN $\approx 16d^2$(取 $d_{ff}=4d$)。两者相等当 $4nd=8d^2$,即 **$n\approx 2d$**。
  - **读数**:短序列时 FFN 是主角(约占 $2/3$);$n\gt 2d$ 后注意力反超;例如 $d=12288$(GPT-3 规模)的交点约 **2.4 万 token**。这正是长上下文一切优化(滑窗、稀疏、线性注意力)的动机。
  - 为什么不是耗时:耗时还取决于该算子是 compute-bound 还是 memory-bound(见 L02/L03)。训练/prefill 时注意力打分基本 compute-bound,decode 时则是 memory-bound。
  - 出处:讲义 P2 引用的图 [meibel.ai](https://www.meibel.ai/post/understanding-the-impact-of-increasing-llm-context-windows)。

- [x] **Q2(p3)｜FlashAttention"不改变复杂度,只改变常数因子,而常数很重要"怎么理解?**
  > 原问(还原):flash attention 不改变复杂度,只改变了所谓的 constant factors , which is really matters,这要如何理解?
  - **一句话**:它的 FLOPs 和朴素注意力**一模一样**(都是 $O(n^2d)$),但它把两个 $n\times n$ 中间矩阵的显存读写彻底省掉了;而长序列下瓶颈是**搬字节**而不是算数,所以常数一降,时间就明显降。
  - **严谨表述**:朴素实现要把 $S=QK^\top$、$P=\mathrm{softmax}(S)$ 写回 HBM 再读出来(访存 $O(n^2)$);FlashAttention 分块 + online-softmax,让它们只在片上 SRAM 里存在,总 HBM 访存降到约 $O(n^2d^2/M)$($M$ 为 SRAM 大小),**FLOPs 不变**。
  - **为什么"常数"这么值钱**:加速器算力(FLOP/s)比带宽(B/s)高约三个数量级(见 L02 roofline),所以"少搬字节"能直接变成时间收益;相反"少算"在 memory-bound 场景几乎没用。
  - 出处:Dao et al. 2022,[arXiv:2205.14135](https://arxiv.org/abs/2205.14135);更通俗的 IO 复杂度分析见原论文第 3 节。

- [x] **Q3(p4)｜"不做 softmax 就叫 linear attention"——这是它的定义吗?**
  > 原问(还原):如果不做 softmax,那么就被称之为 linear attn?这个是 linear attn 的定义吗?
  - **一句话**:方向对,但更准确的定义是"把 softmax 换成**可分解的核**":$\mathrm{Attn}=\phi(Q)\bigl(\phi(K)^\top V\bigr)$,"不做 softmax($\rho$ 取恒等)"只是 $\phi$ 取恒等这个最简特例。
  - **为什么这一点是关键**:softmax 要求"每一行看过所有 key 才能归一化",因此 $QK^\top$ 无法分解;换成可分解的核之后,矩阵乘法可以重排:
    $QK^\top V=Q(K^\top V)$,复杂度从 $n^2d_k+n^2d_v$ 降到 $2nd_kd_v$——**对 $n$ 线性**。
  - **代价与补救**:表达力下降(做不出"只看最相关几个位置"的尖锐选择),所以现代做法要么加门控(Mamba-2 / GDN),要么与全注意力混合(7:1、3:1、H1/H2)。
  - 出处:Shen et al. 2018,[arXiv:1812.01243](https://arxiv.org/abs/1812.01243);Katharopoulos et al. 2020(Linear Transformers),[arXiv:2006.16236](https://arxiv.org/abs/2006.16236)。

- [x] **Q4(p5)｜递推形式怎么来的?$y_t$ 是什么、什么形状?RNN 的好处与"duality"指什么?**
  > 原问(还原):这里具体的推导,为啥这个形式等价于上面的式子。这里的 $y_t$ 就是第 $t$ 位置的输出的 token 吗?$y_t$ 的形状是什么?以及像 RNN 的好处是什么?duality 指的是什么?
  - **推导(等价性)**:把"状态"定义为 $S_t=\sum_{i\le t}k_iv_i^\top$(形状 $d_k\times d_v$),则
    $S_t=S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t=\sum_{i\le t}(q_t^\top k_i)\,v_i$。
    最后这个求和正是**线性注意力的输出**(对历史做加权求和),所以递推形式与原式等价。
  - **$y_t$ 是什么**:不是 token id,而是**该位置在这一层的注意力输出向量**;形状 $d_v$(再经输出投影 $W_O$ 回到模型宽度 $C$)。token 要等到 LM head 把它投影到词表并采样之后才有。
  - **RNN 的好处**:推理时状态大小**恒定**($d_kd_v$,与序列长度无关),每步只需 $O(d_kd_v)$ 且**不需要随 $T$ 增长的 KV cache** ⇒ 长上下文 decode 不再是 memory-bound。
  - **duality(对偶)**:同一个层有**并行(矩阵)形式**和**串行(递推)形式**两种等价写法——训练用前者(二次但可并行,吃满 GPU),推理用后者(线性、每步常数)。讲义 P5 原话:"train efficiently (parallel, quadratic form) and inference efficiently (serial, linear form)"。
  - 顺带:把 $S_{t-1}$ 乘一个衰减 $\gamma$,就得到 RetNet 那一类"带遗忘的线性注意力"。

- [x] **Q5(p5)｜线性注意力与 GQA 的定位有何不同?它只是为了缓解 compute-bound 吗?**
  > 原问(还原):和 GQA 相比,这里的 linear attn 的目的是啥?GQA 我理解是多个 query 共享一个 K,V……那么这里的 linear attn 的目的,能否理解为当系统已经是 compute-bound 的时候,通过降低前向所需要的 flops 从而缓解 compute-bound?
  - **一句话**:不完全是——线性注意力最大的价值恰恰在**decode 的 memory-bound** 场景:它把"随序列长度增长的代价"整项去掉了,而 GQA 只是把它缩小一个常数倍。
  - **严谨对比**:
    - GQA/MQA:KV 从 $H$ 份降到 $g$ 份(常数倍缩小),**每步仍要读 $O(T)$ 个历史 KV** ⇒ 仍随 $T$ 线性增长,只是斜率小了;
    - 线性注意力:状态 $S$ 大小固定 ⇒ 每步读取 $O(d_kd_v)$、**与 $T$ 无关**;注意力 FLOPs 也从 $O(n^2)$ 降到 $O(n)$。
  - 所以它同时改善了 prefill/训练(compute-bound:省 $O(n^2)$ FLOPs)和 decode(memory-bound:去掉随 $T$ 的读取),而后者在长上下文推理里才是主痛点。
  - **代价**:固定大小的状态存不下无限多 key-value,序列超过状态维度就会出现"记忆碰撞"、**精确检索能力下降** ⇒ 所以要用"线性层 + 少量全注意力"的混合来补(讲义 P6/P8/P10 的三个混合例子)。

- [x] **Q6(p7)｜Mamba-2 的详细解释**
  > 原问(还原):Mamba-2 的详细解释。
  - **它在线性注意力上加了什么**:
    $S_t=\gamma_tS_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t+v_t^\top D$,其中 $\gamma_t=f(x_t)\in(0,1]$ 是**数据相关的标量衰减**,$D$ 是输出侧的直通(跳跃)项。
  - **三项改动各自的含义**:① $\gamma_t$ 给状态一个"整体遗忘"旋钮(可学地丢弃旧记忆);② 门控让线性注意力表达力更强(讲义原话:gating is good);③ $v_t^\top D$ 是一条不经状态的短路,类似残差,缓解"所有信息都必须挤过状态"。
  - **仍然保持对偶性**:可以把 $\gamma$ 展开成一个**衰减感知的因果掩码** $\Gamma_{ij}=\gamma_i/\gamma_j$,于是并行形式写成 $O=\bigl((QK^\top)\odot\Gamma\bigr)V$——这就是论文里的 SSD(State Space Duality),训练时可分块(chunkwise)用 matmul 并行。
  - **一句话定位**:Mamba-2 = 线性注意力 + 标量门控,是"SSM 与注意力统一视角"的最好例子。
  - 出处:Dao & Gu 2024,Mamba-2 / SSD,[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)。

- [x] **Q7(p9)｜gated delta net 里的"erase""no input operation"是什么意思?相比 Mamba-2 改进了什么?**
  > 原问(还原):这里说的 erase,还有 no input operation 等我不理解,直接详细解释下 gated delta net 的思想,和相比 mamba-2 的改进。
  - **递推式**:$S_t=\gamma_t\bigl(I-\beta_tk_tk_t^\top\bigr)S_{t-1}+\beta_tk_tv_t^\top$,$y_t=q_t^\top S_t$;$k_t$ 通常归一化($\|k_t\|=1$)。
  - **erase 是什么**:$k_tk_t^\top$ 是**到 $k_t$ 方向的投影**,$(I-\beta_tk_tk_t^\top)$ 做的事情是:先把状态里"与当前 key 相关的那部分旧关联"投影出来,再按强度 $\beta_t$ 扣掉它。也就是 delta 规则——**先擦掉旧的 $k_t\to v_t^{\text{old}}$ 关联,再写新的**,避免盲目堆叠(这正是 DeltaNet 在关联回忆任务上强的原因)。
  - **no input operation 是什么**:当门控学到 $\beta_t=0$ 时,更新退化成 $S_t=\gamma_tS_{t-1}$——**这一步不写入任何新信息**,只做遗忘。模型因此可以学会"这个位置不值得记"。
  - **相比 Mamba-2 的改进**:Mamba-2 的遗忘是**整体均匀**的(所有关联一起被 $\gamma$ 衰减),没法单独修改某个 key 的关联;GDN 同时拥有 ① 整体遗忘($\gamma$)、② **精准覆盖**(delta 项)、③ 完全跳过写入($\beta=0$),所以长上下文检索/关联回忆明显更好。
  - **一个漂亮的等价视角**:这次更新等价于对 $\tfrac12\|S k_t-v_t\|^2$ 做**一步在线 SGD**($\beta_t$ 是学习率),而 $\gamma_t$ 相当于权重衰减 ⇒ GDN 与 "fast weights / test-time training" 是同一族思想。
  - 出处:Yang et al. 2024,Gated Delta Networks,[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)。

- [x] **Q8(p12)｜DSA(DeepSeek Sparse Attention)的详细解释与数学原理**
  > 原问(还原):DSA 的详细解释和数学原理。
  - **两个组件**:
    ① **lightning indexer(闪电索引器)**:给每个历史位置算一个廉价的相关性分数
    $I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\cdot\mathrm{ReLU}\bigl(q^I_{t,j}\cdot k^I_s\bigr)$,
    其中 $H^I$ 是 indexer 头数(公开配置 64、每头 $d^I=128$),$q^I_{t,j},w^I_{t,j}$ 由当前 token 导出,$k^I_s$ 由历史 token 导出,全部用 **FP8** 计算(ReLU 是为了吞吐);
    ② **top-k 选择器**:只保留分数最高的 $k$ 个 KV 条目(这里是 MLA 的潜变量 $c_s$),主注意力只在这 $k$ 个上做。
  - **复杂度**:稠密注意力的 $O(L^2)\to O(Lk)$,$k\ll L$;注意力矩阵从稠密变成结构化稀疏。
  - **为什么"后装"很容易**:它是 **post-hoc** 的——从已训练好的稠密短上下文模型继续训练就能装上(DeepSeek-V3.2 就是从 V3.1-Terminus 持续训练而来,架构上唯一的改动就是 DSA)。
  - **工程细节**:DSA 实例化在 **MLA 的 MQA 模式**下(一个潜变量被所有 query 头共享),这样 kernel 层面每个 KV 条目可以被多个 query 复用。
  - 出处:DeepSeek-V3.2 技术报告,[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)(公式即其式 (1)(2))。

- [x] **Q9(p12)｜$I_{t,s}$ 的计算也是二次复杂度吧?那到底是怎么省时间的?"常数的重要性"怎么理解?**
  > 原问(还原):这里的 $I_{t,s}$ 的生成,也是二次方复杂度的对吗?那么是如何节省时间的?首先是可以做很多 system z-trick 优化;其次已经选了 top-k,所以是 $k^2$ 复杂度而不是 $n^2$,这也是常数的重要性(这里不理解)。
  - **先纠正一处**:indexer 本身**确实是二次的**——每个 query 都要和所有历史位置算分数,总开销 $O(L^2\cdot H^Id^I)$。所以"省时间"不来自 indexer,而来自下面三处。
  - **省时间的三处来源**:
    ① **主注意力被 top-k 截断**:每个 query 只处理 $k$ 个 entry ⇒ 总代价 $O(Lk)$(不是 $O(L^2)$)。你说的"$k^2$ 而不是 $n^2$"更准确的说法是"**每个 query 只做 $k$ 次,总共 $Lk$ 次**";
    ② **indexer 的单位成本极小**:头数少、维度小、FP8、多个 indexer 头共享同一份 key、而且**不需要 V 的加权求和**(只是一次打分);
    ③ **系统层面**:融合 kernel、MLA 的 MQA 共享让 KV 复用率更高。
  - **"常数的重要性"**:在 $L=128\text{k}$ 这种量级,主注意力的二次项已经被 $k$ 截成线性,而剩下的二次项只有 indexer —— 它的系数小到只占总开销百分之几。于是端到端开销**近似随 $L$ 线性增长**,即使严格写出来仍含一个 $L^2$ 项。这就是"复杂度不变、常数决定成败"的典型例子(和 Q2 的 FlashAttention 是同一类论证)。

### B. MoE 的收益与代价(p15–p26)

- [x] **Q10(p15)｜"参数量 4 倍,但前向/反向 FLOPs 与 dense 相同"为什么?**
  > 原问(还原):4 倍参数量,但是前向和反向传播的 flops 和 dense model 相同。#TODO 为啥相同?
  - **一句话**:因为 MoE 只让每个 token 走 **top-k 个专家**——参数量看"全部专家之和",FLOPs 看"每个 token 实际激活的那部分"。
  - **严谨表述**:总参数 $\approx N\times$ 单专家大小(随专家数线性增长);而单 token 的 FLOPs 只与**激活参数**有关:若 $k$ 个被激活专家的规模之和 = 一个 dense FFN,则前向 FLOPs 与 dense 相同。反向同理——**没被激活的专家这一步梯度为 0**(稀疏梯度),所以反向 FLOPs 也不增加。
  - **代价在哪**:显存(要放下所有专家)与通信(EP 的 all-to-all);也就是"用显存/通信换质量"——同 FLOPs 下参数更多、loss 更低(讲义 P16 的图)。
  - 参考:Fedus et al. 2022(Switch Transformer),[arXiv:2101.03961](https://arxiv.org/abs/2101.03961)。

- [x] **Q11(p19)｜expert parallel 到底是什么被并行了?为什么需要?训练还是推理?**
  > 原问(还原):详细解释下 expert parallel?到底是将什么东西并行了?只并行专家,还是将整个模型都并行了?为什么需要并行?训练还是推理的时候并行?具体原理是什么?
  - **并行的对象是"专家"(参数维),不是序列也不是层**:每张卡只放一部分专家,一个 token 要去的专家若不在本卡,就通过 **all-to-all** 把它的激活发过去,算完再 all-to-all 发回来。
  - **为什么需要**:MoE 的总参数量极大(例如 DeepSeek-V3 671B),单卡放不下;而每个 token 只用少数专家,于是天然可以"把专家摊开",做到"总参数很大、每卡显存不大"。
  - **与其它并行的关系**:DP(切数据,每卡一份模型)、TP(切单层矩阵,通信最频繁、通常限机内)、PP(按层切,需要 micro-batch 填流水)、**EP(按专家切,MoE 特有)**。EP 常常与 DP 组合:不同数据副本走不同专家分片。
  - **训练与推理都用**:训练时 EP 是主流(MoE 层用 EP、注意力层用 TP/DP);推理时常把专家**全部驻留**在显存里、按 token 动态路由(或按热点专家做缓存/分卡),EP 主要解决"放不下"。
  - **系统代价**:负载不均会直接拖慢 all-to-all(最慢设备决定时间),通信量随 token 数与 top-k 增长 ⇒ 这正是 P41-p46 那些平衡与通信优化的动机。

- [x] **Q12(p24)｜"为什么 MoE 还没更流行"那几句话到底在说什么?**
  > 原问(还原):这段话没看懂。[Why haven't MoEs been more popular? Infrastructure is complex / advantages on multi node; Training objectives are somewhat heuristic (and sometimes unstable)]
  - **三点翻译**:
    ① **基础设施复杂、且优势集中在多机**:EP + all-to-all + 负载不均只在**大规模多卡多机**时才划算;小规模集群上反而比同 FLOPs 的 dense 更慢更难调(所以早期只有 Google 这类有超算的团队用)。
    ② **训练目标是启发式的**:路由是离散的、不可微,平衡靠"辅助损失"这类启发式(α 要手调),还有容量/丢 token、路由数值不稳定等一堆坑(下文 P36-p51)。
    ③ **收益与代价都是系统级的**:省的是 FLOPs,花的是显存、通信与工程复杂度。
  - 结论:MoE 不是"模型结构的一个小开关",而是一整套"模型 + 系统 + 训练目标"的协同设计。

### C. 路由与门控(p27–p35)

- [x] **Q13(p27)｜三种路由方式,尤其"expert pick token"是什么意思?形状如何?给个数值例子**
  > 原问(还原):详细解释下这三种,特别是 expert pick token,这个具体是啥意思?输入输出形状?通过一个具体数值例子,对比这三者的表现。
  - **共同输入输出**:输入 $x\in\mathbb R^{B\times T\times C}$,输出仍 $(B,T,C)$(每个 token 的表示被"它选中的专家"加工过);路由分数矩阵 $s\in\mathbb R^{B\times T\times N}$($N$ 为专家数)。
  - **① token chooses expert(绝对主流)**:每个 token 自己挑 top-k 个专家。
    优点:每个 token 的计算量固定(好并行、形状规整);缺点:负载可能极不均(热门专家被挤爆 ⇒ 早期要设容量并丢弃超额 token)。
  - **② expert chooses token**:每个**专家**挑自己最喜欢的 top-c 个 token(c 为容量)。
    优点:每个专家恰好处理 c 个 token ⇒ **天然均衡**;缺点:① token 可能一个专家都没选中(需要兜底);② "谁被选中"依赖同批次其他 token ⇒ 破坏"逐 token 因果性"、也要小心评测一致性。
  - **③ global routing(全局优化)**:把整批的"token→专家"分配写成一个**全局匹配/线性规划**问题求解(见 Q15)。
  - **数值例子**(4 专家,$k=1$):token A 分数 $[0.9,0.1,0.05,0.02]$,token B 分数 $[0.05,0.95,0.2,0.1]$。
    - token-choice:A→E1、B→E2,负载均衡 ✓;
    - 若 B 的分数是 $[0.95,0.05,0.2,0.1]$(A、B 都最爱 E1):token-choice 会把**两个都塞给 E1**(E2-E4 闲置,若设容量 1 则有一个 token 被丢弃);expert-choice 在 E1 容量 1 时按分数选 B、A 落到次优专家 ⇒ **天然不爆仓**。

- [x] **Q14(p29)｜hashing 路由具体指什么?怎么做?**
  > 原问(还原):hashing 具体指的是啥?如何做的?
  - **做法**:用一个**固定的(通常是随机)哈希函数**把 token 直接映射到专家,完全不学习参数。例如 Fedus et al. 2022 里的 hash layers:$\text{expert}=h(\text{token id})\bmod N$(也可按位置哈希)。
  - **它的角色**:主要是**基线/消融对照**——看看"完全随机分配"能到什么水平,从而衡量"学习路由"到底带来多少收益;结论是学习路由明显更好,所以没人真用哈希。
  - **优点**:零路由开销、无需门控参数、哈希均匀时天然不爆仓;**缺点**:与内容无关(该去哪不看语义),专家无法形成"专精"。

- [x] **Q15(p30)｜linear assignment 路由是什么?**
  > 原问(还原):解释下这个 linear assignment 具体是啥。
  - **做法**:把路由写成一个**二分匹配/线性分配(linear assignment)问题**:给定 token×专家的分数矩阵,求一个最优分配——每个 token 分到 k 个专家、每个专家最多收 c 个 token——用匈牙利算法、Sinkhorn 或线性规划求解。
  - **好处**:全局最优 + **天然均衡**(约束里就写着容量);**代价**:需要解 solver(慢、不可微,还得处理"进不去"的情形),难以塞进训练循环。
  - 因此实践中很少用,主流仍是 top-k 启发式(讲义 P30 也是把它列为"其它方法")。参考:Clark et al. 2022,[arXiv:2202.01169](https://arxiv.org/abs/2202.01169)。

- [x] **Q16(p31)｜门控公式怎么读($u_t$、$e_i$、$s_{i,t}$)?为什么是内积而不是"MLP+softmax"?**
  > 原问(还原):详细解释下这里的公式,包括 $u_t$、$e_i$ 等等含义。以及这里的 $s_{i,t}$ 是如何得到的。一般来说,MoE 的门控网络的权重不是应该是输入过一个 MLP+softmax 得到的吗?这里为啥是 $u_t$ 和 $e_i$ 做 inner product?这是什么逻辑?
  - **公式**:$s_{i,t}=\mathrm{Softmax}_i\bigl(u_t^\top e_i\bigr)$(DeepSeek V1/V2)或 $\mathrm{Sigmoid}\bigl(u_t^\top e_i\bigr)$(V3);$u_t$ 是第 $t$ 个 token 的门控输入向量,$e_i$ 是第 $i$ 个专家的**中心/embedding 向量**,$s_{i,t}$ 就是"这个 token 与专家 i 的合拍程度"。
  - **为什么内积就等于"线性层 + softmax"**:任何单层线性门控的第 $i$ 个输出就是 $w_i^\top u_t$——把 $e_i$ 令成 $w_i$,立刻得到 $u_t^\top e_i$。所以"内积门控"和"不带 bias、无隐层的线性层门控"**是同一个东西的两种写法**;文献里说"MLP 门控"通常只是允许它再多一层/带非线性,而实践表明单层线性已经够用且最便宜。
  - **为什么选它**:① 计算量相对专家前向可忽略(只有 $N\times d$ 个参数);② $e_i$ 有清晰的几何解释(专家方向);③ 便于加"负载偏置"做 aux-loss-free 平衡(见 Q24)。
  - 出处:DeepSeekMoE(Dai et al. 2024),[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)。

- [x] **Q17(p32)｜共享专家的权重是多少?top-k 权重之和等于 1 吗?要不要归一化?**
  > 原问(还原):一般来说,共享专家的权重是多少?比如有 4 个共享专家,那么这四个专家权重都等于 1?另外,top-k 中,选择了 top-k 后,这 k 个专家的权重之和其实不等于 1 吧,那么这里需要做归一化吗?为什么?
  - **共享专家的权重**:等于**常数 1**,它**不参与门控竞争**——所有 token 都会过共享专家,输出直接相加。所以"4 个共享专家"就是 4 个各自以权重 1 生效的专家。它们承担"通用知识",让路由专家专心"专精"。
  - **top-k 的权重和确实不等于 1**:若门控是 sigmoid(V3),k 个权重之和没什么约束;若是 softmax over 全部 $N$ 个专家,选中的 k 个之和也小于 1(概率质量被没选中的专家分走了)。
  - **需要归一化,而且主流都做**:把选中的 k 个权重**重新归一化到和为 1**——这样输出的尺度不随 k、温度、门控分布漂移,与残差流配合也更稳。DeepSeek-V3 正是"只对选中的专家归一化";Mixtral/DBRX 的"先 top-k 再 softmax"其实也是归一化的一种实现(见 Q19)。
  - 一句话:**共享专家"恒权 1、不竞争";路由专家"竞争 + 选中后归一化"**。

- [x] **Q18(p35)｜表里的 fine-grained ratio 指的是什么?**
  > 原问(还原):这里的 fine-grained ratio 指的是啥?
  - **定义**:单个专家的隐藏维度 ÷ 一个"标准 dense FFN"的隐藏维度(即 $d_{ff}$)。表里 DeepSeek v3 写 1/14,意思是"每个路由专家只有标准 FFN 的 1/14 宽"。
  - **为什么要"细粒度"**:同样多的激活参数,拆成**更多、更小**的专家后,可选组合数 $\binom{N}{k}$ 暴涨 ⇒ 同样的 FLOPs 下模型能表达的"专家组合"更多,质量更好(DeepSeek 的 fine-grained expert segmentation)。
  - **代价**:专家数变多 ⇒ 路由/通信/调度开销上升(需要 P44-P46 那套系统优化),表里 OlMoE(1/8)、Qwen1.5(1/8)、DeepSeek v1(1/4)的比值也在说明"细粒度程度"是一个可调旋钮。

### D. 门控的两种写法、随机路由与负载均衡(p38–p42)

- [x] **Q19(p38)｜"先 softmax 再选 top-k" 与 "先选 top-k 再 softmax" 有何不同?**
  > 原问(还原):这个式子和前面几页 ppt 的公式是否存在不同?前面的是门控网络的输出先 softmax,再选 topk;这里是先选 topk,再 softmax?
  - **是的,这确实是两种不同写法**:
    ① **先 softmax(对全部 $N$ 个专家)再取 top-k**:Switch/GShard 以及 DeepSeek V1-V2 早期风格。选中权重是"全局 softmax 的子集",和小于 1,通常要再重归一化一次。
    ② **先在分数上取 top-k,再对选中的 $k$ 个做 softmax**:Mixtral、DBRX、DeepSeek v3(讲义 P31 原文:"Mixtral, DBRX, DeepSeek v3 softmaxes after the TopK")。权重只由选中的 $k$ 个决定,**和恒为 1**,天然无需再归一化。
  - **影响**:②的权重尺度更干净(不受未选中专家的分数影响),梯度信号也更聚焦;①则要额外处理"和小于 1"的尺度问题。DeepSeek-V3 更进一步:用 **sigmoid** 打分 + 对选中者**归一化**,并给每个专家加路由偏置做无辅助损失的平衡(见 Q24)。
  - 一句话:top-k 与 softmax 的**先后顺序**是不同实现的核心区别之一,现代主流是"先 top-k 再归一化"。

- [x] **Q20(p38)｜stochastic 路由:目标是什么?"robust"与"learn how to rank"是什么意思?**
  > 原问(还原):详细解释下这里的 stochastic 方法,目标是什么?这个是如何实现目标的?这里的 robust 是什么意思?第二点 learn how to rank 又是啥意思?
  - **做法**:在门控分数上加噪声再做 top-k。Shazeer et al. 2017 加**高斯噪声** $\mathcal N(0,\sigma^2)$;Fedus et al. 2022 用**乘性抖动** $r\sim U[1-\epsilon,1+\epsilon]$ 乘到分数上。
  - **目标 1:让专家更 robust(不脆弱)**:路由变成随机的 ⇒ 模型不能依赖"某个 token 必走某个专家"的尖锐分工;噪声相当于数据增强/正则,让专家学到更通用的功能而不是死记少数样本。
  - **目标 2:让"排序"变得可学(learn how to rank K experts)**:top-k 之外的专家本来梯度为 0;加了噪声后,"接近边界"的专家会偶尔被选中、拿到梯度 ⇒ 模型学会的是**给专家排序**这件事(而不是输出无法求导的硬 one-hot)。
  - **后续**:Zoph et al. 2022 发现抖动收益不明显、把它去掉了 ⇒ 现在主流不再用噪声路由,改用平衡损失(见 Q22)与路由器精/偏置(见 Q24)。

- [x] **Q21(p40)｜"不做平衡会让第一轮被选中的专家越来越强"这个理解对吗?**
  > 原问(还原):如果不做 balancing,会导致第一轮的 topk expert 在反向传播得到更多的信号,从而增强这 k 个 expert 的表现,后续也会趋向选择这 k 个 expert,这理解对吗?详细解释下。
  - **理解基本正确**,补两点机制细节:
    ① **梯度是稀疏的**:这一步没被选中的专家拿到 **0 梯度**,所以只有被选中的专家在更新参数——这天然是"强者恒强"的起点;
    ② **门控自己也在学**:门控参数与专家参数同时被优化,被选中的专家输出更好 ⇒ 门控给它的分数更高 ⇒ 更容易再次被选中,形成**正反馈(马太效应 / rich-get-richer)**。
  - **后果**:少数专家过载(系统变慢、容量溢出丢 token、EP 通信不均),其余专家欠训练(参数被浪费);严重时出现"路由坍塌"(几乎所有 token 都涌向少数专家)。
  - 所以平衡的目标不是"公平",而是**系统效率 + 参数利用率**:让每个专家都被充分训练、每台设备负载相当。

- [x] **Q22(p40)｜平衡损失的公式与求导怎么来的?**
  > 原问(还原):详细解释下这里的公式,以及这里的求导是如何求出来的,导数形式和含义。
  - **Switch Transformer 的辅助损失**(Fedus et al. 2022):
    $L_{\text{aux}}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_iP_i$,其中 $f_i=\frac1T\sum_x\mathbb 1[\arg\max p(x)=i]$(专家 $i$ 实际接收的 token 比例)、$P_i=\frac1T\sum_x p_i(x)$(专家 $i$ 的平均门控概率),$N$ 为专家数,$T$ 为该 batch 的 token 数,$\alpha$ 是系数。
  - **求导**:$f_i$ 由 argmax 决定,**不可微,求导时当成常数**;只有 $P_i$ 贡献梯度:
    $\dfrac{\partial L_{\text{aux}}}{\partial p_i(x)}=\dfrac{\alpha N f_i}{T}=\dfrac{\alpha N}{T^2}\sum_x\mathbb 1[\arg\max p(x)=i]$
    —— 这正是讲义 P40 上那条式子。
  - **含义**:门控概率受到的"下压"正比于该专家的负载 $f_i$ ⇒ **越热门的专家被压得越狠**;与此同时,热门专家因为接收的 token 多、被训练得更好 ⇒ 一推一拉之间达到均衡。
  - 实践提醒:该损失与主损失相加,所以 $\alpha$ 要小(否则伤害主任务);它也是"启发式"的典型代表。

- [x] **Q23(p41)｜什么叫 device balancing?为什么专家粒度的均衡还不够?**
  > 原问(还原):这里不理解什么叫 device balancing。
  - **为什么要按设备平衡**:EP 下每张卡承载**好几个专家**;整批计算的耗时由**最慢的设备**决定。所以即便"每个专家"负载均衡,只要某台卡上的专家总和偏重(或 token 都往它那跑),all-to-all 就要等它 ⇒ 系统效率下降。
  - **做法(DeepSeek V1-V2)**:把 Switch 式的平衡损失**按设备聚合**——$f_j,P_j$ 的定义从"专家"改成"设备 $j$ 上的所有专家之和",得到 per-device 平衡损失,与 per-expert 损失一起用(讲义 P41:"per-expert balancing – same as the switch transformer;per-device balancing – the objective above, but aggregated by device")。
  - 一句话:**per-expert 管"专家别闲/别爆",per-device 管"每张卡别闲/别爆"**;后者直接对应通信与吞吐。

- [x] **Q24(p42)｜DeepSeek v3 的 per-expert bias 与"auxiliary-loss-free balancing"具体怎么做?**
  > 原问(还原):没看懂这个做法。
  - **做法**:给每个专家一个可调的偏置 $b_i$,**只加到"选择用"的分数上**,不影响最终权重:$s'_{i,t}=s_{i,t}+b_i$。用 $s'$ 取 top-k,但加权求和仍用原始 $s_{i,t}$。
  - **在线更新(负反馈)**:每隔若干步检查每个专家的负载——**过载就把 $b_i$ 调小、欠载就调大**(常数步长 $\gamma$;DeepSeek-V3 用 $10^{-3}$)。这个更新**不经过反向传播**,所以主损失里不需要再塞一个平衡项。
  - **为什么叫 aux-loss-free**:平衡不再靠梯度式的辅助损失,而是靠"控制论式"的外环调节,不干扰主损失;但讲义也提醒**它并非完全无辅助损失**——V3 仍保留了一个很小的"序列级"辅助损失(系数约 $10^{-4}$)来稳定训练。
  - 旁证:Nemotron 3 Super 也用了同款(sigmoid 路由 + 专家偏置,更新率 $10^{-3}$),说明这是当前主流做法。
  - 出处:DeepSeek-V3 技术报告,[arXiv:2412.19437](https://arxiv.org/abs/2412.19437);Wang et al. 2024(aux-loss-free),[arXiv:2408.15664](https://arxiv.org/abs/2408.15664)。

### E. MoE 的系统、稳定性与训练技巧(p44–p51)

- [x] **Q25(p44)｜并行的几种方式分别是什么?MoE 为什么能很好地并行?**
  > 原问(还原):详细解释下这里提到的不同的 parallelism 的方式,以及 MoE 为啥能很好地并行?MoE 是如何并行的?
  - **四种基本并行**:
    - **DP(数据并行)**:每卡一份完整模型副本、各吃一部分数据,梯度 all-reduce;
    - **TP(张量并行)**:把单层的大矩阵切到多卡(每卡算一部分输出/输入分块),通信最频繁,通常只放在机内高速互联上;
    - **PP(流水线并行)**:按层切分到不同卡,靠 micro-batch 填满流水,减少气泡是难点;
    - **EP(专家并行,MoE 特有)**:按**专家**切分,每卡放一部分专家。
  - **MoE 为什么天然适合并行**:专家彼此**完全独立**(各是一个 FFN)⇒ 计算天然可分;而且一个专家的参数量适合整块放一张卡(不需要像 dense 层那样把单个矩阵切碎)。所以 MoE 层用 EP,注意力层用 TP/DP/PP,组合起来扩展性好。
  - **代价**:路由分发/回收需要 **all-to-all** 通信,通信量与"token 数 × top-k × 维度"成正比;且**负载不均会直接拖慢**(等最慢设备)——这就是平衡损失、容量、通信压缩(如 P46 的 LatentMoE)存在的原因。

- [x] **Q26(p45)｜为什么这里要用分块(块稀疏)矩阵乘法?**
  > 原问(还原):这里的详细解释?为啥要分块矩阵乘法?没看懂。
  - **背景**:MoE 里每个 token 只去 $k$ 个专家 ⇒ 计算图天然是**块稀疏**的(行 = token,列 = 专家)。朴素实现有两条烂路:① 按"每专家容量"把 token 补齐到固定形状再算 ⇒ 大量 padding 白算;② 先 permutation 把同专家的 token 聚在一起,再对每个专家单独发一个 GEMM ⇒ 小 kernel 一大堆、启动开销吃掉收益。
  - **做法(MegaBlocks,Gale et al. 2022)**:用 **block-sparse GEMM / grouped GEMM**——一次 kernel 处理"变长分组"的 token,直接跳过没有 token 的块,既省 padding 又省 kernel 启动,还支持动态容量(**不丢 token**)。
  - 一句话:MoE 的瓶颈常在**数据搬运与 kernel 效率**,而不是算术量;块稀疏 GEMM 就是为它定制的算子。参考:Gale et al. 2022,[arXiv:2211.15841](https://arxiv.org/abs/2211.15841)。

- [x] **Q27(p46)｜"把激活下投影以减少通信"这页到底在讲什么?**
  > 原问(还原):这页完全没看懂。
  - **这页讲的是 LatentMoE(Nemotron 3 的新做法)**:把 token 从 $d$ 维**下投影**到更小的潜空间 $\ell$ 维($W_\downarrow\in\mathbb R^{\ell\times d}$),**路由与专家计算都在潜空间做**,算完再上投影回 $d$ 维($W_\uparrow\in\mathbb R^{d\times\ell}$)。
  - **为什么能省通信**:EP 的 all-to-all 传的就是 token 激活,维度从 $d$ 降到 $\ell$ ⇒ **通信量和专家的权重读取量都降为原来的 $\ell/d$**。
  - **省下来的预算怎么用**:同时把专家总数从 $N$ 增到 $N\cdot d/\ell$、top-k 从 $K$ 增到 $K\cdot d/\ell$——保持"有效非线性预算 $K\cdot m$"不变,所以质量不降反升(报告原话:improving model accuracy at approximately constant inference cost)。
  - **注意**:路由门控、共享专家、非路由层仍留在全维 $d$(它们不是通信瓶颈)。
  - 出处:NVIDIA Nemotron 3 白皮书 / Nemotron 3 Super 技术报告([research.nvidia.com](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf))。

- [x] **Q28(p47)｜expert capacity 是什么?为什么要 drop token?后果是什么?drop-less 怎么做?**
  > 原问(还原):expert capacity 指的是啥?为啥早期的架构需要 drop tokens that exceed capacity?会造成什么后果?目前的 drop-less 架构是如何做到的?
  - **expert capacity(专家容量)**:每个专家在一个 batch 内**最多处理多少个 token**,通常写成 $c=\text{容量因子}\times\frac{kT}{N}$($T$ 为 batch token 数、$N$ 专家数、$k$ 为 top-k)。
  - **为什么要 drop**:早期实现要求每个专家的输入是**固定形状**的张量(才能用规整的 GEMM、静态显存),所以超出容量的 token 只能丢弃;这样系统的形状与时间可预测。
  - **后果**:① 被丢的 token **这一层等于没算 FFN**(质量受损);② 引入**批级随机性**——某个 token 进不进得去,取决于同批次其他人的 query(讲义 P47 的吐槽:"other people's queries can drop your token!"),导致同一条输入在不同 batch 下结果不同、不可复现。
  - **现在怎么做到 drop-less**:靠 ① 块稀疏/grouped GEMM 允许**变长分组**(Q26),② permutation/动态容量让形状随实际负载变化,③ 用路由偏置(aux-loss-free)+合理的容量因子把负载压均衡 ⇒ 容量不再是"硬闸门",而只是一个监控/调节指标。

- [x] **Q29(p48)｜MoE 的稳定性问题是什么?为什么"路由器用 FP32(+ z-loss)"能解决?**
  > 原问(还原):这里提到的问题和解决方式?
  - **问题**:MoE 训练不稳定,常见根源在**路由器的数值**:门控 logits 变大/溢出 ⇒ softmax(sigmoid)饱和 ⇒ 路由决策剧烈跳变 ⇒ 专家负载震荡、loss 尖峰。
  - **解决(Zoph et al. 2022)**:把**路由器单独用 FP32** 计算(其余部分仍是 bf16),必要时再加一个 **z-loss** 抑制 $\log Z=\log\sum_ie^{z_i}$ 变大。
  - **为什么便宜又有效**:路由器参数量占比极小,提到 FP32 的开销可忽略;而它恰好是"指数归一化 + 离散决策"的环节,数值最敏感 ⇒ 收益/成本比极高。
  - 顺带:z-loss 的原理见 L03 笔记(P54 那一节):把 $\log Z$ 往 0 拉,等价于惩罚"logits 整体过大"。

- [x] **Q30(p50)｜MoE 微调会遇到什么问题?两种解决方案在做什么?**
  > 原问(还原):这一页提到的问题和解决方式?
  - **问题**:稀疏 MoE 在小规模微调数据上**容易过拟合**——总参数量大、每个 token 又有稀疏激活的"专用通道",在小数据上很快就会记住训练集。
  - **解法一(Zoph et al. 2022)**:微调时**只训练非 MoE 部分**(把稀疏专家冻结,或干脆换成 dense MLP)——限制可训练参数量,等价于强正则。
  - **解法二(DeepSeek)**:用**足够大的数据**(约 1.4M SFT 样本)让过拟合不成为瓶颈。
  - 一句话:MoE"参数多"在预训练阶段是优点(容量大、同 FLOPs 更好),到小数据微调阶段就变成缺点 ⇒ 要么少训参数,要么给足数据。

- [x] **Q31(p51)｜upcycling 是什么?具体怎么做?**
  > 原问(还原):这里的 upcycling 指的是啥?具体怎么做的?
  - **定义**:用**已训练好的 dense 模型**去初始化 MoE,而不是从零训练 MoE。
  - **具体做法**:把 dense 的 FFN(以及其它层)复制成 $N$ 个专家作为初始权重;门控网络重新随机初始化;通常还会给复制出的专家**加噪/扰动或随机 dropout** 来打破对称性(否则所有专家一模一样、门控学不出分工);然后继续预训练。
  - **收益**:省掉从零训 MoE 的巨额算力,站在 dense checkpoint 上"扩参数"就能拿到增益。讲义给的两个例子:MiniCPM(MiniCPM 模型 upcycle 成 top-2/8 专家,约 520B token 就有效果)、Qwen MoE(从 Qwen 1.8B 初始化成 top-4/60 专家 + 4 共享,是最早被确认成功的 upcycling 之一)。
  - **注意**:upcycle 后需要**足够的继续训练**让专家分化;upcycling 的收益主要来自"同 FLOPs 下参数更多",而不是魔法。

### F. 收尾:MLA(潜变量与 KV cache)(p57)

- [x] **Q32(p57)｜latent $c$ 具体怎么做?什么架构?为什么能省 KV cache?**
  > 原问(还原):这里的 latent c,具体是如何做的?具体是啥架构?为啥能节省 KV cache?
  - **架构(MLA)**:不再为每个头单独算 $K,V$,而是先把输入**下投影**到一个低维潜向量 $c_t^{KV}$(DeepSeek-V3 里是 512 维),再用它**上投影**出各头的 $K,V$:
    $K=W^{UK}c_t^{KV}$、$V=W^{UV}c_t^{KV}$;查询侧同样有潜向量 $c_t^Q$。
  - **为什么省 KV cache**:推理时**只需缓存 $c_t^{KV}$**(512 维),而不是 $H$ 个头各自的 $K,V$(在 V3 里是 $128\times(128+128)$ 量级)⇒ 缓存体积降一个数量级(讲义 P58 的 "KV-caching, we only need to store $c_t^{KV}$")。
  - **与 RoPE 的冲突(重要细节)**:$W^{UK}$ 本来可以**合并进 $Q$ 的投影**($\langle hW^Q,\,W^{UK}c\rangle=\langle hW^QW^{UK},\,c\rangle$),这样连 $K$ 都不必物化;但 RoPE 要作用在 $K$ 上(位置相关),合并后就无法在 $c$ 上旋转。**解法**:留出**少量"非潜变量"的 key 维度专门承载 RoPE**(decoupled RoPE),其余维度走潜变量路径。
  - 出处:DeepSeek-V2 / V3([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)、[arXiv:2412.19437](https://arxiv.org/abs/2412.19437));与 L03 笔记里 MLA 那一节可以对照看。

---

## ② 用自己的话复述本讲核心思想

(看完后随手写,一句话也行。可参考线索:① 注意力的成本随 $n$ 二次增长 ⇒ 两条路——**重排**(线性注意力/SSM:丢掉 softmax 的归一化,用矩阵状态换固定成本)与**裁剪**(稀疏注意力:只算 top-k 的 KV);前者纯线性但**检索弱**,后者保精度但要付 indexer 的代价 ⇒ 主流落点是**混合**。
② MoE 的核心交易是"**用显存/通信换质量**":参数量 ↑ 但激活参数/FLOPs 不变;难点全在"离散路由怎么训、负载怎么均衡、专家怎么并行、数值怎么稳";DeepSeek MoE 的演化(v1 常规 → v2 通信平衡 + top-M 设备路由 → v3 sigmoid 门控 + 无辅助损失偏置 + 细粒度/共享专家)基本是这一讲的答案汇总。)

## ③ 触发的新问题 / 想扩展的知识

- [ ] (示例)想深入:xxx 与贝叶斯/优化理论的关系
- (AI 提示:本讲引出的延伸问题已整理进 [gap-list.md](gap-list.md),其中标注 A 必补的条目建议本周补齐。)

## AI 补充笔记(按日期追加)
(AI 的展开讲解、推导、延伸阅读写在这里;不要删除历史记录)

### 2026-10-10 · Lecture 04 首次答疑(32 条 #TODO 全解)

本节是 ① 里问题的**加长版推导**,按主题分组;① 已经说清的细节这里不再重复。
(诚实说明:下面的 arXiv 编号来自检索与记忆,个别可能有偏差,按标题检索即可确证;凡涉及**公式**的关键结论我都对着原文核对过。)

**补充 1｜线性注意力的账:为什么"重排"能省这么多**
- 原式 $\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V$($\rho$ 是逐行 softmax)。若 $\rho$ 可写成逐元素的核 $\phi$,就有
  $\bigl(\phi(Q)\phi(K)^\top\bigr)V=\phi(Q)\bigl(\phi(K)^\top V\bigr)$。
- 复杂度账(左 vs 右):左 = $n^2d_k$(算 $QK^\top$)+ $n^2d_v$(再乘 $V$);右 = $nd_kd_v$(先算 $K^\top V$)+ $nd_kd_v$(再乘 $Q$)⇒ **$2nd_kd_v$,对 $n$ 线性**。
- 直觉:从"给每个 query 现算一套权重再加权求和"改成"先把所有 KV 压成一个固定大小的矩阵(状态),query 来了直接查表"。
- 代价:softmax 的"归一化 + 尖锐选择"没了;$S\in\mathbb R^{d_k\times d_v}$ 的容量有限,序列太长时不同 key-value 对互相干扰(memory collision),精确检索能力下降。
- **对偶性的两种形式**(同一层的两种算法):
  - 并行:$O=\bigl(\phi(Q)\phi(K)^\top\bigr)V$ —— 二次但全是 matmul,训练时能吃满 GPU;
  - 串行:$S_t=S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t$ —— 每步 $O(d_kd_v)$,与 $T$ 无关,推理时省显存/带宽;
  - 现代实现用 **chunkwise(分块)**:块内走并行形式、块间用递推传状态 ⇒ 兼顾 matmul 密度与线性复杂度。

**补充 2｜Mamba-2 / DeltaNet / GDN 的统一视角**
- 它们都属"矩阵值状态 + 写入门控"的同一家族,差别只在**怎么写、怎么擦**:

| 模型 | 状态更新 | 整体遗忘 | 精准覆盖 | 跳过写入 |
| --- | --- | --- | --- | --- |
| 线性注意力 | $S_t=S_{t-1}+k_tv_t^\top$ | ❌ | ❌ | ❌ |
| RetNet / GLA | $S_t=\gamma_tS_{t-1}+k_tv_t^\top$ | ✅(标量 / 对角) | ❌ | ❌ |
| Mamba-2 | $S_t=\gamma_tS_{t-1}+k_tv_t^\top$,输出加直通 $v_t^\top D$ | ✅ | ❌ | ❌ |
| DeltaNet | $S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | ❌ | ✅ | ❌ |
| **Gated DeltaNet** | $S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | ✅ | ✅ | ✅($\beta_t=0$) |

- **erase 的几何**:设 $\|k_t\|=1$,则 $k_tk_t^\top$ 是"投到 $k_t$ 方向"的投影算子;$(I-\beta_tk_tk_t^\top)$ 把状态在该方向的分量按 $\beta_t$ 扣掉 ⇒ 先擦掉旧的 $k_t\to v_t^{\text{old}}$ 关联,再写入新的,这正是联想记忆里的 delta 规则。
- **在线学习视角**:展开一步可得 $S_t-S_{t-1}^{\text{(decay 后)}}=-\beta_t\bigl(S_{t-1}k_t-v_t\bigr)k_t^\top$,它恰好是对 $\tfrac12\|Sk_t-v_t\|^2$ 的梯度步($\beta_t$ 当学习率),而 $\gamma_t$ 是权重衰减 ⇒ 这就是"把状态当成一个小模型、在推理时在线更新"(fast weights / test-time training)那一族思想。
- **为什么仍要混合**:固定大小的状态无法精确长程检索,所以主流是 $N$ 层线性层 + 1 层全注意力(7:1、3:1、H1/H2)。

**补充 3｜DSA 的成本分解:为什么"复杂度不变、常数决定成败"**
- 总成本 $\approx$ indexer 成本 + 主注意力成本:
  - indexer:$O(L^2\cdot H^Id^I)$(二次,但 $H^I=64$、$d^I=128$、FP8、多 query 头共享同一份 $k^I_s$、且不需要与 $V$ 加权 ⇒ 常数极小);
  - 主注意力(MLA):$O(Lk\cdot(\ldots))$ —— 已被 $k$ 截断。
- $L$ 很大时,第一项被小常数压住、第二项随 $L$ 线性 ⇒ 端到端近似线性增长,而稠密基线的 $L^2$ 会越拉越开。
- 这就是"constants matter":同一个 $O(L^2)$,系数差 3~10 倍就是能否上线的差别;FlashAttention(Q2)则是"复杂度不变、把**访存**常数压下来"的另一个例子。

**补充 4｜MoE 路由三族的形状与代价**

| 路由族 | 打分矩阵 | 谁做 top-k | 每 token 计算量 | 负载 | 批内因果性 |
| --- | --- | --- | --- | --- | --- |
| token-choice | $s\in\mathbb R^{B\times T\times N}$ | 每个 token 选 $k$ 个专家 | 固定(最规整) | 可能不均,需容量/平衡 | 好(只看自己) |
| expert-choice | $s\in\mathbb R^{N\times T}$ | 每个专家选 $c$ 个 token | 随负载变化 | 天然均衡 | 差(依赖同批其它 token) |
| global / matching | 同上 | 全局求解分配 | 求解成本高 | 最优均衡 | 差 |

- 结论:token-choice 赢在**计算规整 + 因果干净**,expert-choice 赢在**天然均衡** ⇒ 主流是 token-choice + 平衡损失/偏置。

**补充 5｜平衡损失推导与 aux-loss-free 的伪代码**
- Switch 平衡损失 $L_{\text{aux}}=\alpha N\sum_i f_iP_i$ 的求导细节:$P_i=\frac1T\sum_xp_i(x)$ ⇒ $\frac{\partial P_i}{\partial p_i(x)}=\frac1T$,又因为 $f_i$ 由 argmax 决定、求导时视为常数,于是
  $\frac{\partial L_{\text{aux}}}{\partial p_i(x)}=\alpha Nf_i\cdot\frac1T=\frac{\alpha N}{T^2}\sum_x\mathbb 1[\arg\max p(x)=i]$ ✓(与讲义 P40 一致)
- aux-loss-free 的做法(每步或每个 batch 末执行一次,不参与反向传播):
  - 统计每个专家的接收量 $c_i$;若 $c_i$ 高于目标 ⇒ $b_i\mathrel{-}=\gamma$;若低于目标 ⇒ $b_i\mathrel{+}=\gamma$($\gamma$ 如 $10^{-3}$);
  - 选 top-k 时用 $s_{i,t}+b_i$,加权求和时用原始(或归一化后的)$s_{i,t}$。
  - 本质是**比例-积分控制**:偏置充当积分项,长期把负载推到均衡,而不占梯度。

**补充 6｜MoE 的通信账,以及 LatentMoE 为什么有效**
- all-to-all 量级:每层每个 token 要把激活发给 $k$ 个专家所在设备(约 $O(BTkd)$),算完再收回来 ⇒ EP 组越大、跨机越多,通信越贵;负载不均时"最慢设备决定时间"。
- LatentMoE:把送进专家的激活维度从 $d$ 降到 $\ell$ ⇒ **通信量与专家权重读取量都乘 $\ell/d$**;省下的预算换成更多专家($N\cdot d/\ell$)与更大 top-k($K\cdot d/\ell$),保持有效非线性预算 $K\cdot m$ 不变。
- 相关细节:块稀疏 GEMM 解决 shape/kernel 效率(Q26),容量与丢 token 的历史与现状(Q28)。

**补充 7｜MoE 训练技巧清单(把 P36-p51 串起来)**
- 路由不可微 ⇒ 三条路:RL(REINFORCE,理论上好但方差大、少用)、随机扰动(Shazeer 2017 高斯噪声 / Fedus 2022 抖动,后来被 Zoph 2022 移除)、**启发式平衡损失**(实践主流)。
- 平衡:per-expert + per-device(DeepSeek V1-V2)→ per-expert bias(aux-loss-free,V3)。
- 数值:路由器 FP32(+ 可选 z-loss)。
- 容量:早期固定容量 + drop token(引入批级随机性)→ 现在块稀疏/动态容量做到 drop-less。
- 微调:小数据易过拟合 ⇒ 只训非 MoE 层,或给足 SFT 数据。
- 初始化:upcycling(从 dense 复制专家 + 打散对称性 + 继续预训练)。

**补充 8｜MLA 里"合并进 Q"与 RoPE 的冲突(把 Q32 的细节推一遍)**
- 不带 RoPE:$\langle hW^Q,\ W^{UK}c^{KV}\rangle=\langle hW^QW^{UK},\ c^{KV}\rangle$,把 $W^{UK}$ 吸收进 $Q$ 的投影 ⇒ 推理时可以只缓存 $c^{KV}$、连 $K$ 都不用物化 ✓
- 带 RoPE:$K$ 需要先乘位置相关的旋转 $R_k$,即 $\langle hW^QR_q,\ R_kW^{UK}c^{KV}\rangle=\langle hW^QR_qR_kW^{UK},\ c^{KV}\rangle$ —— $R_k$ 夹在中间且依赖位置,**无法再吸收** ✗
- 解法:把 key 拆成"潜变量部分(不旋转)"+"少量非潜变量维度(物化后正常加 RoPE)" ⇒ 主缓存仍只是 $c^{KV}$ ✓

**延伸阅读**
- 线性注意力 / 对偶性:Linear Transformers [arXiv:2006.16236](https://arxiv.org/abs/2006.16236);RetNet;FlashAttention [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- SSM / 递推架构:Mamba-2(SSD)[arXiv:2405.21060](https://arxiv.org/abs/2405.21060);Gated DeltaNet [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- 稀疏注意力:DSA / DeepSeek-V3.2 [arXiv:2512.02556](https://arxiv.org/abs/2512.02556);NSA(Native Sparse Attention)
- MoE 基础:Switch Transformer [arXiv:2101.03961](https://arxiv.org/abs/2101.03961);GShard;MoE 综述(Fedus/Zoph)
- MoE 变体与系统:DeepSeekMoE [arXiv:2401.06066](https://arxiv.org/abs/2401.06066);aux-loss-free 平衡 [arXiv:2408.15664](https://arxiv.org/abs/2408.15664);DeepSeek-V3 [arXiv:2412.19437](https://arxiv.org/abs/2412.19437);MegaBlocks [arXiv:2211.15841](https://arxiv.org/abs/2211.15841);OlMoE
- 长上下文与架构:Nemotron 3 白皮书([NVIDIA](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf));MLA(DeepSeek-V2)[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
