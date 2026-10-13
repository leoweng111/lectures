# Lecture 04 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_04.pdf`(静态讲义,共 60 页;用 PDF 阅读器/浏览器打开)
- 状态:进行中(①「卡住的地方」32 条 `#TODO` 已于 2026-10-10 全部作答并勾选;看完后在 [progress.md](progress.md) 更新)
- 说明:本讲 PDF 共 78 条批注,其中 `#TODO` 32 条。每条问题的答案就写在下方;较长推导、公式与延伸阅读统一放在文末「AI 补充笔记」的 2026-10-10 一节
- **记号约定(容易踩坑)**:线性注意力的递推公式统一用**列向量**写法:$q_i,k_i\in\mathbb R^{d_k\times1}$、$v_i\in\mathbb R^{d_v\times1}$(于是外积 $k_iv_i^\top$、输出 $q_t^\top S_t$ 的形状都自然成立,与讲义 P5 和论文原式一致)。注意两点:① 讲义 P4 的 $Q,K,V\in\mathbb R^{n\times d}$ 与 PyTorch 张量 $(B,H,T,d_h)$ 是**行视角**(「第 $i$ 行」= 这里的列向量 $k_i,v_i$ 的**转置**),取出来用时转置一下即可,数值不变;② 列向量约定下投影写作 $q_i=W_Qx_i$、$W_Q\in\mathbb R^{d_k\times C}$(代码里常见的 $x_iW_Q$ 只是它的转置写法)。

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
  - **追问(2026-10-10)｜这是否说明 FlashAttention 只解决 memory-bound?训练是 compute-bound 时它作用大吗?**
    - **准确说法不是"只解决 memory-bound",而是"把注意力里 memory-bound 的环节消掉"。**注意力的 matmul($QK^\top$ 打分、$PV$ 加权)在 $n\gtrsim 2d$ 时是 compute-bound(算术强度 $\sim O(n)$),但它夹着的 **softmax、mask、以及 $n\times n$ 中间矩阵的物化与回读**是彻底的 memory-bound(算术强度 $\sim O(1)$)。注意力这一块是**两种瓶颈混在一起**,FA 的贡献是把后者融合进前者。
    - **训练时作用很大,论文数据最能说明问题**(GPT-2、A100、前向+反向的注意力单项):FLOPs 从 66.6 GFLOP **升到** 75.2 GFLOP(反向要重算,算得更多了),但 HBM 读写从 40.3 GB 降到 4.4 GB(**降到 1/9**),时间从 41.7 ms 降到 7.3 ms(**快 5.7×**)——它赢得完全靠省访存,不靠省算力。
    - **端到端量级**:BERT-large(seq 512)只快 15%(注意力占比本来就小),GPT-2(seq 1K)**3×**,Long-Range-Arena(1K–4K)2.4×;GPT-2 small 训练从 4.7 天 → 2.7 天(约 1.7×)。FA2 更进一步:GPT-3 1.3B、8k 上下文的训练吞吐 72 → 170(FA1)→ 220 TFLOP/s(FA2),相当于 **~70% MFU**。也就是说:FA 系列真正的价值是"把训练里注意力的 MFU 拉到接近峰值"。
    - **为什么 compute-bound 的训练也吃得到红利**:① 省访存 ⇒ 时间下降;② 不物化 $n\times n$ ⇒ **省显存**,于是能开更大 batch / 更长序列,间接提升 MFU;③ 减少 kernel 启动与调度空隙。
    - **什么时候收益小**:序列很短($n\ll 2d$,注意力本来就占比低——BERT 512 只 +15%)、baseline 实现已经很优(FA2 相对 FA1 还能再快约 2×,说明"实现质量"本身就在决定收益)、以及 **decode 阶段**(瓶颈是 KV cache 的读取带宽,得靠 GQA/MLA/量化,FA 只能改善一部分)。
    - 出处:FlashAttention [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)(摘要与 Table 1);FA2 的数字见 [Stanford CRFM 博客](https://crfm.stanford.edu/2023/07/17/flash2.html)。

- [x] **Q3(p4)｜"不做 softmax 就叫 linear attention"——这是它的定义吗?**
  > 原问(还原):如果不做 softmax,那么就被称之为 linear attn?这个是 linear attn 的定义吗?
  - **一句话**:方向对,但更准确的定义是"把 softmax 换成**可分解的核**":$\mathrm{Attn}=\phi(Q)\bigl(\phi(K)^\top V\bigr)$,"不做 softmax($\rho$ 取恒等)"只是 $\phi$ 取恒等这个最简特例。
  - **为什么这一点是关键**:softmax 要求"每一行看过所有 key 才能归一化",因此 $QK^\top$ 无法分解;换成可分解的核之后,矩阵乘法可以重排:
    $QK^\top V=Q(K^\top V)$,复杂度从 $n^2d_k+n^2d_v$ 降到 $2nd_kd_v$——**对 $n$ 线性**。
  - **代价与补救**:表达力下降(做不出"只看最相关几个位置"的尖锐选择),所以现代做法要么加门控(Mamba-2 / GDN),要么与全注意力混合(7:1、3:1、H1/H2)。
  - 出处:Shen et al. 2018,[arXiv:1812.01243](https://arxiv.org/abs/1812.01243);Katharopoulos et al. 2020(Linear Transformers),[arXiv:2006.16236](https://arxiv.org/abs/2006.16236)。
  - **追问(2026-10-10)｜为什么去掉 softmax 表达力就下降?精确原因是什么?**
    - **精确原因不是"少了一个非线性",而是"注意力矩阵被强制做低秩分解"**:
      - softmax 注意力:$A=\mathrm{softmax}(S)$ 可以是**任意正的行归一化矩阵**——给定任意正矩阵 $T$,取 $S=\log T$ 就有 $\mathrm{softmax}(S)=T/\text{rowsum}(T)$(实测完全吻合)。因此它有约 $n^2$ 个自由度,秩可以达到 $n$;
      - 线性注意力:$A=\phi(Q)\phi(K)^\top$ 是两个 $n\times d'$ 矩阵的乘积 ⇒ **$\mathrm{rank}(A)\le d'$(特征维,实践上就是头维 $d$)**,自由度被压到 $O(nd)$。实测($n=6,d=3$):softmax 注意力秩 = 6,线性注意力秩 = 3。
    - **后果就是"记忆容量有上限"**:线性注意力等价于"把整段前缀压进一个固定 $d\times d$ 的状态矩阵",容量 $\sim d$;一旦 key-value 对的数量超过 $d$,不同关联就会互相干扰(memory collision)。实测:$d=3$ 的状态里存 5 对 key-value 再按 key 查询,线性注意力的回忆误差为 2~6(基本失败),而 softmax 注意力权重近似 one-hot、误差 0~0.15(基本成功)。
    - **对应到能力**:这就是为什么线性注意力在**关联回忆 / 复制 / 长文档检索**这类任务上明显偏弱,而在普通语言建模上差距不大——后者不需要"精确检索某一对关联"。
    - **另一层次要影响:归一化。**softmax 输出是**凸组合**(每行权重和为 1,输出范数有界,实测 0.7~1.0);线性注意力是"未归一化的求和",量级随 $n$ 增长(实测范数在 0~5.7 之间乱跳)⇒ 深堆叠时需要额外加归一化(如 RMSNorm)才稳。
    - **补救办法**:① 加门控,让"衰减 / 精确覆盖 / 跳过写入"可选择(Mamba-2 / GDN);② 混入少量全注意力层(H1/H2、7:1、3:1)——本质都是"固定状态不够,就花钱买回 $O(n)$ 的记忆"。
    - 参考:Linear Transformers [arXiv:2006.16236](https://arxiv.org/abs/2006.16236);Gated DeltaNet [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)("memory collisions" 的原文表述);关联回忆的反例可看 Zoology / associative recall 系列工作。

- [x] **Q4(p5)｜递推形式怎么来的?$y_t$ 是什么、什么形状?RNN 的好处与"duality"指什么?**
  > 原问(还原):这里具体的推导,为啥这个形式等价于上面的式子。这里的 $y_t$ 就是第 $t$ 位置的输出的 token 吗?$y_t$ 的形状是什么?以及像 RNN 的好处是什么?duality 指的是什么?
  - **推导(等价性)**:把"状态"定义为 $S_t=\sum_{i\le t}k_iv_i^\top$(形状 $d_k\times d_v$,列向量约定),则
    $S_t=S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t=\sum_{i\le t}(q_t^\top k_i)\,v_i^\top$。
    最后这个求和正是**线性注意力的输出**(对历史做加权求和),所以递推形式与原式等价。
  - **$y_t$ 是什么**:不是 token id,而是**该位置在这一层的注意力输出向量**;形状 $1\times d_v$(行向量;再经输出投影 $W_O$ 回到模型宽度 $C$)。token 要等到 LM head 把它投影到词表并采样之后才有。
  - **RNN 的好处**:推理时状态大小**恒定**($d_kd_v$,与序列长度无关),每步只需 $O(d_kd_v)$ 且**不需要随 $T$ 增长的 KV cache** ⇒ 长上下文 decode 不再是 memory-bound。
  - **duality(对偶)**:同一个层有**并行(矩阵)形式**和**串行(递推)形式**两种等价写法——训练用前者(二次但可并行,吃满 GPU),推理用后者(线性、每步常数)。讲义 P5 原话:"train efficiently (parallel, quadratic form) and inference efficiently (serial, linear form)"。
  - 顺带:把 $S_{t-1}$ 乘一个衰减 $\gamma$,就得到 RetNet 那一类"带遗忘的线性注意力"。
  - **追问(2026-10-10)｜把递推推导再展开一遍:$k_i$、$v_i$ 是什么、什么形状?**
    - **先定记号与形状**(以**单个头、单条序列**为例;真实的 batch 里整体形状是 $B\times H\times T\times d_h$,每个头内部做同一套运算):

      | 符号 | 形状                 | 含义 |
      | --- |--------------------| --- |
      | $x_i$ | $C\times1$(即 $d_{model}$) | 第 $i$ 个 token 的隐藏态(该层 LN 之后、投影之前) |
      | $q_i=W_Qx_i$ | $d_k\times1$      | 第 $i$ 个位置的 **query**:去"提问" |
      | $k_i=W_Kx_i$ | $d_k\times1$      | 第 $i$ 个位置的 **key**:它的"地址 / 标签",用来被别人检索 |
      | $v_i=W_Vx_i$ | $d_v\times1$      | 第 $i$ 个位置的 **value**:被检索到时**提供的内容** |
      | $S_t$ | $d_k\times d_v$    | 状态(关联记忆表),**与 $t$、与 $n$ 都无关** |
      | $y_t=q_t^\top S_t$ | $1\times d_v$      | 第 $t$ 个位置的注意力输出(行向量),之后过 $W_O$ 回到 $C$ |

      在标准多头注意力里 $d_k=d_v=d_h=C/H$(MLA 里两者可以不同);矩阵视角下 $k_i,v_i$ 分别是 $K\in\mathbb R^{n\times d_k}$、$V\in\mathbb R^{n\times d_v}$ 的**第 $i$ 行**,而本节公式把它们当**列向量**用(即第 $i$ 行的转置)——两者只差一个转置,数值一致。
    - **记号约定(2026-10-10 澄清)**：上表的"形状"列用的是**列向量**($q_i,k_i\in\mathbb R^{d_k\times1}$、$v_i\in\mathbb R^{d_v\times1}$),这样外积 $k_iv_i^\top\in\mathbb R^{d_k\times d_v}$、输出 $q_t^\top S_t\in\mathbb R^{1\times d_v}$ 的形状都自然成立(与讲义 P5 的 $S_t=S_{t-1}+k_tv_t^\top$、$q_t^\top S_t$ 一致)。若改用"行向量"写法,公式就要相应变成 $k_i^\top v_i$、$y_t=q_tS_t$——**同一个 $S_t$ 与 $y_t$,只差一个转置**;本节统一按**列向量**写。(早先把表格形状写成 $1\times d_k$ 与公式不一致,已改回 $d_k\times1$。)
    - **三种等价写法**(因果版本、不含 softmax):
      ① 逐位置求和:$y_t=\sum_{i\le t}(q_t^\top k_i)\,v_i^\top$ —— 每个历史位置贡献"相似度 × 它自己的 value"(标量 $q_t^\top k_i$);
      ② 矩阵形式:$O=\bigl(\mathrm{mask}\odot QK^\top\bigr)V$($\mathrm{mask}$ 为下三角 0/1 掩码),它的第 $t$ 行就是 ①;
      ③ 递推形式:$S_t=S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t$,初值 $S_0=0$。
    - **为什么 ③ 等于 ①(这就是"推导")**:先看状态的定义式
      $S_t=\sum_{i=1}^{t}k_iv_i^\top$,
      其中 $k_iv_i^\top$ 是 $d_k\times d_v$ 的**外积**,意为"把这一对 key-value 写进记忆表"。把求和拆出最后一项:
      $S_t=\sum_{i\le t-1}k_iv_i^\top+k_tv_t^\top=S_{t-1}+k_tv_t^\top$ ✓
      再看输出:
      $y_t=q_t^\top S_t=q_t^\top\sum_{i\le t}k_iv_i^\top=\sum_{i\le t}\bigl(q_t^\top k_i\bigr)v_i^\top$ ✓ 与 ① 逐项相同(标量 $q_t^\top k_i$ 可以提到求和里)。
      实测($n=5,d_k=3,d_v=2$):并行因果形式与递推形式的最大差 $4.4\times10^{-16}$(就是浮点舍入)。
    - **一个必须注意的坑**:讲义 P4 那条重排 $QK^\top V=Q(K^\top V)$ 是**双向(非因果)**版本——把整段的 $K^\top V$ 一次算好,等价于"每个位置都能看到未来"。**因果 LM 不能这么用**,必须用**前缀状态** $S_t=\sum_{i\le t}k_iv_i^\top$ 增量累加。实测同一组数据下,非因果版与因果版的最大差达 2.9(完全是两个东西)。这也解释了"训练要并行"该怎么办:用 **chunkwise / prefix-scan**——把序列切块,块内用矩阵形式、块间用递推把状态传下去。
    - **复杂度账**(单头、$n$ 个 token):直接算 $(QK^\top)V$ 需要 $n^2d_k+n^2d_v$ 次乘加;重排成 $Q(K^\top V)$ 只要 $2nd_kd_v$ —— 对 $n$ **线性**。实测 $n=5,d_k=3,d_v=2$:125 次 → 60 次。
    - **状态到底是什么**:$S_t$ 是一张 $d_k\times d_v$ 的"关联记忆表",第 $(a,b)$ 个元素表示"第 $a$ 个 key 维度与第 $b$ 个 value 维度的累计关联强度";$k_iv_i^\top$ 是"写一笔",$y_t=q_t^\top S_t$ 是"用 query 查表"。它是整段前缀的**固定大小充分统计量** —— 好处是推理不需要 KV cache、每步只要 $O(d_kd_v)$;代价是**容量只有 $d_kd_v$**,key-value 对超过它就会互相干扰(即 Q3 追问里的 memory collision)。
    - **"对偶性"再落一句**:同一层有两种**算法**算同一个**函数**——并行形式(全是 matmul,可并行,但含 $n^2$ 项或需要前缀扫描)与串行形式(每步一次外积 + 一次矩阵-向量乘)。训练取前者吃满 GPU,推理取后者拿到"每步常数、无 KV cache 增长"。
  - **追问(2026-10-10)｜线性注意力推理时就不需要 KV cache 了吗?那需要缓存什么?**
    - **你的理解正确,只需把"缓存"换个说法**:推理时确实**不再需要 KV cache**(不再存全部历史的 $K,V$),但每层每个头需要维护一个**固定大小的状态** $S\in\mathbb R^{d_k\times d_v}$,而且它是**每步读-改-写**(在线更新)的,不是只读缓存。
    - **每步只做三件事**(单头):① 用新 token 算出 $q_t,k_t,v_t$;② **更新状态** $S_t=S_{t-1}+k_tv_t^\top$(带门控的版本:$S_t=\gamma_tS_{t-1}+k_tv_t^\top$,或 GDN 的 delta 形式);③ **读出** $y_t=q_t^\top S_t$。
    - **与 KV cache 的三个本质区别**:
      - **大小**:KV cache 随序列长度线性增长($O(T\cdot H d_h)$ 个元素);状态是**常数**($O(Hd_kd_v)$ 个元素,与 $T$ 无关),$L$ 层合计也是常数;
      - **语义**:KV cache 是"把历史**原样存下来**"(无损,能回看任意位置);状态是"把历史**压成一个矩阵**"(有损压缩,容量 $\sim d$——见 ③ 里新增的那一问);
      - **访存/瓶颈**:KV cache 每步要读回 $O(Td)$ 字节(这正是 decode memory-bound 的来源);状态每步只读写 $O(d_kd_v)$ 字节(常数)⇒ 长上下文 decode 不再是 memory-bound ✓。
    - **训练 vs 推理**:训练用**并行/chunkwise** 形式,不需要"跨 step 的递推状态"参与反向(chunk 边界存少量状态检查点、chunk 内重算即可);真正需要维护状态的是**推理(decode)**。
    - **一句话**:线性注意力把"缓存整段历史"换成了"缓存一个固定大小的状态"——**KV cache 消失、状态缓存出现**,而且这个状态是每步都要改写的"活记忆"。

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
  - 出处:Dao & Gu 2024,Mamba-2 / SSD,[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)。(其中 "SSM / SSD / 半可分矩阵" 这几个概念见本笔记 ③ 的 **Q-L4-2**。)

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
  - **追问(2026-10-10)｜把 indexer 公式逐项拆开:$q$、$w$、$k$ 各是什么、什么形状、怎么得到的?**
    - **公式(论文式 (1))**:$I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\cdot\mathrm{ReLU}\bigl(q^I_{t,j}\cdot k^I_s\bigr)$。先注意记号:上标 $I$(讲义幻灯片写作 $l$)只是标记"这属于 indexer",**不是指数/幂**;$I_{t,s}$ 是分数(index score),别和单位矩阵混淆。
    - **先用标准张量形状对齐记号**:设本层输入为 $X\in\mathbb R^{B\times T\times d}$(单样本就是 $T\times d$),$X_t$ 表示第 $t$ 行的 $d$ 维向量。那么 **$h_t=X_t$**(当前正在处理的位置 $t$ 那一行),**$h_s=X_s$**,其中 $s$ 取遍**当前 token 能看到的历史位置**——因果下 $0\le s\le t$(含自身;论文里的 "preceding" 是宽泛措辞;如果你想只指"之前那些 token",写成 $s\lt t$ 即可)✓。

      | 符号 | 形状(单样本) | 含义 | 从哪来 |
      | --- | --- | --- | --- |
      | $h_t$、$h_s$ | $d$ | 当前 query token / 历史 token **进入本层**的隐藏态(与 MLA 同一份输入) | 上一层输出 |
      | $q^I_{t,j}$ | $d^I$ | 第 $j$ 个 indexer 头的**低维 query** | 由 $h_t$ 经轻量投影(实现上复用 MLA 的 query 低秩潜变量),FP8 |
      | $w^I_{t,j}$ | 标量 | 第 $j$ 个头的**权重** | 由 $h_t$ 经一个输出维为 $H^I$ 的线性层得到 |
      | $k^I_s$ | $d^I$ | 历史位置 $s$ 的 indexer key | 由 $h_s$ 投影得到,**所有 indexer 头共享同一份**(MQA 式) |
      | $I_{t,s}$ | 标量 | 该 $(t,s)$ 对的索引分数 | 上式算出,供 top-k 选择 |

    - **批形状(实现视角)**:$q^I\in\mathbb R^{B\times T\times H^I\times d^I}$(每个位置、每个头一个低维 query)、$w^I\in\mathbb R^{B\times T\times H^I}$(每个位置、每个头一个标量权重)、$k^I\in\mathbb R^{B\times T\times d^I}$(**不带头维**,因为所有头共享)、分数 $I\in\mathbb R^{B\times T\times T}$(第 $[b,t,s]$ 个元素是 token $t$ 看 token $s$ 的分数);top-k 沿最后一维取,并屏蔽 $s\gt t$ ✓。

    - **这个式子在做什么**:对每个头算一个**低维内积相似度** $q^I_{t,j}\cdot k^I_s$ → 过 ReLU → 按标量权重 $w$ 加权求和 ⇒ **"多个子空间相似度的加权和"**;产出的 $I_{t,:}$ 就是"当前 token 对每个历史位置的关注优先级"✓。
    - **为什么这么设计(四个取舍)**:
      - **ReLU 而不是 exp**:① 非负 ⇒ 分数可叠加、不需要 softmax 的分母开销;② 吞吐(论文原话 for throughput consideration);③ FP8 友好;
      - **多头 + 每头一个标量权重 $w$**:单个相似度表达力太弱;多个头看不同子空间,再用**数据相关的权重**组合 ⇒ 更强的相关性判别能力;
      - **key 在所有头间共享(MQA 式)**:indexer 的 key cache 只需"每 token 一个 $d^I$ 维向量"(而不是 $H^I$ 份)⇒ 长上下文下预填/解码都便宜;
      - **少头 + 低维 + FP8**:indexer 本身是一次 $O(L^2)$ 的预扫描,必须**远便宜于主注意力**(否则省不下来)——这就是 Q9 里那个"常数决定成败"的来源。
    - **每个量到底怎么得到的(两条线)**
      - **前向**:$(\{q^I_{t,j}\},\{w^I_{t,j}\})=f^I_Q(h_t)$、$k^I_s=f^I_K(h_s)$——就是两个**轻量线性投影**;历史侧的 $k^I_s$ 单独缓存 ✓。
      - **训练(这才是 DSA 最巧的地方)**:从 V3.1-Terminus 的 128K checkpoint 出发做**两阶段继续训练**——
        - **阶段 1 Dense Warm-up(初始化 indexer)**:保持**稠密注意力**,冻结除 indexer 外的全部参数;把主注意力分数**跨头求和**、再沿序列维 **L1 归一化**得到目标分布 $p_{t,:}$,用 **KL 损失**把 indexer 输出对齐到它:$\mathcal L^I=\sum_t D_{\mathrm{KL}}\bigl(p_{t,:}\,\|\,\mathrm{Softmax}(I_{t,:})\bigr)$;lr $10^{-3}$、1000 步 × 16 条 × 128K = **2.1B token** ✓;
        - **阶段 2 Sparse Training(全模型适应稀疏模式)**:引入 top-k 选择,优化全部参数;仍然对齐,但**只在被选中的集合** $S_t$ 上算 KL;且 **indexer 的输入被 detach**(从计算图断开)⇒ **indexer 只由 $\mathcal L^I$ 优化、主模型只由 LM loss 优化**(解耦训练);lr $7.3\times10^{-6}$、15000 步 × 480 条 × 128K = **943.7B token**,每 query 选 **$k=2048$** 个 KV ✓。
    - **为什么用 detach 而不是端到端**(讲义 P13 那张表就是这个问题):detach 侧训练稳(梯度不经"离散 top-k")、但 indexer 不是任务感知的,上界受 indexer 限制;端到端目标统一(只有 LM loss)、理论天花板更高,但要处理**离散选择的梯度**问题(STE / RL),稳定性差 ✓——这正是讲义注里说的"MoE 路由面临的是同一个问题"(对照 Q20-Q24)✓。

**Q8 追问补充：前向伪代码(带详细形状注释)**

```python
# ===== DSA(DeepSeek Sparse Attention)前向：indexer → top-k → 主注意力 =====
# 形状约定(公开配置)：B=batch, T=当前总长, d=模型宽, H=主注意力头数, d_h=每头维度
#                       H_I=indexer 头数(64), d_I=indexer 每头维度(128)
#                       k=每 query 选的 KV 条目数(2048), d_c=MLA 的 KV 潜维(512)
# 每层两份缓存：
#   cache_kI  : [B, T_prev, d_I]   indexer key（每 token 一个，头间共享）
#   cache_cKV : [B, T_prev, d_c]   MLA 的 KV 潜变量（主注意力真正要用的 KV 条目）

h = layer_input                              # [B, T, d] 本层输入；h[:, t, :] 就是 h_t；h[:, s, :] (s ≤ t) 是历史位置
                                             # 因果：位置 t 只能看 s ≤ t（含自身）

# ---- 1) indexer 的 query 侧：只依赖当前 token，逐位置独立 ----
q_I = W_qI(h)                                # [B, T, H_I, d_I]  低维 query（复用 MLA 的 query 低秩潜变量；FP8）
w_I = W_wI(h)                                # [B, T, H_I]       每个头一个标量权重

# ---- 2) indexer 的 key 侧 + MLA 潜变量：每个 token 各一个，追加进缓存 ----
kI_new  = W_kI(h)                            # [B, T, d_I]       注意没有 H_I 维（头间共享，MQA 式）
cKV_new = W_dkv(h)                           # [B, T, d_c]       MLA 的 KV 潜变量
cache_kI  = cat(cache_kI,  kI_new,  dim=1)   # [B, T_cur, d_I]
cache_cKV = cat(cache_cKV, cKV_new, dim=1)   # [B, T_cur, d_c]

# ---- 3) index score：I[b, t, s] = Σ_j w_I[b,t,j] · ReLU(q_I[b,t,j] · kI[b,s]) ----
dot = einsum(q_I, cache_kI, "... t h i, ... s i -> ... t h s")   # [B, T, H_I, T_cur]
dot = relu(dot)                                                  # [B, T, H_I, T_cur]
I   = einsum(dot, w_I, "... t h s, ... t h -> ... t s")          # [B, T, T_cur]  多头加权求和
I   = I.masked_fill(s_idx > t_idx, -inf)                         # 屏蔽未来（s > t）

# ---- 4) top-k 选择：每个 query 只留 k 个 KV 条目 ----
idx   = topk(I, k=k, dim=-1)                 # [B, T, k]         选中的 s 索引
c_sel = gather(cache_cKV, idx)               # [B, T, k, d_c]    只把这些条目的潜变量取出来

# ---- 5) 主注意力（MLA 的 MQA 模式）：只在这 k 个条目上算 ----
q     = W_q(h)                               # [B, T, H, d_h(+d_rope)]
c_sel = rope_on_non_latent_dims(c_sel, idx)  # 只有少数非潜维承载 RoPE（decoupled RoPE）
k_sel = einsum(c_sel, W_uk, "... s c, c h -> ... s h")   # [B, T, k, d_h]  头间共享（MQA）
v_sel = einsum(c_sel, W_uv, "... s c, c h -> ... s h")   # [B, T, k, d_h]
attn  = softmax(q @ k_sel.transpose(-1, -2) / sqrt(d_h)) # [B, T, H, k]    只有 k 列，不是 T 列
o     = attn @ v_sel                                      # [B, T, H, d_h]
o     = merge_heads(o) @ W_o                              # [B, T, d]       回到模型宽

# ===== 训练阶段：indexer 与主注意力分布的“对齐”（两阶段继续训练）=====
# 目标分布：p = L1_normalize( sum_over_heads(主注意力分数), dim=seq )   # [B, T, T_cur]（只保留 s ≤ t）
# 阶段1 Dense Warm-up（冻结除 indexer 外的全部参数）：
#     L_I = KL( p[b, t, s ≤ t] || softmax(I[b, t, s ≤ t]) )           # 逐 query 一个 KL
# 阶段2 Sparse Training（全量参数，且 indexer 输入 detach）：
#     把上面两处都限制在被选中的集合 S_t = top-k(I[b, t, :]) 上
```

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
  - **expert capacity(专家容量)**:每个专家在一个 batch 内**最多处理多少个 token**,通常写成 $c=\lambda\cdot\frac{kT}{N}$($\lambda$ 是容量因子,常见 1.0~1.25;$T$ 为 batch token 数、$N$ 专家数、$k$ 为 top-k)。
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
- [x] **Q-L4-1(延伸 2026-10-10)｜从公式/矩阵推导层面:线性注意力为什么表达力下降?为什么会 memory collision?代价到底是什么?**
  > 原问:开一个新问题,从数学公式、矩阵推导层面上,详细解释下为什么会表达力下降,为什么会 memory collision,linear attn 的代价是什么。
  - **一句话总括**:线性注意力把"整段前缀的注意力矩阵"压成了**两个低秩因子的乘积** $A=\phi(Q)\phi(K)^\top$,并把历史压进一个**固定大小的状态** $d_k\times d_v$——所以注意力矩阵的秩上界是 $d'$、状态的容量 $\sim d$;一旦要存的 key-value 对超过容量,就必然互相干扰(memory collision)。
  - **先补一句:$A=\phi(Q)\phi(K)^\top$ 是怎么来的、$\phi$ 是什么(以及与讲义 P4 的对应)**
    - **讲义 P4 写的是** $\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V$,并说"当 $\rho$ 是恒等映射"时就可以重排 $QK^\top V=Q(K^\top V)$——也就是**不做 softmax** 的版本(该页引用里也标了 "for kernel version":Shen 2018、Katharopoulos 2020)。
    - **一般写法把 $\rho$ 换成"可分解的相似度"**:$\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$,于是 $A=\rho(QK^\top)$ 变成 $A=\phi(Q)\phi(K)^\top$——**讲义那页就是 $\phi=\mathrm{id}$(恒等映射)的特例** ✓(此时 $A=QK^\top$)。
    - **$\phi$ 叫"特征映射"**:把每个 $q,k\in\mathbb R^{d_k}$ 映到 $\mathbb R^{d'}$。常见选择:恒等(讲义情形)、ReLU、$\mathrm{elu}(\cdot)+1$、$\exp$。想让 $A$ 直接当"权重"用需要 $\phi\ge0$;取恒等时 $A$ 里可能有负值,讲义用它只是为了说明"重排"这件事。
    - **为什么必须能分解**(这才是线性注意力的起点):注意力本质是 $y_i=\sum_j\mathrm{sim}(q_i,k_j)\,v_j$。只有 $\mathrm{sim}$ 能写成内积,才能把对 $j$ 的求和**移到里面**:
      $y_i=\phi(q_i)^\top\Bigl(\sum_j\phi(k_j)v_j^\top\Bigr)=\phi(q_i)^\top S$,其中 $S=\phi(K)^\top V\in\mathbb R^{d'\times d_v}$ **与 $n$ 无关**——这就是"线性"的来源;$A=\phi(Q)\phi(K)^\top$ 只是把同一件事写成矩阵形式(实测:$(A)V$ 与 $\phi(Q)(\phi(K)^\top V)$ 完全相等,差 $3.6\times10^{-15}$)。
    - **与 softmax 的距离**:softmax 的核 $\mathrm{sim}=\exp(q^\top k)$ 需要**无穷维**特征映射才能精确分解(Taylor 展开),有限维 $\phi$ 只能近似(Performer 用随机特征逼近)⇒ 这也是一条"表达力下降"的来源——**你只能在一个有限维子空间里度量相似度**。
    - **两个视角别混淆:$A=\phi(Q)\phi(K)^\top$ 只用于"分析",实现里从不把它算出来**(你问的这一点)。它是**概念上的注意力矩阵**——只有写出来才能谈秩/自由度/表达力(见下面 ①);而**真正的计算顺序是"先合 KV、后用 Q"**:
      $S=\phi(K)^\top V\in\mathbb R^{d'\times d_v}$(代价 $n d'd_v$)→ $O=\phi(Q)S\in\mathbb R^{n\times d_v}$(代价 $n d'd_v$)。
      两者相等靠的是矩阵结合律:$\bigl(\phi(Q)\phi(K)^\top\bigr)V=\phi(Q)\bigl(\phi(K)^\top V\bigr)$ ✓(实测差 $1.8\times10^{-15}$)。因果版本同理:$S_t=\sum_{i\le t}\phi(k_i)v_i^\top$ 只累加 K/V 侧,$q_t$ 只在读取时出现 —— 所以**"Q 和 K 相乘"只出现在讲"表达力"那一步(概念分析),不出现在实现里** ✓。
    - **代价对比**(实测 MACs,$d'=d_v=128$):$n=512$ 时 $6.7\times10^7$ vs $1.7\times10^7$(4×)、$n=4096$ 时 $4.3\times10^9$ vs $1.3\times10^8$(**32×**)、$n=32768$ 时 256×;反过来,当 $n\lt 2d'd_v/(d'+d_v)\approx d'$(本例约 128)时,物化 $A$ 反而更便宜 ⇒ **"线性注意力"只在长序列($n\gtrsim d'$)才划算** ✓。
  - **① 表达力下降的精确来源:注意力矩阵被迫低秩**
    - 线性注意力(kernel 形式):$A=\phi(Q)\phi(K)^\top$,其中 $\phi(Q),\phi(K)\in\mathbb R^{n\times d'}$。由秩不等式 $\mathrm{rank}(UV^\top)\le\min(\mathrm{rank}U,\mathrm{rank}V)$ 得 $\mathrm{rank}(A)\le\min(n,d')$——**关键是有个与 $n$ 无关的上界 $d'$**(实践里 $d'$ 取头维 $d_h$ 或与它同量级;讲义 $\phi=\mathrm{id}$ 时 $A=QK^\top$ 且 $d'=d_k$)。
    - **顺带把"秩"说清楚(定义 + 直观)——这是理解"表达力/容量"的钥匙**
      - **定义(三种等价说法,任选一种记)**:① **线性无关的列数**(等于线性无关的行数,两者恒相等);② **像空间维数**:映射 $x\mapsto Ax$ 的所有可能输出张成的空间维数,也就是"$A$ 的输出只能住在一个多低维的子空间里";③ **最小的 $r$**,使得 $A=UV^\top$ 且 $U\in\mathbb R^{m\times r},V\in\mathbb R^{n\times r}$(等价地:**非零奇异值的个数**)。
      - **直观 1:秩 = "这堆数里真正独立的方向有几个"**。$m\times n$ 个数字看似有 $mn$ 个自由度,但秩告诉我们只有 $r$ 个方向是独立的,其余都是它们的线性组合。
      - **直观 2:秩 1 = 一个外积**。$kv^\top$ 的每一行都是 $v^\top$ 的倍数(实测 $4\times3$ 外积的秩 = 1,任意两行逐元素比完全一致);反过来,任何秩 1 矩阵都能写成一列乘一行。所以 $S=\sum_ik_iv_i^\top$ 就是"把若干个秩 1 的块叠起来"——**秩的上限就是能叠加的独立块数**。
      - **直观 3:低秩 = "通道被卡脖子"**。$A=UV^\top$ 意味着"所有输出都必须由 $U$ 的 $r$ 个方向拼出来" ⇒ 无论 $n$ 多大,表达能力都被压在这 $r$ 维里。对注意力矩阵:$A\in\mathbb R^{n\times n}$ 有 $n^2$ 个数,softmax 版秩可达 $n$,而线性注意力 $\mathrm{rank}\le d'$ ⇒ 当 $n\gt d'$ 时**必然存在它表示不出的注意力图案**(例如"把第 1、3、5…个位置分别挑出来、各自配任意权重")。
      - **数值读法**:秩 = 非零奇异值个数(实测 $6\times3$ 随机矩阵秩 3,奇异值 $3.293,1.864,1.234$ 全非零);实际中还要看奇异值谱的"有效秩"(能量/参与比),因为极小的奇异值虽非零但影响有限。
      - **对状态容量的直接含义**(呼应下面 ②):$S=\sum_{i\le t}k_iv_i^\top$ 的秩 $\le\min(d_k,d_v)$,**与 $t$ 无关**。实测:存 10 对时 rank = 3(状态 $4\times3$)、存 100 对时 rank = 8(状态 $8\times8$)、存 **4096 对时 rank 仍然只有 64**(状态 $64\times64$)——即状态最多只能"独立地"记住 $\min(d_k,d_v)$ 对关联,再多的只能重叠(碰撞)✓
    - softmax 注意力:$A=\mathrm{softmax}(S)$ 可以是**任意正的行归一化矩阵**——给定任意正矩阵 $T$,取 $S=\log T$ 即得 $\mathrm{softmax}(S)=T/\text{rowsum}(T)$(即每行除以该行之和),也就是 $n^2$ 个自由度、秩可达 $n$。
    - 实测($n=6,d'=3$):softmax 注意力秩 = 6,线性注意力秩 = 3 ✓。**"自由度 $n^2$ vs $nd'$"就是表达力差距的矩阵层面表述**。
  - **② memory collision 的精确机制:检索时多出一个互扰项**
    - 状态 $S=\sum_{i=1}^{t}k_iv_i^\top$ 就是一张关联记忆。用第 $j$ 个 key 去检索:
      $k_j^\top S=\|k_j\|^2v_j^\top+\sum_{i\ne j}(k_j^\top k_i)v_i^\top$ —— **第一项是信号**(要取回的 $v_j$),**第二项是互扰**(其它 key-value 对混进来的干扰,cross-talk)
      (实测这个拆分与直接计算一致,差 $10^{-15}$)。
    - **要精确取回 $v_j$ 就必须让互扰项为 0 ⇒ 要求所有 key 两两正交**;而 $d_k$ 维空间里最多只有 $d_k$ 个互相正交的向量 ⇒ **容量上界 $\sim d_k$**。实测:$k$ 取正交向量时检索误差**恰好为 0**;取随机 key 时误差 2~22(基本取不出来)。
    - **互扰的量级(这就是"memory collision"的量化)**:把 key 取成单位随机向量,则 $k_j^\top k_i\approx\mathcal N(0,1/d)$,互扰是 $t$ 个独立项叠加 ⇒ 范数 $\approx\sqrt{t/d}\,\|v\|$,于是**相对检索误差 $\approx\sqrt{t/d}$**。实测($d=32$):$t/d=0.2,1,4,16,64$ 对应误差 $0.35,0.99,1.99,4.04,8.10$,与理论值 $0.50,1.00,2.00,4.00,8.00$ 基本吻合 ✓。
    - **结论**:只要 $t\gtrsim d$(序列长超过头维),检索就开始失真——这解释了为什么线性注意力在关联回忆/复制/长文档检索上明显弱,而在普通语言建模(不需要精确检索)上差距不大。
  - **③ 代价清单(完整版)**
    - **表达力**:每层注意力矩阵 $\mathrm{rank}\le d'$;状态容量 $\sim d$ ⇒ 精确检索、长程复制、in-context 取数类任务变差(靠门控/混合层补)。
    - **数值稳定性**:softmax 输出是**凸组合**(权重和为 1、范数有界),线性注意力是**未归一化求和**(量级随 $t$ 增长)⇒ 深堆叠需要额外归一化(RMSNorm)或衰减门控。(实测输出范数在 0~5.7 之间乱跳,而 softmax 稳定在 0.7~1.0。)
    - **训练并行度**:虽然能重排成线性复杂度,但**因果版本不能直接用一次 $K^\top V$**——必须 chunkwise/prefix-scan;而且门控(如 GDN 的 $\beta_t,\gamma_t$ 依赖输入)让"前缀和"不能简单并行 ⇒ 实现复杂度高于注意力。
    - **不可逆性**:状态是有损压缩,不像 KV cache 能回看任意位置;想"精确回看"就必须混入全注意力层。
    - **收益端(对照着看)**:推理每步 $O(d_kd_v)$ 常数、无随 $T$ 增长的 KV cache;prefill $O(n)$;长上下文下显存与吞吐占优 —— 这就是"混合层"成为当下主流的原因:用少量全注意力保住检索能力,其余用线性层换效率。
  - 参考:Gated Delta Net 论文里 "memory collisions" 的原文表述 [arXiv:2412.06464](https://arxiv.org/abs/2412.06464);容量/互扰这套分析也见于 fast weight / associative memory 系列工作。
- [x] **Q-L4-2(延伸 2026-10-10)｜Mamba-2 里的 "state space model(SSM)" 与 "state space duality(SSD)" 到底是什么?**
  > 原问:在讲义中新开一个问题,补充解释下 mamba-2 论文中提到的 state space model, state space duality 等概念的含义。
  - **一句话**:SSM 就是"**带矩阵状态的线性递推**"这种序列模型的标准写法;而 SSD 是 Mamba-2 论文用来证明"**这类 SSM 与某些注意力变体是同一个东西的两种面孔**"的框架——桥是**半可分矩阵**。
  - **① SSM(state space model):从控制论搬过来的标准形式**
    - 连续时间线性系统:$h'(t)=Ah(t)+Bx(t)$、$y(t)=Ch(t)+Dx(t)$;深度学习里用它的**离散化**版本:$h_t=\bar Ah_{t-1}+\bar Bx_t$、$y_t=Ch_t+Dx_t$ —— 就是一个**状态维为 $N$ 的线性递推**($N$ 与序列长度无关)。
    - **同一个层的三张面孔**(三种等价算法):① **递推**:逐步更新,每 token $O(N)$、常数状态;② **卷积**:参数时不变时 $y=x*\bar K$,核 $\bar K=(C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots)$(可用 FFT);③ **矩阵形式**:$y=Mx$,其中 $M$ 是**半可分矩阵**($M_{ij}=C_iA_i\cdots A_{j+1}B_j$ 当 $i\ge j$,否则 0)。
    - **Mamba 系列的取舍**:Mamba-1 让 $\bar A,\bar B,C,\Delta$ **依赖输入**(selective:按内容决定记住/忘掉什么),代价是卷积形式失效、只能 scan;Mamba-2 把 $\bar A$ 简化为**标量 × 单位阵**($\alpha_tI$),于是矩阵形式变成 **1-半可分矩阵** ⇒ 可以分解成矩阵乘 ⇒ 快 2~8× ✓。
  - **② SSD(state space duality):SSM 与注意力的对偶框架**
    - 论文标题就是《Transformers are SSMs》。SSD 用**结构化矩阵**(参数与乘法复杂度都是次二次的矩阵)做桥梁:证明 **SSM 等价于一族半可分矩阵**;把线性注意力推广成 **structured masked attention(SMA)**;再证明 **SSM 与 SMA 有大交集、互为对偶**:同一个层同时拥有"SSM 式线性形式"与"注意力式二次形式" ✓。论文还顺带证明了:**任何具备快速递推形式的核注意力,必然是一个 SSM** ✓。
    - **最简形式的对偶**(最该记住的一条):把 $A_t$ 固定成恒等($\alpha_t=1$),SSM 就退化**正是因果线性注意力**——$B$ 当 key 投影、$C$ 当 query 投影、输入 $X$ 当 value ✓;Mamba-2 只是给这个线性注意力加了一个**标量衰减** $\alpha_t$ ✓。
    - **对偶的二次形式**(Mamba-2 的注意力形式):$y=\bigl(L\circ QK^\top\bigr)V$,其中 $L_{ij}=a_i a_{i-1}\cdots a_{j+1}$(当 $i\ge j$),否则为 0。与标准 softmax 注意力只差两点:① **去掉了 softmax**;② 多了**逐元素乘的衰减掩码 $L$**——论文把它解释为"一种**数据相关的相对位置编码**",取代 Transformer 里启发式的位置嵌入 ✓。
    - **为什么这个对偶很重要**(工程价值):① 训练用二次形式(全是 matmul + chunkwise 分块)⇒ 比 Mamba-1 的 scan 快 2~8×;② 推理用线性形式(每步 $O(1)$、状态常数)⇒ 与线性注意力同样的收益;③ 把 S4/Mamba/H3/RetNet/GLA/线性注意力等整合成同一张"结构化掩码注意力"谱系图 ⇒ "在 SSM 与 attention 之间插值"变成一个连续的设计空间。
    - **代价**:$A_t$ 从对角简化为标量、head dim 从 1 提到 64/128 ⇒ **表达力略降、换训练效率**(论文原话:slightly decreasing the expressive power in return for significant training efficiency improvements)。
  - **③ 别和 Q4 里的 "duality" 搞混**(两个不同层次的对偶)
    - **Q4 的 duality(算法层)**:同一个层有"并行矩阵形式"与"串行递推形式"两种**算法**;
    - **SSD 的 duality(模型/函数类层)**:一**类**模型(结构化 SSM)与另一**类**(注意力变体)是**同一个东西**——它更强,说的是"某些层既能用 SSM 描述、也能用注意力描述"✓。
    - 关系:线性注意力的"并行/串行对偶"是 SSD 的一个特例 ✓。
  - 参考:Mamba-2 / SSD 论文 [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)(摘要、§3 半可分矩阵、§4 SMA、§5 对偶);"递推展成矩阵形式"的直观推导可在搜到的二手资料里找到那形式 $\bigl[\alpha_{i:j}\bigr]\circ(CB^\top)X$ 的下三角矩阵写法。
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
- **在线学习视角**:记衰减后的状态为 $\tilde S_{t-1}=\gamma_tS_{t-1}$,展开一步可得 $S_t-\tilde S_{t-1}=-\beta_t\bigl(\tilde S_{t-1}k_t-v_t\bigr)k_t^\top$ —— 它恰好是对 $\tfrac12\|Sk_t-v_t\|^2$ 的一步梯度下降($\beta_t$ 当学习率、$\gamma_t$ 相当于权重衰减;按 GDN 论文的状态排列)⇒ 这就是"把状态当成一个小模型、在推理时在线更新"(fast weights / test-time training)那一族思想。
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
