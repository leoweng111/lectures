# Lecture 03 · 学习笔记(问题记录与 AI 解答)

（本文件是 lecture_03.md 的**上半部分**：原文件第 1–108 行，即 ① 的 A–D 小节、Q1–Q11。原文件第 109 行起（### E. 训练超参与稳定性及之后）会让 PyCharm 预览的公式渲染整体失效，正在定位；需要时可让我再生成 lecture_03_2.md 存放那部分。）

- 讲义:仓库根目录 `lecture_03.pdf`(静态讲义,共 67 页;PDF 阅读器打开)
- 状态:进行中(看完后在 [progress.md](progress.md) 更新)
- 说明:你的问题以 **#TODO** 标注在 PDF 批注层。批注文本在导出时发生了编码损坏(UTF-8 中文被误读),无法 100% 逐字还原,下列"原问"是我**按语义还原的表述**,措辞如有出入请以你在 PDF 中的原始标注为准、并告诉我修正。

> 主题速览:Lecture 03 = LM 架构与超参的"业界共识"——pre/post-norm、LayerNorm vs RMSNorm、门控激活、serial/parallel 块、位置编码(RoPE)、d_ff/头数/宽深比/词表等超参惯例,以及 z-loss / QK-norm / logit soft-cap 等稳定性技巧、GQA/MQA 与混合注意力。

---

## ① 卡住的地方(PDF #TODO,全部已解答)

### A. 归一化与残差结构

- [x] **Q1(P12)｜为什么 post-LN 会让梯度不稳定,而 pre-LN 不会?**
  > 原问(还原):这里不是很理解为什么 post-LN 会使梯度不稳定,而 pre-LN 就不会。
  - **一句话直觉**:post-LN 把归一化放在"加法路径上",残差捷径被 LN 的缩放反复污染,梯度像穿过层层衰减器传回浅层;pre-LN 把 LN 挪到支路里,捷径保持恒等,**梯度有一条干净的高速公路**。
  - **公式化对比**(块输出 $y$,输入 $x$,块函数 $f$):
    - post-LN:$y=\mathrm{LN}(x+f(x))$,反向 $\dfrac{\partial y}{\partial x}$ 要穿过 $\mathrm{LN}$ 对 $x$ 与 $f(x)$ 两者的缩放;
    - pre-LN:$y=x+f(\mathrm{LN}(x))$,反向 $\dfrac{\partial y}{\partial x}=1+\dfrac{\partial f}{\partial x}\cdot\cdots$——恒等项 1 保证梯度至少以 1 传到更浅层(类似 ResNet 的 $\dfrac{\partial L}{\partial x_l}=\dfrac{\partial L}{\partial x_L}(1+\text{stuff})$ 展开)。
  - **讲义给的两种机制**(Xiong et al. 2020「On Layer Normalization in the Transformer Architecture」,https://arxiv.org/abs/2002.04745;Salazar & Nguyen 2019):① **gradient attenuation**:post-LN 中 LN 的缩放使深层的梯度信号在回传时被逐层衰减,浅层学不动;② **gradient spikes**:post-LN 训练早期容易出现梯度尖峰,需要小心的小 LR/warmup。pre-LN 的附加红利:稳定到可以去掉 warmup、用更大 LR。
  - 补充(2024 共识):pre-norm 让 LN 不影响主残差信号路径;个别例外如 OPT-350M 仍用 post-LN。诚实声明:pre/post 的完整机理近年仍有讨论,上述"残差捷径 + 梯度衰减"是主流且被讲义采用的理解。

- [x] **Q2(P14)｜LayerNorm 的完整公式、batch 形状下的计算过程、以及与 RMSNorm 的本质区别**
  > 原问(还原):详细的 layernorm 公式,配合具体 batch 的样本形状变换和计算说明,以及和 RMSNorm 真正本质的区别。
  - **LayerNorm 公式**(对每个样本、每个特征向量归一化):
    $\mathrm{LN}(x)=\gamma\odot\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\dfrac1d\sum_{j=1}^d x_j,\quad \sigma^2=\dfrac1d\sum_{j=1}^d (x_j-\mu)^2$
    其中 $\gamma,\beta\in\mathbb{R}^d$ 是可学习缩放/偏移;**归一化统计量在特征维 $d$ 上算,与 batch 无关**(这是它与 BatchNorm 的本质区别)。
  - **配合 batch 形状算一遍**:设激活 $X\in\mathbb{R}^{B\times S\times d}$(batch × 序列 × 特征)。LN 对每个 $(b,s)$ 行向量独立做:先按行求 $\mu_{b,s}$、$\sigma^2_{b,s}$(对 $d$ 维求和),再逐元素标准化、乘 $\gamma$ 加 $\beta$。输出形状不变 $B\times S\times d$;FLOPs 量级 $O(B\cdot S\cdot d)$(算均值/方差各一遍 + 归一化),但注意它需要**两次读遍数据**(先算统计量、再归一化),这正是它 runtime 偏高的原因之一(见 P16 你的批注)。
  - **RMSNorm 公式**(Zhang & Sennrich 2019,https://arxiv.org/abs/1910.07467):
    $\mathrm{RMSNorm}(x)=\dfrac{x}{\sqrt{\frac1d\sum_j x_j^2+\epsilon}}\odot\gamma$
    只做"按均方根缩放",**不减均值、不加 bias($\beta$)**。
  - **本质区别**:LayerNorm 移除的是"均值偏移 + 方差尺度"两个自由度;RMSNorm 只处理"尺度"一个自由度——它假设**均值项对深层 LM 收益低**(信号已在残差中零均值化),砍掉 mean/bias 后参数更少、少一次全局归约(runtime 更快),实证性能基本持平(讲义 15-17 页:Narang et al. 2020 还观察到偶有性能提升)。注意 RMSNorm 仍保留 $\gamma$。
  - 记忆:**LayerNorm = 减均值 + 除方差 + 可学 γ/β;RMSNorm = 只除 RMS + 可学 γ**,差在"去不去均值、要不要 bias",都是**按样本按特征维**归一化、与 batch 无关。

- [x] **Q3(P13 批注,作为 Q1 的延伸)｜"训练不稳定就多加几个 LayerNorm,就会稳定一些"**
  > 原问(还原):如果训练不够稳定,那么就多加几个 layernorm,然后训练就会稳定一些。
  - 是的——这就是 P13 的 **double-norm / 非残差 post-norm**:既然把 LN 放进残差流有害,那就把第二个 LN **放到残差流之外**(块的输出侧),如 Grok、Gemma 2(OLMo 2 只做非残差 post-norm)。作用:在不破坏捷径的前提下,额外压一次激活尺度,抑制大 logit/大激活引起的尖峰。要点:加在"支路/输出"上有效且不伤梯度;**别加在残差主路径上**(那是 post-LN 的老问题)。

### B. 激活与 FFN / 块结构

- [x] **Q4(P28)｜parallel 块是什么?与经典 serial 块的区别?(以 GPT-J 说明)**
  > 原问(还原):parallel 是啥?和经典的 serial 架构区别?就用这个 GPT-J 的架构具体说明。
  - **serial(经典,绝大多数模型)**:块内先 attention 后 FFN,各自过归一化并逐段加残差:
    $x' = x+\mathrm{Attn}(\mathrm{LN}_1(x)),\quad x''=x'+\mathrm{FFN}(\mathrm{LN}_2(x'))$(两步,两个 LN,两个加法)。
  - **parallel(GPT-J / PaLM / GPT-NeoX)**:两个分支**同时**从同一个归一化后的输入出发,结果一起加回:
    $y = x+\mathrm{Attn}(\mathrm{LN}(x))+\mathrm{FFN}(\mathrm{LN}(x))$——只有一个 LN(两分支共享),attention 与 FFN 的矩阵乘可**融合成一次大 matmul**(GPU 利用率更好、kernel 启动更少)。
  - 直觉:把"先 A 后 F"的两段流水改成"A、F 并排",层内时延更短、硬件吞吐更高;代价是同一输入同时喂两个分支,两者对残差的贡献可能相互干扰(并行块在部分实验中表现略逊,所以现在主流回到 serial)。讲义原文:近期用 parallel 的只有少数模型(GPT-J/PaLM/NeoX 及个别新模型),serial 是主流。
  - 参考:GPT-J-6B(Wang & Komatsuzaki 2021,https://github.com/kingoflolz/mesh-transformer-jax)

- [x] **Q5(P37)｜$d_{ff}$(d_ff)具体是什么?**
  > 原问(还原):dff 具体是啥?这个 4 倍应该是经验之谈吧?
  - **定义**:$d_{ff}$ = FFN 中间层(隐藏层)的宽度。标准 FFN 为 $d_{model}\to d_{ff}\to d_{model}$(两个线性层夹一个激活),如 $\mathrm{FFN}(x)=\sigma(xW_1)W_2$,其中 $W_1\in\mathbb{R}^{d_{model}\times d_{ff}},\ W_2\in\mathbb{R}^{d_{ff}\times d_{model}}$。
  - **共识值**:$d_{ff}=4\,d_{model}$(讲义 P37);GLU 类门控激活因多一路投影、为控制参数量会缩到约 $\frac83 d_{model}$(≈2.67,讲义 P23/P38,LLaMA/Qwen/DeepSeek 等都在 2.5~3.5 区间)。
  - **"4 倍是经验之谈吗?"**:是经验规则,但**有依据**(讲义 P40):Kaplan et al. 2020 显示 $d_{ff}/d_{model}$ 在 1~10 之间有个"近似最优盆地",4 在其中且最简单;它不是硬约束——T5-11B 用 64 倍也能训(GPU 效率考量),但其改进版 T5 v1.1 又回到 2.5 倍(说明 64 倍非最优,P41)。
  - 参考:https://arxiv.org/abs/2001.08361(Kaplan et al. 2020)

### C. 位置编码:RoPE 深入

- [x] **Q6(P32)｜RoPE 的底层原理**
  > 原问(还原):RoPE 的详细底层原理理解。
  - **目标**:让注意力只依赖相对位置 $i-j$。形式化:要找 $f(x,i)$ 使 $\langle f(x,i),f(y,j)\rangle=g(x,y,i-j)$(讲义 P31)。绝对位置相加做不到(内积出现非相对交叉项),正弦位置编码也做不到(有交叉项),而**内积对旋转不变**——所以把 $x$ 按位置 $i$ 旋转一个角度,内积就会只差"旋转角之差"。
  - **做法**:把 $d$ 维向量按坐标**两两配对**成 $d/2$ 个二维平面,第 $i$ 个 token 的第 $m$ 对坐标旋转角度 $m\theta_m$($\theta_m$ 是随维度递减的频率,类似正弦编码的波长):
    $q'_{2m}=\cos(m\theta_m)\,q_{2m}-\sin(m\theta_m)\,q_{2m+1},\qquad q'_{2m+1}=\sin(m\theta_m)\,q_{2m}+\cos(m\theta_m)\,q_{2m+1}$
    对 $Q,K$ 都旋转后做点积:旋转矩阵正交 → $\langle \mathrm{Rot}(i)q,\ \mathrm{Rot}(j)k\rangle=\langle q,\ \mathrm{Rot}(j-i)k\rangle$,**只差 $j-i$** ✓。
  - **频率怎么取**(Su et al. 2021,https://arxiv.org/abs/2104.09864):$\theta_m=10000^{-2m/d}$(长上下文常把底数调大,如 500k context 用 $10^6$ 量级)。
  - 与正弦/绝对的区别:P34 讲义:RoPE 是**乘性**的(旋转)、无加法交叉项;而且它作用在 **Q/K 上、每次注意力计算前**,不是在 embedding 层加一次(见 Q8 的 Note)。
  - 直觉收尾:旋转角 = 位置"时差",频率 = "刻度";低维转得快(分辨近距)、高维转得慢(覆盖远距)。

- [x] **Q7(P33)｜Gemma 的 p-RoPE 与原始 RoPE 的区别/改进**
  > 原问(还原):Gemma 的 p-RoPE 和原来的 RoPE 的区别?改进点?
  - 诚实声明:讲义此页只有一句 "Gemma 4 alternative: just first 2",未展开;以下结合公开资料,但**Gemma 4 的确切实现细节我无法完全核实**,给你当前最被认可的解读。
  - **p-RoPE = partial/proportional RoPE(部分旋转 RoPE)**:原始 RoPE 旋转全部 $d/2$ 对坐标;p-RoPE 只旋转**一部分**坐标对(讲义图示为"只旋转前 2 个/少数字段"),其余维度不旋转(起 NoPE 作用)。
  - 公开资料里的两种变体语义:① 经典 partial RoPE(GPT-NeoX/ChatGLM 式):只旋转前 $r\cdot d$ 维,**频率序列也压缩进更小的旋转维度**;② Gemma 系 p-RoPE 变体:旋转维度仍是部分,但**频率分母保留完整的 $d_{head}$**,并且**截掉最低频(转得最慢、长上下文下漂移最大)的那几档频率**。
  - **改进动机**:长上下文下,超低频率维度旋转过慢会产生"位置信号漂移/语义通道被位置信息污染"的问题;少旋转一些维度 = 让部分通道**纯粹编码语义(NoPE)**,同时省下旋转的计算;代价是相对位置精度略降(仍靠剩余频率提供)。
  - 结论:本质是**"用一部分通道换位置、其余通道专注语义"的工程折中**,主要用于超长上下文稳定性与效率;若你需要精确到 Gemma 4 配置(旋转哪些维、频率底数多少),建议以 Gemma 4 技术报告/权重 config 为准,我暂不编造具体数字。

- [x] **Q8(P35)｜讲义代码注 "Note: embedding at each attention operation to enforce position invariance" 是什么意思**
  > 原问(还原):这个 Note 是什么意思?
  - 直译:**在"每一次注意力计算处"做位置注入,以强制位置不变性**。即 RoPE 不是像绝对位置编码那样在输入 embedding 处加一次位置向量,而是**在每个 attention 层里、对当层的 $Q$ 与 $K$ 实时旋转**(带位置角)。
  - 为什么必须这样:相对位置性质要在"任意两个位置做内积"那一刻才成立;若只在 embedding 加一次绝对位置,经过多层非线性与残差相加后,会产生绝对位置交叉项、相对性质被破坏。逐层在 Q/K 上旋转,才能保证**每一层**的注意力都只看到 $i-j$。
  - 和 P34 呼应:RoPE 是"乘性、无交叉项";"在每层注意力注入"正是它与 additive 方案的又一关键差异。

### D. 超参:头数、宽深比、词表

- [x] **Q9(P42)｜这一页没看懂:head_dim × num_heads 与 d_model 的关系**
  > 原问(还原):这页没看懂。
  - **它在问什么**:常规多头注意力里,习惯上把 $d_{model}$ **均匀切给 $h$ 个头**:每个头维度 $k=d_{model}/h$,于是 $k\times h=d_{model}$(比例 ratio=1)。这页想说:**这个相等并不是数学必然**——可以设计 $k\times h\ne d_{model}$(如每个头独立投影、总宽度大于 $d_{model}$)。
  - **怎么读表**:ratio = $(k\times h)/d_{model}$。GPT-3/LLaMA2/T5 v1.1 等 ratio≈1(最主流);PaLM 1.48、LaMDA 2、Qwen 3.5 27B 1.2、T5 16(极端)等说明 Google 系模型常让注意力总宽度大于模型宽度(相当于把更多参数量/FLOPs 花在注意力上)。
  - **要点/结论(讲义)**:① 大多数模型遵守 $k\cdot h=d_{model}$(实现最省事,张量形状好组织);② 但这只是**惯例而非定律**,偏离 1 的模型也不少;③ 论文里对"偏离是否带来收益"**验证很弱**(低验证/low-to-no validation)——所以默认按 ratio=1 做即可。你如果卡在"为什么有人偏离",答案就是:多给注意力一点宽度通常无害、个别工作认为有帮助,但证据不硬。

- [x] **Q10(P44)｜aspect ratio(宽深比)指什么?**
  > 原问(还原):模型的 aspect ratio 指的是什么?
  - **定义**:把模型在"多深(n_layers)还是多宽(d_model)"上的选择量化成比例:**aspect ratio = $d_{model}\,/\,\text{n_layers}$**(讲义表即此;也有用参数量/深度表达)。宽深乘积大致决定参数量,比例决定"形状"。
  - **数据**:主流模型这个比值大多落在 61~205(BLOOM 205、GPT-3/LLaMA 100-128、Gemma 4 61),说明大家都不约而同选了"中等偏宽"而不是极端深。
  - **为什么存在甜蜜点**:① 太深(比值过小):难以并行化、层间依赖串行延迟高(Tay et al. 2021,https://arxiv.org/abs/2104.07636),且深模型对归一化/残差稳定性更敏感;② 太宽(比值过大):同样参数量下表达"层数多样性"变少,效果也下降;③ 系统/效率约束常比精度更早定死这个值(P45 你的批注:在 width 上 scale 比在 deep 上 scale 简单、更好组织)。
  - 补充:Kaplan et al. 2020 与 Tay et al. 2021 都做过"同样参数量,深/宽曲线"的实验,结论是中等比例附近最稳(P46 两张图)。

- [x] **Q11(P47)｜词表大小会影响什么?这两张表的区别是?**
  > 原问(还原):词表大小会影响什么?这两张图区别是啥?
  - **"两张图"应是 P47 左右两张表**:左表 = **单语模型**(GPT/T5/LLaMA 等,vocab 约 3 万~10 万,主流 30-50k);右表 = **多语/生产系统**(mT5/PaLM/Gemma 4/DeepSeek/Qwen,vocab 10 万~26 万)。
  - **词表大小影响什么**(与 Lecture 01 的 tokenizer 分析接上):
    1. **参数量与内存/带宽**:embedding 与输出头都是 $V\times d$ 的参数(softmax 每 token 还要过一遍 $V\times d$ 的 matmul),$V$ 直接乘进去;
    2. **压缩率与序列长度**:$V$ 大→单 token 可打包更多字节→序列短、注意力省( Lecture 01 的 compression ratio);
    3. **稀疏性与学习效率**:$V$ 太大→大量 token 出现频次极低、统计学不牢,embedding 尾部浪费(长尾稀疏);
    4. **覆盖面**:多语言需要覆盖更多脚本/字符组合,所以 $V$ 必须大(单语 30-50k 就够);这就是左右两表差异的根本原因。
  - **工程权衡**:单语选 ~30-50k;多语/生产系统 100-250k;再往上收益递减、成本线性涨。例:LLaMA 32k、GPT-2/3 50k、PaLM 256k、Gemma 4 262k。

