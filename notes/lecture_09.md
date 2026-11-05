# Lecture 09 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_09.pdf`(静态讲义,共 57 页;用 PDF 阅读器/浏览器打开)
- 状态:进行中(①「卡住的地方」13 条 `#TODO` 已于 2026-11-04 全部作答并勾选;看完后在 [progress.md](progress.md) 更新)
- 说明:本讲 PDF 共 56 条批注,其中 `#TODO` 13 条。答案写在题目下方;较长推导与延伸阅读放在文末「AI 补充笔记」的 2026-11-04 一节

> **主题速览**:Lecture 9 = **scaling laws 基础**。
> Part 1 数据 scaling 的**历史与理论**:从 VC 上界、Banko & Brill(2001)、Hestness(2017)到"log-log 直线";理论侧用"均值估计 → $1/n$"与"非参数回归 → $n^{-1/d}$ 量级"解释幂律。
> Part 2 **模型工程**:架构(LSTM vs Transformer)、优化器(Adam vs SGD)、深度/宽度、batch size(**critical batch**)、学习率(**μP**);核心手法是"小模型拟合 → 外推大模型"。
> Part 3 **compute-optimal**:joint scaling law、IsoFLOPs、**Kaplan vs Chinchilla** 的口径之争、以及"训练最优 ≠ 部署最优(**over-train**)"。

---

## ① 卡住的地方(按讲义页码排序;13 条已全部作答)

### A. Part 1:数据 scaling 的图景与理论(p14–p19)

- [x] **Q1(p14)｜这张图在讲什么?irreducible error 是什么?**
  > 原问(还原):这张图具体的含义详细解释下。以及这里的 irreducible error 指的是什么?
  - **一句话**:横轴是"训练集大小"(log)、纵轴是"泛化误差"(log);误差随数据量分三段走 —— **太少时只能瞎猜(平台)→ 够多时按幂律下降(直线)→ 再多也撞地板**(不可约误差)✓
  - **三个 region 逐个读**(Hestness et al. 2017 的示意图):
    - **Small Data Region**:数据少到学不出任何东西,模型只能退化成"**最佳猜测**"(例如预测边际分布、多数类)⇒ 误差平在 **Best Guess Error** 上 ✓
    - **Power-Law Region**:数据开始够用,误差按 $n^{-\alpha}$ 下降(双对数坐标下是直线,斜率 $-\alpha$)✓
    - **Irreducible Error Region**:继续加数据,误差趋近一个地板 ✓
  - **irreducible error(不可约误差)= 贝叶斯误差**:即使**有无限数据、用最优模型**,误差也降不到它以下的那部分 ✓。它来自:① **标签/观测噪声**(同一输入对应不同输出);② **分布本身的重叠**(类别在特征上不可分);③ 任务定义里的**歧义**(例如"下一个 token 本来就有多种合理续写")✓ —— 注意:它**不是**"模型不够大"造成的,那是 approximation error ✓
  - **经验上常写成**:$\mathrm{Error}(n)=\left(\dfrac{n_c}{n}\right)^{\alpha}+E_{\mathrm{irr}}$,前一项是"可下降的部分",后一项是地板 ✓(这正是后面 Chinchilla 那种 $a/n^{\alpha}+b/m^{\beta}+c$ 形式的来源)✓
  - **为什么纵轴要取 log**:幂律 $n^{-\alpha}$ 在 log-log 上变成直线 $y=-\alpha x+C$,直线好拟合、好外推 ✓
  - 参考:[Hestness et al. 2017, *Deep Learning Scaling is Predictable, Empirically*](https://arxiv.org/abs/1712.00409)

- [x] **Q2(p16)｜"error should be monotone"是什么意思?为什么误差该是单调的?**
  > 原问(还原):这里是啥意思?monotone 是什么意思?为啥 error should be monotone?
  - **一句话**:monotone = **单调(不增)** —— 数据越多,期望误差不会变大;因为"多给的数据可以不用",所以数据是**只有好处、没有坏处**的资源 ✓
  - **严谨表述**:固定假设类与学习算法,经验风险最小化器的期望风险随样本数 $n$ **非增** ✓。直观证明:拿到 $n+1$ 个样本时,你**总可以**只用前 $n$ 个跑原来的算法 ⇒ 至少不会更差 ✓(严格结论需要一点技术条件,但方向是这样 ✓)
  - **但单调只解释了"下降",解释不了"为什么是直线"** ⇒ 这才是 p16 真正要问的:误差下降的**速率**为什么恰好是幂律(双对数下是直线)✓ 讲义的答案:估计误差"自然地"按**多项式**衰减(见 Q3、Q4)✓
  - **顺便说清楚**:单调是"关于数据量";如果换成"关于模型大小",在过拟合区间也可能出现非单调(小模型欠拟合、大模型过拟合再变小),所以这句话特指**数据** ✓

- [x] **Q3(p17)｜polynomial rate 是什么?这个均值估计的例子在说什么?**
  > 原问(还原):polynomial rate 指的是啥?这里是啥意思?
  - **一句话**:polynomial rate = 误差按 $n$ 的**幂**衰减(形如 $\mathrm{Error}\propto n^{-\alpha}$);均值估计给出 $\alpha=1$,所以 log-log 图上斜率是 $-1$、截距是 $2\log\sigma$ ✓
  - **严谨推导**(讲义 toy example):$x_1,\dots,x_n\sim\mathcal N(\mu,\sigma^2)$,取 $\hat\mu=\frac1n\sum_i x_i$,则 $\mathbb E[(\hat\mu-\mu)^2]=\dfrac{\sigma^2}{n}$ ✓(无偏 + 方差 $\sigma^2/n$)
  - **取对数**:$\log\mathbb E[\text{Error}]=-\log n+2\log\sigma$ ⇒ 这就是一条 $y=-\alpha x+C$ 的形式,其中 **$\alpha=1$、$C=2\log\sigma$** ✓ —— 这正是在 log-log 图上看到的"直线" ✓
  - **推广**:任何 $\mathrm{Error}\propto n^{-\alpha}$(任何多项式速率)**都是** scaling law ✓;$\alpha$ 决定斜率,$\sigma$(数据噪声)只决定**截距** ✓ —— 记住这条:**噪声改截距、机制改斜率** ✓
  - **为什么重要**:它说明"**参数化**的经典估计器给出 $\alpha=1$";而真实 LM 的 $\alpha$ 远小于 1(见 Q4、Q5),这中间的差别是理解 scaling law 的关键 ✓

- [x] **Q4(p19)｜和前面 $1/n$ 的情形相比,是什么不同导致了别的指数?**
  > 原问(还原):和前面得 1/n scaling 的情况,这里是什么不同导致其他系数的情况?
  - **一句话**:区别在于"**要估的东西是不是有限个参数**"。Q3 估的是 1 个数(参数个数固定)⇒ 误差 $\propto 1/n$;这里要估的是 d 维空间里的**一整个函数**,模型复杂度必须**随数据一起增长** ⇒ 误差只按 $n^{-1/d}$ 量级下降,维度越大越慢 ✓
  - **严谨推导(讲义版)**:把 d 维单位立方体切成边长 $h$ 的小格子,共 $h^{-d}$ 个;每个格子平均分到 $n h^{d}$ 个样本 ⇒ ① **偏差** $\sim h$(格子越粗,函数被"阶梯化"得越厉害);② **方差** $\sim \dfrac{1}{n h^{d}}$(格子里的样本越少越吵)⇒ 总误差 $\sim h^2+\dfrac{1}{n h^{d}}$;对 $h$ 求极小得 $h\sim n^{-1/(d+2)}$,误差 $\sim n^{-2/(d+2)}$ ✓
  - **和讲义写法的关系**:讲义把结论写成 $n^{-1/d}$ 并强调"**指数由维度决定**" ✓;更严格地说,指数取决于**光滑性假设**:Lipschitz($\beta=1$)下是 $n^{-2/(2+d)}$,更光滑($\beta$)则是 $n^{-\beta/(2\beta+d)}$ ✓ —— 这里我不确定的只是讲义为何取 $n^{-1/d}$ 这个特定形式(应是教学简化),但"指数 = 维度的函数、不是 1"这个结论是确定的 ✓
  - **所以"别的系数"从哪来**:从"**非参数**"来 —— 假设空间容量随 $n$ 增长(参数个数 $\sim h^{-d}$ 也在长),于是误差衰减慢于 $1/n$ ✓;这也解释了为什么真实 LM 的斜率是 $-0.13$、$-0.30$ 这种小数(对应"有效维度"几十的量级)✓
  - **一句 caution**:把 LM 的 $\alpha$ 直接解释成"内在维度"只是**类比**(p20 自己就说 intrinsic dimension 的估计很 sketchy、不 airtight)✓ 参考:[Bahri et al. 2021, *Explaining Neural Scaling Laws*](https://arxiv.org/abs/2102.06701)

### B. 数据 scaling 的工程含义(p25)

- [x] **Q5(p25)｜几种模型的 slope 差不多,说明了什么?有什么启发?**
  > 原问(还原):这里 slope 差不多说明了什么?有什么启发和作用?
  - **一句话**:斜率相近说明"**数据换性能的效率**"主要由**任务与数据分布**决定,而不是由模型家族决定;换架构/加 ensemble 主要改变**截距**(同等数据下更好),不改斜率 ✓
  - **具体证据**(p25 的图):几种不同模型(包括**集成模型**)在 log-log 上的直线**近乎平行** ✓ —— 集成本该等价于"更多算力/更大有效模型",但它主要把整条线**下移**,而不是掰弯它 ✓
  - **三个直接启发**:
    - **可外推**:斜率稳定 ⇒ "小模型/小数据上拟合的 $\alpha$ 可以外推到大尺度",这是整套 scaling law 工程的前提 ✓
    - **比较方法时看截距**:同预算下谁的 loss 更低(截距),谁的"scaling 速率"就一样;所以"某个改动让斜率变大"是更强的信号,而"只抬截距"是常见情形 ✓
    - **判断架构值不值得押注**:一个架构如果连 scaling 趋势都看不出来,通常说明它不够好 ✓(讲义 p31 的注解就是这个意思)
  - **两句 caution(讲义 p25 原文)**:① scaling law 会被**盲目套用**打破(换数据分布/换 recipe 就可能失效);② 它是**下界**,你总能通过更好的方法做得更好 ✓

### C. Batch size:noise scale 与 critical batch(p37–p39)

- [x] **Q6(p37)｜noise scale 是什么?这张图的横轴、纵轴分别是什么?**
  > 原问(还原):这里的 noise scale 是啥意思?这张图的横坐标和纵坐标分别是啥?
  - **一句话**:noise scale(**梯度噪声尺度**)$B_{\mathrm{noise}}$ 衡量"**梯度估计里噪声有多大**";图里横轴是"你的 batch 相对它有多大"($B/B_{\mathrm{noise}}$),纵轴是"步数相对最小值大了多少"($\varepsilon_{\mathrm{opt}}(B)/\varepsilon_{\max}$)✓
  - **noise scale 的严谨定义**(McCandlish et al. 2018):$B_{\mathrm{noise}}\approx\dfrac{\operatorname{tr}\Sigma}{\lVert G\rVert^{2}}$,其中 $\Sigma$ 是**单样本梯度的协方差**、$G$ 是真实(全量)梯度 ✓。直白说:它 = "**梯度的总噪声 ÷ 信号强度的平方**" ⇒ 就是"梯度信噪比"的倒数尺度 ✓(讲义 p38 底部那句小字 "ratio of the trace of the gradient covariance and squared norm of the gradient" 说的就是它 ✓)
  - **这张图怎么读**(右图 *Predicted Training Speed*):
    - **横轴** $B/B_{\mathrm{noise}}$:你的 batch 是噪声尺度的几倍(log 轴)✓
    - **纵轴** $\varepsilon_{\mathrm{opt}}(B)/\varepsilon_{\max}$:在 batch $B$ 下"达到目标 loss 所需的最优**步数**" ÷ "batch 趋于无穷时的**最少步数**"(log 轴)✓
    - 左段 **Perfect scaling**:斜率约 1 ⇒ batch 翻倍、步数减半(算力不浪费)✓
    - 右段 **Ineffective scaling**:曲线走平 ⇒ 再加 batch 几乎不省步数(算力浪费)✓
    - 拐点正好在 $B/B_{\mathrm{noise}}=1$ ✓(虚线)
  - **左图在说什么**:loss 等值线 + 两条 SGD 轨迹 —— **小 batch(红)** 的梯度噪声大、路径抖动、能"跨过"窄谷;**大 batch(蓝)** 的梯度准、路径平滑但方向更"短视" ✓(直觉:小 batch 的噪声既拖慢收敛,也带来隐式正则/逃离尖锐极小值的效果)✓
  - 参考:[McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162)

- [x] **Q7(p37)｜critical batch 是什么?为什么会有 diminishing returns?**
  > 原问(还原):解释下 critical batch 的含义,以及为啥会有 diminishing returns?
  - **一句话**:critical batch $B_{\mathrm{crit}}$ 就是"**再加大 batch 就开始明显浪费算力**"的那个 batch 大小;它近似等于 noise scale,含义是"**这一步的梯度噪声已经和信号差不多大了**"✓
  - **严谨表述**:在固定目标 loss 下,步数 $S$ 与数据量 $E$ 满足双曲线关系(Q8);$B_{\mathrm{crit}}=E_{\min}/S_{\min}$,并且
    - $B\ll B_{\mathrm{crit}}$:$S\approx S_{\min}\left(1+\dfrac{B_{\mathrm{crit}}}{B}\right)$ ⇒ **batch 翻倍、步数近似减半**(perfect scaling)✓
    - $B\gg B_{\mathrm{crit}}$:$S\approx S_{\min}$、$E\approx E_{\min}\dfrac{B}{B_{\mathrm{crit}}}$ ⇒ **步数几乎不降、数据却线性涨**(ineffective scaling)✓
    - $B=B_{\mathrm{crit}}$:$S=2S_{\min}$、$E=2E_{\min}$ ⇒ 两边都恰好"多付一倍" ✓
  - **为什么必然递减(机制)**:
    - **噪声按 $1/B$ 降,但"有用进展"不是**:梯度估计的方差 $\propto 1/B$,可真实梯度 $G$ 不随 batch 变 ⇒ 当噪声已经远小于信号,**再降噪对方向几乎没帮助** ⇒ 每步的"进展"饱和 ✓
    - **每步代价线性涨**:一步处理 $B$ 个样本 ⇒ 墙钟/算力 $\propto B$;于是"总成本 = 步数 × 每步成本"在 $B\gg B_{\mathrm{crit}}$ 时只增不减 ✓
    - **曲率/二阶信息**:大 batch 相当于"用更准的一阶信息走更短的步",在病态曲率方向上并不能自动加速(需要二阶方法才能吃掉大 batch 的好处)✓
  - **实践含义**:想要"用满算力",就让 $B$ 停在 $B_{\mathrm{crit}}$ 附近(典型做法是**随训练 loss 下降逐步增大 batch**,因为 $B_{\mathrm{crit}}$ 随 loss 下降而变大,见 Q8)✓

- [x] **Q8(p38)｜这个式子是什么意思?$S_{\min}$、$E_{\min}$ 怎么算出来?**
  > 原问(还原):详细解释下这个式子,包括每个符号的含义,和式子的含义和意义等。另外,Smin 和 Emin 是如何计算出来的?
  - **符号**(全部以"达到某个目标 loss"为前提):
    - $S$:**优化步数**(steps)✓;$E$:**样本/token 数**(examples,且 $E=S\cdot B$,讲义也标了 `steps * batch_size = examples` ✓)
    - $S_{\min}$:batch **趋于无穷**时的最少步数(此时每步梯度最准)✓
    - $E_{\min}$:batch **趋于零**时的最少数据量(此时每步最省数据)✓
  - **式子**:$\dfrac{S}{S_{\min}}-1=\left(\dfrac{E}{E_{\min}}-1\right)^{-1}$,等价于 $\left(\dfrac{S}{S_{\min}}-1\right)\left(\dfrac{E}{E_{\min}}-1\right)=1$ ✓
  - **含义(一句话)**:两个"浪费因子"的乘积恒为 1 —— **想省步数(省墙钟)就得多花数据,想省数据就得多走步**,没有免费的两全 ✓;而且它给出一个**双曲线**(不是直线),所以"batch 越大越好"在这里是被否定的 ✓
  - **两个等价的极限形式**(用 $E=S\cdot B$ 化简):
    - $S=S_{\min}\left(1+\dfrac{B_{\mathrm{crit}}}{B}\right)$(小 batch 侧)
    - $E=E_{\min}\left(1+\dfrac{B}{B_{\mathrm{crit}}}\right)$(大 batch 侧)
    - 其中 $B_{\mathrm{crit}}=\dfrac{E_{\min}}{S_{\min}}$ ✓ —— 这正是讲义最后那行 $B_{\mathrm{crit}}=E_{\min}/S_{\min}$ ✓
  - **为什么 $B_{\mathrm{crit}}$ 又等于 $\operatorname{tr}\Sigma/\lVert G\rVert^{2}$**:这是 McCandlish 的**理论**结论 —— 从"每步的有效进展 $\propto$ 信噪比"推出"最优步数对 batch 的依赖",其拐点正是梯度噪声尺度 ✓;所以"$E_{\min}/S_{\min}$(从数据拟合)"与"$\operatorname{tr}\Sigma/\lVert G\rVert^{2}$(从梯度统计)"是**同一个量的两种测法** ✓
  - **$S_{\min}$、$E_{\min}$ 怎么算出来**(实操):① 选定一个目标 loss;② **扫一组 batch**(例如 2^k 个档位),每个 batch 记录"到达该 loss 所需的步数" $S(B)$(可从 loss 曲线插值);③ 令 $E(B)=S(B)\cdot B$;④ 用模型 $S(B)=S_{\min}\left(1+B_{\mathrm{crit}}/B\right)$ 做**最小二乘拟合**,得到两个参数 $S_{\min}$、$B_{\mathrm{crit}}$(从而 $E_{\min}=S_{\min}B_{\mathrm{crit}}$)✓ —— 等价地,直接在 $(S,E)$ 平面上拟合双曲线 ✓
  - **注意(别当定律用)**:这是**经验模型**,受 LR schedule、warmup、优化器、以及"目标 loss 在训练曲线哪一段"的影响;近年有工作指出在 **WSD(恒定 LR)schedule** 下这个关系可能失效 ✓ —— 所以拟合时要说清口径 ✓

### D. 学习率与 μP(p39–p40)

- [x] **Q9(p39)｜这张图和下面的式子到底在说什么?**
  > 原问(还原):详细解释下这张图,以及下方的公式。我没看懂含义和这张图想要说明什么。
  - **先对齐坐标轴(容易读反)**:横轴是 **WebText2 训练 loss**,从左到右是 **10 → 6 → 4 → 3**(即**越来越大→越来越小**,方向要注意 ✓);纵轴是 **critical batch size(单位 tokens,log 轴 10^3–10^6)** ✓
  - **图里有三样东西**:① 两条实线 = 实测的 $B_{\mathrm{crit}}$,分别对应 $N=3$M 与 $N=85$M 的模型;② 绿色小点 = 直接测得的 **noise scale**;③ 灰色虚线 = 拟合出来的幂律 $B_{\mathrm{crit}}=2.1\times10^{8}\cdot L^{-4.8}$ ✓
  - **结论(一句话)**:目标 loss 越低,$B_{\mathrm{crit}}$ 越大 —— 而且是**指数级敏感**:$L$ 每降一半,$B_{\mathrm{crit}}$ 涨 $2^{4.8}\approx28$ 倍 ✓ ⇒ **"想压到更低的 loss,才配得上更大的 batch"**;训练前期用小 batch、后期逐步加大,正是这个道理 ✓
  - **下方公式**:$C_{\min}(C)\equiv\dfrac{C}{1+B/B_{\mathrm{crit}}(L)}$ ✓
    - $C$:名义算力(你实际花掉的 FLOPs);$B$:你选的 batch;$B_{\mathrm{crit}}(L)$:在该目标 loss 下的临界 batch ✓
    - 含义:把"因为 batch 过大而浪费掉的算力"扣掉,得到**有效算力** $C_{\min}$ ✓;当 $B\ll B_{\mathrm{crit}}$ 时 $C_{\min}\approx C$(几乎不浪费)✓;当 $B\gg B_{\mathrm{crit}}$ 时 $C_{\min}\approx C\cdot B_{\mathrm{crit}}/B$(大部分算力白花)✓
  - **这张图的教学意义**:选 batch 不是"越大越好",而应由**目标 loss 水平**决定 —— 这给了一条可操作的规则:先估出 $B_{\mathrm{crit}}(L)$ 曲线,再让训练过程中的 batch 跟着它走 ✓

- [x] **Q10(p40)｜两幅图的区别是什么?μP 到底是什么?**
  > 原问(还原):详细解释下两幅图的区别,以及右图的做法 μP,到底是什么含义?
  - **左图(Standard Practice)**:横轴 $\log_2(\text{Learning Rate})$、纵轴 Training Loss,每条曲线对应一个**宽度**(128 → 8192)⇒ **随着模型变宽,最优点在漂移**(讲义标了 *optimum shifts*)⇒ 小模型上调好的 LR 搬到宽模型就**不再是最好** ✓
  - **右图(Our Work / μP)**:同样的扫描,但不同宽度的最优点几乎**重合**(讲义标了 *optimum stable*)⇒ 可以直接"**小模型调参、大模型照用**" ✓
  - **μP = maximal update parametrization**(Yang et al. 2022,*Tensor Programs V*)✓:一套**逐层的参数化/缩放规则**,使得宽度趋于无穷时**每一层的更新量都保持 $\Theta(1)$**(即"最大化更新"而不是让某些层的更新爆炸/消失)⇒ **最优超参数与宽度无关** ✓
  - **它具体改了什么**(讲义右表 = 论文 Table 2;$M'$ 是 $M$ 宽度的 $r$ 倍):
    - **matrix-like 参数**(隐藏层权重,二维且随宽度增长):初始化方差 $\sigma\to\sigma/r$;**学习率 $l\to l/r$** ✓
    - **"others"**(embedding、bias、norm 等有限维参数):**保持不变** ✓(embedding 虽然有两维,但 vocab 维不随宽度变 ⇒ 按"others"处理 ✓)
    - **输出层乘子**:$\tau\to\tau/r$ ✓;**attention logits** 按 $1/d$ 而不是 $1/\sqrt d$ 缩放 ✓
  - **为什么标准参数化(SP)不行**:SP 的初始化($\text{var}\propto1/\text{fan\_in}$)只保证**初始时**激活尺度与宽度无关;一旦开始训练,更新量随宽度漂移(每层 LR 相同,但累积更新的大小随宽度变),激活尺度就跑了 ⇒ 最优 LR 跟着漂 ✓
  - **μTransfer 的用法**:在小宽度(便宜)模型上调好 LR/初始化/其他超参,再按 μP 规则**直接放大**到大模型,几乎不需要重调 ✓(Cerebras-GPT 就是这么把 40M 上的配置用到 2.7B ✓)
  - **谨慎**:独立复现(ETH 2024)确认"μP 下最优 LR 确实近似与宽度无关",但也发现**不总带来更好的泛化** ✓ 参考:[Tensor Programs V (arXiv:2203.03466)](https://arxiv.org/abs/2203.03466)、[μTransfer 介绍](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)

### E. Compute-optimal:Kaplan vs Chinchilla,以及"训练最优 ≠ 部署最优"(p50–p57)

- [x] **Q11(p50)｜图里的曲线是不是"给定 FLOPs 下的最优参数量"?Kaplan 和 Chinchilla 到底差在哪、为什么?**
  > 原问(还原):这幅图中的曲线指的是在给定的横轴 flops 预算下,最佳的模型参数量吗?也就是让 loss 最小的模型参数量?这里的不同指的具体是什么?kaplan 和 Chinchill 两个研究产生的不同的 scaling law 吗?为什么会产生?结合本页和后面几页,详细解释下这块到底是在讲什么?
  - **你的理解对** ✓:横轴 FLOPs、纵轴 Parameters,曲线就是"**给定算力预算 $C$,使 loss 最小的那个参数量 $N_{\mathrm{opt}}(C)$**" ✓;它和最优数据量是一体的:$C\approx6ND$ ⇒ $D_{\mathrm{opt}}=C/(6N_{\mathrm{opt}})$ ✓(所以"N 大"⇔"D 小",两件事是一件事 ✓)
  - **"不同"指什么**:两条 law 的**指数/常数**不同 ⇒ 外推出的配方差很多:
    - **Kaplan**:$N_{\mathrm{opt}}\propto C^{0.73}$、$D_{\mathrm{opt}}\propto C^{0.27}$ ⇒ tokens/param 随预算增大而**下降**(预算越大越偏"大模型、少数据")✓ 图上虚线更陡 ✓
    - **Chinchilla**:$N_{\mathrm{opt}}\propto C^{0.5}$、$D_{\mathrm{opt}}\propto C^{0.5}$ ⇒ tokens/param 大致**恒定**(经验值约 **20 tokens/param**)✓ 图上三条方法线更平,且几个真实大模型(Chinchilla 70B、Gopher、GPT-3、MT-NLG)贴着它 ✓
    - 直观后果:同一个 $C=10^{24}$ 预算,Kaplan 会说"上个更大的模型",Chinchilla 会说"模型小一点、数据多 8 倍" ✓
  - **为什么会产生这个差异**(pp.51–53 的三条解释):
    - **参数计数口径**:Kaplan 把**最后一层(以及嵌入层)参数**排除在 $N$ 之外 ⇒ $N$ 的度量变了,指数跟着偏 ✓
    - **训练 schedule 的污染**:极小算力预算下 **warmup 占比过高** ⇒ 小模型的 loss 被"人为抬高" ⇒ 拟合出来的斜率/常数偏移 ✓(讲义还提到 decay 在 batch/LR 调好时未必关键 ✓)
    - **非嵌入 vs 总参数 + 小模型上的非线性效应**:把"非嵌入参数"和"总参数"混在一套 law 里,会让小模型段失真 ✓
    - **另外**:Chinchilla 自己的 **method 3(joint fit)在原论文里实现有误**;Besiroglu et al. 2024 通过数据取证复原原始数据后重拟合,结果与 method 1/2 一致 ✓
  - **这一块到底在讲什么(本讲的核心 warning)**:scaling law 的**结论对"口径 + 拟合方式 + 拟合区间"极度敏感** —— 两种做法在各自口径下都"拟合得很好",但外推出的 **compute-optimal recipe 可以差好几倍** ✓ ⇒ 所以不能直接照搬"20 tokens/param";要在**自己的**参数计数、schedule、数据分布下重新拟合 ✓
  - 参考:[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)、[Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556)、[Besiroglu et al. 2024](https://arxiv.org/abs/2404.10102)

- [x] **Q12(p54)｜"over train"是什么意思?为什么要这样做?**
  > 原问(还原):这里为啥说要 over train?这是啥意思,而且为啥要这样做?
  - **一句话**:**over-train = 用超过 compute-optimal 比例的数据去训**(即 tokens/param 明显大于 20);因为 Chinchilla 只算"**训练**算力",而真实部署里 **推理**算力往往占大头 —— 多花训练数据把模型做**小**一点,能在推理上省很多 ✓
  - **账怎么算**:训练成本 $\approx6ND$(一次性);推理成本 $\approx2N\times$ 生成 token 数 $\times$ 请求数(按生命周期累积)⇒ **$N$ 越小,推理越便宜** ⇒ 当预期使用量很大时,最优解从"$N$ 大、$D$ 小"移向"**$N$ 小、$D$ 大**" ✓
  - **讲义给的真实比例**(tokens/param):GPT-3 **2**;Chinchilla **20**;LLaMA-65B **22**;Llama 2 70B **29**;Mistral 7B **110**;Llama 3 70B **215** ✓ —— 越新的、越可能被大量调用的模型,比例越高 ✓
  - **判据**:预期"每个参数被用多少次"越大,越值得 over-train;一次性/低频使用的模型按 Chinchilla 反而更省 ✓(讲义原话:*the more usage we expect, the more it becomes worth it to pay the upfront cost* ✓)
  - **一个 caveat**:over-train 到 100–200+ tokens/param 已经超出原始 law 的拟合区间,\(\alpha\) 可能变化 ⇒ 要用"重数据"区间专门拟合的 law 来定比例 ✓(近年有专门的 over-training scaling law 工作 ✓)

- [x] **Q13(p57)｜为什么说 model scaling 能降低训练成本?**
  > 原问(还原):为什么说 model scaling 可以降低训练成本?
  - **这里的"降低成本"指的是省掉"试错"的钱**:不必真的训一个 GPT-3 级模型才知道"这个 LR/这个架构选错了" ✓ —— 用小模型拟合 law、再**外推**,把"训一遍才知道"变成"算一遍就知道" ✓
  - **具体流程**(讲义 p42 的三步法):① 训几个**小模型**;② 拟合 scaling law(例如 Adam vs SGD 的 law);③ 按 law 的预测**选最优超参**,只在大模型上跑最终一次 ✓
  - **可省的三类决策**(讲义 p29–p40):架构(LSTM vs Transformer)、优化器(Adam vs SGD)、深度/宽度/头数、batch size、学习率(μP)✓ —— 每一项原本都要在大模型上扫一遍 ✓
  - **它的边界**:能预测的是**预训练 loss**;下游指标(accuracy 等)常常不可预测(p41 的 caution)✓;而且 law 会随数据/recipe 变(要重新拟合)✓ ⇒ 它省的是"预训练阶段的扫参成本",不是"训练本身不需要钱" ✓

---

## ② 用自己的话复述本讲核心思想

(看完后随手写,一句话也行。可参考线索:① 数据/模型/算力的 scaling 都是"**log-log 上的直线**",斜率反映"效率",截距反映"方法好坏";② 斜率稳定 ⇒ 可以**小模型拟合、大模型外推**;③ 外推的结论对**口径**(参数怎么数、schedule、拟合方法)极度敏感 —— Kaplan 与 Chinchilla 的差异就是例子;④ 训练最优 ≠ 部署最优,推理成本会把最优解推向 **over-train**。)

## ③ 触发的新问题 / 想扩展的知识

- [ ] (示例)想深入:xxx 与贝叶斯/优化理论的关系
- (AI 提示:本讲引出的延伸问题已整理进 [gap-list.md](gap-list.md);下一讲 L10 会把 scaling law 推进到"如何拟合/如何用于 MoE 与推理成本"。)

## AI 补充笔记(按日期追加)
(AI 的展开讲解、推导、延伸阅读写在这里;不要删除历史记录)

### 2026-11-04 · Lecture 09 展开笔记

#### 补充 1:为什么"幂律"在 log-log 上就是直线,以及那条线什么时候会弯

- **纯幂律**:$E(n)=c\,n^{-\alpha}$ ⇒ $\log E=\log c-\alpha\log n$ ⇒ 斜率 $-\alpha$、截距 $\log c$ ✓
- **带地板的形式**:$E(n)=c\,n^{-\alpha}+E_{\mathrm{irr}}$ ⇒ 只有当 $c n^{-\alpha}\gg E_{\mathrm{irr}}$ 时才近似直线 ⇒ 到了地板附近曲线**向下弯平** ✓ —— 这正是 Q1 图里"Power-Law Region 只在中间一段"的原因 ✓
- **"噪声改截距、机制改斜率"**:若把数据噪声方差从 $\sigma^2$ 换成 $2\sigma^2$(Q3 的均值估计),截距变 $2\log\sigma\to\log2+2\log\sigma$,斜率不变 ✓;而把估计器从"估 1 个参数"换成"估 d 维函数",斜率从 $-1$ 变成 $-\frac{2}{2+d}$(讲义简写 $-1/d$)✓。**所以比较方法时看截距,比较问题难度时看斜率** ✓
- **实操提醒**:拟合时一般要**固定"地板项"**或至少给它留自由度,否则早期/晚期点会把斜率带歪 ✓

#### 补充 2:critical batch 的双曲线模型(完整推导)

设"达到目标 loss"时步数 $S$、样本数 $E$、batch $B$,且 $E=S\cdot B$。经验模型:

$\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1$

令 $x=S/S_{\min}$、$y=E/E_{\min}$,由 $E=S B$ 得 $y=x\cdot\dfrac{B}{B_{\mathrm{crit}}}$(其中 $B_{\mathrm{crit}}\equiv E_{\min}/S_{\min}$)✓。代入解得:

$x=\frac{1}{1-B_{\mathrm{crit}}/B},\qquad y=\frac{1}{1-B/B_{\mathrm{crit}}}$

(当 $B\gg B_{\mathrm{crit}}$ 时把分母展开:$x\approx1+B_{\mathrm{crit}}/B$、$y\approx B/B_{\mathrm{crit}}$,即 Q8 里的两个极限形式 ✓)

**几个特征点**(用公式直算):

| $E/E_{\min}$ | 2 | 3 | 6 | 11 |
| --- | --- | --- | --- | --- |
| $S/S_{\min}$ | 2 | 1.5 | 1.2 | 1.1 |
| $B/B_{\mathrm{crit}}$ | 1 | 2 | 5 | 10 |

- **$B=B_{\mathrm{crit}}$**:步数与数据都恰好是"最小值 ×2" —— 这就是"平衡点"的说法 ✓
- **为什么会有拐点**:模型的来源是"每步进展 $\propto$ 梯度信噪比",而信噪比 $\approx B/B_{\mathrm{noise}}$ 在 $B\ll B_{\mathrm{noise}}$ 时近似 $\propto B$(线性收益),在 $B\gg B_{\mathrm{noise}}$ 时饱和 ✓ ⇒ 双曲线形状 ✓(严格推导见 McCandlish 2018 §2–3 ✓)
- **拟合的坑**:$\varepsilon_{\mathrm{opt}}$ 曲线是"固定目标 loss 下的步数",所以目标 loss 一变,整套 $(S_{\min},E_{\min})$ 都要重拟;跨 loss 比较没有意义 ✓

#### 补充 3:μP 为什么能让"最优 LR 不随宽度变"(动机版)

考虑一个 2 层线性网:$f(x)=W_2W_1x$,$W_1\in\mathbb{R}^{n\times n}$、$W_2\in\mathbb{R}^{1\times n}$(宽度 $n$)。

- **标准参数化(SP)**:$\mathrm{var}(W_1)\propto1/n$,两者用**同一个** LR $\eta$ ✓。一次梯度更新后,第一层输出的变化量 $\Delta h\approx\Delta W_1\,x$,而 $\Delta W_1$ 的每个元素 $\sim\eta\cdot\partial L/\partial W_1$ ⇒ 把 $n$ 个分量加起来,$\lVert\Delta h\rVert$ 大致 $\propto\eta\sqrt{n}$ 或 $\eta n$(取决于相关性)⇒ **宽度越大,同样 $\eta$ 下"动得越多"** ⇒ 最优点随 $n$ 漂移 ✓
- **μP 的两件事**:① 初始化 $\mathrm{var}\propto1/n$(和 SP 一样)② **LR 也 $\propto1/n$**(即讲义表里 $l\to l/r$)⇒ 更新量与宽度无关 ⇒ 激活尺度和最优 $\eta$ 都不随 $n$ 变 ✓
- **再加一条**:输出层(把 $n$ 维映回单值)的乘子要 $\propto1/n$,否则输出尺度会随宽度漂 ✓
- **诚实边界**:上面只是"为什么需要这样缩放"的动机;严格的"μP 是唯一让所有层都保持 $\Theta(1)$ 更新的参数化"结论,要靠 Tensor Programs 的坐标检查/张量程序证明 ✓,这里不展开 ✓

#### 补充 4:Kaplan vs Chinchilla —— 口径对照表

| 维度 | Kaplan et al. 2020 | Hoffmann et al. 2022 (Chinchilla) |
| --- | --- | --- |
| 参数怎么数 | **非嵌入参数**(去掉了 embedding 与最后输出层) | **总参数**(含嵌入) |
| 数据怎么数 | 训练 token 数 | 训练 token 数 |
| 拟合方式 | 固定模型大小扫训练步数;对小预算的 warmup 很敏感 | 三种:min-over-runs / IsoFLOPs / joint fit(其中 method 3 原实现有误) |
| 结论 | $N\propto C^{0.73}$, $D\propto C^{0.27}$(数据相对更少) | $N\propto C^{0.5}$, $D\propto C^{0.5}$(约 20 tokens/param) |
| 后果 | 同预算下偏向"更大的模型" | 同预算下偏向"更小的模型 + 4–8 倍数据" |

- **要点**:两者的差异不是"谁的数据更真",而是**口径不同 + 拟合区间不同 + 一处实现错误** ✓(Besiroglu 2024 复原数据后指出 Chinchilla 的 method 3 数值有问题 ✓)
- **给自己的口径**:做 scaling law 时先写清楚"参数怎么数、数据怎么数、loss 用哪一段、schedule 是什么",否则横向比较没有意义 ✓

#### 补充 5:把推理成本算进去,为什么最优解会"偏小模型 + 多数据"

设训练成本 $\approx6ND$ FLOPs、推理成本 $\approx2N\cdot U$($U$ = 生命周期内生成的总 token 数)$\Rightarrow$ 总成本

$C_{\mathrm{tot}}\approx6ND+2NU=N\,(6D+2U)$

- 若 $U=0$(只关心训练),就是 Chinchilla 的问题:$\min L(N,D)$ s.t. $6ND=C$ ✓
- 若 $U$ 很大,**每增大一点 $N$ 都要为所有推理付费** ⇒ 拉格朗日解会把 $N$ 压小、$D$ 拉大 ✓
- 直观判据:$\dfrac{2U}{6D}$ 越大(用得越多),越该 over-train ✓ —— 这正是讲义 p54 那句 *the more usage we expect, the more it becomes worth it* ✓
- 同一个逻辑也解释了为什么 **MoE** 在推理侧有吸引力:$N_{\mathrm{act}}$ 小 ⇒ 每次推理便宜 ✓(讲义 p36 的"参数 vs 激活参数"讨论 ✓)

#### 补充 6:延伸阅读

- 数据 scaling 的实证史:[Hestness et al. 2017](https://arxiv.org/abs/1712.00409);幂律的理论解释:[Bahri et al. 2021](https://arxiv.org/abs/2102.06701)
- 模型/数据联合 law:[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)、[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)、[Besiroglu et al. 2024(复现)](https://arxiv.org/abs/2404.10102)、[Rosenfeld et al. 2020](https://arxiv.org/abs/1909.12673)
- batch size / critical batch:[McCandlish et al. 2018](https://arxiv.org/abs/1812.06162)
- 超参迁移:[Tensor Programs V / μTransfer(arXiv:2203.03466)](https://arxiv.org/abs/2203.03466)、[`mup` 官方实现](https://pypi.org/project/mup/)
- 课程内:下一讲 L10 继续讲 scaling law 的进阶用法(拟合细节、MoE/推理成本、以及"什么时候 law 会失效")✓


