# Lecture 01 · 学习笔记(问题记录与 AI 解答)

- 讲义:仓库根目录 `lecture_01.py`(可执行讲义)
- 本地查看:前端打开 `http://localhost:5173/?trace=var/traces/lecture_01.json`(启动方法见 [README.md](README.md) 第 2 节)
- 状态:未开始(看完后在 [progress.md](progress.md) 更新)
- 用法:看讲时把内容写进 ①~③;然后把本文件路径填入 [README.md](README.md) 第 6 节的 Prompt 发给 AI,AI 会把回答直接写在问题下方并勾选 [x]

## ① 卡住的地方(尽量带讲义小节号或视频时间戳)
Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. 
这里没有很明白想要表达什么。是否想要说：在小模型上做的比如attention的架构改进，在大模型上可能就不再适用，因为大模型的计算量主要消耗在MLP上，而不是attention上？
  - AI: 你的理解方向对,但这句话的落点是「**直觉不跨规模迁移**」,不只针对某个改进是否失效:它的作用是在第一讲论证——为什么我们不能只在小模型上做实验就把结论搬到大模型,以及为什么这门课把「资源账本(resource accounting)」放在核心位置。更具体地:每个 token 上,attention 的 FLOPs 量级约为 $O(L\cdot d)$(L=序列长度,d=隐藏维度,由 QK^T、AV 等矩阵乘得出),而 MLP 与 embedding 之外的参数部分约 $O(d^2)$;模型scale主要靠增大 $d$,所以 **MLP 占总 FLOPs 的比例随规模上升、attention 占比下降**(数十 B 以上、常见上下文下 attention 通常只占小头)。推论:小模型上「改进 attention」收益明显,大规模时总收益被稀释——所以该看的是目标规模下的占比而非小模型直觉。两个补充:1) 序列变长时 attention 是 $O(L^2)$,占比会回升,长上下文是另一条线;2) 精确比例随架构/超参变,你会在 systems 单元的 resource accounting 和 Assignment 2 里亲手算。出处:讲义引的 Stephen Roller 帖子(https://x.com/stephenroller/status/1579993017234382849)。
flop的定义和计算公式
  - AI: FLOP = Floating-point OPerations,一次浮点运算;工程上常说 MAC(乘加)记 2 FLOPs(一次乘+一次加)。scaling law 里用的标准恒等式是 $C \approx 6ND$ 其中 C=训练总 FLOPs,N=参数量,D=训练 token 数。6N 的来源(每个 token):前向约 2N(每个参数平均被乘加约 2 次),反向约 4N(梯度 + 更新,backward 比 forward 贵约 2 倍),合计 6N。这是对标准 dense Transformer 的近似:忽略了 attention、logits 层、以及激活等二阶项,并假设 batch 足够大;Hoffmann et al. 2022(Chinchilla)用的就是这条近似。真正逐层算清楚是你 Assignment 1/2 里 resource accounting 的内容(讲义有示例:70B × 1T tokens ≈ 4.2e23 FLOPs,即 $6\times 70\times10^9\times10^{12}$)。
Kernels章节中：Kernels和GPU区别？Kernels是啥？另外，其中的Naive和Fused解释下。
  - AI: Kernel ≠ GPU:GPU 是硬件,kernel 是**跑在 GPU 上的一个函数/程序**(由 CPU 启动,内部几千上万个线程并行执行)。你用 PyTorch 时,每个算子(matmul、relu、layernorm…)都会在 GPU 上 launch 一个标准 kernel;kernel 间数据要经过显存 HBM 读写。Naive vs Fused 讲的是**最小化数据搬运**:
  - AI: - Naive:算子 A 的 kernel:读 HBM→算 A→写 HBM;算子 B 的 kernel 再读 HBM→算 B→写 HBM——每个算子都往返一次显存。
  - AI: - Fused:把 A 和 B 合并进一个 kernel:读 HBM 一次→算 A 和 B→写一次。现代 GPU 很多算子是 memory-bound(瓶颈是显存带宽不是计算),省一次读写就是近乎翻倍的收益。讲义的原则就一句:organize computation to minimize data movement。
以及Strategies: operator fusion (matmul + activation), tiling (FlashAttention)
  - AI: 这是把上面原则落地的两种手段:1) operator fusion = 把多个连续算子合成一个 kernel,典型如把 matmul 和它的激活/残差合并,中间结果不落 HBM;2) tiling = 把矩阵/序列切成块(tile),让块在 SM 的 shared memory/寄存器里被反复复用,而不是每次都回 HBM 读——FlashAttention 就是 tiling 的代表:分块计算 softmax-attention,从不需要物化 $L\times L$ 的注意力矩阵,复杂度不变但显存占用从 $O(L^2)$ 降到 $O(L)$。深度细节在 systems 单元/Assignment 2(Triton 手写 kernel)展开。
    
Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers
  - AI: 这些是 GPU 编程/性能术语,先给一句直觉,后面 systems 讲座和 Assignment 2 会逐个练到:
  - AI: - warp:GPU 的基本调度单位 = 32 个线程,同一时刻执行同一条指令(SIMT);
  - AI: - warp divergence:warp 内线程走了不同 if/else 分支,只能分时执行,另一路空转——尽量避免;
  - AI: - memory coalescing:让 warp 中相邻线程访问相邻地址,才能合并成一次内存事务,否则带宽浪费(访存性能的头号规则);
  - AI: - bank conflicts:shared memory 分成约 32 个 bank,同一 warp 同时访问同一 bank 的不同地址会冲突串行化;
  - AI: - occupancy:SM 上驻留的 warp 数占容量的比例,越高越能靠切换线程隐藏访存/指令延迟;
  - AI: - bulk-async transfers:用异步拷贝(cp.async / TMA)把数据搬运与计算重叠,不让 SM 空等。
    
Write kernels in CUDA/Triton/CUTLASS/ThunderKittens
这块也没看懂。
  - AI: 意思是:为了上面那些优化,你可以**绕过 PyTorch 预置算子,自己写 kernel**。四个选项是不同层次的工具:CUDA = NVIDIA 的原生语言(最底层、最灵活也最累);Triton = OpenAI 出的 Python 风格 kernel 语言,自动帮你做 tiling/调度,CS336 的 Assignment 2 就是用 Triton 写 fused RMSNorm; CUTLASS = NVIDIA 的高性能模板库(FlashAttention 官方版基于它);ThunderKittens = MIT 的快速 kernel 框架。前沿大模型基本都重度自研/改写 kernel,课程的态度是「make GPUs go brrr」——默认 kernel 太浪费,能 fuse 就 fuse。

Speculative decoding的含义和原理
  - AI: 一句话:用便宜的小模型先「猜」出 k 个 token,再用大模型一次前向并行校验并接受——**加速且不改变输出分布(精确解码)**。原理分三步:1) draft 模型(小而快)自回归地草拟 k 个 token;2) 大模型对草稿序列做一次并行 forward,得到每个位置的真实分布;3) 逐位置按拒绝采样规则接受草稿(接受概率 $p_{\text{target}}/p_{\text{draft}}$,不一致处截断并修正采样),被接受多少就少跑多少步串行 decode。为什么快:decode 阶段是 memory-bound、一次只出一个 token,而把 k 个候选拼成一个 batch 让大模型并行打分,边际成本远低于 k 次串行;草稿命中率高时加速明显(常见 2-3x)。注意它是「exact」的:拒绝采样保证和直接大模型采样同分布,不是近似。本讲只提及,完整算法与图示在讲义 Lecture 10(inference 单元,speculative_sampling());原始论文:Leviathan et al. 2023(https://arxiv.org/abs/2211.17192)与 Chen et al. 2023(https://arxiv.org/abs/2302.01318)。

Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)这块的核心思想和作用，解决的问题
  - AI: 解决的核心问题:**目标规模(如 $10^{25}$ FLOPs)太贵,不可能直接做超参实验**,那怎么知道「某个 scaling recipe(FLOPs → 超参的规则)」在目标规模上能拿到多少 loss?做法:对同一 recipe,在若干小规模(如到 $10^{24}$)上跑实验得到 (C, loss) 点,用幂律拟合 $L(C)=aC^{-\alpha}+c$ 之类,外推到 $10^{25}$;于是你可以在花大钱之前:1) 预测该 recipe 的最终 loss;2) 用多次小实验比较不同 recipe、挑目标规模上预测最优的。要点:scaling law 不是自动成立的,它要求你小心构造 recipe(参数量/数据量/LR 随规模怎么走的规则都要定好),否则外推不可靠。这对应 Assignment 3:定义 training API→小规模取样→拟合→提交外推超参与 loss 预测。

Predictability is at least as important as optimality!这个的深入理解，这里应该是从scaling raw的角度触发的一个深刻理解
  - AI: 你的感觉对,这是 scaling 视角的核心态度转变:**工程/研究决策真正依赖的是「可预测性」而非「小规模上的最优」**。一个小规模上最优化(loss 最低)的 recipe,如果它的行为无法外推,对你做大规模决策毫无价值;反过来,一个在小规模上略次、但 loss 随规模平滑可预测的 recipe,能让你:1) 在大规模花钱前就可靠预注册(preregister)它的预期 loss,2) 低成本比较几十个候选方案再押注。这也解释了为什么课程强调「careful construction of scaling recipe」并把超参设计成可迁移的形式(→ 你后面问的 hyperparameter transfer / μP):可预测性 = 你可以用计算预算换取信息,把昂贵的全规模实验变成少数几次。讲义里 Marin 的例子就是这个哲学:预注册 loss,训练完回来对答案。

Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)? 这个也没搞懂。为什么需要规定flops预算？ISOFLOP curves是啥？
  - AI: 拆两层:为什么规定预算——现实里你**只有有限的算力/钱/电**(如「这轮训练只有 $10^{23}$ FLOPs」),所以真正要回答的永远是分配问题:给定 C,买更大的模型还是喂更多数据?而 $C=6ND$ 说明在固定 C 下,N 和 D 此消彼长(选大 N 就只能少训数据,反之亦然),不能两全。ISOFLOP curves = **等算力曲线**:把 C 固定为若干档(比如 $10^{19},10^{20},…$),每一档内扫不同的 (N, D) 组合(同一条曲线上 C 相同、N·D 的配比不同),画出 loss;每条等算力曲线的最低点就是该预算下的最优配比。把所有档的最优点连起来,就是讲义里那根 compute-optimal frontier。经典结论(Chinchilla):最优点大致满足 D ≈ 20N(见下)。配套直觉:早期 Kaplan 2020 的配比对数据偏少(模型偏大、欠训);Chinchilla 用等算力曲线更系统地重估后发现应该用**更小的模型 + 更多数据**。
![img.png](img.png)compute-optimal scaling laws
![img_1.png](img_1.png)compute-optimal frontier可以理解为：这个frontier上的flops是固定的，一个frontier上的点的不同是hyparams的不同，比如D和N的不同组合。
然后这个frontier上的点的不同组合是通过调节超参数得到的？
  - AI: 对,基本正确。每条等算力曲线上 C 固定,点的差异来自 (N, D) 的配比——实操里通过选不同的模型规模 N(网络宽度/深度)和配套的训练 token 数 D(以及随规模一起调的 LR、batch size 等)得到;画图时横轴 N、纵轴 loss(或 D),每个点是一次真实小规模训练。frontier = 每档 C 下 loss 最低的点连成的线,「compute-optimal」指的就是这条线上每个点都是在花同样的钱时拿到了最低 loss。注意这些点不是玄学调参,而是同一套 **scaling recipe**(超参随规模如何伸缩的规则)下的扫描结果——这正是上一条说的 recipe 思想。
TL;DR: D = 20 N is roughly optimal如何理解
  - AI: 直译:给定算力预算,**每 1 个参数大约配 20 个训练 token 时最划算**(对 Chinchilla 的实验设定而言)。例:70B 模型应训约 70e9×20 = 1.4T tokens。来龙去脉:在 $C=6ND$ 下找使 loss 最小的 (N,D),解出约 $D/N\approx20$,等价地 $N^*\approx\sqrt{C/120},\ D^*\approx20N^*$。为什么反直觉:业界早年倾向把模型做大、数据相对不足(GPT-3 175B 只训 300B tokens,D/N≈1.7,远低于 20),Chinchilla 说明同样算力下把规模缩小一点、多喂几倍数据,loss 反而更低。注意事项(讲义也提醒):这是**训练视角**的最优,没算推理成本——若在意部署(推理贵、想要小模型),会刻意「过训练」到 D/N>20。

Now that the model should be reasonable, we can improve it further from weak supervision.这要如何理解？weak supervision.的概念？
  - AI: 这句在 alignment 单元的引入处(讲义 455-458 行),先澄清:此处的 weak supervision **不是**半监督/弱标注数据集那种老概念,而是指对齐阶段的**反馈信号来自「评判」而非「标准答案」**。对比:预训练是强监督——每个位置的下一 token 就是 ground truth,直接做 token 级监督;但「写出好的对话/代码」没有标准答案,**generate 很难、critique 很容易**——人更擅长对两个回答排序、裁判能判对错、代码能跑就算对。于是流程变成模板三步:1) 让模型生成候选回答;2) 用 {人、可验证器(代码执行/数学判题)、LM judge} 打分;3) 更新模型偏好高分回答(RLHF 的 PPO / 更简的 DPO / 免 value 函数的 GRPO)。这个评分信号比 token 级标注弱得多也便宜得多,但足以把模型从「会说话」提升到「会办事」。句首的 now that… 是前提:先靠预训练把基础能力训到 reasonable,再靠弱反馈调行为。

compute-constrained和data-constrained的关系和区别？为什么说Tomorrow, we will become data-constrained...？
  - AI: 意思:决定模型质量的**瓶颈资源**会随时间变化,不同约束下最优决策完全不同。compute-constrained = 算力(钱/GPU/电)是稀缺方:今天(2026 前沿)算力增长快、互联网高质量文本快被用尽,所以讲义里几乎所有设计决策都在「抠算力」——tokenizer 压缩省序列长度、MoE 省计算、数据过滤别把算力浪费在坏数据上、scaling laws 用小算力做超参调优。data-constrained = 高质量文本数据才是稀缺方:当算力继续涨、语料逼近上限(还有版权/许可、合成数据、非文本模态等问题),你变成「有算力没数据」——明天的决策围绕数据效率:重复使用(epochs/上采样)、合成数据、自训练、把字节/多模态也纳入。经典参考:Muennighoff et al. 2023「Scaling Data-Constrained Language Models」(https://arxiv.org/abs/2305.16264):数据不足时重复训练到约 4 epochs 仍是「免费」的,超过则收益骤减。关系:同一个「最大化给定资源下的质量」框架,只是哪根弦先绷紧;课程先教 compute-constrained 的今天,预告 data-constrained 的明天。

tokenizer的compression_ratio是如何计算的？结合 get_compression_ratio 函数解释。是否是compression_ratio = (len(text) / len(tokenizer.encode(text)))，即原始文本长度除以tokenizer编码后的长度？
  - AI: 方向对,但**单位是字节而非字符**。讲义定义(Lecture 01 第 567-571 行):$\text{compression\_ratio}=\frac{\text{len(bytes(string, utf-8))}}{\text{len(indices)}}=\frac{\text{UTF-8 字节数}}{\text{token 数}}$ 即「平均每个 token 打包多少字节」。你的公式只有把 len(text) 理解为 **UTF-8 编码后的字节数**(而非字符数)才成立——纯 ASCII 两者相同,遇到中文/emoji 就不同:"你好" 是 2 字符但 6 字节;🌍 是 1 字符但 4 字节。意义(讲义 610 行):ratio 越大 = 同样内容 token 越少 = 序列越短,对 O(L²) 的 attention 越友好。参考值:ByteTokenizer 恒为 1(1 字节=1 token,最差);BPE/GPT 分词器约 3~5;word 级最高。验证:在 trace-viewer 里跑 lecture_01,对同一字符串看 `get_compression_ratio` 在 char/byte/word/BPE 下的返回值差异。

One could increase compression ratio by increasing vocabulary size (number of possible token values increases)
这句话能否这样理解：增加vocab size就相当于增加了词语的种类，使得我们更能将一句话划分为更少的词语，从而降低tokenizer编码后的长度，从而提高compression ratio？
  - AI: 直觉大体正确:vocab 越大(允许更多、更长的 merge),常见片段更容易整段打包成一个 token,平均序列更短 → ratio 升高。但这是第一层,讲义紧接着给反面(原词):**leading to sparsity(稀疏)**。两个必须知道的代价:1) 长尾稀疏——词表里大量 token 出现频次极低,模型学不到可靠统计(每次见到都近乎新词),embedding 白白占显存;2) 收益递减——压缩率与词表大小大致是对数/亚线性关系,10k→100k 提升明显,再翻几倍收益很小,却让每个 token 更稀有、更难估计分布。所以词表大小是**平衡点**:大到能压住常见文本,又别让尾部 token 变成噪声。BPE 训练的本质就是选 merge 数量,在 ratio 与稀疏/复杂度之间取舍——Assignment 1 你会亲手调 vocab size 感受它。

CharacterTokenizer、ByteTokenizer、word_tokenizer各自的原理。它们分别是如何把句子分成token的？每个token代表什么？具体举例说明。
  - AI: 用同一句 "Hello, 🌍! 你好!" 举例(讲义里就是它):
  - AI: - CharacterTokenizer:一个 token = 一个 Unicode 码点(ord)。encode 把每个字符换成整数码点:'H'→72、'你'→20320、'🌍'→127757。优点:简单无损;缺点:词表巨大(Unicode 约 15 万字符、大量极稀有)且压缩率最低——这句 14 个字符就是 14 个 token(讲义原话:worst of both worlds,大词表+低压缩)。
  - AI: - ByteTokenizer:一个 token = 一个 UTF-8 字节(0-255)。encode 先转 UTF-8 字节再逐字节转整数:ASCII 每字符 1 token('H'=72);'你' 3 字节 → 3 token;🌍 4 字节 → 4 token。优点:词表固定 256、极小、可表示任意文本;缺点:压缩率恒为 1,序列很长,而注意力是 O(L²)(讲义原话:terrible)。
  - AI: - word_tokenizer:正则 \w+|. 切词/非词碎片:"I'll say supercalifragilisticexpialidocious!" → 'I'、'll'、'say'、'supercalifragilisticexpialidocious'、'!' 等。每个 token ≈ 一个词,语义强、压缩率高;但词表 = 语料中的不同词,可能几十万且开放:罕见词学不动、新词只能落到 UNK 占位符(影响困惑度),词表大小也不固定。共性:**token 是模型序列的基本原子**,理想切分要在「token 有意义」与「序列别太长」间取平衡——这正是 BPE 的目标(下一问呼应)。建议在讲义 trace 里跑这三个类看 indices,最直观。

Chunks should be variable (allocate more model capacity to interesting chunks)。这里我想要更加细致地理解下：如果我们地tokenizer不够好，将很多没有意义的chunk分成了token，会如何？
  - AI: 先理解这句设计目标:模型对**每个 token 位置分配的容量/算力基本相同**(每个位置都过一次 attention/FFN),所以把长文本切得越短,能留给「每单位信息」的容量就越充足。这里 chunk 的"interesting"指**信息密度高、值得模型精细加工的部分**——理想 tokenizer 让常见/低信息片段(如 " the ")塌缩成少数字符少的 token,把位置预算省给真正难的内容,等价于可变粒度 = 自适应计算。
  - AI: 反过来,若 tokenizer 不好、把很多无意义碎片切成 token,后果是链条式的:1) 序列显著变长 → 注意力 O(L²) 的计算/内存成本暴涨,同样上下文窗口能装的内容变少;2) 模型被迫把容量花在字符级琐碎规律(拼写、空格)上,长程依赖被拉长稀释,学语言结构更难;3) 语义被切碎(一个完整词拆成三四个 merge),模型要自己学会"重新组合",统计负担变重;4) 词汇碎片若跨语言/领域分布差,还会放大稀疏性。这正是为什么 byte/char 分词性能吃亏、而 BPE 能赢:它在字节的完备性与词的语义性之间找一个数据驱动的中间粒度。若你还想深挖"interesting"的信息论含义(高熵=高信息),值得补一点熵/交叉熵直觉——已记入 gap-list。
## ② 用自己的话复述本讲核心思想
tokenization methods、scale recipe等等

## ③ 触发的新问题 / 想扩展的知识
SwiGLU主要讲的是啥？
  - AI: SwiGLU 是一种**门控激活函数**,把 FFN 里的「线性+ReLU/GeLU」换成了两个投影相乘、其中一个过 Swish 当门:$\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_1)\odot (xW_2)W_3$(忽略 bias;也可以理解为 FFN 增加了一条门控支路,参数略增但通常去掉 bias 抵消)。直觉:乘性门控让网络能按输入内容“决定传多少信号”,表达力强于逐元素的固定非线性。效果:Shazeer 2020(GLU variants, https://arxiv.org/abs/2002.05202)及后续 PaLM/LLaMA/Qwen 等实证,同参数量下比 ReLU/GeLU FFN 困惑度更低(常有 ~0.1 量级的 PPL 改善量级,随实现而异)。你会在架构单元/Assignment 1 里亲手实现它;讲义把它的出现归为「实验结果先于理论」(那张 divine-benevolence 图的梗)。
in-context learning是什么含义？
  - AI: in-context learning(ICL,上下文学习/情境学习)= **在推理时不更新任何参数,只把示例/任务说明写进 prompt 上下文,模型就能按示例执行新任务**。典型:few-shot——给 2~3 个 (输入→输出) 例子再丢新输入,GPT-3(175B)时代首次被系统观察到并命名(讲义把它标为 GPT-3 的标志性能力,区别于 BERT 时代「必须 fine-tune」)。本质仍有争议(它到底激活了哪些已学机制、是否等价于隐式梯度等),但对使用者它是「把模型当条件分布用」:p(输出 | 输入 + 示例)。和你的领域相关:排序/召回里拿 LLM 做零样本/少样本特征或 reranker 时,ICL 设计(示例选择、格式)直接影响效果。后续讲义会在 agents/长上下文里继续用这个概念。
The dream: tokenizer-free model architectures, which operate directly on bytes: 这类的研究一般怎么做的
  - AI: 动机:tokenizer 是套在字节流上的“人工切分”,带来脆性(拼写错误、未知词、多语言不平衡、序列长短畸变),于是有「直接对字节建模」的研究线,大致三派(都是讲义引用的工作):
  - AI: 1) 直接 byte 序列 + 改造架构消化长序列:ByT5(https://arxiv.org/abs/2105.13626)把 T5 改成字节级,性能接近 token 级但速度明显慢(序列长 ~4 倍)——说明“纯 byte”的主要代价是序列长度;
  - AI: 2) 层次/局部压缩:Megabyte(https://arxiv.org/abs/2305.07185)把字节序列按局部 patch 处理,深层只对 patch 摘要做全局建模,缓解长度问题;
  - AI: 3) 动态分块(自适应):BLT / Byte Latent Transformer(https://arxiv.org/abs/2412.09871)让模型自己按“信息量”(entropy)把字节聚成大小不一的 patch——和你前面问的「chunks should be variable」直接呼应,高熵(难)的地方切细、低熵(好猜)的地方切粗。
  - AI: 还有 MambaByte(字节级 SSM)、以及讲义 refs 里的 tfree_2024/hnet_2025 等后续工作。现状(讲义原话):promising,但**还没有被 scale 到前沿水平**——tokenizer 带来的长度压缩收益目前仍大于它的坏处;这条线值得作为你后续深入方向之一跟踪。
How to Scale Your Model 的文章
  - AI: 这不是一篇论文,而是 **Google JAX 团队写的免费在线书/教程**,标题就叫 How to Scale Your Model(https://jax-ml.github.io/scaling-book/)。讲义在 systems 单元把它列为推荐读物:“Nicely lays out how to approach systems for LLMs conceptually”;注意它来自 Google,所以以 **TPU/JAX 生态**为主,但「参数怎么分布、数据怎么搬运、从单卡到集群怎么推演」的高层概念通用,正好衔接你后面 Assignment 2 的并行/资源账本。可以把它当作 systems 单元的中英对照资料:遇到讲义里 kernel/并行/显存概念卡住,去翻这本书对应章节。
hyperparameter transfer: 在small scale模型上调好的超参数，能否直接迁移到大模型上使用？或者至少能有一个关联关系，将small scale模型的超参数调优结果作为大模型的参考？
  - AI: 能,但**不是免费自然发生的,需要刻意设计参数化方式**——这正是讲义 line「Parameterize the model in a way to get hyperparameter transfer」(配的引用是 μP 论文)的意思。
  - AI: 标准做法:μP / μTransfer(Yang et al., https://arxiv.org/abs/2203.03466,开源包 muzero? 实为 microsoft/mup):把权重初始化、各层学习率按宽度(和残差分支等)做“maximal update”缩放(比如输出侧权重按 1/宽度 缩放、隐藏层 LR 独立于宽度等),使得**小模型上调到最优的学习率/初始化/部分超参,在更宽的模型上仍然接近最优**——小模型(如宽度 512)调好,宽度 8192 直接继承,近乎零成本迁移。经验上对“变宽”最可靠,对“变深”与数据量变化需要额外规则(深层有 depth-* 技巧,仍在研究)。工业界更朴素的变体:直接在固定深度下用 √d 或固定 LR 的经验规则(很多开源模型就是这么定的,细节常不公开)。
  - AI: 和 Lecture 01 的呼应:hyperparameter transfer 是让「scaling recipe 可预测/可外推」的关键拼图——没有它,你只能在每个规模重新调参,scaling law 也就无从谈起。若想深入,μP 论文 + scaling-book 的调参章节足够入门;这块(前沿实操配方)有“工业界不透明”的成分,论文能给的确定性到此为止,我不会假装更确定。

## AI 补充笔记(按日期追加)

### 2026-09-07 · Lecture 01 首次答疑

本次已将 ①/③ 中全部问题就地回答(每题下方 2 空格 `- AI:` 行)。要点速记:

1. **资源账本是第一讲主线**:attention 与 MLP 的 FLOPs 占比随 $d$ 增大而此消彼长;训一次 LLM ≈ $C\approx6ND$(前向 2N + 反向 4N 每 token)。
2. **Scaling 心智**:固定预算下在 (N,D) 间取舍;isoFLOP 曲线最低点 → compute-optimal frontier → Chinchilla $D\approx20N$。最小推导:$D=20N\Rightarrow C=6N\cdot20N=120N^2\Rightarrow N^*=\sqrt{C/120}$,例如 $C=10^{23}$ 时 $N^*\approx2.9\times10^{10}$(≈29B)、$D^*\approx5.8\times10^{11}$(≈580B token);70B 模型对应约 1.4T tokens。
3. **Predictability > optimality**:可外推的 recipe 才能用小算力换取大规模决策信息。
4. **Tokenizer = 字节 ↔ 整数序列**:compression_ratio = UTF-8 字节数/token 数;char(大词表低压缩)与 byte(压缩=1)两头差,BPE 取中间;词表大小在压缩与稀疏之间取舍。
5. **补充几个“后续讲会深挖”的钩子**:speculative decoding(L10)、kernel/并行(Assignment 2 + Lecture 13/14)、alignment 的 PPO/DPO/GRPO(L15-17)、数据约束下的 scaling(Assignment 3/4)。

延伸阅读(按重要度):
- Kaplan et al. 2020, Scaling Laws for Neural LMs: https://arxiv.org/abs/2001.08361
- Hoffmann et al. 2022 (Chinchilla): https://arxiv.org/abs/2203.15556
- Karpathy 讲 tokenizer 的视频(讲义推荐): https://www.youtube.com/watch?v=zduSFxRajkE
- 互动 tokenizer 演示(tiktokenizer): https://tiktokenizer.vercel.app/?encoder=gpt2
- Shazeer 2020, GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Yang et al. 2022, Tensor Programs V (μP/μTransfer): https://arxiv.org/abs/2203.03466
- Google 免费书 How to Scale Your Model: https://jax-ml.github.io/scaling-book/
- Speculative decoding: https://arxiv.org/abs/2211.17192 · https://arxiv.org/abs/2302.01318
- ByT5/Megabyte/BLT(字节级模型): https://arxiv.org/abs/2105.13626 · https://arxiv.org/abs/2305.07185 · https://arxiv.org/abs/2412.09871
- Muennighoff et al. 2023, Scaling Data-Constrained LMs: https://arxiv.org/abs/2305.16264

### 2026-09-07 · 追问详解:scaling recipe、isoFLOP 外推与 Percy 推文

**Q1: 为什么 recipe 是“FLOPs → 超参的规则”?**
- 先区分两组概念:一组**固定的超参数**是一个点;而 recipe 是一套**“随预算伸缩的决策规则”**,输入目标算力 C(以及数据等资源),输出完整训练配置 {N, D, 宽深比, LR, batch size, 学习率调度, warmup, 初始化, …}。你的理解“给定 FLOPs → 该用什么超参”基本正确,只是把“一套数”换成“一套函数/规则”。
- 为什么必须是“规则”而不是“一组数”:scaling law 实验需要在很多不同 C 下各跑一次,且每次都要配置完整、口径一致;每个小实验就是**同一个 recipe 在 C=Cᵢ 处的取值**。只有“不同规模共享同一套伸缩规则”,loss(C) 才是平滑、可外推的曲线;单一模型/单一配置根本不构成可外推的对象。所以推文原话才说:要的不是单个模型,而是 a training recipe that scales reliably。
- recipe 内部有两类成分:**分析可定的**(如 N/D 配比——Chinchilla/isoFLOP 给出)、**经验/可迁移的**(LR、init、schedule 等——靠 μP 这类 hyperparameter transfer 从小规模继承),两者拼起来再验证外推是否成立。

**Q2: isoFLOP 曲线能“直接”从小预算外推出大预算的超参吗?**
- 拆成三层看,答案分层:
  - 单条 isoFLOP 曲线(固定 C):只回答“这个预算内 (N,D) 怎么配最优”(U 形最低点),不跨预算。
  - compute-optimal frontier + 幂律拟合:由若干小 C 档的最优点外推到任意 C —— Chinchilla 的输出是 D ≈ 20N(与 C 近似无关的比例),于是给 C=10²³ 就能算 N*≈√(C/120)、D*≈20N*。对 **N 和 D 这两维,能外推**。
  - 其余超参(LR/batch/schedule/初始化/宽深配比):**isoFLOP 不回答**。它们要靠“可迁移参数化”(μP 使小模型调好的 LR 等在大宽度下仍接近最优)或写进 recipe 规则,否则每个规模都得重调。Assignment 3 的 leaderboard 就是“给预算→交整套超参与预测 loss”的完整走一遍。
- 所以准确说法:外推能力集中在 (N, D, loss) 这类“分析维度”,而完整训练配置 = 外推的 N/D + 可迁移超参 + recipe 规则。精度还依赖三点:小预算点够密、所有规模共用同一 recipe、拟合形式选对——所以“预测完要实跑对答案”(见 Q3)是科学闭环,不是可有可无。

**Q3: Percy 那条推文(https://x.com/percyliang/status/2034367256277533100,2026-03-18)在讲什么实验?**
- 背景:Marin = Percy Liang 发起的开放开发(open development)大模型项目,特点是所有实验**先在 GitHub issue 预注册**(假设、配置、预测指标写清楚再跑),全程公开 checkpoint 与日志,避免“跑完再讲故事”。
- 配图标题 Delphi Scaling Suite Results:Delphi suite 是 EleutherAI Pythia 的“现代化版”(同一套数据与顺序、多个规模、带中间 checkpoint 的公开套件),专门用来做 scaling 研究。
- 图怎么读:横轴 = training tokens(对数),纵轴 = Paloma macro loss(跨多领域 perplexity 的宏平均,越靠右下越好)。每条彩色 U 形曲线 = 一个固定算力预算 C(3e18 ~ 1e22 FLOPs)的 isoFLOP 扫描:同一 C 下把 N 与 D 配比从一边扫到另一边,loss 先降后升形成 U 形(偏左=模型大但欠训;偏右=模型小、容量不足)。X 标记(图例 Forecast)= 该预算下拟合出的最优点;所有 X 连成灰色虚线 = compute-optimal frontier(等价于 Chinchilla 的 D≈20N 轨迹)。两处 Δloss = +0.011 / +0.005 的箭头:标注的是选定配置偏离(或拟合相对)精确最优点时的 loss 代价——数字很小,说明 **frontier 附近相当平坦**,不必为“精确最优”钻牛角尖,这也是“predictability 比 optimality 重要”的实证底色。
- 实验步骤(推文原文要点):已在 ≤1e22 FLOPs 训过一批模型并拟合出 scaling law → 把“在 1e23 FLOPs 时 loss 会是多少”的预测**预注册到 GitHub**(后续推文给的链接:github.com/marin-community/marin/issues/1337)→ @WilliamBarrHeld 正在跑 1e23 那次训练(即图右下角那个孤立的、标注 Currently training! 的 X)→ 几天后对答案。
- 它示范了什么(与讲义 scaling_laws() 的 Live example 完全对应):1) “用小规模拟合 → 外推到目标规模 → 预注册 → 大规模实跑验证”的完整闭环,正是你之前问的 fit a scaling law to predict loss at target scale 的现实版;2) “要 recipe 不要单个模型”;3) 预注册 = 先写预测再跑,让外推质量可以被证伪。
- 诚实声明:我目前没有核实到这次 1e23 预测 vs 实跑结果的后续(推文说“几天后见分晓”);需要的话我可以去查 marin issue #1337 或后续推文,把“预测准不准”的答案补进来。
