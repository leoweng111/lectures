# CS336 学习笔记体系 · Spring 2026

本目录是 **Stanford CS336「Language Modeling from Scratch」(Spring 2026 讲义仓库 `stanford-cs336/lectures`)** 的个人学习笔记体系,由笔记文件 + 一套"AI 学伴"工作流组成。

- 仓库本地路径:`D:\work_repo\lectures`(克隆自 `https://github.com/stanford-cs336/lectures`,官方上游默认分支 `main`)
- 个人分支:本仓库当前在 `notes` 分支上工作,`main` 保持与上游同步
- 运行环境:conda 环境 `cs336`(Python 3.12,含 edtrace/tiktoken/einops/mmh3/torch-CPU 等)

---

## 1. 仓库内容速览

| 类型 | 讲次 | 说明 |
|---|---|---|
| 可执行讲义 `lecture_XX.py` | 01, 02, 06, 07, 10, 12, 13, 14, 17 | 交互式讲义,已预生成 trace 在 `var/traces/lecture_XX.json`,可用前端本地查看 |
| 静态讲义 `lecture_XX.pdf` | 03, 04, 05, 08, 09, 11, 15, 16 | 直接用 PDF 阅读器打开 |

## 2. 本地查看讲义

```bash
# 方式 A:静态讲义(03/04/05/08/09/11/15/16)
# 直接用浏览器/PDF 阅读器打开 D:\work_repo\lectures\lecture_XX.pdf

# 方式 B:可执行讲义(01/02/06/07/10/12/13/14/17) —— 本地前端(trace-viewer)
cd D:\work_repo\lectures
npm run --prefix=edtrace/frontend dev        # 启动本地服务器(默认 5173;被占用会自动顺延,以启动日志为准)
# 然后浏览器打开(把 XX 换成讲次,端口换成实际端口):
#   http://localhost:<端口>/?trace=var/traces/lecture_XX.json
# 示例(本机当前):http://localhost:5174/?trace=var/traces/lecture_01.json
```

运行 conda 环境中的 Python:

```bash
conda activate cs336          # 或: conda run -n cs336 python ...
python lecture_01.py          # 重新生成 trace(一般不需要,var/traces 已有)
```

## 3. 每讲学习流程(单讲约 50 分钟)

1. 打开本讲讲义/前端 + YouTube 官方录像,用 1.25x 观看;
2. 在本目录 `lecture_XX.md`(XX=讲次)里**只写三类内容**:
   - ① 卡住的地方(尽量带讲义小节号或视频时间戳);
   - ② 用自己的话复述本讲核心思想;
   - ③ 触发的新问题/想扩展的知识;
3. 看完当场把该 `lecture_XX.md` 交给 AI 答疑(用第 5 节的 Prompt),AI 会把回答**直接写回文件**(见各文件头部约定),不懂的当场打掉。

## 4. 每周日"AI 对谈"(固定 15 分钟,让 AI 跟进你的进度)

进度跟进的机制:**不靠工具的魔法,靠把进度变成文件,每次让 AI 从文件恢复状态。**

1. 把本周看过的几讲 md + `gap-list.md` 一起交给 AI;
2. 让 AI 压缩成本周**半页总结**(写回 `progress.md`),出 3~5 道主动回忆题(先不给答案,你自答后再核对);
3. 让 AI 更新 `gap-list.md`(新增缺口按 A/B/C 分类,已解决的标记关闭);
4. 下一阶段开始时,先把 `progress.md` + `gap-list.md` 喂给 AI,它就能"记得"你学到哪、缺口在哪。

## 5. 文件地图

| 文件 | 作用 | 谁写 |
|---|---|---|
| `README.md`(本文件) | 学习方法总纲 + AI Prompt | 你维护,AI 只读 |
| `lecture_01.md` ~ `lecture_17.md` | 每讲一问一答笔记 | 你看讲时写问题;AI 直接把回答写回 |
| `progress.md` | 进度台账 + 每周半页总结 | AI 每周对谈时更新(你审核) |
| `gap-list.md` | 缺口清单(A 必补 / B 延后 / C 放弃) | AI 按答疑中暴露的缺口更新(你审核) |

规则:AI 只能修改你明确允许的文件(`lecture_XX.md` / `progress.md` / `gap-list.md`),且改动要可追溯(勾选问题、追加日期段落),不要删除你的原始问题记录。

---

## 6. 给 AI 的答疑 Prompt(复制即用)

把下面整段发给任意 AI(Claude / ChatGPT / 其他 agent,包括我),把 `<问题文件路径>` 替换成具体文件即可:

```text
你是我的 LLM 学习助教,正在协助我自学 Stanford CS336《Language Modeling from Scratch》(Spring 2026)。

【学习资料仓库】本地路径:D:\work_repo\lectures(克隆自 https://github.com/stanford-cs336/lectures,
内含 lecture_01.py~lecture_17.py 可执行讲义与 lecture_XX.pdf 静态讲义,以及我个人的笔记目录 D:\work_repo\lectures\notes\)。
必要时请阅读仓库里的讲义文件、var/traces 下的内容,并允许联网搜索(如讲义里引用的论文、术语的权威解释),以给出准确回答。

【本次任务】请阅读我的问题笔记文件:<问题文件路径,例如 notes\lecture_01.md>,
回答文件中记录的所有问题(主要是在“卡住的地方”，且是未勾选的条目;若已勾选则跳过)。回答要求:
1. 直接编辑该 md 文件:在每个问题条目下方追加回答(问题有多问就分条列),并把该条目勾选为 [x];
   若单次答疑内容较多,把展开讲解、公式推导、延伸阅读写在文件末尾的"AI 补充笔记"区,按日期追加,不要覆盖我写的内容。
2. 回答以中文为主,公式统一用行内 LaTeX `$...$`(不要用 `$$...$$` 显示公式:我的 Markdown 渲染器不认它;公式里表示「小于」一律用 \lt,不要写小于号紧跟字母的连写——会让整篇笔记的公式全部渲染失败,详见第 8 节),关键结论给一句话直觉 + 严谨表述两层;
3. 优先基于讲义/课程材料回答;涉及课外知识(论文、数学背景)时先联网核实再作答,并注明出处链接;
4. 如果我的问题暴露了前置数学/知识缺口(如贝叶斯、优化、信息论),顺手在 D:\work_repo\lectures\notes\gap-list.md 里
   新增对应条目(标注来源讲座、分类 A 必补/B 延后/C 放弃、建议学习资源);
5. 如果有多个问题一次答不完,先回答最影响理解主线的 3 个,并告诉我其余已记入 backlog;
6. 不确定的地方要明说"这里我不确定",不要编造。
7. 美化一下总体Q&A的排版，可以将我的问题用更加准确的语言表述，并且做成每个回答的小标题。回答的格式也尽量分点、格式美观一些。

完成后,简短汇报:回答了哪些问题、在哪些小节写入了回答、补充了哪些 gap-list 条目(用 3~5 行)。
```

> 每周日对谈用同款 Prompt 开头,但把任务换成第 4 节的"总结 + 自测 + 更新缺口",并附上 `progress.md` 与本周 `lecture_XX.md` 列表。

---

## 7. edtrace 前端重建(换机器 / 误删 / 升级工具)

`edtrace/` 是第三方工具([percyliang/edtrace](https://github.com/percyliang/edtrace)),**已去版本化**(本地无 .git),被外层仓库 .gitignore 忽略,不会也不应被 push。它损坏或想升级时,直接删除整个 `edtrace` 目录后按下面重建(约 1~2 分钟,node_modules 会重新安装):

```bash
cd D:\work_repo\lectures
git clone --depth 1 https://github.com/percyliang/edtrace   # 重新拉取工具
npm install --prefix edtrace/frontend                       # 安装前端依赖

# 建立 var/images 目录联接(让 trace-viewer 能读取仓库根目录的讲义数据)
node -e "const fs=require('fs'); for (const d of ['var','images']) fs.symlinkSync('D:/work_repo/lectures/'+d, 'edtrace/frontend/'+d, 'junction');"

npm run --prefix=edtrace/frontend dev   # 启动,浏览器打开 http://localhost:<端口>/?trace=var/traces/lecture_XX.json
```

---

## 8. 写笔记的公式格式约定与踩坑记录(重要)

### 8.1 两条硬规则

1. **公式统一用行内写法**(单个美元号定界),不要用双美元号显示公式——我的 PyCharm Markdown 预览不认后者;
2. **公式里表示「小于」一律用 `\lt`**,不要写「小于号紧跟字母」的连写(例如把 n 小于 d 写成中间没有空格的连写)。

### 8.2 为什么第 2 条这么要命(2026-09-24 实测)

- **症状**:只要某一条公式里出现「小于号紧跟字母」,坏的**不是那一条**,而是**整篇笔记的所有公式**都退化成裸 LaTeX;刷新预览、重开标签都无效。
- **原因**:PyCharm 预览的链路是「Java 侧识别公式 → JS 侧用 MathJax 渲染」。这种写法会被 HTML 当成一个标签的开头,把页面结构的解析搞乱,于是整篇渲染流程中断,后面的公式全部停在"源码"状态。
- **证据**:`lecture_02.md` 全文 **0 处**这种写法(小于号后面要么是空格、要么是反斜杠),一直渲染正常;`lecture_03.md` 只有 **2 处**连写,却让整篇崩掉;把这两处改成 `\lt` 后立刻恢复(242 行整篇正常)。
- **安全写法**:小于号后面接**空格**,或接**反斜杠命令**(如 `\lt`),都是安全的;只有"紧跟字母"会出事。

### 8.3 自查方法

在 notes 目录执行下面这行,输出为空即安全:

```powershell
Get-ChildItem *.md | Select-String -Pattern '<[A-Za-z]'
```

### 8.4 排查这类「整篇渲染失败」的套路(备忘)

1. 先做**段落增量试**:把文件按小节切开、逐段加回去,每加一段看一次,问题会快速收敛到某一小段;
2. 再做**单变量对照**:把可疑的那一行复制成几个变体,每份只改一个写法,逐份看哪个变体坏;
3. **注意一个坑**:自己造的对照文件里,不要把"缩进 4 空格的行"放在文件顶层——Markdown 会把它当成缩进代码块,里面的公式当然不渲染,会得出假结论(这次就因此误判过一轮)。
