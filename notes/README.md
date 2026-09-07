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
回答文件中记录的所有问题(未勾选的条目;若已勾选则跳过)。回答要求:
1. 直接编辑该 md 文件:在每个问题条目下方追加一行"  - **AI:** ..."(问题有多问就分条列),并把该条目勾选为 [x];
   若单次答疑内容较多,把展开讲解、公式推导、延伸阅读写在文件末尾的"AI 补充笔记"区,按日期追加,不要覆盖我写的内容。
2. 回答以中文为主,公式用 LaTeX($...$ / $$...$$),关键结论给一句话直觉 + 严谨表述两层;
3. 优先基于讲义/课程材料回答;涉及课外知识(论文、数学背景)时先联网核实再作答,并注明出处链接;
4. 如果我的问题暴露了前置数学/知识缺口(如贝叶斯、优化、信息论),顺手在 D:\work_repo\lectures\notes\gap-list.md 里
   新增对应条目(标注来源讲座、分类 A 必补/B 延后/C 放弃、建议学习资源);
5. 如果有多个问题一次答不完,先回答最影响理解主线的 3 个,并告诉我其余已记入 backlog;
6. 不确定的地方要明说"这里我不确定",不要编造。

完成后,简短汇报:回答了哪些问题、在哪些小节写入了回答、补充了哪些 gap-list 条目(用 3~5 行)。
```

> 每周日对谈用同款 Prompt 开头,但把任务换成第 4 节的"总结 + 自测 + 更新缺口",并附上 `progress.md` 与本周 `lecture_XX.md` 列表。
