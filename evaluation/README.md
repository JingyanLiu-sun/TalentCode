# KamaClaude 上下文压缩（compact）评测体系

> 回答一个核心问题：**压缩省的钱**与**压缩掉的能力**各是多少？
> 三层评测体系，双轨分工：
> - **合成门禁（Tier1/Tier2）**：零成本确定性回归，每次提交跑（`make eval`）
> - **公开观测（Tier0）**：LoCoMo 真实对话分布的压缩保留率 + 裁判校准（M1/M6）
> - **τ-bench 续跑（Tier3）**：真实工具环境双臂对照任务完成率（M2/M4）
>
> 合成轨与公开轨分开产出、分开引用，**绝不做加权混合分**——
> 混合会同时掩盖回归与分布偏移。

## 目录结构

```
evaluation/
├── README.md                  # 本文档：方法论、指标口径、复现方式
├── compact/
│   ├── generate_cases.py      # Tier1 数据集生成器（确定性，无随机源）
│   └── cases.jsonl            # 入库数据集：24 cases（4 场景 × 3 尺寸 × 2 seed）
├── e2e/
│   ├── tasks.jsonl            # Tier2 任务集：4 个"压缩后续跑"任务
│   └── taubench_adapter.py    # Tier3：τ-bench 官方 Agent 的压缩注入子类
└── public/
    ├── prepare_datasets.py    # LoCoMo/SummEval 下载、sha256 校验、固定子集切片
    ├── locomo_adapter.py      # LoCoMo → KamaClaude 消息格式 + 埋点位置校验
    └── data/                  # 数据缓存（.gitignore，许可证原因不入库）

scripts/
├── run_compact_eval.py        # Tier1 harness（离线默认 / --live 真实摘要）
├── run_e2e_eval.py            # Tier2 harness（离线校验 / --live 真实续跑）
├── run_public_eval.py         # Tier0 harness（离线校验 / --live / --calibrate）
├── run_taubench_eval.py       # Tier3 harness（双臂对照 + 成本熔断）
└── benchmark_compact.py       # 早期单 fixture 基准（保留，作为快速冒烟）

tests/unit/
├── test_compact_eval.py       # Tier1 口径固化（确定性 / 埋点位置 / 100% 门禁）
├── test_e2e_eval.py           # Tier2 管道连通性（脚本化 provider 驱动完整 AgentLoop）
├── test_public_eval.py        # Tier0 转换/校验/确定性口径零成本单测
└── test_taubench_adapter.py   # Tier3 格式转换与压缩注入切分（τ-bench 缺失时跳过）

output/evaluation/             # 报告输出（不入库）
├── compact-eval-report.{json,md}
├── e2e-eval-report.{json,md}
├── public-eval-report.{json,md}
└── taubench-eval-report.{json,md}
```

## 指标体系（6 个互补指标）

| 指标 | 定义 | 计算方式 | 门禁 | 适用边界 |
|---|---|---|---|---|
| **M1 信息保留质量** | 压缩后仅凭摘要+tail 答对金标 QA 的比例 | LoCoMo 子集：正确数/可答题数（相对不压缩基线的保留率），按单跳/多跳/时序分类 | 观察值；相对上次改动降幅 >10pp 报警 | 生活对话域，无工具消息；不外推代码域 |
| **M2 任务完成率保留** | 压缩对端到端任务成功率的影响 | τ-bench Δpass^1 = pass^1(压缩) − pass^1(基线) | ≥ −5pp 且压缩臂 ≥ 基线 80% | 客服域确定性环境；用户模拟有方差，固定 seed |
| **M3 压缩成本** | 省下的 token 与真实花费 | 预处理压缩比 + 端到端压缩比 + τ-bench 美元累计（含压缩调用开销） | 预处理比 <50%（CI 硬门禁）；端到端 ≥2x（观察） | 依赖 tokenizer 与价目表版本，报告带版本戳 |
| **M4 续跑可靠性** | 压缩后续跑的一致性与信息找回代价 | ① pass^k：同任务 k 次全通比例；② 重读率代理：压缩臂工具调用数/基线工具调用数（>1.5 报警） | pass^3 ≥ 0.6 × pass^1（观察）；重读率增幅 >50% 报警 | τ-bench 环境 |
| **M5 效率与回归风险** | 压缩链路自身的稳定性 | ① 压缩失败率（`compact_messages` 返回 None）；② 压缩延迟 p95；③ 合成集全部门禁 | 失败率 = 0（硬门禁）；合成门禁全部保留进 `make eval` | 失败率门禁对所有数据源生效 |
| **M6 裁判可信度** | LLM-as-judge 与人工/确定性口径的一致性 | SummEval 子集（人工标签）或 LoCoMo 自校准（确定性 EM）的 Cohen's kappa | ≥ 0.6 才允许引用语义类指标 | 校准集与对话域存在偏移，只做方向性校准 |

## Tier1：静态信息保留率

**思路**：在合成多轮对话中预埋 6 类"埋点事实"，直接调用
`Compactor.compact_messages()`，检查事实是否幸存。不跑 Agent 循环，变量只有压缩本身。

### 数据集设计（`evaluation/compact/cases.jsonl`）

每个 case = 10/14/18 轮真实对话（每轮含超长 bash 输出、read_file 代码文件、
错误结果），预埋 6 类事实：

| 事实类型 | 校验口径 | 埋点位置 |
|---|---|---|
| `path`（文件路径） | 字面：字符串精确匹配 | read_file 工具调用 |
| `critical_id`（关键 ID） | 字面 | tool_use input |
| `error`（报错原文） | 字面 | is_error 的 tool_result |
| `constraint`（编码约束） | 语义：LLM-as-judge | 用户消息 |
| `decision`（技术决策） | 语义 | 用户消息 |
| `todo`（未决事项） | 语义 | 用户消息 |

关键约束：**所有埋点轮次 ≤ total − 4**，保证事实落在 history 区
（最近 `TAIL_ROUNDS=3` 轮之外）——必须经过摘要才能幸存，测的是压缩质量
而不是"恰好还在 tail 里"。深度分桶（early < 34% / mid < 67% / deep）
用于输出**保留率随埋点深度的衰减曲线**。

### 指标口径

| 指标 | 口径 | 门禁 |
|---|---|---|
| 字面事实预处理保留率 | 确定性预处理后，字面事实仍可字符串匹配的比例 | **= 100%**（确定性手段不允许丢字面信息），由单测固化 |
| 预处理压缩比 | `count_messages_tokens(预处理后) / 预处理前` | < 50% |
| 端到端压缩比（--live） | 原文 token /（摘要 + tail token），tiktoken 计量 | 观察值 |
| 字面保留率（--live） | 摘要+tail 中字面事实可匹配比例 | 观察值 |
| 语义保留率（--live） | LLM-as-judge 逐事实判定"可从摘要恢复"的比例 | 观察值 |
| 六章节结构完整性（--live） | 摘要含 `## 1.` ~ `## 6.` 全部章节 | 观察值 |

字面/语义双口径的设计动机：字面校验零成本、完全确定，进 CI 当门禁；
语义校验必须花 LLM 调用，作为定期质量观测。**裁判提示词要求：
对 ID/路径/错误，"保留"必须是精确值可恢复**——防止裁判放水。

### 运行

```bash
uv run python -m evaluation.compact.generate_cases     # 重新生成数据集
uv run python scripts/run_compact_eval.py              # 离线（零成本，可进 CI）
uv run python scripts/run_compact_eval.py --live       # 含真实摘要质量
uv run python scripts/run_compact_eval.py --live --limit 4   # 冒烟
```

## Tier2：端到端任务评测（合成续跑）

**思路**：压缩的最终目的不是"摘要写得好"，而是"续跑做得对"。每个任务验证：
早轮预埋的关键信息，在压缩后仅凭摘要 + 最近 3 轮，续跑的模型能否正确使用。

### 四个任务（`evaluation/e2e/tasks.jsonl`）

| 任务 | 验证的信息类型 | 校验方式 |
|---|---|---|
| e2e-01-constraint | 早轮编码约束（类型标注+docstring）在压缩后仍被遵守 | regex 检查续跑产出的函数签名 |
| e2e-02-critical-data | 关键 ID 逐字恢复 | contains 精确匹配 |
| e2e-03-error-recall | 报错原文逐字召回（`[ERROR-VERBATIM]` 机制的端到端验证） | contains 精确匹配 |
| e2e-04-todo-continuity | 压缩后仍知道下一步该做什么 | LLM-as-judge |

### 执行链路（--live，每任务 2~3 次 LLM 调用）

```
构造预埋历史 → 真实 compact → 仅凭摘要+tail 续跑提问 → 确定性校验（含/正则/裁判）
```

离线模式做任务集 schema 校验 + 埋点位置校验；**管道连通性**
（水位触发压缩 → 摘要进上下文 → 循环正常终止）由
`tests/unit/test_e2e_eval.py` 用脚本化 provider 驱动**完整 AgentLoop + Compactor**
覆盖，零 API 成本、可进 CI。

```bash
uv run python scripts/run_e2e_eval.py            # 离线校验
uv run python scripts/run_e2e_eval.py --live     # 真实续跑评测
```

## Tier0：公开数据集观测（LoCoMo，M1/M6）

**动机**：合成集分布单一（模板对话、受控埋点），压缩器对真实噪声分布
（跑题、冗余、口语化）的鲁棒性未被检验。LoCoMo（Snap Research, ACL 2024）
提供 10 段 ~300 轮人工校对的长对话 + 金标答案 QA，正补这块。

### 数据集选型与不采纳理由

| 数据集 | 结论 | 理由 |
|---|---|---|
| **LoCoMo** | 采纳（M1） | 真实分布长对话 + 金标 QA；每段 ~9K token，成本极低 |
| **SummEval 子集** | 采纳（M6 校准） | 1600 条人工事实一致性标注；校准现有裁判提示词 |
| **τ-bench** | 采纳（Tier3） | 确定性工具环境 + 数据库状态校验，见下节 |
| SWE-bench | 不采纳 | 单任务需 Docker 沙箱、小时级、数美元；成败由代码能力主导，压缩单变量信噪比太低 |
| LongMemEval | 不采纳 | _M 档 1.5M token 不可跑；_S 档成本不可控；且测"检索式记忆"而非"摘要压缩"，机制错位 |
| CNN/DM、XSum + ROUGE | 不采纳 | 压缩目标是"事实可恢复、续跑可执行"，不是摘要措辞质量；ROUGE 与目标不相关 |

### 数据准备与复现（`evaluation/public/prepare_datasets.py`）

- 原始数据**不入库**（LoCoMo 研究许可）：一次性下载 → sha256 校验 →
  切出固定子集（2 段对话 × 每类 ≤15 题）缓存到 `evaluation/public/data/`
- 子集哈希写入 `manifest.json`，离线校验每次核对——**子集锁死防刷榜，
  禁止针对子集调 `_COMPACT_PROMPT`**
- 网络不可达时支持手动放置原始数据（脚本只校验哈希）

### 评分口径

- 格式转换：`locomo_adapter.py` 把 LoCoMo session 展平成 user/assistant
  交替消息；**证据全部落在 history 区（复用 `_find_tail_boundary` 校验）
  的题目才入选**——保证测的是"经过摘要才能答对"
- 单跳题：确定性 EM / 归一化 token F1（SQuAD 式）为主；多跳/时序题：
  judge 对照金标为辅口径
- `--with-baseline`：同批题在完整未压缩历史上作答作为上界，
  **保留率 = 压缩后正确数 / 未压缩正确数**，消除题目难度噪声

```bash
uv run python -m evaluation.public.prepare_datasets      # 下载校验 + 子集切片（一次性）
uv run python scripts/run_public_eval.py                 # 离线校验（零成本，可进 CI）
uv run python scripts/run_public_eval.py --live          # M1 压缩保留率
uv run python scripts/run_public_eval.py --live --with-baseline
uv run python scripts/run_public_eval.py --calibrate     # M6 裁判校准（kappa ≥ 0.6 门禁）
```

## Tier3：τ-bench 压缩续跑双臂评测（M2/M4）

**动机**：生产续跑是全工具 ReAct 循环，而 Tier2 只测到无工具单次问答；
压缩后模型是否因信息缺失反复重调工具（成本反噬），需要真实工具环境证据。

### 注入方式（`evaluation/e2e/taubench_adapter.py`）

不把 KamaClaude AgentLoop 塞进 τ-bench（工具 schema 不匹配），而是反向：
`CompactingToolCallingAgent` 继承官方 `ToolCallingAgent`，在消息装配处注入——
历史 token 超阈值时，最近 3 轮原样保留（工具调用链不动），更早历史转
KamaClaude 格式交给 `Compactor.compact_messages()`，替换为 [摘要, 确认] 消息对。
**被测对象是压缩器本体**；任务环境、评分（数据库状态校验）全部复用
τ-bench 官方实现，结果可与社区对照。压缩失败不中断任务，记入失败率门禁。

```bash
# τ-bench 为可选依赖（不进 pyproject，避免 git 源进 uv.lock 破坏 CI）
uv pip install git+https://github.com/sierra-research/tau-bench.git

uv run python scripts/run_taubench_eval.py                    # 冒烟：3 任务 × 2 臂
uv run python scripts/run_taubench_eval.py --num-tasks 20     # 完整子集
uv run python scripts/run_taubench_eval.py --repeat 3         # pass^3
```

内置 `--max-cost` 美元熔断（agent + user simulator 累计）；固定 `--seed`。
注意：litellm 价目表不认识的模型名（如代理端自定义模型）会记成本为 0，
此时熔断线不生效，需按墙钟时间人工控制规模。

## 公开数据与合成数据的共存策略

- **双轨分工，不合并总分**：合成集 = 每次提交的零成本确定性回归门禁
  （`make eval`）；公开集 = 改压缩策略时 + 每周一次的外部有效性观测。
  两套报告分开产出、分开引用。
- **分歧处理规则**：合成过/公开挂 → 优先信公开（分布更接近生产），触发归因；
  合成挂/公开过 → 门禁照旧阻断（确定性口径优先）。
- **反"只看榜单"**：公开集分数只回答"相对基线掉了多少"（Δ 指标），
  不追求绝对排名——被测的是压缩器而非模型本身。
- 报告强制记录：数据集/子集哈希、模型版本、裁判版本、采样 seed。

## 与单测 / CI 的关系

| 层 | 内容 | 成本 | 何时跑 |
|---|---|---|---|
| 单测 | 数据集结构、埋点位置、100% 字面门禁、管道连通性、公开集转换口径 | 零 | 每次提交 |
| Tier1/Tier2 离线 | 24 cases 确定性指标 + 任务集校验 | 零 | 每次提交（`make eval`） |
| Tier0 离线 | 公开子集哈希/角色交替/证据位置校验 | 零 | 每次提交（`make eval` 可选接入） |
| Tier1/2 --live | 摘要质量、深度衰减、续跑通过率 | 低 | 改压缩策略后手动跑 |
| Tier0 --live/--calibrate | M1 保留率、M6 裁判 kappa | 低 | 改压缩策略后 / 每周 |
| Tier3 | τ-bench 双臂对照 | 中（有熔断） | 改压缩策略后，先冒烟再放子集 |

## 口径声明（务必随报告一起引用）

- 合成轨为**受控合成数据**：埋点位置、类型、深度完全可控，用于验证
  压缩策略的相对质量与回归门禁，**不代表生产环境真实会话效果**；
- 公开轨（LoCoMo）为生活对话域、无工具消息，不外推到代码域；
  其已知缺陷（部分对话模型生成、事实漂移）通过子集锁死 + 确定性
  EM 主口径 + 相对基线保留率缓解；
- 语义保留率依赖裁判模型：**M6 kappa ≥ 0.6 才允许引用语义类指标**，
  低于阈值时报告自动降级为字面/EM 口径；
- 全工具循环下的**重读率**（模型为找回信息额外付出的工具调用）已在
  Tier3 以工具调用倍率代理落地（M4）；
- 合成数据集入库且逐字节确定性（单测校验生成器同步）；公开子集哈希锁死；
  所有报告记录数据集哈希、模型版本、采样 seed，保证可复现。
