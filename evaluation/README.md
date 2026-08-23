# KamaClaude 上下文压缩（compact）评测体系

> 回答一个核心问题：**压缩省的钱**与**压缩掉的能力**各是多少？
> 两级评测：Tier1 静态信息保留率（可进 CI，离线门禁零成本）+ Tier2 端到端任务评测（真实续跑）。

## 目录结构

```
evaluation/
├── README.md                  # 本文档：方法论、指标口径、复现方式
├── compact/
│   ├── generate_cases.py      # Tier1 数据集生成器（确定性，无随机源）
│   └── cases.jsonl            # 入库数据集：24 cases（4 场景 × 3 尺寸 × 2 seed）
└── e2e/
    └── tasks.jsonl            # Tier2 任务集：4 个"压缩后续跑"任务

scripts/
├── run_compact_eval.py        # Tier1 harness（离线默认 / --live 真实摘要）
├── run_e2e_eval.py            # Tier2 harness（离线校验 / --live 真实续跑）
└── benchmark_compact.py       # 早期单 fixture 基准（保留，作为快速冒烟）

tests/unit/
├── test_compact_eval.py       # Tier1 口径固化（确定性 / 埋点位置 / 100% 门禁）
└── test_e2e_eval.py           # Tier2 管道连通性（脚本化 provider 驱动完整 AgentLoop）

output/evaluation/             # 报告输出（不入库）
├── compact-eval-report.{json,md}
└── e2e-eval-report.{json,md}
```

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

## Tier2：端到端任务评测

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

## 与单测 / CI 的关系

| 层 | 内容 | 成本 | 何时跑 |
|---|---|---|---|
| 单测 | 数据集结构、埋点位置、100% 字面门禁、管道连通性 | 零 | 每次提交 |
| Tier1 离线 | 24 cases 确定性指标 | 零 | 每次提交（`make eval`） |
| Tier1 --live | 摘要质量、深度衰减曲线 | 低 | 改压缩策略后手动跑 |
| Tier2 --live | 续跑任务通过率 | 低 | 改压缩策略后手动跑 |

## 口径声明（务必随报告一起引用）

- 历史对话为**受控合成数据**：事实埋点位置、类型、深度完全可控，用于验证
  压缩策略的相对质量与回归门禁，**不代表生产环境真实会话效果**；
- 语义保留率依赖裁判模型，存在判定噪声；字面指标为确定性口径，优先引用；
- 全工具循环下的**重读率**（模型为找回信息额外付出的工具调用）需要真实
  工具环境，列入后续扩展；
- 数据集入库且逐字节确定性（`tests/unit/test_compact_eval.py` 校验
  `cases.jsonl` 与生成器同步），报告记录数据集 sha256，保证可复现。
