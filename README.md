# HTTT

HTTT 是一个面向科研与技术探索场景的命令行 Agent。它不试图直接给出“标准答案”，而是把一个模糊、失败驱动或指标驱动的问题，拆成可检索、可迁移、可执行的研究路线。

这个项目的出发点很直接：很多真实问题并不是“缺一个论文总结”，而是缺一个能把失败现象、工程约束、跨领域机制和落地风险连起来的工作流。HTTT 的目标就是把这条链路做成一个可复用的 CLI agent。

## Why This Project Exists

科研和工程探索里常见的卡点通常不是信息太少，而是信息太散：

- 问题描述是口语化的，不能直接检索。
- 同领域论文很多，但真正可迁移的方法很少。
- 方案看起来新颖，真正落地时却被算力、数据、时延或场景变化卡死。
- 每次新项目都从零开始，没有把之前踩过的坑沉淀成可调用记忆。

HTTT 把这些问题统一成一个流程：先抽象问题，再做跨领域类比，再检索证据，再保守评估迁移性，最后给出分阶段行动建议。

## Core Approach

HTTT 的主流程由几个模块串联：

1. `Parser`
   把用户输入解析成任务、目标、失败现象、约束和不确定性。
2. `Abstraction`
   把具体问题提炼成更通用的问题结构，避免检索只停留在原始领域关键词。
3. `Analogy`
   让多个“专家视角”围绕机制、约束、失败模式和反例展开讨论，产出跨领域 query family。
4. `Retriever`
   基于 Semantic Scholar 和本地 PDF/RAG 记忆检索候选论文。
5. `Judge`
   对候选方法做保守的迁移性评估，而不是只给“看起来相关”的论文列表。
6. `Composer`
   把证据整理成可执行路线，而不是停在论文摘要级别。
7. `Reflection`
   对结果做自检，必要时补检索或重跑。

## Core Innovations

和普通“问答式文献助手”相比，这个项目的重点不在聊天，而在工作流设计：

- 问题驱动，而不是论文驱动
  输入可以是失败现象、目标指标或研究想法，不要求用户先组织成标准 query。
- 类比驱动检索
  检索词不是人工硬写，而是从问题结构和多角色类比讨论里生成。
- 保守迁移评估
  系统会显式输出可迁移组件、迁移风险和适配成本，而不是默认“相关就可用”。
- 本地记忆与 workspace
  每个项目都有独立 workspace，便于积累上下文、案例、PDF 与会话状态。
- Skill / Soul 机制
  可以给 agent 追加方法偏好、检索策略或人格视角，而不必改主流程代码。

## Current Status

当前版本已经具备这些能力：

- 单次运行与交互式会话模式
- 本地项目级 workspace 和记忆存储
- Skill / Soul 加载与脚本触发
- PDF 下载、抽取与向量化 RAG
- Judge 追问
- 会话内模型切换
- CLI 阶段进度输出，避免长时间无反馈

## Implementation

### 1. Requirements

- Python `>= 3.10`
- 一个兼容 OpenAI Chat Completions / Embeddings 的 API
- 建议在 macOS / Linux / Windows PowerShell 下使用虚拟环境

### 2. Install

```bash
git clone https://github.com/friendlyCamel/TTQQ.git
cd TTQQ
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### 3. Configure LLM

你可以二选一：

方式 A，使用本地配置文件：

```bash
cp config/llm.example.json config/llm.json
```

然后填入自己的 `base_url`、`api_key`、`model`。

方式 B，直接用环境变量：

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_MODEL="qwen3.5-flash"
export OPENAI_EMBEDDING_MODEL="text-embedding-v3"
```

如果没有 `config/llm.json`，CLI 会优先读取这些环境变量。


### 4. Run

交互模式：

```bash
httt
```

或：

```bash
httt session --project default
```

单次运行：

```bash
httt run --query "毫米波手势识别换房间后精度大幅下降，想提升跨场景泛化并保证边缘实时。"
```

执行时 CLI 会输出阶段信息，例如：

```text
[parse] Parsing research problem
[abstract] Building reusable problem structures
[retrieve] Retrieving papers (round 1)
[judge] Assessing transferability (round 1)
```


### 5. Session Commands

常用命令：

- `/help`
- `/type idea-driven|failure-driven|metric-driven`
- `/model <model-name>`
- `/context <text>`
- `/skills`
- `/skill <name|off>`
- `/skillrun <args>`
- `/souls`
- `/soul <name|off>`
- `/status`
- `/exit`

### 6. Workspace Layout

默认项目目录：

```text
workspace/<project>/
```

典型内容：

- `workspace/<project>/memory/session_state.json`
- `workspace/<project>/memory/cases.jsonl`
- `workspace/<project>/memory/rag_vectors.db`
- `workspace/<project>/papers/*.pdf`


### 7. Skills And Souls

默认目录：

- `skills/`
- `souls/`

兼容两种组织方式：

- `skills/<name>.md`
- `skills/<name>/SKILL.md`

`souls/` 同理。

Skill 用来改变检索与执行策略，Soul 用来改变评审视角或表达偏好。

### 8. Project Structure

```text
src/agent_v1/
  cli.py
  orchestrator.py
  llm.py
  memory.py
  session_memory.py
  modules/
  retrieval/
skills/
souls/
config/
```


## Notes

- 当前检索依赖 Semantic Scholar API，可用性受网络环境影响。
- 这是一个偏工程化的研究助手，不是通用聊天机器人。
