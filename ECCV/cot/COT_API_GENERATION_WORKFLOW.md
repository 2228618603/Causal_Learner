# API 生成 CoT-QA：最终数据样例与生成流程（当前代码版本）

本文档基于当前仓库代码（`ECCV/cot/generate_cot_dataset_api.py` + `ECCV/cot/validate_cot_dataset.py`），详细说明：

- **最终产出的 CoT-QA(JSONL) 长什么样**（包含完整字段示例）
- **API 生成 CoT 的逻辑结构与流程**
- **API 侧的输入/输出约束与 prompt 结构**（system/user prompt 模板）
- **质量保障策略**（锚点句、答案原样拷贝、泄漏检测、重试/丢弃、离线校验）

> 重要口径（已固化在代码中）：
> - `Q`：**只保留单行自然问句**（不包含任何 `fields.*` 行；不在 Q 中引导模型如何写 CoT；对需要候选步骤输入的任务（如重排/坏计划诊断/修复），候选步骤以单行 inline 列表出现在 Q 中）。
> - `A`：在原始答案最前面添加 CoT，并用 `<think>...</think>` 包裹；`</think>` 后**紧跟原始答案**，答案必须逐字一致。

---

## 1) 输入（从三阶段产物读取）

CoT 生成器的输入是三阶段管线的最终产物目录（`--input-root`），其下每个视频 item 的最小结构为：

```
<input_root>/<video_id>/
  causal_plan_with_keyframes.json
  01_<slug>/ 02_<slug>/ ...            # step folders（用于取关键帧路径）
    step_meta.json
    frame_###_ts_XX.XXs.jpg            # 关键帧图
  stage1/                              # 全视频帧池（用于 Task_19 的 head/tail glimpses）
    frame_manifest.json                # 推荐：帧索引清单（保证顺序一致）
    sampled_frames/
  cumulative_last_frame_segments/      # 可选：prefix mp4（影响 prefix 相关任务）
  stage2/step_segments.json            # 可选：step→帧池索引（用于更精确对齐 Task_19 head/tail）
```

关键点：

- `causal_plan_with_keyframes.json` **必须严格符合** `ECCV/three_stage/prompts.py` 的 schema；否则生成器会跳过该 video（避免垃圾输入）。
- 生成器会按任务需要构造 evidence（`image`/`video` 字段）：
  - 多数任务使用 step folder 的关键帧图（1–4 张）
  - `Task_19` 优先使用 `stage1` 帧池做 head/tail glimpses（通常 8 张）；若缺失 Stage2 对齐信息则退化为按 step 顺序近似对齐
  - prefix 相关任务（如 `Task_18/20/21/22/23/27`）可使用 `cumulative_last_frame_segments/` 的 prefix mp4（`--require-video-prefix` 可强制要求）

---

## 2) 生成流程概览（从代码结构看）

入口脚本：`ECCV/cot/generate_cot_dataset_api.py`

核心流程（简化）：

1) 遍历 `<input_root>/<video_id>/`
2) 读取并做最小 schema 校验：`causal_plan_with_keyframes.json`
3) 依据 Task_17–Task_27（planning-only）的规则，构造若干 `BaseSample`：
   - `human_q`：单行自然问句（最终写入 JSONL 的 Q）
   - `context`：结构化 gold 上下文（只用于 API prompt；不直接写入 JSONL）
   - `fields`：用于自检/训练的关键字段（写入 `meta.fields`）
   - `answer_block`：精确 gold Answer（要求 API **原样拷贝**）
   - `required_anchors`：若干“必须逐字包含”的锚点句（强约束 grounded 的 CoT）
4) 对每条 `BaseSample` 调用 API：
   - 系统提示词：`SYSTEM_PROMPT_API_COT`
   - 用户提示词：`_build_api_user_prompt(sample)`
   - 要求输出严格 JSON：`{"assistant_text": "<think>...</think> + <answer>"}`
5) 对 API 输出做强校验 `_validate_api_payload()`：
   - JSON schema 正确
   - `<think>` 单段落（无换行）
   - 必须包含全部锚点句（逐字匹配）
   - `</think>` 后的答案必须与 `answer_block` **完全一致**
   - 文本禁止泄漏 frame/path/timestamp 等
   - 不通过则自动重试（`--max-sample-attempts`），仍失败则丢弃该样本
6) 写出 JSONL：`<output_dir>/<task_name>/data.jsonl`
7) 可选：`--post-validate` 触发离线严格校验（见第 6 节）

---

## 2.1 任务覆盖、Q 口径与答案形态（Task_17–Task_27）

生成器支持的任务集合为 Task_17–Task_27（均从 `causal_plan_with_keyframes.json` 派生）。在**当前口径**下：

- 所有任务的 `conversations[0].value`（Q）都是**单行自然问句**
- 所有任务的 `conversations[1].value`（A）都满足：`<think>...</think>\\n<gold_answer>\\n`
- **gold_answer 由生成器提供并强制模型原样拷贝**（模型主要负责生成合格的 CoT）

常见任务的输出形态（按代码实际 prompt）：

- `Task_17`：跨步依赖解释；Answer 为 1 句解释文本（生成器拼接得到；用于对齐 effect→precondition）
- `Task_18/27`：前缀预测下一步；Answer 为一个 `step_goal`（严格匹配）
- `Task_19/20/21/23`：补全/多步预测/重排/修复；Answer 为多行编号列表 `1) ...`（严格匹配）
- `Task_22`：坏计划缺陷定位；Answer 为一行 label（`FlawStep=...; FlawType=...; Reason=...`）
- `Task_24/25`：反事实结果预测；Answer 为 `expected_challenge_outcome`（严格匹配）
- `Task_26`：失败恢复策略；Answer 为 `failure_reflecting.recovery_strategy`（严格匹配）

> 样本并非“每个视频都有每个任务”：若无法构造足够证据/候选项/依赖关系，生成器会跳过对应样本（见第 5 节示例后的说明与第 6 节质量策略）。

---

## 3) API Prompt 结构（system + user）

### 3.1 System prompt（代码原文）

来自 `ECCV/cot/generate_cot_dataset_api.py: SYSTEM_PROMPT_API_COT`：

```text
You are an expert dataset annotation assistant.

You will be given:
1) A task question (human message).
2) Structured context extracted from a gold three-stage annotation.
3) An exact required final Answer text that MUST be preserved verbatim.

Your job:
- Write high-quality reasoning in English, strictly grounded in the provided context only.
- The reasoning MUST be one coherent natural-language paragraph (no bullet lists, no line breaks).
- In the reasoning, explicitly cover causal planning based on:
  1) spatial + affordance preconditions,
  2) spatial + affordance effects,
  3) failure reflecting (failure reason + recovery strategy),
  and conclude with why the final answer follows.
- Do NOT invent new objects, relations, affordances, steps, tools, failure modes, or outcomes.
- Do NOT reference frames/images/timestamps/file paths or any indexing.

Output format:
- Return STRICT JSON only (no markdown, no extra text).
- Schema: {"assistant_text": "<think>...</think> + <answer>"}.
- assistant_text MUST start with "<think>" and contain exactly one reasoning paragraph inside <think>...</think>.
- After </think>, output the exact required Answer text verbatim (do NOT add "Answer:" or any extra prefixes).
```

### 3.2 User prompt 模板（结构）

`_build_api_user_prompt(sample)` 会按如下结构组织（中间会插入 `sample.context` 的 JSON）：

```text
Task name: <task_name>

Task question (human message):
<single-line natural question>

Structured context (gold; do NOT invent new facts):
<pretty-printed JSON>

Exact required final Answer text (copy verbatim after </think>; do NOT change anything inside it):
<answer_block>

You MUST include the following exact sentences verbatim in the <think> paragraph:
- <anchor sentence #1>
- <anchor sentence #2>
...

Constraints:
- Output STRICT JSON only, schema: {"assistant_text": "..."}
- assistant_text MUST start with "<think>" and contain exactly one paragraph inside <think>...</think>.
- The CoT inside <think> must be ONE paragraph (single line; no bullet lists; no extra headings).
- In the CoT, analyze: spatial+affordance preconditions -> spatial+affordance effects -> failure reflecting (failure+recovery) -> conclude.
- After </think>, output the exact required Answer text above verbatim (no 'Answer:' prefix).
- Do NOT mention frames/images/timestamps/file paths or any indexing
```

### 3.3 “锚点句（required anchors）”如何生成

锚点句来自 `ECCV/cot/generate_cot_dataset_api.py: _build_required_anchors()`，用于强约束 CoT 的 grounded 性与结构顺序。

对某个 step（或某两个 step）会抽取：

1) 第 1 条 spatial precondition → 生成句式：`Spatially, ... must ... .`
2) 第 1 条 affordance precondition → `Functionally, ... must be ... .`
3) 第 1 条 spatial effect → `After the action, spatially, ... will ... .`
4) 第 1 条 affordance effect → `After the action, functionally, ... will be ... .`
5) `failure_reflecting.reason` → `A likely failure is that ... .`
6) `failure_reflecting.recovery_strategy` → `If that happens, ... .`

这些句子要求出现在 `<think>...</think>` 内，并且是**逐字匹配**（大小写/空格/标点需一致）。

可读性建议（影响 CoT 自然度）：

- 生成器会把 `failure_reflecting.reason` 拼进 `A likely failure is that ... .`；把 `recovery_strategy` 拼进 `If that happens, ... .`  
  因此更推荐把两者写成**不带句号的短语/从句**，并尽量以小写开头（例如 `the jar cannot be aligned...` / `clear the area...`），避免出现 `that The ...` 或 `If that happens, Clear ...` 这类大小写不自然的句子。
- 锚点句会从 `causal_chain.causal_*` 的**第一条编号项**抽取，并加上 `Spatially/Functionally/After the action ...` 前缀；不会对对象名做额外改写（保持原字符串的命名）。

---

## 4) 最终输出（CoT-QA JSONL）长什么样

### 4.1 目录结构（按 task 分目录）

输出目录结构：

```
<cot_root>/
  Task_18_Next_Step_Goal_Prediction_From_Prefix/
    data.jsonl
  Task_20_Next_K_Steps_Prediction_From_Prefix_QA/
    data.jsonl
  ...
```

每个 `<task_name>/data.jsonl` 是 **JSONL**（每行 1 个 JSON object，UTF-8）。

### 4.2 单条样本的 JSON Schema（ShareGPT 风格）

每一行是一个 JSON 对象，核心字段如下（伪类型）：

```ts
type ConversationTurn = { from: "human" | "gpt"; value: string };

type AssistantGenerator = {
  type: "api_generate_v1";
  api_base_url: string;
  model_provider_id: string;
  model_name: string;
};

type CotEntry = {
  id: string;                 // uuid4
  image: string[];            // 非空；证据图路径（相对 input_root 或绝对路径）
  video?: string;             // 可选；证据视频路径（相对 input_root 或绝对路径）
  conversations: [            // 固定长度为 2
    { from: "human"; value: string }, // Q（单行自然问句）
    { from: "gpt"; value: string }    // A（<think>...</think> + 原答案）
  ];
  meta: {
    task_name: string;        // 任务名（应与目录名一致）
    item_type: "three_stage"; // 固定为 three_stage
    evidence_type: string;    // 如 keyframe_single / images_uniform_scene / video_prefix
    source_path: string;      // 指向 <video_id>/causal_plan_with_keyframes.json（相对 input_root 或绝对）
    step_index: number;       // 与样本对应的 step（通常是 step_id 或其位置）
    fields: Record<string, any>; // 任务特定字段（如 next_step_goal/label 等）
    evidence_files?: string[];   // 自动写入：等于 image + (video) 的路径集合
    assistant_generator: AssistantGenerator;  // 生成器写入，用于审计
  };
};
```

### 4.3 路径口径（相对 / 绝对）

- 默认：`image/video/source_path` 写相对路径（相对 `--input-root`）。
- 若开启 `--abs-paths`：`image/video/source_path` 写绝对路径（更方便跨目录调试）。

### 4.4 对话文本硬约束（Q/A）

**Q（conversations[0].value）**

- 必须是**单行自然问句**（不得包含换行）。
- 不包含任何 `fields.*` 上下文行。
- 对需要候选步骤输入的任务，候选步骤以**单行 inline 列表**出现在 Q 中（同时也会冗余写入 `meta.fields` 便于一致性校验）。
- 不在 Q 中引导模型如何生成 CoT。

**A（conversations[1].value）**

- 必须满足：`<think>...</think>` + **原始答案文本**（答案在末尾；不得改写）。
- `<think>...</think>` 内必须是**单段落**自然语言（禁止换行、禁止 bullet list）。
- `<think>` 里必须按因果规划顺序组织：spatial/affordance preconditions → spatial/affordance effects → failure_reflecting（失败原因+恢复策略）→ 自然过渡到答案成立。
- `</think>` 后必须紧跟 **gold Answer**，并与生成器提供的 `answer_block` 逐字一致（不得加 `Answer:` 前缀/不得改写）。

**泄漏禁止**

- `conversations[*].value` 文本禁止泄漏任何索引/路径信息：`frame_### / sample_### / ts_... / .jpg/.mp4 / Frame 12 / Image 12` 等。
- 证据文件路径只允许出现在条目字段 `image` / `video` / `meta.evidence_files` 中。

### 4.5 Answer 形态（按任务）

注意：本生成器会在 prompt 中提供**精确的 gold Answer text**，并要求模型在 `</think>` 之后**原样拷贝**；因此模型主要负责生成高质量 CoT，答案本身不应被改写。

常见 Answer 形式（以 `generate_cot_dataset_api.py` 的 task prompt 为准；更完整的任务口径见第 2.1 节）：

- `Task_17/24/25/26`：`<text>`
- `Task_18/27`：`<step_goal text>`
- `Task_19/20/21/23`：多行编号列表（如 `1) ...`）
- `Task_22`：`FlawStep=<int>; FlawType=<type>; Reason=<one sentence>`

### 4.6 核心字段说明（便于快速理解）

每行一个 JSON object（ShareGPT 风格），常用关键字段含义如下：

- `image`: evidence 图片路径列表（允许包含 `frame_###_ts_...jpg`；但这些字符串 **不得** 出现在对话文本里）
- `conversations[0]`: `{"from":"human","value":"<single-line Q>"}`（单行问句）
- `conversations[1]`: `{"from":"gpt","value":"<think>CoT</think>\\n<Answer>\\n"}`（CoT + 原答案）
- `meta.source_path`: 指向该 video 的 `causal_plan_with_keyframes.json`
- `meta.fields`: 任务相关字段（如 `dependency_support`、`next_step_goal`、`next_k_step_goals`、`label` 等），用于训练/评测时拼装输入或做一致性校验

> 说明：`sample.context` 只用于 API prompt，不会直接写入 JSONL；需要时可通过 `meta.source_path` 回读三阶段产物获取完整上下文。

---

## 5) 最终数据示例（真实输出形态）

### 示例 1：Task_18（前缀预测下一步 step_goal，文本答案）

```json
{
  "id": "2b0a1fe3-3b7e-4e19-9d6d-2b1b2a6e0c1f",
  "image": [
    "video_0001/01_loosen_and_remove_the_jar_lid/frame_005_ts_12.34s.jpg"
  ],
  "video": "video_0001/cumulative_last_frame_segments/segment_start_to_step01_last.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "Context: High-level goal: \"Open a jar and transfer its contents into a bowl so the contents are ready to be used.\" Last completed step (in this prefix): \"Loosen and remove the jar lid.\" What is the next step goal?"
    },
    {
      "from": "gpt",
      "value": "<think>Spatially, the jar is stably supported on the countertop. Functionally, the jar lid is loosened enough to be removable. After the action, spatially, the jar opening is unobstructed by the lid. After the action, functionally, the jar is in an open state ready for pouring. A likely failure is that the lid remains partially threaded and cannot be removed cleanly. If that happens, regrip the lid, apply steady counter-rotation, and ensure the threads fully disengage. With the jar now open and the pourable affordance established, the next step goal is to position the open jar above the bowl.</think>\nPosition the open jar above the bowl.\n"
    }
  ],
  "meta": {
    "task_name": "Task_18_Next_Step_Goal_Prediction_From_Prefix",
    "item_type": "three_stage",
    "evidence_type": "video_prefix",
    "source_path": "video_0001/causal_plan_with_keyframes.json",
    "step_index": 1,
    "fields": {
      "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
      "prefix_end_step": 1,
      "prefix_end_step_goal": "Loosen and remove the jar lid.",
      "next_step_goal": "Position the open jar above the bowl."
    },
    "evidence_files": [
      "video_0001/01_loosen_and_remove_the_jar_lid/frame_005_ts_12.34s.jpg",
      "video_0001/cumulative_last_frame_segments/segment_start_to_step01_last.mp4"
    ],
    "assistant_generator": {
      "type": "api_generate_v1",
      "api_base_url": "http://model.example/v1",
      "model_provider_id": "vertex_ai",
      "model_name": "gemini-3-pro-preview"
    }
  }
}
```

核对点：

- Q 只有一行自然问句
- `<think>` 只有一段（无换行），并包含 required anchors（逐字匹配）
- `</think>` 后面的 Answer 必须与 `meta.fields.next_step_goal` 完全一致（本任务由生成器强制“原样拷贝”）

### 示例 2：Task_27（恢复后下一步选择，文本答案）

```json
{
  "id": "c6dbbd3c-1ad8-4d76-a4c9-0e6f0b7cf0f4",
  "image": ["video_0007/stage1/sampled_frames/sample_010_ts_12.00s.jpg"],
  "video": "video_0007/cumulative_last_frame_segments/segment_start_to_step06_last.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "Context: High-level goal: \"Open a jar and transfer its contents into a bowl so the contents are ready to be used.\" Failure reason: \"the jar cannot be aligned cleanly over the bowl opening\" Recovery strategy: \"clear the area around the bowl opening and re-align the jar while maintaining a stable hold\" After applying the recovery strategy, what is the most appropriate next step? Answer as a single step_goal."
    },
    {
      "from": "gpt",
      "value": "<think>Spatially, the jar is positioned close enough to the bowl to be aligned. Functionally, the jar is open and can be tilted to pour. After the action, spatially, the jar opening will be above the bowl opening. After the action, functionally, the contents can be transferred into the bowl. A likely failure is that the jar cannot be aligned cleanly over the bowl opening. If that happens, clear the area around the bowl opening and re-align the jar while maintaining a stable hold. Once alignment is restored, the next step is to proceed with positioning the open jar above the bowl so the subsequent pouring action is feasible.</think>\nPosition the open jar above the bowl.\n"
    }
  ],
  "meta": {
    "task_name": "Task_27_Next_Step_After_Recovery_QA",
    "item_type": "three_stage",
    "evidence_type": "video_prefix",
    "source_path": "video_0007/causal_plan_with_keyframes.json",
    "step_index": 6,
    "fields": {
      "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
      "failure_reason": "the jar cannot be aligned cleanly over the bowl opening",
      "recovery_strategy": "clear the area around the bowl opening and re-align the jar while maintaining a stable hold",
      "prefix_end_step": 6,
      "prefix_end_step_goal": "Position the open jar above the bowl.",
      "current_step_goal": "Position the open jar above the bowl.",
      "next_step_goal": "Tilt the jar to pour the contents into the bowl.",
      "decision": "retry_current_step",
      "gold_next_step_goal": "Position the open jar above the bowl."
    },
    "evidence_files": [
      "video_0007/stage1/sampled_frames/sample_010_ts_12.00s.jpg",
      "video_0007/cumulative_last_frame_segments/segment_start_to_step06_last.mp4"
    ],
    "assistant_generator": {
      "type": "api_generate_v1",
      "api_base_url": "http://model.example/v1",
      "model_provider_id": "vertex_ai",
      "model_name": "gemini-3-pro-preview"
    }
  }
}
```

---

## 6) 质量保障与校验策略（强约束）

### 6.1 API 输出强校验（生成阶段）

生成器在每次 API 返回后都会做 `_validate_api_payload()` 校验：

- 必须是 JSON object 且只能包含 `assistant_text`
- `assistant_text` 必须以 `<think>` 开头，且包含 `</think>`
- `<think>...</think>` 内必须是**单段落**（禁止 `\\n`/`\\r`）
- `<think>` 内必须包含全部 required anchors（逐字匹配），并且**按给定顺序出现**
- `</think>` 后的 Answer 必须与 `answer_block` **完全一致**（允许末尾换行差异）
- 对话文本禁止泄漏：`frame_### / sample_### / ts_... / .jpg/.mp4 / Frame 12 / Image 12` 等

不通过 → 自动重试；超出 `--max-sample-attempts` → 丢弃该样本（保证数据“宁缺毋滥”）。

### 6.2 离线严格校验（生成后）

脚本：`ECCV/cot/validate_cot_dataset.py`

额外硬约束：

- human 必须是**单行**，且不得包含 `fields.` 字样
- assistant 必须符合 `<think>...</think>` + answer
- `<think>` 内必须显式体现既定 CoT 风格标记：`Spatially,` / `Functionally,` / `After the action, ...` / `A likely failure is that ...` / `If that happens, ...`
- 默认会根据 `meta.source_path` 回读三阶段产物并重算 required anchors，要求 `<think>` 内逐字包含且顺序一致（可用 `--no-anchor-check` 关闭）
- 对每个 task，答案必须与 `meta.fields` 一致（如 `Task_18` 必须等于 `meta.fields.next_step_goal`；`Task_22` 必须等于 `meta.fields.label`；`Task_27` 必须等于 `meta.fields.gold_next_step_goal`；列表任务必须与对应的 `meta.fields.*_steps` 一致）
- `id` 必须是有效的 UUID 字符串；`meta.evidence_files`（若存在）必须严格等于 `image + (video)` 的路径序列
- `--strict` 会检查 `image/video/source_path` 指向的文件是否存在

---

## 7) 如何运行（推荐命令）

建议先对三阶段产物做一次先验校验（强建议）：

```bash
python3 ECCV/three_stage/validate_three_stage_output.py --video-output-dir <three_stage_output_root>/<video_id>
```

生成（API-only）：

```bash
python3 ECCV/cot/generate_cot_dataset_api.py \
  --input-root <three_stage_output_root> \
  --output-dir <cot_root> \
  --api-base http://model.mify.ai.srv/v1 \
  --provider <MODEL_PROVIDER_ID> \
  --model <MODEL_NAME> \
  --api-key <API_KEY> \
  --seed 42 \
  --post-validate
```

仅做离线校验：

```bash
python3 ECCV/cot/validate_cot_dataset.py \
  --input-root <three_stage_output_root> \
  --cot-root <cot_root> \
  --strict
```

常用参数说明（见脚本 `--help`）：

- `--tasks`：选择生成的任务子集（默认 Task_17–Task_27 全部）
- `--require-video-prefix`：缺少 prefix mp4 时跳过 prefix 相关样本（更一致）
- `--abs-paths`：输出路径用绝对路径（便于跨目录调试）
- `--max-sample-attempts`：单样本 API 失败重试上限
- `--max-tokens`：单样本最大输出 tokens（CoT 被约束为单段落，通常不需要很大）

---

## 8) FAQ（为什么 Q 这么“短”）

### Q: 为什么 Q 不包含 `fields.*` 行？

因为当前数据口径要求 **Q 必须是“单行自然问句”**，不引导模型写 CoT，也不以“字段行/结构化 schema”形式把 `meta.fields` 展开到 Q 里。

但对需要显式候选步骤输入的任务（如重排/坏计划诊断/修复），候选步骤会以**单行 inline 列表**出现在 Q 中，否则任务本身不可定义。

- 生成阶段：`sample.context` 会被注入到 API prompt 中（gold 上下文）
- 输出数据：关键信息写入 `meta.fields`，并提供 `meta.source_path` 可回读完整 `causal_plan_with_keyframes.json`
