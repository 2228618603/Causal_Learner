# API 生成 CoT-QA：最终数据样例与生成流程（当前代码版本）

本文档基于当前仓库代码（`ECCV/cot/generate_cot_dataset_api.py` + `ECCV/cot/validate_cot_dataset.py`），详细说明：

- **最终产出的 CoT-QA(JSONL) 长什么样**（包含完整字段示例）
- **API 生成 CoT 的逻辑结构与流程**
- **API 侧的输入/输出约束与 prompt 结构**（system/user prompt 模板）
- **质量保障策略**（锚点句、答案原样拷贝、泄漏检测、重试/丢弃、离线校验）

> 重要口径（已固化在代码中）：
> - `Q`：**只保留单行自然问句**（不包含任何 `fields.*` 行；不包含 options/candidates 列表；不在 Q 中引导模型如何写 CoT）。
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
  cumulative_last_frame_segments/      # 可选：prefix mp4（影响 prefix 相关任务）
  stage1/sampled_frames/               # 可选：场景均匀帧（用于补充 evidence）
```

关键点：

- `causal_plan_with_keyframes.json` **必须严格符合** `ECCV/three_stage/prompts.py` 的 schema；否则生成器会跳过该 video（避免垃圾输入）。
- 生成器会按任务需要，从 step folder 中选择 1–4 张图作为 evidence（`image` 字段）；有些任务会额外使用 prefix 视频（`video` 字段）。

---

## 2) 生成流程概览（从代码结构看）

入口脚本：`ECCV/cot/generate_cot_dataset_api.py`

核心流程（简化）：

1) 遍历 `<input_root>/<video_id>/`
2) 读取并做最小 schema 校验：`causal_plan_with_keyframes.json`
3) 依据 Task_28–Task_42 的规则，构造若干 `BaseSample`：
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

## 2.1 任务覆盖、Q 口径与答案形态（Task_28–Task_42）

生成器支持的任务集合为 Task_28–Task_42（均从 `causal_plan_with_keyframes.json` 派生）。在**当前口径**下：

- 所有任务的 `conversations[0].value`（Q）都是**单行自然问句**
- 所有任务的 `conversations[1].value`（A）都满足：`<think>...</think>\\n<gold_answer>\\n`
- **gold_answer 由生成器提供并强制模型原样拷贝**（模型主要负责生成合格的 CoT）

常见任务的输出形态（按代码实际 prompt）：

- `Task_28`：Q 固定；Answer 为 1 句解释文本（生成器拼接得到；用于对齐跨步依赖）
- `Task_29/30`：Q 固定；Answer 为下一步 `step_goal`（严格匹配）
- `Task_31`：Q 固定；Answer 为整数 `step_id`
- `Task_32/34/36`：Q 固定；Answer 为多行编号列表 `1) ...`（严格匹配）
- `Task_33`：Q 固定（包含具体数值 k）；Answer 为逗号分隔的大写字母集合（如 `A,C,E`）
- `Task_35`：Q 固定；Answer 为一行 label（`FlawStep=...; FlawType=...; Reason=...`）
- `Task_37`：Q 来自 `counterfactual_challenge_question`（单行化后输出）；Answer 为 `expected_challenge_outcome`（严格匹配）
- `Task_38`：Q 为单行问句（由 `counterfactual_challenge_question` 规范化得到的 `What is the most likely outcome if ...?`）；Answer 为 `A/B/C/D`（注意：4 个候选项放在 `meta.fields.options`，不出现在 Q 中）
- `Task_40/42`：Q 固定；Answer 为 `A/B/C/D`（注意：4 个候选项放在 `meta.fields.options`，不出现在 Q 中）
- `Task_39`：Q 固定（会把失败原因嵌入问句）；Answer 为 `failure_reflecting.recovery_strategy`（严格匹配）
- `Task_41`：Q 固定；Answer 为二分类 `retry_current_step|continue_next_step`

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
- 对象名会做 `_`→空格 的 humanize（如 `jar_lid` → `jar lid`），锚点句以该结果为准。

---

## 4) 最终输出（CoT-QA JSONL）长什么样

### 4.1 目录结构（按 task 分目录）

输出目录结构：

```
<cot_root>/
  Task_29_Next_Action_Prediction/
    data.jsonl
  Task_33_Next_K_Steps_MultiSelect_From_Prefix/
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
    fields: Record<string, any>; // 任务特定字段（如 options/label 等）
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
- 不包含 options/candidates 等列表（多选/四选一候选项放在 `meta.fields.options`）。
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

- `Task_28/29/30/37`：`<text>`
- `Task_31`：`<int>`
- `Task_33`：`<comma-separated letters like A,C,E>`
- `Task_32/34/36`：多行编号列表（如 `1) ...`）
- `Task_35`：`FlawStep=<int>; FlawType=<type>; Reason=<one sentence>`
- `Task_38/40/42`：`<A/B/C/D>`
- `Task_39`：`<recovery_strategy>`
- `Task_41`：`retry_current_step|continue_next_step`

### 4.6 核心字段说明（便于快速理解）

每行一个 JSON object（ShareGPT 风格），常用关键字段含义如下：

- `image`: evidence 图片路径列表（允许包含 `frame_###_ts_...jpg`；但这些字符串 **不得** 出现在对话文本里）
- `conversations[0]`: `{"from":"human","value":"<single-line Q>"}`（单行问句）
- `conversations[1]`: `{"from":"gpt","value":"<think>CoT</think>\\n<Answer>\\n"}`（CoT + 原答案）
- `meta.source_path`: 指向该 video 的 `causal_plan_with_keyframes.json`
- `meta.fields`: 任务相关字段（如 next_step_goal、options、label 等），用于训练/评测时拼装输入或做一致性校验

> 说明：`sample.context` 只用于 API prompt，不会直接写入 JSONL；需要时可通过 `meta.source_path` 回读三阶段产物获取完整上下文。

---

## 5) 最终数据示例（真实输出形态）

### 示例 1：Task_29（下一步动作预测，文本答案）

```json
{
  "id": "2b0a1fe3-3b7e-4e19-9d6d-2b1b2a6e0c1f",
  "image": [
    "video_0001/01_loosen_and_remove_the_jar_lid/frame_005_ts_12.34s.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "What is the next planned action?"
    },
    {
      "from": "gpt",
      "value": "<think>Spatially, the hands must be holding the jar. Functionally, the jar must be portable. After the action, spatially, the jar will be above the bowl. After the action, functionally, the jar will be pourable. A likely failure is that The jar cannot be aligned cleanly over the bowl opening. If that happens, Clear the area around the bowl opening and re-align the jar while maintaining a stable hold. With these preconditions and intended effects established, the next planned action is to position the open jar above the bowl.</think>\nPosition the open jar above the bowl.\n"
    }
  ],
  "meta": {
    "task_name": "Task_29_Next_Action_Prediction",
    "item_type": "three_stage",
    "evidence_type": "keyframe_single",
    "source_path": "video_0001/causal_plan_with_keyframes.json",
    "step_index": 1,
    "fields": {
      "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
      "current_step_goal": "Loosen and remove the jar lid.",
      "next_step_goal": "Position the open jar above the bowl."
    },
    "evidence_files": [
      "video_0001/01_loosen_and_remove_the_jar_lid/frame_005_ts_12.34s.jpg"
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

### 示例 2：Task_42（恢复后下一步选择，MCQ，答案为字母）

> 本任务的 Q 仍然是单行自然问句；4 个候选项不会写在 Q 里，而是放在 `meta.fields.options` 中。

```json
{
  "id": "c6dbbd3c-1ad8-4d76-a4c9-0e6f0b7cf0f4",
  "image": ["video_0007/stage1/sampled_frames/sample_010_ts_12.00s.jpg"],
  "video": "video_0007/cumulative_last_frame_segments/segment_start_to_step06_last.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "After applying the recovery strategy, what is the most appropriate next step?"
    },
    {
      "from": "gpt",
      "value": "<think>Spatially, the hands must be holding the jar. Functionally, the jar must be portable. After the action, spatially, the jar will be above the bowl. After the action, functionally, the jar will be pourable. A likely failure is that the jar cannot be aligned cleanly over the bowl opening. If that happens, clear the area around the bowl opening and re-align the jar while maintaining a stable hold. Since the recovery restores the preconditions and preserves the intended effects, the best next step is the option that continues the plan from this corrected state.</think>\nA\n"
    }
  ],
  "meta": {
    "task_name": "Task_42_Next_Step_After_Recovery",
    "item_type": "three_stage",
    "evidence_type": "video_prefix",
    "source_path": "video_0007/causal_plan_with_keyframes.json",
    "step_index": 6,
    "fields": {
      "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
      "failure_reason": "the jar cannot be aligned cleanly over the bowl opening",
      "recovery_strategy": "clear the area around the bowl opening and re-align the jar while maintaining a stable hold",
      "prefix_end_step": 6,
      "options": [
        "Position the open jar above the bowl.",
        "Tilt the jar to pour the contents into the bowl.",
        "Put the jar back on the shelf.",
        "Leave the workspace and stop."
      ],
      "label": "A",
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

> 注意：由于 MCQ 任务需要构造足够的“干扰项”，如果某个视频/计划的可用候选不足，生成器会直接跳过该样本（避免低质量或不满足约束的数据）。

---

## 6) 质量保障与校验策略（强约束）

### 6.1 API 输出强校验（生成阶段）

生成器在每次 API 返回后都会做 `_validate_api_payload()` 校验：

- 必须是 JSON object 且只能包含 `assistant_text`
- `assistant_text` 必须以 `<think>` 开头，且包含 `</think>`
- `<think>...</think>` 内必须是**单段落**（禁止 `\\n`/`\\r`）
- `<think>` 内必须包含全部 required anchors（逐字匹配）
- `</think>` 后的 Answer 必须与 `answer_block` **完全一致**（允许末尾换行差异）
- 对话文本禁止泄漏：`frame_### / sample_### / ts_... / .jpg/.mp4 / Frame 12 / Image 12` 等

不通过 → 自动重试；超出 `--max-sample-attempts` → 丢弃该样本（保证数据“宁缺毋滥”）。

### 6.2 离线严格校验（生成后）

脚本：`ECCV/cot/validate_cot_dataset.py`

额外硬约束：

- human 必须是**单行**，且不得包含 `fields.` 字样
- assistant 必须符合 `<think>...</think>` + answer
- 对每个 task，答案必须与 `meta.fields` 一致（如 Task_29/30 必须等于 `meta.fields.next_step_goal`；MCQ 必须等于 `meta.fields.label` 等）
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

- `--tasks`：选择生成的任务子集（默认 Task_28–Task_42 全部）
- `--require-video-prefix`：缺少 prefix mp4 时跳过 prefix 相关样本（更一致）
- `--abs-paths`：输出路径用绝对路径（便于跨目录调试）
- `--max-sample-attempts`：单样本 API 失败重试上限
- `--max-tokens`：单样本最大输出 tokens（CoT 被约束为单段落，通常不需要很大）

---

## 8) FAQ（为什么 Q 这么“短”）

### Q: 为什么 Q 不包含 fields/options/plan steps？

因为当前数据口径要求 **Q 必须是“单行自然问句”**，不引导模型写 CoT，也不携带结构化字段；结构化信息由以下位置承载：

- 生成阶段：`sample.context` 会被注入到 API prompt 中（gold 上下文）
- 输出数据：关键信息写入 `meta.fields`，并提供 `meta.source_path` 可回读完整 `causal_plan_with_keyframes.json`

如果你在训练时需要把 `meta.fields.options` 或 `plan steps` 拼回到模型输入，可由 data loader 在训练前自行组装（但不改变 `conversations[0].value` 的原始 Q）。
