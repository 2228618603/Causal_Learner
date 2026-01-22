# CoT 数据格式说明（API 生成版，Task_28–Task_42）

本文档说明 `ECCV/cot/generate_cot_dataset_api.py` 产出的 **CoT(JSONL)** 的数据类型与文本约束（API-only）。

---

## 1) 目录结构（按 task 分目录）

输出目录固定为：

```
<cot_root>/<task_name>/data.jsonl
```

- `<cot_root>`：`--output-dir` 指定的目录
- `<task_name>`：如 `Task_29_Next_Action_Prediction`（必须与 `meta.task_name` 对齐）
- `data.jsonl`：每行 1 个样本（UTF-8，JSON object）

---

## 2) 单条样本的 JSON Schema（ShareGPT 风格）

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
    { from: "human"; value: string }, // prompt
    { from: "gpt"; value: string }    // CoT + Answer（见第 3 节模板）
  ];
  meta: {
    task_name: string;        // 任务名（应与目录名一致）
    item_type: "three_stage"; // 固定为 three_stage
    evidence_type: string;    // 如 keyframe_single / images_uniform_scene / video_prefix / video_clip
    source_path: string;      // 指向 <video_id>/causal_plan_with_keyframes.json（相对 input_root 或绝对）
    step_index: number;       // 与样本对应的 step（通常是 step_id 或其位置）
    fields: Record<string, any>; // 任务特定字段（如 options/label 等）
    evidence_files?: string[];   // 自动写入：等于 image + (video) 的路径集合
    assistant_generator: AssistantGenerator;  // 生成器写入，用于审计
  };
};
```

### 路径口径

- `--abs-paths`：若开启，`image/video/source_path` 写绝对路径。
- 未开启：`image/video/source_path` 写相对路径（相对 `--input-root`）。

---

## 3) conversations[1].value（assistant 文本模板）

### 3.1 通用硬约束

1) assistant 文本必须是：`<think>...</think>` + **原始答案文本**（答案在末尾；不得改写）。
2) `conversations[*].value` 文本 **禁止** 泄漏任何索引/路径信息，例如：
   - `frame_###` / `sample_###` / `ts_...`
   - `.jpg/.png/.mp4`
   - `Frame 12` / `Image 12` 等

证据文件路径只允许出现在条目字段 `image` / `video` / `meta.evidence_files` 中。

补充说明：

- `conversations[0].value`（human）必须是**单行**自然问句：不包含任何 `fields.*` 上下文行；不包含 options/candidates 等列表；不在 Q 中引导如何生成 CoT。

---

### 3.2 `<think>` CoT 段落要求（单段落 + 因果规划 + 失败反思）

- **单段落**：`<think>...</think>` 内禁止出现换行（可包含多句，但必须是同一段/同一行）；禁止使用列表/条目格式。
- **因果规划顺序**：CoT 必须按如下顺序组织并覆盖：
  1) spatial preconditions
  2) affordance preconditions
  3) spatial effects
  4) affordance effects
  5) failure_reflecting（失败原因 + 恢复策略）
  6) 自然过渡到“为什么答案成立”（但不要输出 `Answer:` 前缀）
- **严格 grounded**：不得凭空新增对象/工具/动作/因果关系/失败模式/结果等。
- 生成器默认使用英文提示词生成英文 CoT/答案（如需中文需要自行修改生成器提示词，但仍必须满足上述结构与硬约束）。

### 3.3 必须逐字包含的“锚点句”（required anchors）

为保证 planning CoT 与三阶段标注严格对齐，生成器会从 `causal_plan_with_keyframes.json` 中抽取并注入若干句 **必须逐字包含** 的锚点句（要求出现在 `<think>...</think>` 内的 CoT 段落中，大小写/标点/空格需一致）。典型形式如下：

- `Spatially, ... must ... .`（来自 `causal_chain.causal_precondition_on_spatial`）
- `Functionally, ... must ... .`（来自 `causal_chain.causal_precondition_on_affordance`）
- `After the action, spatially, ... will ... .`（来自 `causal_chain.causal_effect_on_spatial`）
- `After the action, functionally, ... will ... .`（来自 `causal_chain.causal_effect_on_affordance`）
- `A likely failure is that ... .`（来自 `failure_reflecting.reason`）
- `If that happens, ... .`（来自 `failure_reflecting.recovery_strategy`）

若 API 输出不满足锚点句/格式/泄漏约束，生成器会自动重试；达到 `--max-sample-attempts` 上限仍失败则丢弃该样本（避免引入低质量数据）。

### 3.4 Answer 段格式（按任务）

注意：本生成器会在提示中提供**精确的 gold Answer text**，并要求模型在 `</think>` 之后**原样拷贝**；因此模型主要负责生成高质量 CoT，答案本身不应被改写。

常见 Answer 形式（以 `generate_cot_dataset_api.py` 的 task prompt 为准）：

- `Task_28/29/30/37`：`<text>`
- `Task_31`：`<int>`
- `Task_33`：`<comma-separated letters like A,C,E>`
- `Task_32/34/36`：多行编号列表（如 `1) ...`）
- `Task_35`：`FlawStep=<int>; FlawType=<type>; Reason=<one sentence>`
- `Task_38/40/42`：`<A/B/C/D>`
- `Task_39`：`<recovery_strategy>`
- `Task_41`：`retry_current_step|continue_next_step`

---

## 4) 样例（单条 JSONL）

以下为示例（省略/替换了部分内容；示例仅用于说明格式，真实路径以实际产物为准）：

```json
{
  "id": "6b6bb0dd-7e0c-4c1b-a0c9-0c9b7b7ad2e5",
  "image": ["video_001/01_step/frame_001_ts_00.00s.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "What is the next planned action?"
    },
    {
      "from": "gpt",
      "value": "<think>Spatially, the cup must be on the table. Functionally, the cup must be graspable. After the action, spatially, the cup will be in the sink. After the action, functionally, the cup will be clean. A likely failure is that the cup slips. If that happens, regrasp the cup securely, which restores the preconditions and keeps the effects aligned so the planned outcome can still be achieved.</think>\\n<gold_next_step_goal>\\n"
    }
  ],
  "meta": {
    "task_name": "Task_29_Next_Action_Prediction",
    "item_type": "three_stage",
    "evidence_type": "keyframe_single",
    "source_path": "video_001/causal_plan_with_keyframes.json",
    "step_index": 1,
    "fields": {"next_step_goal": "<gold_next_step_goal>"},
    "evidence_files": ["video_001/01_step/frame_001_ts_00.00s.jpg"],
    "assistant_generator": {
      "type": "api_generate_v1",
      "api_base_url": "http://.../v1",
      "model_provider_id": "...",
      "model_name": "..."
    }
  }
}
```

---

## 5) 校验方式（推荐）

生成后可用严格校验器检查字段与文本约束：

```bash
python3 ECCV/cot/validate_cot_dataset.py \
  --input-root <input_root> \
  --cot-root <cot_root> \
  --strict
```

---

## 6) 参考

- 输出生成入口：`ECCV/cot/generate_cot_dataset_api.py`
- 严格校验器：`ECCV/cot/validate_cot_dataset.py`
- 更完整的生成/标注指南：`ECCV/cot/COT_ANNOTATION_GUIDE.md`
