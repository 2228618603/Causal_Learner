# mani_longvideo 多模态任务清单（causal reasoning / failure reflecting 扩展）

本文档定义一套可落地生成的多任务多模态 QA/监督数据规范（扩展任务：`Task_28–Task_35`），面向 `three_stage/pipeline.py` 产出的单视频 item 目录（核心标注：`causal_plan_with_keyframes.json`）。

本文包含三部分：
- `## 1. 核心目标与统一口径`
- `## 2. 任务体系`：扩展任务集合（字段来源 + 证据形态 + 输出约束）；
- `## 3. 任务卡片`：逐任务的字段来源、证据来源、样本构造、QA 范例；

---

## 1. 核心目标与统一口径

### 1.1 最终核心能力

- **因果规划（causal planning）**
- **失败反思（failure reflecting）**

### 1.2 证据形态（统一 4 类）

为兼容训练落地与可控性，`meta.evidence_type` 建议统一为 4 类：

1) `keyframe_single`：关键帧图像（通常 1 张；多图时写入 `meta.evidence_files`；不暴露帧序号/时间戳）
2) `images_uniform_scene`：完整视频均匀抽帧多图
3) `video_clip`：步骤间局部片段 mp4
4) `video_prefix`：累积前缀 mp4

### 1.3 Schema 摘要（与 three_stage 产物对齐）

本文默认 item 的核心标注文件为：`<ITEM_DIR>/causal_plan_with_keyframes.json`，关键字段：

- 顶层：`high_level_goal: str`, `steps: List[Step]`
- Step：`step_id`, `step_goal`, `rationale`, `causal_chain`, `counterfactual_challenge_question`, `expected_challenge_outcome`, `failure_reflecting`, `critical_frames`
- Step.causal_chain（macro）：`agent`, `action`, `patient`, `causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`
- Step.critical_frames：每步 **固定 2 张**，按时间顺序排列（`critical_frames[0]` 更早，`critical_frames[1]` 更晚）
  - CriticalFrame：`frame_index`, `action_state_change_description`, `causal_chain`, `interaction`
  - CriticalFrame.causal_chain（micro）：仅 4 个文本字段：`causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`
  - interaction：`description`, `affordance_type`, `mechanism`
- failure_reflecting：`reason`, `recovery_strategy`

提醒：
- `critical_frames[*].frame_index` 为 **1-based** 索引，索引空间对齐到该 step 使用的抽帧序列（通常是 step clip 的抽帧池，而非全局抽帧池）。

---

## 2. 任务体系（扩展任务集合：Task_28–Task_35）

本节列出“扩展任务集合”。每个任务均绑定：

- **字段（JSONPath）**：构造 `meta.fields` 与 label 的唯一允许来源
- **证据形态**：从 `keyframe_single / images_uniform_scene / video_clip / video_prefix` 中选择
- **输出约束**：尽量可自动评分/可控格式

### 2.1 核心/支撑任务分层（建议配比口径）

- 核心（causal reasoning）：`Task_28–Task_31`（跨时间前置→后效闭环、macro↔micro 一致性）
- 支撑（failure reflecting）：`Task_32–Task_35`（反向推断、条件绑定检验，便于构造 hard negatives）

---

### Task_28_BeforeAfter_PrePost_Quadruple_Report（两关键帧：前置/后效四联报告）

- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - before：`steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial`, `steps[i].critical_frames[0].causal_chain.causal_precondition_on_affordance`
  - after：`steps[i].critical_frames[1].causal_chain.causal_effect_on_spatial`, `steps[i].critical_frames[1].causal_chain.causal_effect_on_affordance`
- **证据形态**：`keyframe_single`（两图写入 `meta.evidence_files`）
- **输出约束**：固定 4 行英文：`Before(spatial): ... / Before(affordance): ... / After(spatial): ... / After(affordance): ...`
- **负样本**：替换 after 关键帧或替换 after 文本，构造“因果不闭合”的 hard negative。

### Task_29_Micro_to_Macro_CausalChain_Summarization（micro→macro：两关键帧总结 step-level 因果链）

- **字段（JSONPath）**：label 为 `steps[i].causal_chain`（7 fields）；上下文：`steps[i].step_goal`
- **证据形态**：`keyframe_single`（两图写入 `meta.evidence_files`）
- **输出约束**：输出 JSON object（7 keys）：`agent/action/patient/causal_precondition_on_spatial/causal_precondition_on_affordance/causal_effect_on_spatial/causal_effect_on_affordance`
- **负样本**：只替换单个字段（如 patient 或 causal_effect_on_affordance）为相邻 step 的字段，形成“单一扰动”一致性对比。

### Task_30_Macro_to_Micro_CausalChain_Detailing（macro→micro：给定 step-level 因果链细化到关键帧）

- **字段（JSONPath）**：
  - 输入上下文：`steps[i].causal_chain`（macro）
  - label：`steps[i].critical_frames[j].causal_chain`（micro）
- **证据形态**：`keyframe_single`
- **输出约束**：输出 JSON object（4 keys）：`causal_precondition_on_spatial/causal_precondition_on_affordance/causal_effect_on_spatial/causal_effect_on_affordance`
- **负样本**：把 label 替换为同一步另一张关键帧或其它 step 同位关键帧。

### Task_31_CausalChain_Consistency_Check（关键帧↔因果链一致性二分类）

- **字段（JSONPath）**：
  - 正例：`steps[i].critical_frames[j].causal_chain`
  - 负例：其它 step/帧的 `critical_frames[*].causal_chain`
- **证据形态**：`keyframe_single`
- **输出约束**：`match / mismatch`

### Task_32_FailureReason_From_RecoveryStrategy（恢复策略→失败原因：反向失败反思）

- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - 输入：`steps[i].failure_reflecting.recovery_strategy`
  - label：`steps[i].failure_reflecting.reason`
- **证据形态**：`keyframe_single`（建议 step-init 关键帧；也可纯文本）
- **输出约束**：自由文本 1 句（尽量与 gold 一致）
- **负样本**：把 recovery_strategy 替换成其它 step 的策略。

### Task_33_CounterfactualQuestion_From_ExpectedOutcome（反事实结果→反事实问题：反向生成）

- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - 输入：`steps[i].expected_challenge_outcome`
  - label：`steps[i].counterfactual_challenge_question`
- **证据形态**：`keyframe_single`（可选；纯文本也成立）
- **输出约束**：必须以英文 `What if ...?` 的问句形式输出 1 句

### Task_34_FailureReason_Matching_Binary（失败原因匹配检验：二分类）

- **字段来源**：
  - 正例：`steps[i].failure_reflecting.reason`
  - 负例：其它 step 的 `failure_reflecting.reason`
- **证据形态**：`keyframe_single`
- **输出约束**：`match / mismatch`

### Task_35_RecoveryStrategy_Matching_Binary（恢复策略匹配检验：二分类）

- **字段来源**：
  - 正例：`steps[i].failure_reflecting.recovery_strategy`
  - 负例：其它 step 的 `failure_reflecting.recovery_strategy`
- **证据形态**：`keyframe_single`
- **输出约束**：`match / mismatch`

---

## 3. 任务卡片（逐任务：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：构造 `meta.fields` 与 label 的唯一允许来源
- **多模态证据来源**：必须严格从指定路径取图/取片段（路径不进入模型输入）
- **样本构造规则**：如何在 step/frame 上取样、如何造负样本
- **QA 范例**：仅用于展示输入/输出格式（路径仅示意）

统一约定：

- `ITEM_DIR = <ITEM_DIR>`
- `SOURCE_JSON = <ITEM_DIR>/causal_plan_with_keyframes.json`

### Task_28_BeforeAfter_PrePost_Quadruple_Report

- **任务说明**：用同一步的两张关键帧，输出“前置→后效”的四联报告，显式绑定 spatial/affordance 维度，训练模型做跨时间的因果闭环表达。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[0].causal_chain.causal_precondition_on_affordance`
  - `steps[i].critical_frames[1].causal_chain.causal_effect_on_spatial`
  - `steps[i].critical_frames[1].causal_chain.causal_effect_on_affordance`
- **证据来源（严格优先级）**：
  1) `keyframe_single`（两图）：step i 的 `critical_frames[0]` 与 `critical_frames[1]` 对应图片文件（由生成器用 `frame_index` 映射到文件路径；路径不进入 prompt）
- **样本构造规则**：
  - 每 step 1 条；
  - `meta.evidence_files = [init_keyframe_path, end_keyframe_path]`；
  - 可选负样本：用其它 step 的 end_keyframe 替换 `end_keyframe_path`（写 `meta.neg_sample=true`，label 不变或改为 mismatch 版 Task_31）。
- **meta.fields**：`step_goal`, `before_spatial`, `before_affordance`, `after_spatial`, `after_affordance`, `neg_sample`
- **范例**：

```text
Images (pair): <ITEM_DIR>/.../frame_init.jpg ; <ITEM_DIR>/.../frame_end.jpg
step_goal = "Loosen and remove the jar lid."
label.before_spatial = "The hands are contacting the jar_lid while the jar body is supported or stabilized."
label.before_affordance = "The jar_lid rim provides a grasp_point for the fingers to resist slipping during torque application."
label.after_spatial = "The jar opening is exposed with no object in contact with the rim."
label.after_affordance = "The jar becomes open and ready for the subsequent pouring action."
Q: Given the two keyframes of the same step (init then completion), output the causal preconditions before and the causal effects after in the required 4-line format.
A:
Before(spatial): The hands are contacting the jar_lid while the jar body is supported or stabilized.
Before(affordance): The jar_lid rim provides a grasp_point for the fingers to resist slipping during torque application.
After(spatial): The jar opening is exposed with no object in contact with the rim.
After(affordance): The jar becomes open and ready for the subsequent pouring action.
```

### Task_29_Micro_to_Macro_CausalChain_Summarization

- **任务说明**：从两张关键帧（起始/完成）生成 step-level macro 因果链，训练“跨时间观察→宏观因果总结”。
- **字段（JSONPath）**：label 为 `steps[i].causal_chain`（7 fields）；上下文：`steps[i].step_goal`
- **证据来源（严格优先级）**：
  1) `keyframe_single`（两图）：同 Task_28
- **样本构造规则**：
  - 每 step 1 条；
  - 输出必须是固定 7-key JSON；
  - 可选负样本：单字段扰动（只替换 patient 或某个 causal_* 字段），写 `meta.neg_sample=true` 并把 label 设为二分类 `match/mismatch`（可复用 Task_31）。
- **meta.fields**：`step_goal`, `label_step_causal_chain`, `neg_sample`
- **范例**：

```text
Images (pair): <ITEM_DIR>/.../frame_init.jpg ; <ITEM_DIR>/.../frame_end.jpg
label.step_causal_chain.agent = "hands"
label.step_causal_chain.action = "apply torque to loosen"
label.step_causal_chain.patient = "jar_lid"
label.step_causal_chain.causal_precondition_on_spatial = "...(string)..."
label.step_causal_chain.causal_precondition_on_affordance = "...(string)..."
label.step_causal_chain.causal_effect_on_spatial = "...(string)..."
label.step_causal_chain.causal_effect_on_affordance = "...(string)..."
Q: Summarize the step-level causal chain as a 7-key JSON object.
A: {"agent":"hands","action":"apply torque to loosen","patient":"jar_lid","causal_precondition_on_spatial":"...","causal_precondition_on_affordance":"...","causal_effect_on_spatial":"...","causal_effect_on_affordance":"..."}
```

### Task_30_Macro_to_Micro_CausalChain_Detailing

- **任务说明**：给定 step-level macro 因果链与单张关键帧，输出该关键帧对应的 micro 因果链（4-key JSON），训练“宏观约束→时刻化解释”分解能力。
- **字段（JSONPath）**：
  - 输入上下文：`steps[i].causal_chain`
  - label：`steps[i].critical_frames[j].causal_chain`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：step i 的 `critical_frames[j]` 对应图片文件
- **样本构造规则**：
  - 每个 critical_frame 1 条（每 step 共 2 条）；
  - 输入提供 `step_goal` + `step_causal_chain`（7-key JSON）；
  - label 为该帧的 4-key causal_chain JSON；
  - 可选负样本：交换 j（init/end）或跨 step 替换 causal_chain。
- **meta.fields**：`step_goal`, `step_causal_chain`, `label_frame_causal_chain`, `frame_slot(j)`, `neg_sample`
- **范例**：

```text
Image: <ITEM_DIR>/.../frame_005.jpg
step_goal = "Loosen and remove the jar lid."
step_causal_chain = {"agent":"hands","action":"apply torque to loosen","patient":"jar_lid", ...}
label.frame_causal_chain.causal_precondition_on_spatial = "...(string)..."
label.frame_causal_chain.causal_precondition_on_affordance = "...(string)..."
label.frame_causal_chain.causal_effect_on_spatial = "...(string)..."
label.frame_causal_chain.causal_effect_on_affordance = "...(string)..."
Q: Given the step-level causal chain and this keyframe, output the frame-level causal chain as a 4-key JSON object.
A: {"causal_precondition_on_spatial":"...","causal_precondition_on_affordance":"...","causal_effect_on_spatial":"...","causal_effect_on_affordance":"..."}
```

### Task_31_CausalChain_Consistency_Check

- **任务说明**：给定关键帧与一个候选 frame-level 因果链，判断其是否匹配当前关键时刻（match/mismatch）。该任务可自动构造负样本，且更直接训练 causal grounding。
- **字段（JSONPath）**：
  - 正例：`steps[i].critical_frames[j].causal_chain`
  - 负例：其它 step/帧的 `critical_frames[*].causal_chain`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：step i 的 `critical_frames[j]` 对应图片文件
- **样本构造规则**：
  - 每个 (i,j) 造 1 条正例 + N 条负例；
  - 负例优先：同 item 内其它 step 的同位关键帧（hard negative），其次跨 item 采样；
  - 输出严格为 `match/mismatch`。
- **meta.fields**：`step_goal`, `candidate_frame_causal_chain`, `label_match`, `neg_sample_source`
- **范例**：

```text
Image: <ITEM_DIR>/.../frame_002.jpg
step_goal = "Loosen and remove the jar lid."
candidate_frame_causal_chain = {"causal_precondition_on_spatial":"The jar is positioned above the bowl ...", ...}
Q: Does the candidate causal chain describe this keyframe? Answer with match or mismatch.
A (label_match): mismatch
```

### Task_32_FailureReason_From_RecoveryStrategy

- **任务说明**：给定 recovery strategy 反推它想修复的 failure reason，训练“从干预推断根因”的失败反思能力。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - input：`steps[i].failure_reflecting.recovery_strategy`
  - label：`steps[i].failure_reflecting.reason`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：建议 step-init 关键帧 `critical_frames[0]`（可选）
- **样本构造规则**：
  - 每 step 1 条；
  - 可选负样本：把 recovery_strategy 替换成其它 step 的策略（写 `meta.neg_sample=true`，label 不变或改为 mismatch 版 Task_35）。
- **meta.fields**：`step_goal`, `recovery_strategy`, `label_failure_reason`, `neg_sample`
- **范例**：

```text
Image (optional): <ITEM_DIR>/.../frame_init.jpg
step_goal = "Loosen and remove the jar lid."
recovery_strategy = "Reposition the grip to a higher-friction region and increase normal force while stabilizing the jar body to prevent co-rotation."
label.failure_reason = "The jar_lid slips under the fingers so torque does not transfer effectively to the lid."
Q: Given the recovery strategy, what failure reason is it most directly addressing?
A: The jar_lid slips under the fingers so torque does not transfer effectively to the lid.
```

### Task_33_CounterfactualQuestion_From_ExpectedOutcome

- **任务说明**：给定 `expected_challenge_outcome` 反推 `counterfactual_challenge_question`，强调 outcome↔cause 的一致性（反向反事实生成）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - input：`steps[i].expected_challenge_outcome`
  - label：`steps[i].counterfactual_challenge_question`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：可选 step-init 关键帧（不影响 label）
- **样本构造规则**：
  - 每 step 1 条；
  - 输出必须是英文问句且以 `What if` 开头。
- **meta.fields**：`step_goal`, `expected_challenge_outcome`, `label_counterfactual_question`
- **范例**：

```text
step_goal = "Tilt the jar and pour its contents into the bowl."
expected_challenge_outcome = "The jar_contents would remain inside the jar despite the tilt, so the transfer would stall until additional agitation or a larger tilt angle is applied."
label.counterfactual_challenge_question = "What if the jar_contents are clumped and do not flow when the jar is tilted?"
Q: Given the expected challenge outcome, write the most appropriate counterfactual challenge question in one English sentence starting with "What if".
A: What if the jar_contents are clumped and do not flow when the jar is tilted?
```

### Task_34_FailureReason_Matching_Binary

- **任务说明**：failure reason 的条件绑定检验：给定 step_goal + 关键帧 + 候选 reason，判断是否匹配当前 step（match/mismatch），支持自动构造负样本。
- **字段来源**：
  - 正例：`steps[i].failure_reflecting.reason`
  - 负例：其它 step 的 `failure_reflecting.reason`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：建议 step-init 关键帧
- **样本构造规则**：
  - 每 step 1 条正例 + N 条负例（同 item 内优先）；
  - 输出严格为 `match/mismatch`。
- **meta.fields**：`step_goal`, `candidate_failure_reason`, `label_match`, `neg_sample_source`
- **范例**：

```text
Image: <ITEM_DIR>/.../frame_init.jpg
step_goal = "Loosen and remove the jar lid."
candidate_failure_reason = "The jar is not tilted far enough so the mouth does not become the lowest path for the jar_contents to exit."
Q: Does this failure reason match the current step and keyframe? Answer with match or mismatch.
A (label_match): mismatch
```

### Task_35_RecoveryStrategy_Matching_Binary

- **任务说明**：恢复策略的条件绑定检验：给定 step_goal + 关键帧 + 候选 recovery strategy，判断是否匹配当前 step 的失败场景（match/mismatch）。
- **字段来源**：
  - 正例：`steps[i].failure_reflecting.recovery_strategy`
  - 负例：其它 step 的 `failure_reflecting.recovery_strategy`
- **证据来源（严格优先级）**：
  1) `keyframe_single`：建议 step-init 关键帧
- **样本构造规则**：
  - 每 step 1 条正例 + N 条负例（同 item 内优先）；
  - 输出严格为 `match/mismatch`。
- **meta.fields**：`step_goal`, `candidate_recovery_strategy`, `label_match`, `neg_sample_source`
- **范例**：

```text
Image: <ITEM_DIR>/.../frame_init.jpg
step_goal = "Tilt the jar and pour its contents into the bowl."
candidate_recovery_strategy = "Reposition the grip to a higher-friction region and increase normal force while stabilizing the jar body to prevent co-rotation."
Q: Does this recovery strategy match the current step's likely failure scenario? Answer with match or mismatch.
A (label_match): mismatch
```

