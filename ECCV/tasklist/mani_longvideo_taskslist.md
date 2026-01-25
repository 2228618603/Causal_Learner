# mani_longvideo 多模态任务清单

本文档定义一套可落地生成的多任务多模态 QA/监督数据规范，面向 `mani_longvideo.py` / `three_stage/pipeline.py` 生成的单视频 item 目录。

本文包含三部分：
- `## 1. 核心目标与统一口径`
- `## 2. 任务体系`：最终任务集合（字段来源 + 证据形态 + 输出约束）；
- `## 3. 任务卡片`：逐任务的字段来源、证据来源、样本构造、QA 范例；

---

## 1. 核心目标与统一口径

### 1.1 最终核心能力

- **因果规划（causal planning）**
- **失败反思（failure reflecting）**

### 1.2 证据形态（统一 4 类）

为兼容训练落地与可控性，`meta.evidence_type` 建议统一为 4 类：

1) `keyframe_single`：关键帧图像（通常 1 张；部分任务会用 2 张对比或 4 张 A/B/C/D；多图时写入 `meta.evidence_files`；不暴露帧序号/时间戳）
2) `images_uniform_scene`：完整视频均匀抽帧多图
3) `video_clip`：步骤间局部片段 mp4
4) `video_prefix`：累积前缀 mp4

### 1.3 Schema 摘要（与 three_stage 产物对齐）

本文默认 item 的核心标注文件为：`<ITEM_DIR>/causal_plan_with_keyframes.json`，关键字段：

- 顶层：`high_level_goal: str`, `steps: List[Step]`
- Step：`step_id`, `step_goal`, `rationale`, `causal_chain`, `counterfactual_challenge_question`, `expected_challenge_outcome`, `failure_reflecting`, `critical_frames`
- CausalChain：`agent`, `action`, `patient`, `causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`
- CriticalFrame：`frame_index`, `action_state_change_description`, `causal_chain`, `interaction.tools/materials/hotspot`

提醒：
- `critical_frames[*].frame_index` 是 **step clip 内局部索引**，不等于全局 `sampled_frames/` 的序号。
---

## 2. 任务体系（最终任务集合）

本节列出“最终任务集合”。每个任务均绑定：

### 2.1 核心/支撑任务分层（建议配比口径）
---

### Task_01_Goal_Recognition_From_Full_Video（完整视频高阶目标识别）

- **字段（JSONPath）**：`high_level_goal`
- **证据形态**：优先 `video_prefix`（全长前缀=完整视频）；无 mp4 时用 `images_uniform_scene`
- **输出约束**：自由文本 1 句（尽量为“整体目标+最终结果”）。

### Task_02_Macro_Anchor_Extraction（场景锚点/关键对象集合）

- **字段（JSONPath）**：
  - `high_level_goal`
  - `steps[*].critical_frames[*].interaction.tools[*]`
  - `steps[*].critical_frames[*].interaction.materials[*]`
  - `steps[*].critical_frames[*].causal_chain.causal_precondition_on_spatial[*].objects[*]`
  - `steps[*].critical_frames[*].causal_chain.causal_precondition_on_affordance[*].object_name`
  - `steps[*].critical_frames[*].causal_chain.agent`
  - `steps[*].critical_frames[*].causal_chain.patient`
- **证据形态**：`images_uniform_scene`
- **输出约束**：
  - 给定 `high_level_goal` + 场景均匀抽帧，在候选对象中选择“对达成 high_level_goal 必不可少的 planning key_objects”；
  - 推荐输出为 **对象名列表**（必须是候选子集、去重、snake_case）。

### Task_03_Clip_to_StepGoal_Statement（给定 clip 概括/生成 step_goal）

- **字段（JSONPath）**：
  - `high_level_goal`（上下文，可选）
  - label：`steps[i].step_goal`
- **证据形态**：`video_clip`（对齐 step i 的执行片段；无 clip 时 fallback 为 step 关键帧）
- **输出约束**：严格只输出该片段对应的 `step_goal`（1 句；尽量与标注完全一致）。
- **负样本**：可用“错配 clip↔step_goal”的方式构造 mismatch（写 `meta.neg_sample=true`）。
- **备注**：若想做严格客观题，可把候选 step_goal 做成四选一（思路同 `Task_05`，证据替换为 `video_clip`）。

### Task_04_Plan_Execution_Alignment（计划-执行一致性判别）

- **字段（JSONPath）**：`steps[i].step_goal`（可选 `steps[i].causal_chain.causal_effect_on_*` 作为辅助解释）
- **证据形态**：`video_clip`（尽量对齐 step i 的执行片段）
- **输出约束**：首字段输出三分类标签之一：`match / partial match / mismatch`；可选补充 0–1 句可见证据或说明不可判断（评分默认只看标签）。
- **负样本**：跨 step 替换 step_goal 或打乱 clip 对齐关系（写入 `meta.neg_sample=true`）。

### Task_05_Keyframe_to_StepGoal_Matching_MC（关键帧对应 step_goal 四选一；强监督）

- **字段（JSONPath）**：
  - 正确项：`steps[s].step_goal`
  - 干扰项：同 item 其他 step 的 `step_goal`
- **证据形态**：`keyframe_single`（优先 `steps[s].critical_frames[0]`）
- **输出约束**：只输出 `A/B/C/D`

### Task_06_StepGoal_to_Keyframe_Matching_MC（step_goal → 关键帧四选一）

- **字段来源**：`steps[s].step_goal`（query），正例图来自 `steps[s].critical_frames[0]`；负例图来自其他 steps 的关键帧。
- **证据形态**：`keyframe_single`（4 张 keyframe 图：A/B/C/D）
- **输出约束**：只输出 `A/B/C/D`
- **负样本**：hard negatives 优先选“同工具/同材料/相近动作”的 step 的关键帧。

### Task_07_Init_vs_Complete_Keyframe_Order（同一步两关键帧：起始/完成判别）

- **字段来源**：同一步 `critical_frames[0]` 与 `critical_frames[1]` 的两张图（标签由生成器记录，**不暴露 frame_index**）
- **证据形态**：`keyframe_single`（两张图）
- **输出约束**：二分类 `A_is_initiation` 或 `B_is_initiation`

### Task_08_Entity_Role_Identification（工具/材料角色识别）

- **字段（JSONPath）**：`steps[i].critical_frames[*].interaction.tools/materials`（可附 `steps[i].step_goal`）
- **证据形态**：`keyframe_single`
- **输出约束**：
  - 自由文本版：明确区分 tools vs materials（可输出两类列表或 1–2 句说明）；
  - 客观题变体（推荐）：Yes/No（给定实体，问其在该 step 是否为 tool；只输出 `Yes` 或 `No`）。

### Task_09_Patient_Identification_MCQ（`patient` 识别：四选一/多选）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.patient`
- **证据形态**：`keyframe_single`
- **输出约束**：`A/B/C/D`（优先单选，patient 明显为单实体时）

### Task_10_Action_Phrase_MCQ（`causal_chain.action` 四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.action`（或 step-level `steps[i].causal_chain.action`）
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_11_Hotspot_AffordanceType_MCQ（热点 affordance_type 四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.affordance_type`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_12_Hotspot_Mechanism_MCQ（热点机制四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`
- **建议**：对 mechanism 做轻量标准化（短句），否则 label 噪声大。

### Task_13_Micro_Affordance_Visual_Semantics（热点可供性语义：描述/机制）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.description/affordance_type/mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；回答需覆盖 hotspot 的 `affordance_type` 与 `mechanism`（可简述外观特征）；可与 `Task_11/12`（客观题）形成互补。

### Task_14_State_Evolution_Description（关键帧动作-状态变化事件描述）

- **字段（JSONPath）**：`steps[i].critical_frames[j].action_state_change_description`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本 1–2 句；用于支持时序类任务（如 `Task_26`）。

### Task_15_Holistic_Causal_Chain_Analysis（关键帧物理因果链解释）

- **字段（JSONPath）**：
  - `steps[i].critical_frames[j].causal_chain.agent/action/patient`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
  - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：输出为英文单句长句（不分段；建议 1 个句号，可用分号/从句/括号），必须覆盖 preconditions(spatial+affordance) → agent/action/patient → mechanism → effects(spatial+affordance)；无法从单帧直接支持的前置条件/后效写 `not directly observable`。

### Task_16_Strategic_Rationale_Justification（步骤动机/必要性解释）

- **字段（JSONPath）**：`steps[i].rationale`（可附 `high_level_goal`, `step_goal`）
- **证据形态**：`keyframe_single`
- **输出约束**：尽量短（1–2 句），强调“对后续步骤的因果贡献”。

### Task_17_Spatial_Precondition_Description（空间前置条件：描述 precondition）

- **字段（JSONPath）**：
  - `steps[i].causal_chain.causal_precondition_on_spatial[*].relation/objects/truth`
  - 可选：`steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial[*]`（用于补充 initiation 视角）
- **证据形态**：`keyframe_single`（建议 step-init 关键帧）
- **输出约束**：用自然语言列出 1–4 条空间 precondition（必须包含 relation + objects；若 truth=false 则显式表达否定）；看不清则写 `not directly observable`。
- **备注**：空间 precondition 的客观“选择”版本用 `Task_18`。

### Task_18_Spatial_Precondition_MCQ（空间前置条件四选一）

- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_precondition_on_spatial[*].relation/objects/truth`
- **证据形态**：`keyframe_single`（建议 step-init 关键帧）
- **输出约束**：`A/B/C/D`
- **负样本**：
  - 从其它 step 抽取 precondition 作为干扰项；或仅做单一扰动（替换 objects / 变更 relation / 翻转 truth）。

### Task_19_Affordance_Precondition_Description（可供性前置条件：描述 precondition）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_precondition_on_affordance[*].object_name/affordance_types/reasons`
- **证据形态**：`keyframe_single`（建议 step-init 关键帧）
- **输出约束**：列出 1–4 条可供性 precondition（对象 + affordance_types；`reasons` 可用 0–1 句概括）；看不清则写 `not directly observable`。
- **备注**：可供性 precondition 的客观“选择”版本用 `Task_20`。

### Task_20_Affordance_Precondition_MCQ（可供性前置条件四选一）

- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_precondition_on_affordance[*].object_name/affordance_types/reasons`
- **证据形态**：`keyframe_single`（建议 step-init 关键帧）
- **输出约束**：`A/B/C/D`
- **负样本**：
  - 从其它 step 抽取 affordance precondition 作为干扰项；或仅做单一扰动（替换 object_name / 替换 affordance_types）。

### Task_21_Physical_Feasibility_Verification_MCQ（可行性核验：四选一）

- **字段（JSONPath）**：
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
  - 上下文：`steps[i].step_goal`
- **证据形态**：`keyframe_single`（建议 step-init 关键帧 `critical_frames[0]`）
- **输出约束**：`A/B/C/D`；每个选项需同时表达 feasibility + 1 条空间前置条件 + 1 条可供性前置条件 + 它们在该帧上的核验结果。
- **负样本**：按“关键工具/对象缺失”或“关系反转且可核验”构造 `not feasible`。

### Task_22_Spatial_Postcondition_Description（空间后置条件：描述 postcondition）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据形态**：`keyframe_single`（建议 step-end 关键帧）
- **输出约束**：用自然语言列出 1–4 条空间 postcondition（必须包含 relation + objects；若 truth=false 则显式表达否定）；看不清则写 `not directly observable`。
- **备注**：空间 postcondition 的客观“选择”版本用 `Task_23`。

### Task_23_Spatial_Postcondition_MCQ（空间后置条件四选一）

- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据形态**：`keyframe_single`（建议 step-end 关键帧）
- **输出约束**：`A/B/C/D`
- **负样本**：
  - 从其它 step 抽取 spatial effect 作为干扰项；或仅做单一扰动（替换 objects / 变更 relation / 翻转 truth）。

### Task_24_Affordance_Postcondition_Description（可供性后置条件：描述 postcondition）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types/reasons`
- **证据形态**：`keyframe_single`（建议 step-end 关键帧）
- **输出约束**：列出 1–4 条可供性 postcondition（对象 + affordance_types；`reasons` 可用 0–1 句概括）；看不清则写 `not directly observable`。
- **备注**：可供性 postcondition 的客观“选择”版本用 `Task_25`。

### Task_25_Affordance_Postcondition_MCQ（可供性后置条件四选一）

- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types/reasons`
- **证据形态**：`keyframe_single`（建议 step-end 关键帧）
- **输出约束**：`A/B/C/D`
- **负样本**：
  - 从其它 step 抽取 affordance effect 作为干扰项；或仅做单一扰动（替换 object_name / 替换 affordance_types）。

### Task_26_Temporal_Order_Check_AB（两事件先后判别；严格可评分）

- **字段来源**：
  - 事件文本：`steps[a].critical_frames[x].action_state_change_description` 与 `steps[b].critical_frames[y].action_state_change_description`
  - 标签依据：两关键帧对应的真实时间顺序（由生成器在后台用 `ts` 或其它对齐信息比较得出；**不进入 prompt**）
- **证据形态**：`keyframe_single`（允许 2 张 keyframes；写入 `meta.evidence_files=[imgA,imgB]`）
- **输出约束**：只输出 `A` 或 `B`（哪一个更早）。

### Task_27_Stage2_Temporal_Localization_Check（可选：基于 Stage2 的客观时间定位）

仅当 item 内存在可读的 Stage2 产物时启用。

- **字段来源**：Stage2 预测的 `start_frame_index/end_frame_index`
- **证据形态**：`images_uniform_scene`（全局抽帧）
- **输出约束**：输出 `start_frame_index/end_frame_index`（整数；1-based；start inclusive；end exclusive；满足 `1 <= start < end <= num_frames`，且 `end_frame_index` 不允许为 `num_frames + 1`）

### Task_28_Inter_Step_Dependency_Analysis（跨步依赖解释）

- **字段（JSONPath）**：
  - `steps[i].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
  - `steps[i+1].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - 上下文：`steps[i].step_goal`, `steps[i+1].step_goal`, `high_level_goal`
- **证据形态**：`keyframe_single`（建议 step i 尾关键帧；可选追加 step i+1 首关键帧作为第二张图，写入 `meta.evidence_files=[imgA,imgB]`）
- **输出约束**：回答 1–2 句；句式建议：“Step i 导致 X (effect)，从而满足 step i+1 需要的 Y (precondition)。”

### Task_29_Next_Action_Prediction（给定 high_level_goal + 当前 step_goal + 关键帧：预测下一步）

- **字段（JSONPath）**：
  - 输入侧：`high_level_goal`, `steps[i].step_goal`
  - 标签侧：`steps[i+1].step_goal`
- **证据形态**：`keyframe_single`（建议 step i 尾关键帧）
- **输出约束**：严格只输出下一步 `step_goal`（不输出其它步骤文本）。
- **备注**：若有 `video_prefix` 且希望更强的长时序证据，优先使用 `Task_30`。

### Task_30_Next_Step_Goal_Prediction_From_Prefix（前缀预测下一步 step_goal）

- **字段（JSONPath）**：label 为 `steps[i+1].step_goal`；可选输入 `steps[i].step_goal`
- **证据形态**：`video_prefix`（前缀到 step i 尾）
- **输出约束**：严格只输出下一步 `step_goal`（不输出其它步骤）。

### Task_31_Prefix_Completed_Steps_MultiSelect（前缀已完成步骤多选）

- **字段（JSONPath）**：`steps[*].step_goal`（作为“完整计划列表”放进 prompt）
- **证据形态**：`video_prefix`（到 step i 尾）
- **输出约束**：
  - 变体 A（推荐）：输出最大已完成 `step_id`
  - 变体 B：多选（输出已完成 step_id 集合；评分用集合匹配/F1）

### Task_32_Middle_Steps_Infill_From_Head_Tail（头尾证据 → 中间步骤补全）

- **字段（JSONPath）**：`high_level_goal` + `steps[*].step_goal`（middle steps 标签）
- **证据形态**：`images_uniform_scene`（head-tail 拼接）
- **输出约束**：输出中间 step_goal 序列（建议 1..M 编号）。

---

### Task_33_Next_K_Steps_MultiSelect_From_Prefix（未来 K 步多选）

- **字段（JSONPath）**：`steps[i+1:i+K].step_goal`（gold 集合）
- **证据形态**：`video_prefix`
- **输出约束**：多选输出（推荐 `A/B/C/...` 字母集合，或输出 step_goal 列表；必须为候选子集）。
- **负样本**：候选中混入同视频其它 step_goal 或跨视频的干扰 step_goal（写 `meta.neg_sample=true`）。

### Task_34_Next_K_Steps_Reordering_From_Prefix（未来 K 步重排）

- **字段（JSONPath）**：`steps[i+1:i+K].step_goal`（gold 顺序）
- **证据形态**：`video_prefix`
- **输出约束**：输出 K 条 step_goal 的正确顺序（建议 1..K 编号）。

### Task_35_Failed_Planning_Flaw_Pointing（规划缺陷定位：单一扰动）

- **字段（JSONPath）**：
  - `high_level_goal`
  - `steps[*].step_goal`（用于构造 gold plan）
  - 可选：`steps[*].causal_chain.causal_precondition_on_*` / `causal_effect_on_*`（用于构造“依赖违反”）
- **证据形态**：`video_prefix`（到 prefix_end_step）
- **输出约束**：固定格式（便于评分）：`FlawStep=<int>; FlawType=<type>; Reason=<one sentence>`
- **样本构造（推荐）**：
  - 取 gold 未来窗口 `K∈[3,6]` 的 step_goals；
  - **只修改其中一个 step**，把它替换为一个“错误 sub-plan”（或不合理动作），形成 `bad_plan`；
  - `FlawType` 可取：`tool_mismatch | order_violation | precondition_missing | hallucinated_object | goal_mismatch`。

### Task_36_Plan_Repair_From_Flaw（坏计划修复：输出纠正后的计划）

- **字段来源**：Task_35 的 `bad_plan_steps` 与 gold `steps[i+1:i+K].step_goal`
- **证据形态**：`video_prefix`（同 Task_35）
- **输出约束**：输出纠正后的 K 步序列（严格匹配 gold 顺序；或先输出修复后的候选集合再排序）
- **评分**：exact match（序列）

### Task_37_Counterfactual_Prediction（反事实挑战与结果；自由文本）

- **字段（JSONPath）**：`steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；客观题版本见 `Task_38`。

### Task_38_Counterfactual_Outcome_MCQ（反事实结果四选一；客观化 Task_37）

- **字段（JSONPath）**：`steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_39_Failure_Recovery_Protocol（失败模式与恢复策略；自由文本）

- **字段（JSONPath）**：`steps[i].failure_reflecting.reason`, `steps[i].failure_reflecting.recovery_strategy`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；客观题版本见 `Task_40`。

### Task_40_Recovery_Strategy_MCQ（恢复策略四选一；客观化 Task_39）

- **字段（JSONPath）**：`steps[i].failure_reflecting.recovery_strategy`
- **证据形态**：`keyframe_single` 或 `video_prefix`
- **输出约束**：只输出 `A/B/C/D`

### Task_41_Recovery_then_Retry_or_Continue（二分类：恢复后重试本步/继续下一步）

- **字段来源**：`failure_reflecting.reason/recovery_strategy` + `steps[i].step_goal` + `steps[i+1].step_goal`
- **证据形态**：`video_prefix` 或 `keyframe_single`
- **输出约束**：`retry_current_step` / `continue_next_step`
- **弱监督标签建议**（无需新增字段）：
  - 如果 recovery_strategy 明显是“让本步可完成”的修复（如“稳住/重新抓取/增加摩擦”），倾向 `retry_current_step`；
  - 如果 recovery_strategy 本身等价于“本步完成的最后动作”，倾向 `continue_next_step`。
- **备注**：这条任务的 label 仍可能是弱监督，但比“永远选 i+1”更合理。

### Task_42_Next_Step_After_Recovery（失败驱动重规划：恢复后下一步选择）

- **字段来源**：
  - 失败描述：`steps[i].failure_reflecting.reason`
  - 恢复策略（可给定或先由 Task_40 选择）：`steps[i].failure_reflecting.recovery_strategy`
  - 下一步标签：`steps[i+1].step_goal`（或从 `steps[i+1:i+K]` 中选最合理的下一步）
- **证据形态**：`video_prefix`（到 step i 尾）或 `keyframe_single`
- **输出约束**：`A/B/C/D`（在候选 next step_goal 中选择）

---

## 3. 任务卡片（逐任务：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：构造 `meta.fields` 与 label 的唯一允许来源
- **多模态证据来源**：必须严格从指定路径取图/取片段（路径不进入模型输入）
- **样本构造规则**：如何在 step/frame/segment 上取样，如何造负样本
- **QA 范例**：示例字段参考 `causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json` 的写法（可能与实际 item 不同，但格式固定可复用）

统一约定：

- `ITEM_DIR = causal_spafa_plan_dataset_long/P01_01_part1`
- `SOURCE_JSON = <ITEM_DIR>/causal_plan_with_keyframes.json`

### Task_01_Goal_Recognition_From_Full_Video

- **任务说明**：基于完整视频（或全局均匀抽帧）概括该视频的 `high_level_goal`，输出 1 句覆盖整体目标与预期结果的描述。
- **字段（JSONPath）**：`high_level_goal`
- **证据来源（严格优先级）**：
  1) `video_prefix`：`<ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step{last:02d}_last.mp4`（近似全长）
  2) fallback：`images_uniform_scene`（全局 sampled_frames 等距取 6–10 张）
- **样本构造规则**：每 item 1 条；不把 high_level_goal 放进输入。
- **meta.fields**：`high_level_goal`
- **范例**：

```text
Evidence: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step08_last.mp4
label.high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."
Q: Based on the full video, what is the most appropriate high-level goal?
A: Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board.
```

### Task_02_Macro_Anchor_Extraction

- **任务说明**：给定 `high_level_goal` 与全局均匀抽帧多图，从候选对象中选择与 `high_level_goal` 直接相关、且计划中会用到的 planning key_objects（去重输出）。
- **字段（JSONPath）**：
  - `high_level_goal`
  - 候选池（用于构造 options/label）：同 `## 2`
- **证据来源（严格优先级）**：
  1) `images_uniform_scene`：`<ITEM_DIR>/sampled_frames/sample_*.jpg`（等距取 4–8 张）
  2) 若无 `sampled_frames/`：用“每步最早关键帧集合”代理，仍采样到 4–8 张
- **样本构造规则**：
  - 每个 item 1 条；
  - 正例 key_objects：从 schema 候选池去重得到；
  - 候选池需过滤明显非“规划物件”的词项（例如人体/肢体/泛指：`hand/hands/person` 等）；
  - 负例 distractors：从跨 item 的常见物体词表/其它 item 的对象池采样（必须不在正例集合中）；
  - 组成 `options`（建议 8–14 个）并打乱；
  - 要求模型输出为 **options 的子集**（避免凭空编造）。
- **meta.fields（建议最小集）**：`high_level_goal`, `options`, `label_key_objects`
- **范例**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... (8 images)
high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."
options = ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet","microwave","dish_soap"]
Q: Select the key objects that are directly relevant to the high-level goal and will be used for planning.
A (label_key_objects): ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet"]
```

### Task_03_Clip_to_StepGoal_Statement

- **任务说明**：给定对齐的 step 执行片段（clip），概括其对应的 `step_goal`（1 句），用于训练“片段→步骤意图”的对齐能力。
- **字段（JSONPath）**：
  - `high_level_goal`（可选上下文）
  - label：`steps[i].step_goal`
- **证据来源（严格优先级）**：
  1) `video_clip`：`<ITEM_DIR>/last_frame_segments/segment_step{i-1:02d}_to_step{i:02d}.mp4`（i>1）
  2) step 1：`<ITEM_DIR>/last_frame_segments/segment_start_to_step01.mp4`（若有）
  3) fallback：step i 关键帧（`keyframe_single`）
- **样本构造规则**：
  - 只输出该 clip 对应的 step_goal（尽量与 gold 完全一致）；
  - 可选负样本：用错配 clip↔step_goal（写 `meta.neg_sample=true`）。
- **meta.fields**：`high_level_goal`（可选）, `label_step_goal`, `segment_label`, `neg_sample`
- **范例**：

```text
Video clip: <ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4
label.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
Q: What is the step goal of this clip?
A: Wash the cucumber and carrot under running water and place them on the countertop.
```

### Task_04_Plan_Execution_Alignment

- **任务说明**：判断给定片段是否与指定 `step_goal` 的执行内容一致（match/partial match/mismatch），并给出可见证据或说明不可判断。
- **字段（JSONPath）**：`steps[i].step_goal`
- **证据来源**：`video_clip`（对齐 step i 执行片段）
- **样本构造规则**：
  - 输出 `match/partial match/mismatch`；
  - 负样本：用其他 step_goal 作为 query（写 `meta.neg_sample=true`）。
- **meta.fields**：`step_goal`, `label`, `segment_label`, `neg_sample`
- **范例**：

```text
Video clip: <ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4
fields.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
label = "match"
Q: Does the clip align with the step goal?
A: match
```

### Task_05_Keyframe_to_StepGoal_Matching_MC

- **任务说明**：给定单张关键帧，四选一选择最匹配的 `step_goal`（强监督），用于提升“图像语义→步骤语义”的对齐能力。
- **字段（JSONPath）**：同 `## 2`
- **证据来源**：`keyframe_single`
- **样本构造规则**：同 item 取 1 正确 + 3 干扰，打乱为 A/B/C/D。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/03_gather_a_cutting_board_and_a_knife_and_place_them_on_the_countertop/frame_014_ts_46.80s.jpg
Q: Which step goal best matches what is happening in this image?
  A) Retrieve a carrot and a cucumber from the refrigerator.
  B) Gather a cutting board and a knife and place them on the countertop.
  C) Wash the cucumber and carrot under running water and place them on the countertop.
  D) Slice the cucumber into circular pieces on the cutting board.
A (label): B
```

### Task_06_StepGoal_to_Keyframe_Matching_MC

- **任务说明**：给定 `step_goal` 文本，在 4 张关键帧图像中选择最匹配的一张，用于训练“文本→图像”的检索/对齐能力（对称补齐 Task_05）。
- **字段来源**：`steps[s].step_goal`（query），正例图来自 `steps[s].critical_frames[0]`；负例图来自其他 steps 的关键帧。
- **证据来源**：`keyframe_single`（4 张图，写入 `meta.evidence_files=[imgA,imgB,imgC,imgD]`）
- **样本构造规则**：
  - 每个 step 0–1 条；
  - 负例优先 hard negatives：同工具/同材料/相近动作的 step；
  - 输出仅 `A/B/C/D`。
- **meta.fields**：`step_goal`, `label`
- **范例**：

```text
Query step_goal: "Retrieve a carrot and a cucumber from the refrigerator."
Image A: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Image B: <ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg
Image C: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
Image D: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Which image best matches the step goal?
A (label): B
```

### Task_07_Init_vs_Complete_Keyframe_Order

- **任务说明**：给定同一步的两张关键帧（顺序可能被打乱），判断哪张是步骤的起始（initiation），用于训练“步骤内部阶段”理解。
- **字段来源**：同一步两关键帧
- **证据来源**：`keyframe_single`（两张图）
- **样本构造规则**：随机打乱两张图的呈现顺序，问哪张是 initiation。
- **meta.fields**：`label`
- **范例**：

```text
Evidence A: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_018_ts_62.10s.jpg
Evidence B: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
Q: Which image shows the initiation of the step, A or B?
A (label): A_is_initiation
```

### Task_08_Entity_Role_Identification

- **任务说明**：在某一步的证据中区分工具（tools）与材料（materials）的角色归属，可做自由文本解释或二分类核验。
- **字段（JSONPath）**：`steps[i].critical_frames[*].interaction.tools/materials`（可附 `steps[i].step_goal`）
- **证据来源**：`keyframe_single`（该 step 最早关键帧）
- **样本构造规则**：
  - 自由文本：输出 tools/materials 的角色区分；
  - Yes/No 变体：随机抽一个实体 e，问 “Is e a tool in this step?”（label 由 tools 集合决定；materials 同理）。
- **meta.fields**：`tools`, `materials`, `step_goal`（或 `entity`, `label`）
- **范例（自由文本）**：

```text
Image: <ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg
fields.tools = ["refrigerator"]
fields.materials = ["cucumber","carrot"]
fields.step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: In the step Retrieve a carrot and a cucumber from the refrigerator, which items function as tools, and which are the materials being acted upon?
A: The refrigerator functions as the tool or container being accessed, while the cucumber and carrot are the materials being grasped and removed.
```

### Task_09_Patient_Identification_MCQ

- **任务说明**：识别关键帧中被作用的主要对象（patient），以四选一/多选形式训练对象角色定位能力。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.patient`
- **证据来源**：`keyframe_single`
- **样本构造规则**：候选来自同 item 的对象集合（可由 Task_02 去重得到）。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: What is the primary patient object being acted on in this image?
  A) light_switch
  B) refrigerator
  C) cucumber
  D) cutting_board
A (label): A
```

### Task_10_Action_Phrase_MCQ

- **任务说明**：四选一选择最符合该关键帧的动作短语 `causal_chain.action`，用于学习“视觉动作→动作语义短语”的映射。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.action`（或 step-level `steps[i].causal_chain.action`）
- **证据来源**：`keyframe_single`
- **样本构造规则**：动作短语尽量简短一致；干扰项来自其它 step。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Which action phrase best matches what the agent is doing?
  A) apply downward pressure to press
  B) twist to unscrew a lid
  C) tilt to pour liquid
  D) scrape to grate a surface
A (label): A
```

### Task_11_Hotspot_AffordanceType_MCQ

- **任务说明**：四选一识别关键帧交互热点的 `affordance_type`，把可供性类别学习变成可评分监督。
- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.affordance_type`
- **证据来源**：`keyframe_single`
- **样本构造规则**：正确项来自当前关键帧；干扰项来自其它关键帧/其它 step。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Which affordance_type best describes the interaction hotspot in this image?
  A) graspable_handle
  B) pressable_surface
  C) cuttable_edge
  D) pour_spout
A (label): B
```

### Task_12_Hotspot_Mechanism_MCQ

- **任务说明**：四选一选择最符合该关键帧交互的物理机制解释（mechanism），用于强化“可供性→机制”理解与推理。
- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据来源**：`keyframe_single`
- **样本构造规则**：机制句子建议短句化；干扰项来自其它机制类别。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Which mechanism best explains how the hotspot functions?
  A) The sharp edge concentrates force to cut the material.
  B) Pressing transfers force to an internal toggle mechanism to complete a circuit.
  C) Rotating a threaded interface converts torque into linear motion.
  D) Heating transfers thermal energy to change the material state.
A (label): B
```

### Task_13_Micro_Affordance_Visual_Semantics

- **任务说明**：在单张关键帧中定位交互热点（hotspot），并描述其可供性类别与物理机制（为什么这个区域“能被这样用”）。
- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.description/affordance_type/mechanism`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每关键帧 0–1 条。
- **meta.fields**：`affordance_type`, `description`, `mechanism`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.affordance_type = "pressable_surface"
fields.description = "The raised surface of the rocker switch designed for finger contact."
fields.mechanism = "Applying force on the surface actuates the internal toggle mechanism, completing the circuit and turning on the light."
Q: Which specific region affords pressable_surface, and how does it visually appear and physically function?
A: The raised rocker surface is the pressable region; it provides a clear finger-contact target so applied force actuates the internal toggle mechanism and turns on the lights.
```

### Task_14_State_Evolution_Description

- **任务说明**：描述关键帧中正在发生的动作及其导致的即时状态变化（事件级描述），用于训练“动作→状态变化”的表达与理解。
- **字段（JSONPath）**：`steps[i].critical_frames[j].action_state_change_description`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每关键帧 0–1 条；答案 1–2 句。
- **meta.fields**：`action_state_change_description`
- **范例**：

```text
Image: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
fields.action_state_change_description = "The person holds a cucumber under running water and rubs it, causing its surface to be cleaned as water and friction remove contaminants."
Q: What ongoing action is occurring, and what immediate state change does it cause?
A: The person is rubbing the cucumber under running water, which immediately cleans its surface as friction and water remove contaminants.
```

### Task_15_Holistic_Causal_Chain_Analysis

- **任务说明**：基于关键帧解释物理因果链：空间/可供性前置条件 → 动作与机制 → 空间/可供性后效，强调“可被证据支持的因果闭环”。
- **字段（JSONPath）**：
  - `steps[i].critical_frames[j].causal_chain.agent/action/patient`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
  - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据来源**：`keyframe_single`
- **样本构造规则**：
  - 每关键帧 0–1 条；
  - 输出为英文单句长句（不分段；建议 1 个句号，可用分号/从句/括号）；
  - 必须覆盖：preconditions(spatial+affordance) → agent/action/patient → mechanism → effects(spatial+affordance)；
  - 无法从单帧直接支持的前置条件/后效必须写 `not directly observable`（不要硬猜）；
  - 推荐单句模板：With <spatial preconditions> and <affordance preconditions>, <agent> <action> on <patient>; because <hotspot mechanism>, this results in <spatial effects> and <affordance effects>, while anything not visibly supported by the frame is not directly observable.
- **meta.fields**：`agent`, `action`, `patient`, `preconditions`, `effects`, `mechanism`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg
fields.agent = "hands"
fields.action = "apply downward cutting force"
fields.patient = "cucumber"
Q: Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects.
A: With the cucumber stabilized on the cutting board and the knife edge contacting it (any internal material properties are not directly observable), the hands apply downward cutting force to the cucumber; because the sharp edge concentrates force at the hotspot to cut through the material, the cucumber separates and a new slice is produced.
```

### Task_16_Strategic_Rationale_Justification

- **任务说明**：解释该步骤为什么必要、如何支撑整体目标（从“动机/必要性”的因果角度给出简短说明）。
- **字段（JSONPath）**：`steps[i].rationale`（可附 `high_level_goal`, `step_goal`）
- **证据来源**：`keyframe_single`（该 step 最早关键帧）
- **样本构造规则**：每 step 1 条；答案尽量 1–2 句。
- **meta.fields**：`rationale`, `step_goal`, `high_level_goal`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.rationale = "This step improves visibility and enables safe manipulation for all subsequent steps."
Q: Why is the step Enter the kitchen and turn on the light to illuminate the workspace necessary for the overall goal?
A: It provides sufficient lighting so later navigation and object manipulation can be done safely and accurately.
```

### Task_17_Spatial_Precondition_Description

- **任务说明**：基于 step-init 关键帧，用自然语言描述该步执行前必须满足的空间前置条件（relation/objects/truth），用于训练“precondition 表达与对齐”。
- **字段（JSONPath）**：
  - `steps[i].causal_chain.causal_precondition_on_spatial[*].relation/objects/truth`
  - 可选：`steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial[*]`
- **证据来源**：`keyframe_single`（建议使用 step-init 关键帧 `critical_frames[0]`）
- **样本构造规则**：
  - 从 precondition 列表抽 1–4 条；
  - 输出必须覆盖 relation + objects（truth=false 时要表达否定）；
  - 不可从图中判断时，必须写 `not directly observable`（不要硬猜）。
- **meta.fields**：`step_goal`, `preconditions_spatial`
- **范例**：

```text
Image (step-init keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.preconditions_spatial = [{"relation":"contacting","objects":["hand","light_switch"],"truth":true}]
Q: Describe the spatial preconditions that must hold before executing this step.
A: The hand should be contacting the light_switch so it can apply force to toggle it.
```

### Task_18_Spatial_Precondition_MCQ

- **任务说明**：给定 step-init 关键帧与当前 `step_goal`，在 4 个空间前置条件候选中选择“最符合该步执行所需空间布置”的一项（四选一）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_precondition_on_spatial[*].relation/objects/truth`
- **证据来源**：`keyframe_single`（建议 step-init 关键帧）
- **样本构造规则**：
  - 正确项：从当前 step 的 spatial preconditions 抽 1 条（优先可由关键帧核验的关系）；
  - 干扰项：从其它 step 的 spatial preconditions 抽取或做单一扰动（替换 objects/变更 relation/翻转 truth）；
  - 输出仅 `A/B/C/D`。
- **meta.fields**：`step_goal`, `options`, `label`
- **范例**：

```text
Image (step-init keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
Q: Which spatial precondition is the best match for executing this step in the current scene?
  A) hand contacting light_switch (true)
  B) cucumber on_top_of cutting_board (true)
  C) refrigerator door closed (false)
  D) knife inside drawer (true)
A (label): A
```

### Task_19_Affordance_Precondition_Description

- **任务说明**：基于 step-init 关键帧，用自然语言描述该步执行前必须满足的可供性前置条件（objects + affordance_types），可选简述 reasons，用于训练 affordance precondition 的表达。
- **字段（JSONPath）**：`steps[i].causal_chain.causal_precondition_on_affordance[*].object_name/affordance_types/reasons`
- **证据来源**：`keyframe_single`（step-init 关键帧，优先 `critical_frames[0]`）
- **样本构造规则**：
  - 从 affordance preconditions 抽 1–4 条；
  - 不可判断时写 `not directly observable`；
  - reasons 只做可见/可解释的简短补充，避免编造。
- **meta.fields**：`step_goal`, `preconditions_affordance`
- **范例**：

```text
Image (step-init keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.preconditions_affordance = [{"object_name":"light_switch","affordance_types":["pressable_surface"]}]
Q: Describe the affordance preconditions that must hold before executing this step.
A: The light_switch should provide a pressable surface so it can be actuated by the hand.
```

### Task_20_Affordance_Precondition_MCQ

- **任务说明**：给定 step-init 关键帧与当前 `step_goal`，在 4 个可供性前置条件候选中选择“最符合该步执行所需对象可操作性/功能状态”的一项（四选一）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_precondition_on_affordance[*].object_name/affordance_types/reasons`
- **证据来源**：`keyframe_single`（建议 step-init 关键帧）
- **样本构造规则**：
  - 正确项：从当前 step 的 affordance preconditions 抽 1 条（object_name + affordance_types）；
  - 干扰项：从其它 step 抽取或做单一扰动（替换 object_name/替换 affordance_types）；
  - 输出仅 `A/B/C/D`。
- **meta.fields**：`step_goal`, `options`, `label`
- **范例**：

```text
Image (step-init keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
Q: Which affordance precondition is the best match for executing this step?
  A) light_switch pressable_surface
  B) light_switch submerged_in_water
  C) light_switch cuttable_surface
  D) light_switch pourable_container
A (label): A
```

### Task_21_Physical_Feasibility_Verification_MCQ

- **任务说明**：在单张关键帧上做“此刻是否可执行”的物理可行性判别，并且必须**同时指出**支持该判别的 1 条空间前置条件与 1 条可供性前置条件（并给出该帧对两条条件的核验结果），从而把“可行性决策+依据”客观化为可评分四选一。
- **判别口径（与三态一致）**：
  - 若存在可由该帧直接核验的关键前置条件被违反 → `not feasible`
  - 若关键前置条件无法从单帧核验（例如内部状态/被遮挡）且未出现明确违反 → `not directly observable`
  - 关键空间与可供性前置条件均满足 → `feasible`
- **字段（JSONPath）**：同 `## 2`
- **证据来源**：`keyframe_single`（建议 step-init 关键帧 `critical_frames[0]` 以便核验 preconditions）
- **样本构造规则**：
  - 从当前 step 在该关键帧的 preconditions 中各抽 1 条：`precondition_spatial_focus` + `precondition_affordance_focus`（优先选择“可由该关键帧核验”的条目；否则直接标为 `not directly observable`）；
  - 构造 4 个选项（A/B/C/D），每个选项须写 1–2 句自然语言，并同时表达：feasibility + 1 条 spatial precondition + 1 条 affordance precondition + 它们在该帧上的核验结果（`satisfied/violated/not directly observable`，但需用自然语言写出“可见/不可见/被违反”等）；
  - 正确项：feasibility 与两条核验结果必须与 gold 一致。
  - 干扰项：从其它 step 抽取 precondition，或对正确项做单一扰动（替换 `objects` / 变更 `relation` / 翻转 `truth`；替换 `object_name` / 替换 `affordance_types`；或把核验结果改成错误的 `satisfied/violated`）。
  - **反例构造（label = `not feasible`, 推荐）**：让“关键条件被违反”在单帧上**显而易见**，优先用以下两类：
    - **关键工具/对象缺失**：选取与当前 `step_goal` 不匹配的关键帧（来自其它 step），使该步所需的 tool/patient 在图中明显不存在；在正确选项里指出缺失并判为 `not feasible`。（例如需要 `cuttable_edge` 但画面中没有刀具）
    - **关系反转且可核验**：保持同一关键帧，但将 `precondition_spatial_focus` 做单一扰动为“显然与画面相反”的关系（例如把 `on_top_of` 换成 `inside_`，或把 `contacting` 换成 `not_contacting` 且画面清晰可见），从而使正确答案稳定为 `not feasible`。
  - 避免用内部状态/难以从单帧判断的 affordance（例如 locked/pressable）内部结构来做 `not feasible`，否则更合理的 label 往往会变成 `not directly observable`。
  - **配比约束（必须）**：所有 `meta.neg_sample=true` 且 label=`not feasible` 的反例中，上述两类反例需保持 1:1：
    - neg_type=missing_entity：关键工具/对象缺失
    - neg_type=relation_flip：关系反转且可核验
  - 要求严格 1:1（两类样本必须数量相等，故总反例数必须为偶数），推荐实现为“每个 step 生成 2 条反例（两类各 1 条）”，对无法满足其中一类的 step 从其它 step 补齐，直到两类样本数相等。
  - 负样本按上述反例方式构造时写 `meta.neg_sample=true`。
- **meta.fields**：`step_goal`, `options`, `label`, `precondition_spatial_focus`, `precondition_affordance_focus`, `neg_sample`, `neg_type`（仅当 `neg_sample=true` 时必填）
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
Q: Which option best describes whether the step is physically feasible now, and which spatial+affordance preconditions (and their statuses) justify the decision?
  A) It is feasible now because the hand is visibly touching the light switch (spatial precondition satisfied) and the light switch provides a pressable surface (affordance precondition satisfied).
  B) It is not feasible now because the hand is not touching the light switch (spatial precondition violated), even though the light switch still has a pressable surface (affordance precondition satisfied).
  C) It is feasible now because a cucumber is on the cutting board (spatial precondition satisfied) and the light switch provides a pressable surface (affordance precondition satisfied).
  D) It is feasible now because the hand is touching the light switch (spatial precondition satisfied) and the light switch has a cuttable surface (affordance precondition satisfied).
A (label): A
```

### Task_22_Spatial_Postcondition_Description

- **任务说明**：基于 step-end 关键帧，用自然语言描述该步导致的空间后置条件（postconditions on spatial），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据来源**：`keyframe_single`（step-end 关键帧，优先 `critical_frames[-1]`）
- **样本构造规则**：
  - 从 spatial postconditions 抽 1–4 条；
  - 输出必须覆盖 relation + objects（truth=false 要表达否定）；
  - 不可判断时写 `not directly observable`。
- **meta.fields**：`step_goal`, `postconditions_spatial`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
fields.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
fields.postconditions_spatial = [{"relation":"on_top_of","objects":["cucumber","countertop"],"truth":true}]
Q: Describe the spatial postconditions that should hold after completing this step.
A: The cucumber should be on_top_of the countertop after it is placed there.
```

### Task_23_Spatial_Postcondition_MCQ

- **任务说明**：给定 step-end 关键帧与当前 `step_goal`，在 4 个空间后置条件候选中选择“最符合该步执行后空间结果”的一项（四选一）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据来源**：`keyframe_single`（建议 step-end 关键帧）
- **样本构造规则**：
  - 正确项：从当前 step 的 spatial postconditions 抽 1 条（优先可由 step-end 关键帧核验的关系）；
  - 干扰项：从其它 step 的 postconditions 抽取或做单一扰动（替换 objects/变更 relation/翻转 truth）；
  - 输出仅 `A/B/C/D`。
- **meta.fields**：`step_goal`, `options`, `label`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
Q: Which spatial postcondition is the best match after completing this step?
  A) cucumber on_top_of countertop (true)
  B) cucumber inside refrigerator (true)
  C) knife contacting cucumber (true)
  D) carrot under_running_water (true)
A (label): A
```

### Task_24_Affordance_Postcondition_Description

- **任务说明**：基于 step-end 关键帧，用自然语言描述该步导致的可供性后置条件（postconditions on affordance），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types/reasons`
- **证据来源**：`keyframe_single`（step-end 关键帧，优先 `critical_frames[-1]`）
- **样本构造规则**：
  - 从 affordance postconditions 抽 1–4 条；
  - 不可判断时写 `not directly observable`；
  - reasons 只做可见/可解释的简短补充，避免编造。
- **meta.fields**：`step_goal`, `postconditions_affordance`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.postconditions_affordance = [{"object_name":"light_switch","affordance_types":["switched_on"]}]
Q: Describe the affordance postconditions that should hold after completing this step.
A: The light_switch should be in a switched_on state after being pressed, though the internal state may be not directly observable from a single frame.
```

### Task_25_Affordance_Postcondition_MCQ

- **任务说明**：给定 step-end 关键帧与当前 `step_goal`，在 4 个可供性后置条件候选中选择“最符合该步执行后对象功能/状态变化”的一项（四选一）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types/reasons`
- **证据来源**：`keyframe_single`（建议 step-end 关键帧）
- **样本构造规则**：
  - 正确项：从当前 step 的 affordance postconditions 抽 1 条（object_name + affordance_types）；
  - 干扰项：从其它 step 的 affordance effects 抽取或做单一扰动（替换 object_name/替换 affordance_types）；
  - 输出仅 `A/B/C/D`。
- **meta.fields**：`step_goal`, `options`, `label`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
Q: Which affordance/state postcondition is the best match after completing this step?
  A) light_switch switched_on
  B) light_switch submerged_in_water
  C) light_switch sliced_into_pieces
  D) light_switch placed_inside_refrigerator
A (label): A
```

### Task_26_Temporal_Order_Check_AB

- **任务说明**：给定两张关键帧（A/B）及对应事件描述，判断哪一个事件在视频中更早发生（输出 A 或 B），用于训练跨步时间顺序理解。
- **字段（JSONPath）**：两条 `action_state_change_description`
- **证据来源**：`keyframe_single`（两张图）
- **样本构造规则**：
  - 生成器在后台根据 `ts` 决定 label（A earlier / B earlier），但 `ts` 不进入 prompt；
  - prompt 只给两张图（A/B）与两条事件描述。
- **meta.fields**：`event_a`, `event_b`, `label`
- **范例**：

```text
Evidence A: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Evidence B: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
event_a = "A person's hand presses a rocker-style light switch."
event_b = "The person rubs a cucumber under running water."
label = "A"
Q: Which event happens earlier in the video, A or B?
A: A
```

### Task_27_Stage2_Temporal_Localization_Check（可选）

- **任务说明**：（可选）基于全局均匀抽帧，预测/核验某一步在 sampled frame indices 上的起止边界（start/end），用于更严格的时间定位评测。
- **字段来源**：Stage2 产物中的 `start_frame_index/end_frame_index`
- **证据来源**：`images_uniform_scene`（`<ITEM_DIR>/sampled_frames/sample_*.jpg`）
- **样本构造规则**：每 step 1 条；输出两个整数（1-based；start inclusive；end exclusive；满足 `1 <= start < end <= num_frames`，且 `end_frame_index` 不允许为 `num_frames + 1`）。
- **meta.fields**：`step_id`, `start_frame_index`, `end_frame_index`
- **范例**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... sample_050_ts_??.??s.jpg
Q: Predict the start_frame_index and end_frame_index for step_id=4.
A (label): {"start_frame_index": 18, "end_frame_index": 26}
```

### Task_28_Inter_Step_Dependency_Analysis

- **任务说明**：解释跨步依赖：上一动作的后果如何满足下一步的前置条件，强调“effect → precondition”的可解释链接。
- **字段（JSONPath）**：同 `## 2`
- **证据来源**：`keyframe_single`（建议 step i 尾关键帧；可选追加 step i+1 首关键帧作为第二张图，写入 `meta.evidence_files=[imgA,imgB]`）
- **输出约束**：回答 1–2 句，句式建议：“Step i 导致 X (effect)，从而满足 step i+1 需要的 Y (precondition)。”
- **样本构造规则**：每个相邻 step 对 (i, i+1) 0–1 条；要求 `step_n_effects` 与 `step_next_preconditions` 均存在且可建立依赖解释；若字段缺失或无法建立依赖则跳过。
- **meta.fields**：`step_n_goal`, `step_n_effects`, `step_next_goal`, `step_next_preconditions`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_n_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.step_next_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: How does the outcome of the previous step satisfy the preconditions for the next step?
A: Step 1 illuminates the workspace (effect), thereby satisfying the visibility/safety precondition needed to locate and access the refrigerator in Step 2 (precondition).
```

### Task_29_Next_Action_Prediction

- **任务说明**：给定 `high_level_goal` + 当前 `step_goal` + 当前步关键帧（建议 step-end），预测下一步应该执行的动作（以 `step_goal` 形式输出），用于训练“局部状态 + 全局目标 → 下一步规划”。
- **字段（JSONPath）**：
  - 输入侧：`high_level_goal`, `steps[i].step_goal`
  - 标签侧：`steps[i+1].step_goal`
- **证据来源**：`keyframe_single`（建议 step i 尾关键帧）
- **样本构造规则**：严格只输出下一步 `step_goal`（完全匹配，不输出多余文本）。
- **meta.fields**：`high_level_goal`, `current_step_goal`, `next_step_goal`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."
fields.current_step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
label.next_step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: What is the next planned action?
A: Retrieve a carrot and a cucumber from the refrigerator.
```

### Task_30_Next_Step_Goal_Prediction_From_Prefix

- **任务说明**：基于视频前缀预测下一步 `step_goal`（严格只输出下一步），用于训练长时序“前缀→下一步”的规划能力。
- **字段（JSONPath）**：label 为 `steps[i+1].step_goal`
- **证据来源**：`video_prefix`（到 step i 尾）
- **样本构造规则**：严格只输出下一步 step_goal。
- **meta.fields**：`current_step_goal`（可选）, `next_step_goal`, `prefix_end_step`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
label.next_step_goal = "Gather a cutting board and a knife and place them on the countertop."
Q: What is the next step goal?
A: Gather a cutting board and a knife and place them on the countertop.
```

### Task_31_Prefix_Completed_Steps_MultiSelect

- **任务说明**：给定视频前缀与“完整计划 step 列表”，判断当前前缀已经完成到哪一步（推荐输出最大已完成 `step_id`），用于可评分的长时序进度理解。
- **字段（JSONPath）**：`steps[*].step_goal`
- **证据来源**：`video_prefix`
- **样本构造规则（推荐最大 step_id 变体）**：
  - 取 prefix_end_step=i；
  - label 为 `i`（即 `prefix_end_step`，表示已完成到 step_id=i）。
- **meta.fields**：`all_steps`, `prefix_end_step`, `label`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
Plan steps:
  1) Enter the kitchen and turn on the light to illuminate the workspace.
  2) Retrieve a carrot and a cucumber from the refrigerator.
  3) Gather a cutting board and a knife and place them on the countertop.
  4) Wash the cucumber and carrot under running water and place them on the countertop.
  ...
Q: Up to which step_id has the plan been completed in this prefix?
A (label): 3
```

### Task_32_Middle_Steps_Infill_From_Head_Tail

- **任务说明**：给定视频头尾证据与整体目标，补全中间缺失的步骤序列（按顺序输出），用于训练长时序“补全/插值”规划能力。
- **字段（JSONPath）**：`high_level_goal` + `steps[*].step_goal`
- **证据来源**：`images_uniform_scene`（head-tail 拼接）
- **样本构造规则**：输入 head+tail 若干帧，输出中间步骤序列。
- **meta.fields**：`head_frames`, `tail_frames`, `middle_steps`
- **范例**：

```text
Images (head-tail): ["<ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg", "...", "<ITEM_DIR>/sampled_frames/sample_050_ts_??.??s.jpg"]
fields.high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them."
Q: Based on the beginning/end glimpses of the video, infer the missing middle steps in order.
A: 1) Retrieve the vegetables from the refrigerator. 2) Gather a cutting board and a knife. 3) Wash the vegetables under running water.
```

### Task_33_Next_K_Steps_MultiSelect_From_Prefix

- **任务说明**：给定前缀与一组未来候选步骤，在不要求顺序的情况下多选出“接下来 K 步会发生的步骤集合”，用于训练更稳健的未来步骤识别（弱化排序难度）。
- **字段（JSONPath）**：`steps[i+1:i+K].step_goal`
- **证据来源**：`video_prefix`
- **样本构造规则**：
  - 取 gold 未来窗口 `K∈[3,6]`；
  - 构造候选 `options`：包含 gold 的 K 步 + 额外 2–4 个干扰 step_goal（来自同视频其它步或跨视频）；
  - 输出为 options 的子集（建议输出字母集合，集合评分）。
- **meta.fields**：`prefix_end_step`, `K`, `options`, `label_set`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
options:
  A) Retrieve vegetables from the refrigerator.
  B) Gather a cutting board and a knife.
  C) Wash the vegetables under running water.
  D) Turn off the kitchen light and leave.
  E) Put vegetables back into the refrigerator and stop.
Q: Select all steps that will occur in the next K steps (order not required).
A (label_set): A,B,C
```

### Task_34_Next_K_Steps_Reordering_From_Prefix

- **任务说明**：给定前缀与一组被打乱的未来候选步骤，要求重排为最合理的时间顺序（输出序列），用于训练多步规划与顺序推断。
- **字段（JSONPath）**：`steps[i+1:i+K].step_goal`
- **证据来源**：`video_prefix`
- **样本构造规则**：打乱候选，模型输出正确顺序。
- **meta.fields**：`prefix_end_step`, `K`, `shuffled_candidates`, `ordered_steps`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
Candidates (shuffled): ["Wash the vegetables.", "Gather a cutting board and a knife.", "Retrieve vegetables from the refrigerator."]
Q: Reorder the shuffled candidate steps into the most plausible next-step sequence.
A: 1) Retrieve vegetables from the refrigerator. 2) Gather a cutting board and a knife. 3) Wash the vegetables.
```

### Task_35_Failed_Planning_Flaw_Pointing

- **任务说明**：对一个含“单一错误”的坏计划进行缺陷定位：指出错误步骤、错误类型并给出一句话理由，强调可自动评分与可归因。
- **字段（JSONPath）**：同 `## 2`
- **证据来源**：`video_prefix`（到 prefix_end_step）
- **样本构造规则**：
  - 从 gold 未来 steps 中选 1 个 step_goal，替换为明显错误 sub-plan；
  - 要求输出 `FlawStep/FlawType/Reason`。
- **meta.fields**：`prefix_end_step`, `bad_plan_steps`, `gold_plan_steps`, `flaw_step`, `flaw_type`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
bad_plan_steps = [
  "Slice the cucumber on the cutting board.",
  "Open the refrigerator and retrieve a carrot.",
  "Wash the vegetables under running water."
]
Q: Identify the flaw in the bad plan.
A: FlawStep=1; FlawType=precondition_missing; Reason=You cannot slice the cucumber before retrieving it and preparing the cutting board and knife.
```

### Task_36_Plan_Repair_From_Flaw

- **任务说明**：给定视频前缀与一个“只含单一扰动”的坏计划（bad_plan），输出纠正后的正确计划序列，用于训练失败反思中的“纠错→重规划”能力（Task_35 的后续闭环）。
- **字段来源**：Task_35 生成的 `bad_plan_steps` 与 gold `steps[i+1:i+K].step_goal`
- **证据来源**：`video_prefix`（同 Task_35）
- **样本构造规则**：
  - bad_plan 由 gold 未来窗口 K 步中“只改 1 步”得到；
  - 输出纠正后的 K 步（严格匹配 gold 顺序，便于 exact match 评分）。
- **meta.fields**：`high_level_goal`, `bad_plan_steps`, `gold_plan_steps`, `label`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."
bad_plan_steps =
  1) "Wash the cucumber and carrot under running water and place them on the countertop."
  2) "Put the vegetables back into the refrigerator and stop."
  3) "Slice the cucumber into circular pieces on the cutting board."
Q: Repair the plan by outputting the corrected 3-step sequence.
A (label):
  1) "Wash the cucumber and carrot under running water and place them on the countertop."
  2) "Gather a cutting board and a knife and place them on the countertop."
  3) "Slice the cucumber into circular pieces on the cutting board."
```

### Task_37_Counterfactual_Prediction

- **任务说明**：给定该步骤的反事实挑战问题（what-if），生成对物理后果的预测与合理应对（自由文本）。
- **字段（JSONPath）**：`steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每 step 0–1 条；自由文本回答。
- **meta.fields**：`question`, `outcome`, `step_goal`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
fields.question = "What if the cutting board was slippery on the countertop?"
fields.outcome = "The board could slide during cutting, making the action unstable; adding a damp cloth underneath would increase friction."
Q: What if the cutting board was slippery on the countertop?
A: The board might slide under the applied cutting force, increasing instability and risk. Placing a damp cloth or non-slip mat underneath would raise friction and stabilize it.
```

### Task_38_Counterfactual_Outcome_MCQ

- **任务说明**：把反事实结果从自由文本改为四选一客观题：在候选结果中选择正确的 `expected_challenge_outcome`。
- **字段（JSONPath）**：同 `Task_37`
- **证据来源**：`keyframe_single`
- **样本构造规则**：正确项为 `expected_challenge_outcome`；干扰项来自其它 step/item。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: What is the most likely outcome if the cutting board is slippery on the countertop?
  A) The cucumber becomes cleaner under running water.
  B) The board may slide, making cutting unstable; increasing friction can stabilize it.
  C) The knife becomes dull due to heat transfer.
  D) The refrigerator door cannot be opened due to suction.
A (label): B
```

### Task_39_Failure_Recovery_Protocol

- **任务说明**：围绕该步骤可能出现的失败原因，给出可执行的恢复策略（自由文本），用于训练失败反思与纠错表达。
- **字段（JSONPath）**：`steps[i].failure_reflecting.reason`, `steps[i].failure_reflecting.recovery_strategy`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每 step 0–1 条；自由文本回答。
- **meta.fields**：`reason`, `recovery_strategy`, `step_goal`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
fields.reason = "The cucumber rolls during cutting because it is not stabilized."
fields.recovery_strategy = "Hold the cucumber firmly or cut a flat side first to create a stable base."
Q: If the step fails because the cucumber is unstable, what is a plausible recovery strategy?
A: Cut a flat side to prevent rolling, then stabilize it with the non-cutting hand while slicing.
```

### Task_40_Recovery_Strategy_MCQ

- **任务说明**：把恢复策略从自由文本改为四选一客观题：在候选策略中选择正确的 `recovery_strategy`，用于可评分的失败反思训练。
- **字段（JSONPath）**：`steps[i].failure_reflecting.recovery_strategy`
- **证据来源**：`keyframe_single` 或 `video_prefix`
- **样本构造规则**：正确项为 recovery_strategy；干扰项来自其它 step/item。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Which recovery strategy best resolves the failure where the cucumber keeps rolling during cutting?
  A) Place a damp cloth under the cutting board to increase friction.
  B) Cut a flat side on the cucumber and stabilize it with the non-cutting hand.
  C) Turn off the kitchen light to reduce glare.
  D) Put the cucumber back into the refrigerator to cool it down.
A (label): B
```

### Task_41_Recovery_then_Retry_or_Continue

- **任务说明**：给定失败原因与恢复策略，判断恢复后应当“重试当前步”还是“继续下一步”，用于缓解“恢复后永远选 i+1”的弱标问题。
- **字段来源**：`failure_reflecting.reason/recovery_strategy` + `steps[i].step_goal` + `steps[i+1].step_goal`
- **证据来源**：`video_prefix`（到 step i 尾）或 `keyframe_single`
- **样本构造规则**：
  - 输出二分类：`retry_current_step` / `continue_next_step`；
  - 标签可基于 recovery_strategy 的语义做弱监督（见 `## 2` 的建议）。
- **meta.fields**：`failure_reason`, `recovery_strategy`, `current_step_goal`, `next_step_goal`, `label`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step06_last.mp4
failure_reason = "The cutting board is sliding during cutting."
recovery_strategy = "Place a damp cloth under the cutting board to increase friction."
current_step_goal = "Slice the cucumber into circular pieces on the cutting board."
next_step_goal = "Continue slicing the cucumber into uniform pieces."
Q: After applying the recovery strategy, should we retry the current step or continue to the next step?
A (label): retry_current_step
```

### Task_42_Next_Step_After_Recovery

- **任务说明**：失败驱动重规划：给定失败原因与恢复策略，在候选 next step_goal 中选择“恢复之后最合适的下一步”，形成失败反思闭环。
- **字段来源**：`failure_reflecting.reason/recovery_strategy` + 下一步 `step_goal`
- **证据来源**：`video_prefix`（到 step i 尾）
- **样本构造规则**：给 4 个候选 next step_goal，选最合理的下一步（考虑 recovery_strategy）。
- **meta.fields**：`options`, `label`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step06_last.mp4
failure_reason = "The cutting board is sliding during cutting."
recovery_strategy = "Place a damp cloth under the cutting board to increase friction."
Q: After applying the recovery strategy, what is the most appropriate next step?
  A) Slice the cucumber into circular pieces on the cutting board.
  B) Put the knife into the sink and leave the kitchen.
  C) Turn off the kitchen light.
  D) Put the cucumber back into the refrigerator and stop.
A (label): A
```

---
