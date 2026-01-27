# mani_longvideo 多模态任务清单

本文档定义一套 QA-only 的多任务多模态监督数据规范，面向“单视频 item”级别的数据组织。

---

## 1. 核心目标与统一口径

### 1.1 最终核心能力

- **因果规划（causal planning）**
- **失败反思（failure reflecting）**

### 1.2 证据形态（统一 4 类）

为兼容训练落地与可控性，`meta.evidence_type` 建议统一为 4 类：

1) `keyframe_single`：关键帧图像（通常 1 张；部分任务会用 2 张对比或 4 张多图；多图时写入 `meta.evidence_files`；不暴露帧序号/时间戳）
2) `images_uniform_scene`：完整视频均匀抽帧多图
3) `video_clip`：步骤间局部片段 mp4
4) `video_prefix`：累积前缀 mp4


### 1.3 Schema 摘要（字段与 `mani_longvideo` schema 对齐）

本任务集假设每个 item 有一份标注 JSON（文件名仅示例；数据可来自二阶段或三阶段等不同生成管线），其字段结构与 `ECCV/two_stage_new/mani_longvideo.py` 的 dataclass 以及 `ECCV/three_stage/prompts.py` 的最终产物 schema 保持一致。

关键字段（字段名以 JSON 为准；类型为概念描述）：

- 顶层：`high_level_goal: str`, `steps: List[PlanningStep]`
- PlanningStep：`step_id: int`, `step_goal: str`, `rationale: str`, `causal_chain: StepCausalChain`, `counterfactual_challenge_question: str`, `expected_challenge_outcome: str`, `failure_reflecting: FailureReflecting`, `critical_frames: List[CriticalFrameAnnotation]`
- 约束（本任务集默认）：每个 step 的 `critical_frames` **固定为 2**，且按时间顺序排列（`critical_frames[0]` 更早，`critical_frames[1]` 更晚）
- StepCausalChain：`agent: str`, `action: str`, `patient: str`，以及 4 个文本字段：
  - `causal_precondition_on_spatial: str`
  - `causal_precondition_on_affordance: str`
  - `causal_effect_on_spatial: str`
  - `causal_effect_on_affordance: str`
  - 语义：以上 4 个 `causal_*` 字段是 **step-level 的 MACRO 总结**（覆盖整个 step 的前置/效果），不是某一瞬间的描述。
- CriticalFrameAnnotation：`frame_index: int`, `action_state_change_description: str`, `causal_chain: FrameCausalChain`, `interaction: Interaction`
  - 约束：`critical_frames[*].causal_chain` **只包含**上述 4 个 `causal_*` 文本字段（不含 `agent/action/patient`）
- FrameCausalChain：`causal_precondition_on_spatial: str`, `causal_precondition_on_affordance: str`, `causal_effect_on_spatial: str`, `causal_effect_on_affordance: str`
  - 语义：frame-level 的 `causal_precondition_on_*` 描述该关键帧 **这一时刻** 为真/所需的状态；`causal_effect_on_*` 描述该帧对应 micro-action 完成后 **立即、局部** 的预测效果（不一定已在当前帧可见）。
- Interaction：`description: str`, `affordance_type: str`, `mechanism: str`（不包含 `tools/materials`，也不嵌套 `hotspot` 对象）
- FailureReflecting：`reason: str`, `recovery_strategy: str`

提醒：
- `critical_frames[*].frame_index` 为 **1-based** 索引，索引空间对齐到该 step 使用的抽帧序列（例如步骤 clip 的抽帧池，或全局均匀抽帧池）。

## 2. 任务卡片（最终任务集合：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：用于对齐标注 JSON 的字段使用（可用于实现生成器或做数据审计）。
- **Multimodal_input 类型（四类）**：明确该任务使用 `keyframe_single / images_uniform_scene / video_clip / video_prefix` 中的哪一种（或少量可选 fallback）。
- **QA 范例**：只展示 `Multimodal_input / Q / A`，用于训练时的“多模态输入→文本输出”对齐。范例中的路径仅作示意，不作为硬约束。

统一约定（仅用于范例表达）：

- 用 `<ITEM_DIR>` 表示某个单视频样本目录。

### Task_01_Goal_Recognition_From_Full_Video（原始 Task_01_Goal_Recognition_From_Full_Video：完整视频高阶目标识别）

- **任务说明**：基于完整视频（或全局均匀抽帧）概括该视频的 `high_level_goal`，输出 1 句覆盖整体目标与预期结果的描述。
- **字段（JSONPath）**：`high_level_goal`
- **Multimodal_input 类型（四类）**：`video_prefix` / `images_uniform_scene`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=08): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step08_last.mp4
Q: Based on the full video, what is the most appropriate high-level goal?
A: Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board.
```

### Task_02_Macro_Anchor_Extraction（原始 Task_02_Macro_Anchor_Extraction：场景锚点/关键对象集合）

- **任务说明**：给定 `high_level_goal` 与全局均匀抽帧多图，从候选对象中选择与 `high_level_goal` 直接相关、且计划中会用到的 planning key_objects（去重输出）。
- **字段（JSONPath）**：
  - `high_level_goal`
  - 候选池（用于构造 options/label）：
    - `steps[*].step_goal`（抽取名词短语）
    - `steps[*].causal_chain.patient`（主受体对象）
    - `steps[*].critical_frames[*].interaction.description`（热点交互部位/区域描述里出现的对象）
- **Multimodal_input 类型（四类）**：`images_uniform_scene`
- **范例**：

```text
Multimodal_input:
- images_uniform_scene(samples=8, uniform): <ITEM_DIR>/sampled_frames/sample_*.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." From the candidate objects ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet","microwave","dish_soap"], list the key objects that are directly relevant to the goal and will be used for planning.
A: ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet"]
```

### Task_03_Clip_to_StepGoal_Statement（原始 Task_03_Clip_to_StepGoal_Statement：给定 clip 概括/生成 step_goal）

- **任务说明**：给定对齐的 step 执行片段（clip），概括其对应的 `step_goal`（1 句），用于训练“片段→步骤意图”的对齐能力。
- **字段（JSONPath）**：
  - `high_level_goal`（可选上下文）
  - label：`steps[i].step_goal`
- **Multimodal_input 类型（四类）**：`video_clip` / `keyframe_single`
- **范例**：

```text
Multimodal_input:
- video_clip(step_id=04): <ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." What is the step goal of this clip?
A: Wash the cucumber and carrot under running water and place them on the countertop.
```

### Task_04_Patient_Identification_QA（原始 Task_09_Patient_Identification_MCQ：`patient` 识别：四选一/多选）

- **任务说明**：识别关键帧中被作用的主要对象（patient），以 QA 形式输出对象名称/短语。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].causal_chain.patient`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." In this image, what is the primary patient object being acted on?
A: light_switch
```

### Task_05_Action_Phrase_QA（原始 Task_10_Action_Phrase_MCQ：`causal_chain.action` 四选一）

- **任务说明**：识别该关键帧中最贴近的动作短语 `causal_chain.action`，以 QA 形式输出动作短语。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].causal_chain.action`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." What is the action phrase (causal_chain.action) in this keyframe?
A: apply downward pressure to press
```

### Task_06_Hotspot_AffordanceType_QA（原始 Task_11_Hotspot_AffordanceType_MCQ：热点 affordance_type 四选一）

- **任务说明**：识别关键帧交互热点的 `affordance_type`，以 QA 形式输出可供性类别短语。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].interaction.affordance_type`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." What is the affordance_type of the interaction hotspot in this image?
A: pressable_surface
```

### Task_07_Hotspot_Mechanism_QA（原始 Task_12_Hotspot_Mechanism_MCQ：热点机制四选一）

- **任务说明**：描述该关键帧交互热点的物理机制解释（mechanism），以 QA 形式输出短句。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].interaction.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Briefly describe the physical mechanism of the interaction hotspot in this image.
A: Pressing transfers force to an internal toggle mechanism to complete a circuit.
```

### Task_08_Micro_Affordance_Visual_Semantics（原始 Task_13_Micro_Affordance_Visual_Semantics：热点可供性语义：描述/机制）

- **任务说明**：在单张关键帧中定位交互热点（hotspot），并描述其可供性类别与物理机制（为什么这个区域“能被这样用”）。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - label：`steps[i].critical_frames[j].interaction.description`, `steps[i].critical_frames[j].interaction.affordance_type`, `steps[i].critical_frames[j].interaction.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Locate the interaction hotspot area in the image first, then describe its affordance_type and mechanism.
A: The hotspot is the raised rocker surface of the light switch where the finger contacts. It affords pressable_surface, and pressing it actuates the internal toggle mechanism to complete the circuit and turn on the light.
```

### Task_09_State_Evolution_Description（原始 Task_14_State_Evolution_Description：关键帧动作-状态变化事件描述）

- **任务说明**：描述关键帧中正在发生的动作及其导致的即时状态变化（事件级描述），用于训练“动作→状态变化”的表达与理解。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].action_state_change_description`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=04, frame_index=020): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
Q: Context: Step goal: "Wash the cucumber and carrot under running water and place them on the countertop." What ongoing action is occurring, and what immediate state change does it cause?
A: The person is rubbing the cucumber under running water, which immediately cleans its surface as friction and water remove contaminants.
```

### Task_10_Holistic_Causal_Chain_Analysis（原始 Task_15_Holistic_Causal_Chain_Analysis：关键帧物理因果链解释）

- **任务说明**：基于关键帧解释物理因果链：空间/可供性前置条件 → 动作与机制 → 空间/可供性后效，强调“可被证据支持的因果闭环”。答案需为英文单句长句（不分段）。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`（必给），`high_level_goal`（可选）
  - label 组合（不应作为 prompt context 提供）：`steps[i].causal_chain.agent/action/patient` + `steps[i].critical_frames[j].causal_chain`（4 个 causal_* 字段） + `steps[i].critical_frames[j].interaction.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=039): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects. Answer in one English sentence.
A: With the cucumber stabilized on the cutting board and the knife edge contacting it (any internal material properties are not directly observable), the hands apply downward cutting force to the cucumber; because the sharp edge concentrates force at the hotspot to cut through the material, the cucumber separates and a new slice is produced.
```

### Task_11_Strategic_Rationale_Justification（原始 Task_16_Strategic_Rationale_Justification：步骤动机/必要性解释）

- **任务说明**：解释该步骤为什么必要、如何支撑整体目标（从“动机/必要性”的因果角度给出简短说明）。
- **字段（JSONPath）**：
  - 上下文：`high_level_goal`（必给），`steps[i].step_goal`（必给）
  - label：`steps[i].rationale`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Why is this step necessary for the overall goal?
A: It provides sufficient lighting so later navigation and object manipulation can be done safely and accurately.
```

### Task_12_Spatial_Precondition_Description（原始 Task_17_Spatial_Precondition_Description：空间前置条件：描述 precondition）

- **任务说明**：基于**早关键帧**（`critical_frames[0]`），输出该关键帧对应的空间前置条件描述（来自 frame-level `causal_chain.causal_precondition_on_spatial`），用于训练“precondition 表达与对齐”。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - label：`steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the spatial preconditions that must hold before executing this step.
A: The hand should be contacting the light_switch so it can apply force to toggle it.
```

### Task_13_Affordance_Precondition_Description（原始 Task_19_Affordance_Precondition_Description：可供性前置条件：描述 precondition）

- **任务说明**：基于**早关键帧**，用自然语言描述该关键时刻对应的可供性/状态前置条件（frame-level `causal_precondition_on_affordance`），可选简述 reasons，用于训练 affordance precondition 的表达。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - label：`steps[i].critical_frames[0].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the affordance preconditions that must hold before executing this step.
A: The light_switch should provide a pressable surface so it can be actuated by the hand.
```

### Task_14_Physical_Feasibility_Verification_QA（原始 Task_21_Physical_Feasibility_Verification_MCQ：可行性核验：四选一）

- **任务说明**：在单张关键帧上做“此刻是否可执行”的物理可行性判别，并且必须同时指出支持该判别的 1 条空间前置条件与 1 条可供性前置条件（含核验结果），输出为 1 句英文自然语言句子。本任务以 QA 形式呈现，不使用四选一选项。
- **与前置条件描述类任务的区别**：前置条件类任务更偏“描述/对齐具体前置条件条目”，本任务对 `step_goal` 做整体可行性判断，并要求给出一条空间与一条可供性依据，更贴近执行决策。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`（建议使用 step-init 关键帧 `critical_frames[0]`）
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Is this step physically feasible now? Answer in one English sentence, and justify the decision by stating one spatial precondition and one affordance precondition, and whether each is satisfied/violated/not directly observable in this frame.
A: It is feasible now because the hand is visibly touching the light switch (spatial precondition satisfied) and the light switch provides a pressable surface (affordance precondition satisfied).
```

### Task_15_Spatial_Postcondition_Description（原始 Task_22_Spatial_Postcondition_Description：空间后置条件：描述 postcondition）

- **任务说明**：基于**晚关键帧**（`critical_frames[1]`），用自然语言描述该步导致的空间后置条件（postconditions on spatial），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - label：`steps[i].critical_frames[1].causal_chain.causal_effect_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=04, frame_index=025): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
Q: Step goal: "Wash the cucumber and carrot under running water and place them on the countertop." Describe the spatial postconditions that should hold after completing this step.
A: The cucumber should be on_top_of the countertop after it is placed there.
```

### Task_16_Affordance_Postcondition_Description（原始 Task_24_Affordance_Postcondition_Description：可供性后置条件：描述 postcondition）

- **任务说明**：基于**晚关键帧**，用自然语言描述该步导致的可供性后置条件（postconditions on affordance），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：
  - 上下文：`steps[i].step_goal`
  - label：`steps[i].critical_frames[1].causal_chain.causal_effect_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the affordance postconditions that should hold after completing this step.
A: The light_switch should be in a switched_on state after being pressed, though the internal state may be not directly observable from a single frame.
```

### Task_17_Inter_Step_Dependency_Analysis（原始 Task_28_Inter_Step_Dependency_Analysis：跨步依赖解释）

- **任务说明**：解释跨步依赖：上一动作的后果如何满足下一步的前置条件，强调“effect → precondition”的可解释链接（建议 1–2 句）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_spatial`
  - `steps[i].causal_chain.causal_effect_on_affordance`
  - `steps[i+1].step_goal`
  - `steps[i+1].causal_chain.causal_precondition_on_spatial`
  - `steps[i+1].causal_chain.causal_precondition_on_affordance`
  - `high_level_goal`
- **Multimodal_input 类型（四类）**：`keyframe_single`（建议 step i 尾关键帧；可选追加 step i+1 首关键帧）
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Previous step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Next step goal: "Retrieve a carrot and a cucumber from the refrigerator." How does the outcome of the previous step satisfy the preconditions for the next step?
A: Step 1 illuminates the workspace (effect), thereby satisfying the visibility/safety precondition needed to locate and access the refrigerator in Step 2 (precondition).
```

### Task_18_Next_Step_Goal_Prediction_From_Prefix（原始 Task_30_Next_Step_Goal_Prediction_From_Prefix：前缀预测下一步 step_goal）

- **任务说明**：基于视频前缀预测下一步 `step_goal`（严格只输出下一步），用于训练长时序“前缀→下一步”的规划能力。
- **字段（JSONPath）**：
  - 上下文：`high_level_goal`
  - 上下文：`steps[i].step_goal`（前缀最后完成的 step_goal；必给）
  - label：`steps[i+1].step_goal`（下一步）
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=02): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Last completed step (in this prefix): "Retrieve a carrot and a cucumber from the refrigerator." What is the next step goal?
A: Gather a cutting board and a knife and place them on the countertop.
```

### Task_19_Middle_Steps_Infill_From_Head_Tail（原始 Task_32_Middle_Steps_Infill_From_Head_Tail：头尾证据 → 中间步骤补全）

- **任务说明**：给定视频头尾证据与整体目标，补全中间缺失的步骤序列（按顺序输出），用于训练长时序“补全/插值”规划能力。
- **字段（JSONPath）**：
  - 上下文：`high_level_goal`
  - label：`steps[*].step_goal`（按顺序输出缺失的中间步骤子序列）
- **Multimodal_input 类型（四类）**：`images_uniform_scene`（head-tail 子集）
- **范例**：

```text
Multimodal_input:
- images_uniform_scene(head=4, tail=4): <ITEM_DIR>/sampled_frames/sample_*.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them." Based on the beginning/end glimpses of the video, infer the missing middle steps in order.
A: 1) Retrieve the vegetables from the refrigerator. 2) Gather a cutting board and a knife. 3) Wash the vegetables under running water.
```

### Task_20_Next_K_Steps_Prediction_From_Prefix_QA（原始 Task_33_Next_K_Steps_MultiSelect_From_Prefix：未来 K 步多选）

- **任务说明**：给定视频前缀与 `high_level_goal`，预测接下来 `K` 个 `step_goal`（按时间顺序输出），用于训练未来多步规划能力。
- **字段（JSONPath）**：
  - 上下文：`high_level_goal`
  - 上下文：`steps[i].step_goal`（前缀最后完成的 step_goal；必给）
  - label：`steps[i+1:i+K].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=01): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them." Last completed step (in this prefix): "Enter the kitchen and turn on the light to illuminate the workspace." Based on this prefix, predict the next K=3 step goals in order.
A: 1) Retrieve a carrot and a cucumber from the refrigerator. 2) Gather a cutting board and a knife and place them on the countertop. 3) Wash the cucumber and carrot under running water and place them on the countertop.
```

### Task_22_Plan_Repair_From_Flaw（原始 Task_36_Plan_Repair_From_Flaw：坏计划修复：输出纠正后的计划）

- **任务说明**：给定视频前缀与一个“只含单一扰动”的坏计划（bad_plan），输出纠正后的正确计划序列，用于训练失败反思中的“纠错→重规划”能力（Task_28 的后续闭环）。
- **字段来源**：Task_28 生成的 `bad_plan_steps` 与 gold `steps[i+1:i+K].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=03): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Based on this prefix, bad_plan_steps are proposed as the next steps: 1) "Wash the cucumber and carrot under running water and place them on the countertop." 2) "Put the vegetables back into the refrigerator and stop." 3) "Slice the cucumber into circular pieces on the cutting board." Repair the plan by outputting the corrected 3-step sequence.
A: 1) "Wash the cucumber and carrot under running water and place them on the countertop." 2) "Gather a cutting board and a knife and place them on the countertop." 3) "Slice the cucumber into circular pieces on the cutting board."
```

### Task_23_Counterfactual_Prediction（原始 Task_37_Counterfactual_Prediction：反事实挑战与结果；自由文本）

- **任务说明**：给定该步骤的反事实挑战问题（what-if），从 **spatial + affordance** 角度预测物理后果（自由文本）。只做结果预测，不提出任何恢复/修复策略。
- **字段（JSONPath）**：`steps[i].step_goal`, `steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=032): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." Counterfactual: What if the cutting board was slippery on the countertop? From a spatial & affordance perspective, what would likely happen? Only predict the outcome; do not propose any recovery actions.
A: The board might slide under the applied cutting force, making the cutting setup unstable and increasing the risk of the knife slipping.
```

### Task_24_Counterfactual_Outcome_QA（原始 Task_38_Counterfactual_Outcome_MCQ：反事实结果四选一；客观化 Task_37）

- **任务说明**：给定反事实挑战问题（what-if），从 **spatial + affordance** 角度生成最可能的 `expected_challenge_outcome`（QA 短回答）。只做结果预测，不提出任何恢复/修复策略。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].counterfactual_challenge_question`
  - `steps[i].expected_challenge_outcome`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=032): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." What is the most likely outcome if the cutting board is slippery on the countertop? Answer with a short outcome prediction grounded in spatial setup and affordance, and do not propose any recovery actions.
A: The board may slide when cutting force is applied, making the knife motion unstable because the low-friction contact cannot resist lateral forces.
```

### Task_25_Failure_Recovery_Protocol（原始 Task_39_Failure_Recovery_Protocol：失败模式与恢复策略；自由文本 + Task_40_Recovery_Strategy_MCQ：恢复策略四选一；客观化 Task_39）

- **任务说明**：围绕该步骤可能出现的失败原因，给出可执行的恢复策略（自由文本），并从 **spatial + affordance** 角度说明其有效性。
- **字段（JSONPath）**：`steps[i].step_goal`, `steps[i].failure_reflecting.reason`, `steps[i].failure_reflecting.recovery_strategy`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=032): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." Failure reason: "The cucumber rolls during cutting because it is not stabilized." What is a plausible recovery strategy? Explain briefly using spatial stability and affordance/mechanism.
A: Cut a flat side to prevent rolling, then stabilize it with the non-cutting hand while slicing.
```
### Task_26_Next_Step_After_Recovery_QA（原始 Task_42_Next_Step_After_Recovery：失败驱动重规划：恢复后下一步选择）

- **任务说明**：失败驱动重规划：给定失败原因与恢复策略，输出“恢复之后最合适的下一步要做什么”（用 1 句 `step_goal` 表达），形成失败反思闭环。
- **字段来源**：
  - `high_level_goal`
  - `steps[i].failure_reflecting.reason`
  - `steps[i].failure_reflecting.recovery_strategy`
  - label：`steps[i].step_goal`（默认重试本步）或 `steps[i+1].step_goal`（少量样本可选：恢复策略等价于“本步已完成/可跳过”时）
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=06): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step06_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Failure reason: "The cutting board is sliding during cutting." Recovery strategy: "Place a damp cloth under the cutting board to increase friction." After applying the recovery strategy, what is the most appropriate next step? Answer as a single step_goal.
A: Slice the cucumber into circular pieces on the cutting board.
```

---
