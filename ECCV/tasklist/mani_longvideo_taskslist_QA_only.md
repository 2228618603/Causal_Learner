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

本任务集假设每个 item 有一份标注 JSON（文件名仅示例；数据可来自二阶段或三阶段等不同生成管线），其字段结构与 `ECCV/two_stage_new/mani_longvideo.py` 的 dataclass 保持一致。

关键字段（字段名以 JSON 为准；类型为概念描述）：

- 顶层：`high_level_goal: str`, `steps: List[PlanningStep]`
- PlanningStep：`step_id: int`, `step_goal: str`, `rationale: str`, `causal_chain: CausalChain`, `counterfactual_challenge_question: str`, `expected_challenge_outcome: str`, `failure_reflecting: FailureReflecting`, `critical_frames: List[CriticalFrameAnnotation]`
- 约束（本任务集默认）：每个 step 的 `critical_frames` **固定为 2**，且按时间顺序排列（`critical_frames[0]` 更早，`critical_frames[1]` 更晚）
- CausalChain：`agent: str`, `action: str`, `patient: str`，以及四组 list：
  - `causal_precondition_on_spatial: List[SpatialRelation]`
  - `causal_precondition_on_affordance: List[AffordanceState]`
  - `causal_effect_on_spatial: List[SpatialRelation]`
  - `causal_effect_on_affordance: List[AffordanceState]`
- SpatialRelation：`relation: str`, `objects: List[str]`, `truth: bool`（用于核验类任务的 Yes/No 标签）
- AffordanceState：`object_name: str`, `affordance_types: List[str]`, `reasons: str`（可选）
- CriticalFrameAnnotation：`frame_index: int`, `action_state_change_description: str`, `causal_chain: CausalChain`, `interaction: Interaction`
- Interaction：`tools: List[str]`, `materials: List[str]`, `hotspot: Hotspot`
- Hotspot：`description: str`, `affordance_type: str`, `mechanism: str`
- FailureReflecting：`reason: str`, `recovery_strategy: str`

提醒：
- `critical_frames[*].frame_index` 为 **1-based** 索引，默认对齐到全局均匀抽帧序列（`images_uniform_scene`）的顺序位置。

## 2. 任务卡片（最终任务集合：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：用于对齐标注 JSON 的字段使用（可用于实现生成器或做数据审计）。
- **Multimodal_input 类型（四类）**：明确该任务使用 `keyframe_single / images_uniform_scene / video_clip / video_prefix` 中的哪一种（或少量可选 fallback）。
- **QA 范例**：只展示 `Multimodal_input / Q / A`，用于训练时的“多模态输入→文本输出”对齐。范例中的路径仅作示意，不作为硬约束。

统一约定（仅用于范例表达）：

- 用 `<ITEM_DIR>` 表示某个单视频样本目录。

### Task_01_Goal_Recognition_From_Full_Video

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

### Task_02_Macro_Anchor_Extraction

- **任务说明**：给定 `high_level_goal` 与全局均匀抽帧多图，从候选对象中选择与 `high_level_goal` 直接相关、且计划中会用到的 planning key_objects（去重输出）。
- **字段（JSONPath）**：
  - `high_level_goal`
  - 候选池（用于构造 options/label）：
    - `steps[*].step_goal`
    - `steps[*].causal_chain.agent`
    - `steps[*].causal_chain.patient`
    - `steps[*].causal_chain.causal_precondition_on_spatial[*].objects[*]`
    - `steps[*].causal_chain.causal_precondition_on_affordance[*].object_name`
    - `steps[*].causal_chain.causal_effect_on_spatial[*].objects[*]`
    - `steps[*].causal_chain.causal_effect_on_affordance[*].object_name`
    - `steps[*].critical_frames[*].interaction.tools[*]`
    - `steps[*].critical_frames[*].interaction.materials[*]`
    - `steps[*].critical_frames[*].interaction.hotspot.description`
    - `steps[*].critical_frames[*].action_state_change_description`
- **Multimodal_input 类型（四类）**：`images_uniform_scene`
- **范例**：

```text
Multimodal_input:
- images_uniform_scene(samples=8, uniform): <ITEM_DIR>/sampled_frames/sample_*.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." From the candidate objects ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet","microwave","dish_soap"], list the key objects that are directly relevant to the goal and will be used for planning.
A: ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet"]
```

### Task_03_Clip_to_StepGoal_Statement

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

### Task_04_Patient_Identification_QA

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

### Task_05_Action_Phrase_QA

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

### Task_06_Hotspot_AffordanceType_QA

- **任务说明**：识别关键帧交互热点的 `affordance_type`，以 QA 形式输出可供性类别短语。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].interaction.hotspot.affordance_type`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." What is the affordance_type of the interaction hotspot in this image?
A: pressable_surface
```

### Task_07_Hotspot_Mechanism_QA

- **任务说明**：描述该关键帧交互热点的物理机制解释（mechanism），以 QA 形式输出短句。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Briefly describe the physical mechanism of the interaction hotspot in this image.
A: Pressing transfers force to an internal toggle mechanism to complete a circuit.
```

### Task_08_Micro_Affordance_Visual_Semantics

- **任务说明**：在单张关键帧中定位交互热点（hotspot），并描述其可供性类别与物理机制（为什么这个区域“能被这样用”）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - `steps[i].critical_frames[j].interaction.hotspot.description`
  - `steps[i].critical_frames[j].interaction.hotspot.affordance_type`
  - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Locate the interaction hotspot area in the image first, then describe its affordance_type and mechanism.
A: The hotspot is the raised rocker surface of the light switch where the finger contacts. It affords pressable_surface, and pressing it actuates the internal toggle mechanism to complete the circuit and turn on the light.
```

### Task_09_State_Evolution_Description

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

### Task_10_Holistic_Causal_Chain_Analysis

- **任务说明**：基于关键帧解释物理因果链：空间/可供性前置条件 → 动作与机制 → 空间/可供性后效，强调“可被证据支持的因果闭环”。
- **字段（JSONPath）**：
  - `high_level_goal`（上下文）
  - `steps[i].step_goal`（上下文）
  - `steps[i].causal_chain.agent`
  - `steps[i].causal_chain.action`
  - `steps[i].causal_chain.patient`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_affordance`
  - `steps[i].critical_frames[j].interaction.hotspot.description`
  - `steps[i].critical_frames[j].interaction.hotspot.affordance_type`
  - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=039): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Step goal: "Slice the cucumber into circular pieces on the cutting board." Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects.
A: The cucumber is stabilized on the cutting board while the knife edge contacts it, creating a controlled cutting setup. The sharp blade concentrates force along a thin edge, exceeding the cucumber’s shear strength. As a result, the cucumber separates and a new slice is produced.
```

### Task_11_Strategic_Rationale_Justification

- **任务说明**：解释该步骤为什么必要、如何支撑整体目标（从“动机/必要性”的因果角度给出简短说明）。
- **字段（JSONPath）**：`steps[i].rationale`（可附 `high_level_goal`, `step_goal`）
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Why is this step necessary for the overall goal?
A: It provides sufficient lighting so later navigation and object manipulation can be done safely and accurately.
```

### Task_12_Spatial_Precondition_Description

- **任务说明**：基于**早关键帧**（`critical_frames[0]`），输出该关键帧对应的空间前置条件描述（来自 frame-level `causal_chain.causal_precondition_on_spatial`），用于训练“precondition 表达与对齐”。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the spatial preconditions that must hold before executing this step.
A: The hand should be contacting the light_switch so it can apply force to toggle it.
```

### Task_13_Spatial_Precondition_Verification_QA

- **任务说明**：给定**早关键帧**与当前 `step_goal`，再给定 1 条候选空间前置条件陈述，判断该陈述是否正确（QA 形式回答）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Candidate spatial precondition: "hand contacting light_switch". Is this spatial precondition correct for executing the step in the current scene? Answer Yes/No/not directly observable.
A: Yes
```

### Task_14_Affordance_Precondition_Description

- **任务说明**：基于**早关键帧**，用自然语言描述该关键时刻对应的可供性/状态前置条件（frame-level `causal_precondition_on_affordance`），可选简述 reasons，用于训练 affordance precondition 的表达。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the affordance preconditions that must hold before executing this step.
A: The light_switch should provide a pressable surface so it can be actuated by the hand.
```

### Task_15_Affordance_Precondition_Verification_QA

- **任务说明**：给定**早关键帧**与当前 `step_goal`，再给定 1 条候选可供性前置条件陈述，判断该陈述是否正确（QA 形式回答）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Candidate affordance precondition: "light_switch pressable_surface". Is this affordance precondition correct for executing the step? Answer Yes/No/not directly observable.
A: Yes
```

### Task_16_Physical_Feasibility_Verification

- **任务说明**：基于关键帧中的空间与可供性前置条件，判断该步骤此刻是否物理可行（可行/不可行/不可观测），并要求依据证据作答。
- **与 Task_12/13/14/15 的区别**：`Task_12/13/14/15` 更偏“描述/核验具体前置条件条目”，本任务是对 `step_goal` 做整体可行性三态判断（允许不可观测），更贴近真实执行决策。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Is this step physically feasible now based on the visible spatial and affordance preconditions?
A: feasible
```

### Task_17_Spatial_Postcondition_Description

- **任务说明**：基于**晚关键帧**（`critical_frames[1]`），用自然语言描述该步导致的空间后置条件（postconditions on spatial），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=04, frame_index=025): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
Q: Step goal: "Wash the cucumber and carrot under running water and place them on the countertop." Describe the spatial postconditions that should hold after completing this step.
A: The cucumber should be on_top_of the countertop after it is placed there.
```

### Task_18_Spatial_Postcondition_Verification_QA

- **任务说明**：给定**晚关键帧**与当前 `step_goal`，再给定 1 条候选空间后置条件陈述，判断该陈述是否正确（QA 形式回答）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=04, frame_index=025): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
Q: Step goal: "Wash the cucumber and carrot under running water and place them on the countertop." Candidate spatial postcondition: "cucumber on_top_of countertop". Is this spatial postcondition correct after completing the step? Answer Yes/No/not directly observable.
A: Yes
```

### Task_19_Affordance_Postcondition_Description

- **任务说明**：基于**晚关键帧**，用自然语言描述该步导致的可供性后置条件（postconditions on affordance），并对不可从证据判断的后置条件显式标注不可观测。
- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_effect_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Describe the affordance postconditions that should hold after completing this step.
A: The light_switch should be in a switched_on state after being pressed, though the internal state may be not directly observable from a single frame.
```

### Task_20_Affordance_Postcondition_Verification_QA

- **任务说明**：给定**晚关键帧**与当前 `step_goal`，再给定 1 条候选可供性后置条件陈述，判断该陈述是否正确（QA 形式回答）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_affordance`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Candidate affordance postcondition: "light_switch switched_on". Is this affordance postcondition correct after completing the step? Answer Yes/No/not directly observable.
A: Yes
```

### Task_21_Temporal_Order_Check_AB

- **任务说明**：给定两张关键帧（A/B）及对应事件描述，判断哪一个事件在视频中更早发生（输出更早事件的描述文本），用于训练跨步时间顺序理解。
- **字段（JSONPath）**：
  - `high_level_goal`（上下文，可选）
  - `steps[a].critical_frames[x].action_state_change_description`
  - `steps[b].critical_frames[y].action_state_change_description`
- **Multimodal_input 类型（四类）**：`keyframe_single`（2 张关键帧图像，记为 A/B）
- **范例**：

```text
Multimodal_input:
- keyframe_single(A; step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
- keyframe_single(B; step_id=04, frame_index=020): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Event A: "A person's hand presses a rocker-style light switch." Event B: "The person rubs a cucumber under running water." Which event happens earlier in the video? Answer with the earlier event description verbatim.
A: A person's hand presses a rocker-style light switch.
```

### Task_22_Stage2_FrameIndex_Localization_Check

- **任务说明**：（可选）基于全局均匀抽帧多图与某个 critical frame 的文本标注，预测/核验该关键时刻对应的 `frame_index`（1-based），用于更严格的时间定位/对齐评测。
- **字段（JSONPath）**：
  - `steps[i].step_goal`（上下文）
  - label：`steps[i].critical_frames[j].frame_index`
  - query 文本：
    - `steps[i].critical_frames[j].action_state_change_description`
    - `steps[i].critical_frames[j].interaction.hotspot.description`
    - `steps[i].critical_frames[j].interaction.hotspot.affordance_type`
    - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
  - 可选约束：
    - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
    - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
- **Multimodal_input 类型（四类）**：`images_uniform_scene`
- **范例**：

```text
Multimodal_input:
- images_uniform_scene(full N images, numbered Frame 1..N): <ITEM_DIR>/sampled_frames/sample_*.jpg
Q: Context: Step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Critical frame description: "The person presses the light switch and the light turns on." What is the frame_index (1-based) that best matches this moment?
A: 18
```

### Task_23_Inter_Step_Dependency_Analysis

- **任务说明**：解释跨步依赖：上一动作的后果如何满足下一步的前置条件（尽量引用重合对象/affordance 作为依赖证据）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].causal_chain.causal_effect_on_spatial`
  - `steps[i].causal_chain.causal_effect_on_affordance`
  - `steps[i+1].step_goal`
  - `steps[i+1].causal_chain.causal_precondition_on_spatial`
  - `steps[i+1].causal_chain.causal_precondition_on_affordance`
  - `high_level_goal`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=01, frame_index=002): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Previous step goal: "Enter the kitchen and turn on the light to illuminate the workspace." Next step goal: "Retrieve a carrot and a cucumber from the refrigerator." How does the outcome of the previous step satisfy the preconditions for the next step?
A: Turning on the light makes the workspace visible and safe, enabling the person to locate and access the refrigerator to retrieve the vegetables.
```

### Task_24_Next_Step_Goal_Prediction_From_Prefix

- **任务说明**：基于视频前缀预测下一步 `step_goal`（严格只输出下一步），用于训练长时序“前缀→下一步”的规划能力。
- **字段（JSONPath）**：
  - `high_level_goal`（上下文）
  - `steps[i].step_goal`（上下文，可选）
  - label：`steps[i+1].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=02): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Last completed step (in this prefix): "Retrieve a carrot and a cucumber from the refrigerator." What is the next step goal?
A: Gather a cutting board and a knife and place them on the countertop.
```

### Task_25_Prefix_Completed_Steps_QA

- **任务说明**：给定视频前缀与“完整计划 step 列表”，判断当前前缀已经完成到哪一步（推荐输出最大已完成 `step_id`），用于可评分的长时序进度理解。
- **字段（JSONPath）**：`high_level_goal`, `steps[*].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=03): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."
  Plan steps:
    1) Enter the kitchen and turn on the light to illuminate the workspace.
    2) Retrieve a carrot and a cucumber from the refrigerator.
    3) Gather a cutting board and a knife and place them on the countertop.
    4) Wash the cucumber and carrot under running water and place them on the countertop.
    ...
  Up to which step number has the plan been completed in this prefix? Answer with an integer (1-based in the plan list).
A: 3
```

### Task_26_Middle_Steps_Infill_From_Head_Tail

- **任务说明**：给定视频头尾证据与整体目标，补全中间缺失的步骤序列（按顺序输出），用于训练长时序“补全/插值”规划能力。
- **字段（JSONPath）**：
  - `high_level_goal`
  - `steps[*].step_id`
  - `steps[*].step_goal`
- **Multimodal_input 类型（四类）**：`images_uniform_scene`（head-tail 子集）
- **范例**：

```text
Multimodal_input:
- images_uniform_scene(head=4, tail=4): <ITEM_DIR>/sampled_frames/sample_*.jpg
Q: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them." Based on the beginning/end glimpses of the video, infer the missing middle steps in order.
A: 1) Retrieve the vegetables from the refrigerator. 2) Gather a cutting board and a knife. 3) Wash the vegetables under running water.
```

### Task_27_Next_K_Steps_Prediction_From_Prefix_QA

- **任务说明**：给定视频前缀与 `high_level_goal`，预测接下来 `K` 个 `step_goal`（按时间顺序输出），用于训练未来多步规划能力。
- **字段（JSONPath）**：
  - `high_level_goal`（上下文）
  - `steps[i].step_goal`（上下文，可选）
  - label：`steps[i+1:i+K].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=01): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them." Last completed step (in this prefix): "Enter the kitchen and turn on the light to illuminate the workspace." Based on this prefix, predict the next K=3 step goals in order.
A: 1) Retrieve a carrot and a cucumber from the refrigerator. 2) Gather a cutting board and a knife and place them on the countertop. 3) Wash the cucumber and carrot under running water and place them on the countertop.
```

### Task_28_Next_K_Steps_Reordering_From_Prefix

- **任务说明**：给定前缀与一组被打乱的未来候选步骤，要求重排为最合理的时间顺序（输出序列），用于训练多步规划与顺序推断。
- **字段（JSONPath）**：
  - `high_level_goal`（上下文）
  - `steps[i].step_goal`（上下文，可选）
  - label：`steps[i+1:i+K].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=01): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them." Last completed step (in this prefix): "Enter the kitchen and turn on the light to illuminate the workspace." Reorder the shuffled candidate steps ["Wash the cucumber and carrot under running water and place them on the countertop.", "Retrieve a carrot and a cucumber from the refrigerator.", "Gather a cutting board and a knife and place them on the countertop."] into the most plausible next-step sequence.
A: 1) Retrieve a carrot and a cucumber from the refrigerator. 2) Gather a cutting board and a knife and place them on the countertop. 3) Wash the cucumber and carrot under running water and place them on the countertop.
```

### Task_29_Failed_Planning_Flaw_Pointing

- **任务说明**：对一个含“单一错误”的坏计划进行缺陷定位：指出错误步骤、错误类型并给出一句话理由，强调可自动评分与可归因。
- **字段（JSONPath）**：
  - `high_level_goal`
  - `steps[*].step_goal`
  - 可选（用于构造“依赖违反”）：
    - `steps[*].causal_chain.causal_precondition_on_spatial`
    - `steps[*].causal_chain.causal_precondition_on_affordance`
    - `steps[*].causal_chain.causal_effect_on_spatial`
    - `steps[*].causal_chain.causal_effect_on_affordance`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=02): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Based on this prefix, the following bad_plan_steps are proposed as the next steps: 1) "Slice the cucumber on the cutting board." 2) "Open the refrigerator and retrieve a carrot." 3) "Wash the vegetables under running water." Identify the flaw in the bad plan.
A: FlawStep=1; FlawType=precondition_missing; Reason=You cannot slice the cucumber before retrieving it and preparing the cutting board and knife.
```

### Task_30_Plan_Repair_From_Flaw

- **任务说明**：给定视频前缀与一个“只含单一扰动”的坏计划（bad_plan），输出纠正后的正确计划序列，用于训练失败反思中的“纠错→重规划”能力（Task_29 的后续闭环）。
- **字段来源**：Task_29 生成的 `bad_plan_steps` 与 gold `steps[i+1:i+K].step_goal`
- **Multimodal_input 类型（四类）**：`video_prefix`
- **范例**：

```text
Multimodal_input:
- video_prefix(prefix_end_step=03): <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
Q: Context: High-level goal: "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board." Based on this prefix, bad_plan_steps are proposed as the next steps: 1) "Wash the cucumber and carrot under running water and place them on the countertop." 2) "Put the vegetables back into the refrigerator and stop." 3) "Slice the cucumber into circular pieces on the cutting board." Repair the plan by outputting the corrected 3-step sequence.
A: 1) "Wash the cucumber and carrot under running water and place them on the countertop." 2) "Gather a cutting board and a knife and place them on the countertop." 3) "Slice the cucumber into circular pieces on the cutting board."
```

### Task_31_Counterfactual_Prediction

- **任务说明**：给定该步骤的反事实挑战问题（what-if），从 **spatial + affordance** 角度预测物理后果（自由文本）。
- **字段（JSONPath）**：`steps[i].step_goal`, `steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=032): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." Counterfactual: What if the cutting board was slippery on the countertop? From a spatial & affordance perspective, what would likely happen?
A: The board might slide under the applied cutting force, making the cutting setup unstable and increasing the risk of the knife slipping.
```

### Task_32_Counterfactual_Outcome_QA

- **任务说明**：给定反事实挑战问题（what-if），从 **spatial + affordance** 角度生成最可能的 `expected_challenge_outcome`（QA 短回答）。
- **字段（JSONPath）**：
  - `steps[i].step_goal`
  - `steps[i].counterfactual_challenge_question`
  - `steps[i].expected_challenge_outcome`
- **Multimodal_input 类型（四类）**：`keyframe_single`
- **范例**：

```text
Multimodal_input:
- keyframe_single(step_id=07, frame_index=032): <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
Q: Context: Step goal: "Slice the cucumber into circular pieces on the cutting board." What is the most likely outcome if the cutting board is slippery on the countertop? Answer with a short outcome prediction grounded in spatial setup and affordance.
A: The board may slide when cutting force is applied, making the knife motion unstable because the low-friction contact cannot resist lateral forces.
```

### Task_33_Failure_Recovery_Protocol

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
### Task_34_Next_Step_After_Recovery_QA

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
