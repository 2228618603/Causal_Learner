# mani_longvideo 多模态任务清单 v6（融合 useful.md 修改意见）

本文档目标：结合

- `ECCV/chat/useful.md` 的修改意见（聚焦“因果规划 + 失败反思”，强调可评分客观题、减少低增益任务、统一证据口径）
- `ECCV/chat/mani_longvideo_tasks_plan_final.md` 的既有 Task_01~Task_30 任务类型与工程约定

汇总并重新生成一个“最终可落地”的任务清单版本（v6），并提供：

- `## 5. 任务体系`：最终任务集合（含删改/合并后的口径）；
- `## 9. 任务卡片`：逐任务的字段来源、证据来源、样本构造、QA 范例；
- `## 10. 当前无约束 QA 示例`：逐任务自由文本版本（便于指令微调/开放式生成）。

> 重要：本文档只定义“任务与样本构造规范”。实际数据生成时，**不要把文件路径/目录名/`ts_XX.XXs`/step slug** 暴露给模型输入（它们会泄漏 step_goal/时间顺序）。

---

## 0. 核心目标与统一口径

### 0.1 最终核心能力（与 useful.md 对齐）

- **因果规划（causal planning）**：显式状态/约束（spatial + affordance）→ 可行性 → 因果后果 → 跨步依赖 → 长时序规划（prefix / reorder / infill）
- **失败反思（failure reflecting）**：不一致检测 → 缺陷类型定位（flaw type）→ 恢复策略选择 → 失败驱动重规划（replanning）

### 0.2 证据形态（统一 4 类）

为兼容训练落地与可控性，`meta.evidence_type` 建议统一为 4 类：

1) `keyframe_single`：关键帧图像（通常 1 张；允许 2 张用于 pair 任务，但不要暴露帧序号/时间戳）
2) `images_uniform_scene`：全局均匀抽帧多图（来自 `<ITEM_DIR>/sampled_frames/`）
3) `video_clip`：局部片段 mp4（来自 `<ITEM_DIR>/last_frame_segments/`）
4) `video_prefix`：累积前缀 mp4（来自 `<ITEM_DIR>/cumulative_last_frame_segments/`）

工程实现细节：

- 如果当前模型不支持 mp4，可在生成器中把 `video_clip/video_prefix` 预抽帧成 image 序列，但 **`meta.evidence_type` 仍保持 `video_clip/video_prefix`**（把抽帧文件写进 `meta.evidence_files`）。
- 原 v5 中的 `images_uniform_clip` 在 v6 中视作 `video_clip/video_prefix` 的实现细节，不再作为独立 evidence_type。

### 0.3 Schema 摘要（与 three_stage 产物对齐）

本文默认 item 的核心标注文件为：`<ITEM_DIR>/causal_plan_with_keyframes.json`，关键字段：

- 顶层：`high_level_goal: str`, `steps: List[Step]`
- Step：`step_id`, `step_goal`, `rationale`, `causal_chain`, `counterfactual_challenge_question`, `expected_challenge_outcome`, `failure_reflecting`, `critical_frames`
- CausalChain：`agent`, `action`, `patient`, `causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`
- CriticalFrame：`frame_index`, `action_state_change_description`, `causal_chain`, `interaction.tools/materials/hotspot`

关键提醒：

- `critical_frames[*].frame_index` 是 **step clip 内局部索引**，不等于全局 `sampled_frames/` 的序号。
- 关键帧图片路径不要依赖 `keyframe_image_path`（脚本可能生成绝对路径不可移植）；应按 step 目录内 `frame_###_ts_*.jpg` 解析。

---

## 5. 任务体系 v6（最终任务集合）

本节列出 v6 的“最终任务集合”。每个任务均绑定：

- **字段来源（JSONPath）**：用于构造 `meta.fields` 与 label；
- **证据形态（统一 4 类）**：`keyframe_single / images_uniform_scene / video_clip / video_prefix`；
- **输出约束**：优先客观题（Yes/No/三态/ABCD/分类）；
- **负样本规则**：尽量“单一扰动”，便于评分与归因。

### 5.0 核心/支撑任务分层（建议配比口径）

如果训练与评测最终核心严格落在 “因果规划 + 失败反思”，建议按如下分层使用（不强行改编号，优先在配比与指标层面裁剪）：

- **核心 · 因果规划（Causal Planning）**
  - `Task_12/13/16/18/24/29/30/35/37/38/39/42`
- **核心 · 失败反思（Failure Reflecting）**
  - `Task_22/28/40/41/44`
- **支撑 · Grounding（对象/机制/动作短语）**
  - `Task_01/03/04/05/06/27/31/32/33/34/36`

### 5.1 v6 明确移除/替代的 v5 任务

基于 `useful.md` 的修改意见，v6 做如下裁剪（原因：冗余/低增益/难评分/易泄漏）：

- **移除**：`Task_07`（由 `Task_23` 的“完整视频目标识别”取代）
- **移除**：`Task_17`（Why/How 综合过宽；机制/How 由 `Task_03/06/33/34/35/37` 提供）
- **移除**：`Task_19`（由 `Task_38/39` 的可评分后置核验取代）
- **移除**：`Task_20/21/25`（偏辅助叙述，非核心指标；进度类由 `Task_42` 取代）
- **替代**：`Task_09 + Task_18(v5)` → **`Task_18(v6)`**（把“前置条件陈述/三态核验”改为“可评分的 precondition 选择/核验”）
- **建议降权/仅作补充**：`Task_02/10/11/13/14/15`（其中 14/15 通过 40/41 客观化）

> 注意：v6 不要求删除 v5 文档；但在“最终任务清单”里不再将上述任务作为核心可生成项。

---

### Task_01_Macro_Anchor_Extraction（场景锚点/关键对象集合）

- **字段（JSONPath）**：
  - `steps[*].critical_frames[*].interaction.tools[*]`
  - `steps[*].critical_frames[*].interaction.materials[*]`
  - `steps[*].critical_frames[*].causal_chain.causal_precondition_on_spatial[*].objects[*]`
  - `steps[*].critical_frames[*].causal_chain.causal_precondition_on_affordance[*].object_name`
  - `steps[*].critical_frames[*].causal_chain.agent`
  - `steps[*].critical_frames[*].causal_chain.patient`
- **证据形态**：`images_uniform_scene`
- **输出约束**：自由文本或列表均可；建议输出去重对象列表（snake_case）。

### Task_02_Transient_Geometric_Verification（空间关系自然语言复述；低优先级）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial[k].relation/objects/truth`
- **证据形态**：`keyframe_single`
- **输出约束**：只复述关系，不做真假判别；真假核验用 `Task_27`。
- **备注**：与 `Task_27` 高度重叠，v6 建议降权。

### Task_03_Micro_Affordance_Visual_Semantics（热点可供性语义：描述/机制）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.description/affordance_type/mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；可与 `Task_33/34`（客观题）形成互补。

### Task_04_Entity_Role_Identification（工具/材料角色识别）

- **字段（JSONPath）**：`steps[i].critical_frames[*].interaction.tools/materials`（可附 `steps[i].step_goal`）
- **证据形态**：`keyframe_single`
- **输出约束**：
  - 自由文本版：区分 tools vs materials；
  - 客观题变体（推荐）：Yes/No（给定实体，问其在该 step 是否为 tool）。

### Task_05_State_Evolution_Description（关键帧动作-状态变化事件描述）

- **字段（JSONPath）**：`steps[i].critical_frames[j].action_state_change_description`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本 1–2 句；用于支持时序类任务（如 `Task_26`）。

### Task_06_Holistic_Causal_Chain_Analysis（关键帧物理因果链解释）

- **字段（JSONPath）**：
  - `steps[i].critical_frames[j].causal_chain.agent/action/patient`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - `steps[i].critical_frames[j].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
  - `steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本，但建议固定为两段（Spatial+Affordance → Effects）。

### Task_08_Strategic_Rationale_Justification（步骤动机/必要性解释）

- **字段（JSONPath）**：`steps[i].rationale`（可附 `high_level_goal`, `step_goal`）
- **证据形态**：`keyframe_single`
- **输出约束**：尽量短（1–2 句），强调“对后续步骤的因果贡献”。

### Task_10_Step_Execution_Statement（步骤执行动作描述；补充任务）

- **字段（JSONPath）**：`steps[i].step_goal`
- **证据形态**：`video_clip`（推荐）或 `keyframe_single`（fallback）
- **输出约束**：
  - 允许列 2–5 个可观测微动作；看不清则写 `not clearly observable`。
- **备注**：严格可评分版本请用 `Task_31`（step_goal 四选一匹配）。

### Task_11_Expected_Physical_Effects（期望效果/后置状态：自由文本补充）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
- **证据形态**：`keyframe_single`（step-end 关键帧）
- **输出约束**：自由文本列出 2–6 条“期望效果”；对不可观测效果必须显式写 `not directly observable`。
- **备注**：v6 的核心可评分核验请使用 `Task_38/39`；本任务仅作为表达补充，不建议作为核心评测指标。

### Task_12_Inter_Step_Dependency_Analysis（跨步依赖解释）

- **字段（JSONPath）**：
  - `steps[i].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
  - `steps[i+1].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - 上下文：`steps[i].step_goal`, `steps[i+1].step_goal`, `high_level_goal`
- **证据形态**：`keyframe_single`（建议 step i 尾关键帧）
- **输出约束**：必须引用“重合对象/affordance”作为依赖证据；无重合则不生成样本。

### Task_13_Next_Action_Prediction（下一步 step_goal 预测：计划/弱监督 baseline）

- **字段（JSONPath）**：
  - 输入侧：`steps[i].step_goal`（可选）
  - 标签侧：`steps[i+1].step_goal`
- **证据形态**：`keyframe_single`（step i 尾关键帧；可选）或 `video_prefix`
- **输出约束**：严格只输出下一步 `step_goal`（不输出其它步骤文本）。
- **备注**：若有 `video_prefix`，优先使用 `Task_24`；本任务更适合作为“计划文本/弱监督”对照组。

### Task_14_Counterfactual_Prediction（反事实挑战与结果；自由文本）

- **字段（JSONPath）**：`steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；客观题版本见 `Task_40`。

### Task_15_Failure_Recovery_Protocol（失败模式与恢复策略；自由文本）

- **字段（JSONPath）**：`steps[i].failure_reflecting.reason`, `steps[i].failure_reflecting.recovery_strategy`
- **证据形态**：`keyframe_single`
- **输出约束**：自由文本；客观题版本见 `Task_41`。

### Task_16_Physical_Feasibility_Verification（可行性核验：三态）

- **字段（JSONPath）**：
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial`
  - `steps[i].critical_frames[j].causal_chain.causal_precondition_on_affordance`
  - 上下文：`steps[i].step_goal`
- **证据形态**：`keyframe_single`
- **输出约束**：三分类 `feasible / not feasible / not directly observable`。
- **负样本**：可通过“替换 patient/tool 或翻转关键空间关系”构造 `not feasible`。

### Task_18_Precondition_Check_or_MCQ（前置条件可评分核验/选择；v6 替代 Task_09+Task_18v5）

把“列举前置条件/三态核验”改为 **可评分的选择题/核验题**，减少开放式噪声。

- **字段（JSONPath）**：
  - 真实 preconditions：`steps[i].causal_chain.causal_precondition_on_spatial/causal_precondition_on_affordance`
  - 上下文：`steps[i].step_goal`, `high_level_goal`
- **证据形态**：`video_prefix`（到 step i 开始前；若无法对齐则用到 step i-1 尾）
- **推荐两种变体（二选一或同时做）**：
  1) **Yes/No（单条核验）**：给定一个 precondition（来自真实或扰动），判断 `Yes/No/Not observable`
  2) **ABCD（单选）**：4 个候选 precondition，选“最可能为真且与 step_goal 相关”的一项

### Task_22_Plan_Execution_Alignment（计划-执行一致性判别）

- **字段（JSONPath）**：`steps[i].step_goal`（可选 `steps[i].causal_chain.causal_effect_on_*` 作为辅助解释）
- **证据形态**：`video_clip`（尽量对齐 step i 的执行片段）
- **输出约束**：三分类 `match / partial match / mismatch`，并要求给出可见证据或声明不可判断。
- **负样本**：跨 step 替换 step_goal 或打乱 clip 对齐关系（写入 `meta.neg_sample=true`）。

### Task_23_Goal_Recognition_From_Full_Video（完整视频高阶目标识别；v6 替代 Task_07）

- **字段（JSONPath）**：`high_level_goal`
- **证据形态**：优先 `video_prefix`（全长前缀=完整视频）；无 mp4 时用 `images_uniform_scene`
- **输出约束**：自由文本 1 句（尽量为“整体目标+最终结果”）。

### Task_24_Next_Step_Goal_Prediction_From_Prefix（前缀预测下一步 step_goal）

- **字段（JSONPath）**：label 为 `steps[i+1].step_goal`；可选输入 `steps[i].step_goal`
- **证据形态**：`video_prefix`（前缀到 step i 尾）
- **输出约束**：严格只输出下一步 `step_goal`（不输出其它步骤）。

### Task_26_Temporal_Order_Check_AB（两事件先后判别；严格可评分）

- **字段来源**：
  - 事件文本：`steps[a].critical_frames[x].action_state_change_description` 与 `steps[b].critical_frames[y].action_state_change_description`
  - 标签依据：两关键帧对应的真实时间顺序（由生成器在后台用 `ts` 或其它对齐信息比较得出；**不进入 prompt**）
- **证据形态**：`keyframe_single`（允许 2 张 keyframes；写入 `meta.evidence_files=[imgA,imgB]`）
- **输出约束**：只输出 `A` 或 `B`（哪一个更早）。

### Task_27_Visual_Spatial_Relation_Check（视觉空间关系真假核验）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial[k].relation/objects/truth`
- **证据形态**：`keyframe_single`
- **输出约束**：推荐三态 `true / false / not directly observable`；或严格 Yes/No（仅取可观测子集）。
- **负样本**：反转 truth 或替换 objects（写 `meta.neg_sample=true`）。

### Task_28_Failed_Planning_Flaw_Pointing（规划缺陷定位：单一扰动）

- **字段（JSONPath）**：
  - `high_level_goal`
  - `steps[*].step_goal`（用于构造 gold plan）
  - 可选：`steps[*].causal_chain.causal_precondition_on_*` / `causal_effect_on_*`（用于构造“依赖违反”）
- **证据形态**：`video_prefix`（到 prefix_end_step）
- **输出约束**：固定格式（便于评分）：`FlawStep=<int>; FlawType=<type>; Reason=<one sentence>`
- **样本构造（v6 推荐）**：
  - 取 gold 未来窗口 `K∈[3,6]` 的 step_goals；
  - **只修改其中一个 step**，把它替换为一个“错误 sub-plan”（或不合理动作），形成 `bad_plan`；
  - `FlawType` 可取：`tool_mismatch | order_violation | precondition_missing | hallucinated_object | goal_mismatch`。

### Task_29_Next_K_Steps_Reordering_From_Prefix（未来 K 步重排）

- **字段（JSONPath）**：`steps[i+1:i+K].step_goal`（gold 顺序）
- **证据形态**：`video_prefix`
- **输出约束**：输出 K 条 step_goal 的正确顺序（建议 1..K 编号）。

### Task_30_Middle_Steps_Infill_From_Head_Tail（头尾证据 → 中间步骤补全）

- **字段（JSONPath）**：`high_level_goal` + `steps[*].step_goal`（middle steps 标签）
- **证据形态**：`images_uniform_scene`（head-tail 拼接）
- **输出约束**：输出中间 step_goal 序列（建议 1..M 编号）。

---

### Task_31_Keyframe_to_StepGoal_Matching_MC（关键帧对应 step_goal 四选一；强监督）

- **字段（JSONPath）**：
  - 正确项：`steps[s].step_goal`
  - 干扰项：同 item 其他 step 的 `step_goal`
- **证据形态**：`keyframe_single`（优先 `steps[s].critical_frames[0]`）
- **输出约束**：只输出 `A/B/C/D`

### Task_32_Init_vs_Complete_Keyframe_Order（同一步两关键帧：起始/完成判别）

- **字段来源**：同一步 `critical_frames[0]` 与 `critical_frames[1]` 的两张图（标签由生成器记录，**不暴露 frame_index**）
- **证据形态**：`keyframe_single`（两张图）
- **输出约束**：二分类 `A_is_initiation` 或 `B_is_initiation`

### Task_33_Hotspot_AffordanceType_MCQ（热点 affordance_type 四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.affordance_type`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_34_Hotspot_Mechanism_MCQ（热点机制四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`
- **建议**：对 mechanism 做轻量标准化（短句），否则 label 噪声大。

### Task_35_Action_Phrase_MCQ（`causal_chain.action` 四选一）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.action`（或 step-level `steps[i].causal_chain.action`）
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_36_Patient_Identification_MCQ（`patient` 识别：四选一/多选）

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.patient`
- **证据形态**：`keyframe_single`
- **输出约束**：`A/B/C/D`（优先单选，patient 明显为单实体时）

### Task_37_Step_vs_Keyframe_CausalChain_Consistency（步级↔关键帧因果链一致性判别）

- **字段来源**：
  - 正例：`steps[i].causal_chain` 与 `steps[i].critical_frames[j].causal_chain`
  - 负例：把关键帧 causal_chain 替换为其它 step 的 causal_chain（或只替换 patient/action）
- **证据形态**：`keyframe_single`
- **输出约束**：三分类 `consistent / inconsistent / not directly observable`

### Task_38_Spatial_Postcondition_Check（空间后置状态核验：三态/YesNo）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据形态**：`keyframe_single`（step-end 关键帧）
- **输出约束**：
  - 默认三态：`supported / contradicted / not observable`
  - 若想严格 Yes/No：只采样“可观测” effect，并过滤掉 `not observable` 场景。

### Task_39_Affordance_Postcondition_Check（可供性后置状态核验：三态/YesNo/MCQ）

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types`
- **证据形态**：`keyframe_single`（step-end 关键帧）
- **输出约束**：
  - 默认三态：`supported / contradicted / not observable`
  - 客观题变体：给定 object，四选一选择正确 affordance_type

### Task_40_Counterfactual_Outcome_MCQ（反事实结果四选一；客观化 Task_14）

- **字段（JSONPath）**：`steps[i].counterfactual_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据形态**：`keyframe_single`
- **输出约束**：只输出 `A/B/C/D`

### Task_41_Recovery_Strategy_MCQ（恢复策略四选一；客观化 Task_15）

- **字段（JSONPath）**：`steps[i].failure_reflecting.recovery_strategy`
- **证据形态**：`keyframe_single` 或 `video_prefix`
- **输出约束**：只输出 `A/B/C/D`

### Task_42_Prefix_Completed_Steps_MultiSelect（前缀已完成步骤多选；替代 Task_25）

- **字段（JSONPath）**：gold 为 `steps[0..i].step_goal`（不要塞进 prompt）
- **证据形态**：`video_prefix`（到 step i 尾）
- **输出约束**：
  - 变体 A（推荐）：输出最大已完成 `step_id`（客观、好评测）
  - 变体 B：多选（输出一组 step_id 或 step_goal；评分需集合匹配）

### Task_43_Stage2_Temporal_Localization_Check（可选：基于 Stage2 的客观时间定位）

仅当 item 内存在可读的 Stage2 产物时启用。

- **字段来源**：Stage2 预测的 `start_frame_index/end_frame_index`
- **证据形态**：`images_uniform_scene`（全局抽帧）
- **输出约束**：输出 `start_frame_index/end_frame_index`（整数）

### Task_44_Next_Step_After_Recovery（失败驱动重规划：恢复后下一步选择）

- **字段来源**：
  - 失败描述：`steps[i].failure_reflecting.reason`
  - 恢复策略（可给定或先由 Task_41 选择）：`steps[i].failure_reflecting.recovery_strategy`
  - 下一步标签：`steps[i+1].step_goal`（或从 `steps[i+1:i+K]` 中选最合理的下一步）
- **证据形态**：`video_prefix`（到 step i 尾）或 `keyframe_single`
- **输出约束**：`A/B/C/D`（在候选 next step_goal 中选择）

---

## 9. 任务卡片（逐任务：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：构造 `meta.fields` 与 label 的唯一允许来源
- **多模态证据来源**：必须严格从指定路径取图/取片段（路径不进入模型输入）
- **样本构造规则**：如何在 step/frame/segment 上取样，如何造负样本
- **QA 范例**：示例字段参考 `causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json` 的写法（可能与实际 item 不同，但格式固定可复用）

统一约定：

- `ITEM_DIR = causal_spafa_plan_dataset_long/P01_01_part1`
- `SOURCE_JSON = <ITEM_DIR>/causal_plan_with_keyframes.json`

### Task_01_Macro_Anchor_Extraction

- **字段（JSONPath）**：同 `## 5`
- **证据来源（严格优先级）**：
  1) `images_uniform_scene`：`<ITEM_DIR>/sampled_frames/sample_*.jpg`（等距取 4–8 张）
  2) 若无 `sampled_frames/`：用“每步最早关键帧集合”代理，仍采样到 4–8 张
- **样本构造规则**：每个 item 1 条；对对象名可做去重与 snake_case 规范化。
- **meta.fields（建议最小集）**：`key_objects_for_planning`
- **范例**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... (8 images)
fields.key_objects_for_planning ~= ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet"]
Q: Which stable objects are the task-relevant anchors for planning in this scene?
A: The task-relevant anchors include the light switch, refrigerator, vegetables such as the cucumber and carrot, the cutting board and knife, and the sink and faucet used for washing.
```

### Task_02_Transient_Geometric_Verification

- **字段（JSONPath）**：`steps[i].critical_frames[j].causal_chain.causal_precondition_on_spatial[k].relation/objects/truth`
- **证据来源**：`keyframe_single`：在 step 目录内解析 `frame_###_ts_*.jpg`
- **样本构造规则**：每关键帧最多抽 1–2 条 relation；只做自然语言复述。
- **meta.fields**：`relation`, `objects`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.relation = "contacting"
fields.objects = ["hand","light_switch"]
Q: What is the precise spatial relationship between hand and light_switch in this frame?
A: In this frame, the hand is contacting the light_switch.
```

### Task_03_Micro_Affordance_Visual_Semantics

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

### Task_04_Entity_Role_Identification

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

### Task_05_State_Evolution_Description

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

### Task_06_Holistic_Causal_Chain_Analysis

- **字段（JSONPath）**：同 `## 5`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每关键帧 0–1 条；建议输出两段（setup → mechanism/effects）。
- **meta.fields**：`agent`, `action`, `patient`, `preconditions`, `effects`, `mechanism`
- **范例**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg
fields.agent = "hands"
fields.action = "apply downward cutting force"
fields.patient = "cucumber"
Q: Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects.
A: The cucumber is stabilized on the cutting board while the knife edge contacts it, creating a controlled cutting setup. The sharp blade concentrates force along a thin edge, exceeding the cucumber’s shear strength. As a result, the cucumber separates and a new slice is produced.
```

### Task_08_Strategic_Rationale_Justification

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

### Task_10_Step_Execution_Statement

- **字段（JSONPath）**：`steps[i].step_goal`
- **证据来源（严格优先级）**：
  1) `video_clip`：`<ITEM_DIR>/last_frame_segments/segment_step{i-1:02d}_to_step{i:02d}.mp4`（i>1）
  2) step 1：`<ITEM_DIR>/last_frame_segments/segment_start_to_step01.mp4`（若有）
  3) fallback：step i 最早关键帧（`keyframe_single`）
- **样本构造规则**：允许 2–5 个微动作；看不清则标注 `not clearly observable`。
- **meta.fields**：`step_goal`, `segment_label`
- **范例**：

```text
Video clip: <ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4
fields.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
Q: Describe the observable micro-actions required to execute this step.
A: The person moves the vegetables to the sink, turns on the faucet, rinses and rubs the surfaces under running water, then transfers the vegetables to the countertop.
```

### Task_11_Expected_Physical_Effects

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial/causal_effect_on_affordance`
- **证据来源**：`keyframe_single`（该 step 最后一张关键帧 `critical_frames[-1]`）
- **样本构造规则**：
  - 每个 step 0–1 条（建议只取“可观测/可部分观测”的 effect）；
  - 对明显不可从单帧判断的 effect（如 clean/ready），答案必须显式写 `not directly observable`，避免硬判导致噪声。
- **meta.fields**：`expected_effects`, `step_goal`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
fields.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
Q: Upon completion of this step, what physical effects should be expected for the environment and objects?
A: The vegetables should end up on the countertop if their placement is visible. Whether they are fully clean can be not directly observable from a single frame.
```

### Task_12_Inter_Step_Dependency_Analysis

- **字段（JSONPath）**：同 `## 5`
- **证据来源**：`keyframe_single`（建议 step i 尾关键帧）
- **样本构造规则**：仅当 effect↔precondition 有词项重合/包含时生成。
- **meta.fields**：`step_n_goal`, `step_n_effects`, `step_next_goal`, `step_next_preconditions`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_n_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.step_next_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: How does the outcome of the previous step satisfy the preconditions for the next step?
A: Turning on the light makes the workspace visible and safe, enabling the person to locate and access the refrigerator to retrieve the vegetables.
```

### Task_13_Next_Action_Prediction

- **字段（JSONPath）**：
  - 输入侧：`steps[i].step_goal`（可选）
  - 标签侧：`steps[i+1].step_goal`
- **证据来源**：
  - 计划版本：`keyframe_single`（step i 尾关键帧）
  - 视觉版本：若已有 prefix，优先用 `Task_24`（本任务不重复造轮子）
- **样本构造规则**：严格只输出下一步 `step_goal`（完全匹配，不输出多余文本）。
- **meta.fields**：`current_step_goal`, `next_step_goal`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.current_step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
label.next_step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: What is the next planned action?
A: Retrieve a carrot and a cucumber from the refrigerator.
```

### Task_14_Counterfactual_Prediction

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

### Task_15_Failure_Recovery_Protocol

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

### Task_16_Physical_Feasibility_Verification

- **字段（JSONPath）**：同 `## 5`
- **证据来源**：`keyframe_single`
- **样本构造规则**：
  - 正样本：使用原字段；
  - 负样本：替换 patient/tool 或反转关键空间关系（写 `meta.neg_sample=true`）。
- **meta.fields**：`preconditions_spatial`, `preconditions_affordance`, `step_goal`, `label`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.label = "feasible"
Q: Is this step physically feasible now based on the visible spatial and affordance preconditions?
A: feasible
```

### Task_18_Precondition_Check_or_MCQ

- **字段（JSONPath）**：同 `## 5`
- **证据来源（严格优先级）**：
  1) `video_prefix`：`<ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step{i-1:02d}_last.mp4`（i>1）
  2) i=1 时：用 `images_uniform_scene` 的前若干帧代理
- **样本构造规则**（推荐 Yes/No 单条核验）：
  - 从真实 preconditions 抽 1 条作为正例；
  - 通过替换 objects/affordance_type 构造 1 条负例；
  - 输出三态 `Yes/No/Not observable`。
- **meta.fields**：`step_goal`, `precondition`, `label`, `neg_sample`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4
fields.step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
fields.precondition = "hand is near refrigerator_handle"
label = "Not observable"
Q: Based on the provided prefix, is the precondition \"hand is near refrigerator_handle\" satisfied?
A: Not observable
```

### Task_22_Plan_Execution_Alignment

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

### Task_23_Goal_Recognition_From_Full_Video

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

### Task_24_Next_Step_Goal_Prediction_From_Prefix

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

### Task_26_Temporal_Order_Check_AB

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

### Task_27_Visual_Spatial_Relation_Check

- **字段（JSONPath）**：`relation/objects/truth`
- **证据来源**：`keyframe_single`
- **样本构造规则**：可造弱负样本（反转 truth 或替换 objects），必须写 `meta.neg_sample=true`。
- **meta.fields**：`relation`, `objects`, `truth`, `neg_sample`
- **范例**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
relation = "contacting"
objects = ["hand","light_switch"]
truth = true
Q: Is it true that hand is contacting light_switch in this image?
A: true
```

### Task_28_Failed_Planning_Flaw_Pointing

- **字段（JSONPath）**：同 `## 5`
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

### Task_29_Next_K_Steps_Reordering_From_Prefix

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

### Task_30_Middle_Steps_Infill_From_Head_Tail

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

### Task_31_Keyframe_to_StepGoal_Matching_MC

- **字段（JSONPath）**：同 `## 5`
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

### Task_32_Init_vs_Complete_Keyframe_Order

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

### Task_33_Hotspot_AffordanceType_MCQ

- **字段（JSONPath）**：`interaction.hotspot.affordance_type`
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

### Task_34_Hotspot_Mechanism_MCQ

- **字段（JSONPath）**：`interaction.hotspot.mechanism`
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

### Task_35_Action_Phrase_MCQ

- **字段（JSONPath）**：`causal_chain.action`
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

### Task_36_Patient_Identification_MCQ

- **字段（JSONPath）**：`causal_chain.patient`
- **证据来源**：`keyframe_single`
- **样本构造规则**：候选来自同 item 的对象集合（可由 Task_01 去重得到）。
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

### Task_37_Step_vs_Keyframe_CausalChain_Consistency

- **字段来源**：同 `## 5`
- **证据来源**：`keyframe_single`
- **样本构造规则**：
  - 正例：用同 step 的 step-level chain + keyframe-level chain；
  - 负例：把 keyframe-level chain 换成其它 step 的 chain（写 `meta.neg_sample=true`）。
- **meta.fields**：`label`, `neg_sample`
- **范例**：

```text
Image: <ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg
Given step-level patient = "refrigerator" and keyframe-level patient = "refrigerator"
Q: Are the step-level causal_chain and the keyframe causal_chain consistent?
A: consistent
```

### Task_38_Spatial_Postcondition_Check

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_spatial[*]`
- **证据来源**：`keyframe_single`（step-end 关键帧）
- **样本构造规则**：
  - 从 effect 列表抽 1 条；
  - 输出三态 `supported/contradicted/not observable`；
  - 可造负样本：替换 objects 或反转 truth（写 `meta.neg_sample=true`）。
- **meta.fields**：`effect_relation`, `effect_objects`, `effect_truth`, `label`
- **范例**：

```text
Image (step-end keyframe): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
effect = {"relation":"on_top_of","objects":["cucumber","countertop"],"truth":true}
Q: Is the spatial postcondition \"cucumber on_top_of countertop\" supported by this image?
A: supported
```

### Task_39_Affordance_Postcondition_Check

- **字段（JSONPath）**：`steps[i].causal_chain.causal_effect_on_affordance[*]`
- **证据来源**：`keyframe_single`（step-end 关键帧）
- **样本构造规则**：
  - 抽 1 条 object+affordance_type；
  - 输出三态；
  - 或做 object→affordance_type 的四选一。
- **meta.fields**：`object_name`, `affordance_types`, `label`
- **范例（四选一变体）**：

```text
Image (step-end keyframe): <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
object_name = "light_switch"
Q: After this step, which affordance/state is most plausibly true for light_switch?
  A) switched_on
  B) submerged_in_water
  C) sliced_into_pieces
  D) placed_inside_refrigerator
A (label): A
```

### Task_40_Counterfactual_Outcome_MCQ

- **字段（JSONPath）**：同 `Task_14`
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

### Task_41_Recovery_Strategy_MCQ

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

### Task_42_Prefix_Completed_Steps_MultiSelect

- **字段（JSONPath）**：`steps[0..i].step_goal`（gold）
- **证据来源**：`video_prefix`
- **样本构造规则（推荐最大 step_id 变体）**：
  - 取 prefix_end_step=i；
  - label 为 `i`（或 `i+1`，看你的定义：已完成到第几步）。
- **meta.fields**：`prefix_end_step`, `label`
- **范例**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step03_last.mp4
Q: Up to which step_id has the plan been completed in this prefix?
A (label): 3
```

### Task_43_Stage2_Temporal_Localization_Check（可选）

- **字段来源**：Stage2 产物中的 `start_frame_index/end_frame_index`
- **证据来源**：`images_uniform_scene`（`<ITEM_DIR>/sampled_frames/sample_*.jpg`）
- **样本构造规则**：每 step 1 条；输出两个整数。
- **meta.fields**：`step_id`, `start_frame_index`, `end_frame_index`
- **范例**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... sample_050_ts_??.??s.jpg
Q: Predict the start_frame_index and end_frame_index for step_id=4.
A (label): {"start_frame_index": 18, "end_frame_index": 26}
```

### Task_44_Next_Step_After_Recovery

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

## 10. 当前无约束 QA 示例（逐任务，完全自由文本）

本节给出“不引入受控标签输出（例如不要求 Yes/No/ABCD）”时，每个任务的自由文本问答示例。

重要说明：

- 示例中的图片/视频路径只作为“证据引用写法”，实际训练时不要把路径字符串喂给模型。

统一定义：

- `SOURCE_JSON = <ITEM_DIR>/causal_plan_with_keyframes.json`
- `HIGH_LEVEL_GOAL = "Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board."`

### Task_01_Macro_Anchor_Extraction（自由文本示例）

- Evidence（`images_uniform_scene`）：`<ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ...`
- Q: Which stable objects are the task-relevant anchors for planning in this scene?
- A: The anchors include the light switch, refrigerator, vegetables, the cutting board and knife, and the sink and faucet.

### Task_02_Transient_Geometric_Verification（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: What is the spatial relationship between the hand and the light_switch in this frame?
- A: The hand is contacting the light_switch.

### Task_03_Micro_Affordance_Visual_Semantics（自由文本示例）

- Evidence（`keyframe_single`）：同上
- Q: Which region is the functional hotspot, and what affordance and mechanism does it support?
- A: The rocker surface is the hotspot; it affords pressing and transfers force to an internal toggle mechanism to complete the circuit.

### Task_04_Entity_Role_Identification（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg`
- Q: Which items are tools and which are materials in this step?
- A: The refrigerator is the tool/container being accessed, while the cucumber and carrot are the materials being retrieved.

### Task_05_State_Evolution_Description（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg`
- Q: What action is occurring and what immediate state change does it cause?
- A: The person rubs the cucumber under running water, which cleans its surface.

### Task_06_Holistic_Causal_Chain_Analysis（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg`
- Q: Explain the physical causal chain in this keyframe.
- A: The cucumber is stabilized while the knife edge contacts it. Force is concentrated along the blade edge, causing the cucumber to shear and separate into a slice.

### Task_08_Strategic_Rationale_Justification（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Why is turning on the light necessary for the overall goal?
- A: It improves visibility, enabling safe and accurate manipulation in later steps.

### Task_10_Step_Execution_Statement（自由文本示例）

- Evidence（`video_clip`）：`<ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4`
- Q: Describe the observable micro-actions for this step.
- A: The person moves vegetables to the sink, turns on water, rinses and rubs them, then transfers them to the countertop.

### Task_11_Expected_Physical_Effects（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg`
- Q: What physical effects should be expected upon completion of this step?
- A: The vegetables should end up on the countertop if their placement is visible, while claims like “fully clean” can be not directly observable from a single image.

### Task_12_Inter_Step_Dependency_Analysis（自由文本示例）

- Evidence（`keyframe_single`）：step i 尾关键帧
- Q: How does the outcome of step i enable step i+1?
- A: Step i produces the spatial/affordance state required by step i+1, such as making the workspace visible or making an object available for manipulation.

### Task_13_Next_Action_Prediction（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: What is the next planned action after turning on the light?
- A: Retrieve a carrot and a cucumber from the refrigerator.

### Task_14_Counterfactual_Prediction（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg`
- Q: What if the cutting board was slippery on the countertop?
- A: The board could slide under cutting forces, making the action unstable; increasing friction with a damp cloth would help.

### Task_15_Failure_Recovery_Protocol（自由文本示例）

- Evidence（`keyframe_single`）：同上
- Q: If the cucumber rolls during cutting, how can the agent recover?
- A: Create a flat base by cutting one side, then stabilize the cucumber with the non-cutting hand.

### Task_16_Physical_Feasibility_Verification（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Is the step physically feasible now? Explain briefly.
- A: It appears feasible because the hand is within reach of the switch and the switch affords pressing to toggle the light.

### Task_18_Precondition_Check_or_MCQ（自由文本示例）

- Evidence（`video_prefix`）：`<ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4`
- Q: Which preconditions for retrieving vegetables are clearly satisfied in the prefix, and which are not observable?
- A: Being in the kitchen may be observable, but whether the refrigerator contains the vegetables is not directly observable unless the interior is shown.

### Task_22_Plan_Execution_Alignment（自由文本示例）

- Evidence（`video_clip`）：`<ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4`
- Q: Does this clip match the step goal of washing the vegetables and placing them on the countertop?
- A: It mostly matches if the clip shows the sink washing action and then transferring vegetables toward the countertop.

### Task_23_Goal_Recognition_From_Full_Video（自由文本示例）

- Evidence（`video_prefix`）：完整前缀/全视频
- Q: What is the high-level goal of this full video?
- A: Prepare for cooking by turning on the light, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board.

### Task_24_Next_Step_Goal_Prediction_From_Prefix（自由文本示例）

- Evidence（`video_prefix`）：前缀到 step i 尾
- Q: What is the next step goal?
- A: Gather a cutting board and a knife and place them on the countertop.

### Task_26_Temporal_Order_Check_AB（自由文本示例）

- Evidence（`keyframe_single` 两张图）：A/B
- Q: Which event happens earlier in the video, A or B?
- A: Event A happens earlier.

### Task_27_Visual_Spatial_Relation_Check（自由文本示例）

- Evidence（`keyframe_single`）：`<ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Is the hand contacting the light switch?
- A: Yes, the hand appears to be in contact with the switch.

### Task_28_Failed_Planning_Flaw_Pointing（自由文本示例）

- Evidence（`video_prefix`）：到某一步
- Q: Which step in the proposed plan is flawed and why?
- A: The slicing step is flawed because it assumes the cucumber and cutting tools are already available, which is not supported by the prefix.

### Task_29_Next_K_Steps_Reordering_From_Prefix（自由文本示例）

- Evidence（`video_prefix`）
- Q: Reorder the shuffled candidate steps into the most plausible next-step sequence.
- A: Retrieve vegetables → gather tools → wash vegetables.

### Task_30_Middle_Steps_Infill_From_Head_Tail（自由文本示例）

- Evidence（`images_uniform_scene` head-tail）
- Q: Infer the missing middle steps.
- A: Retrieve vegetables, gather tools, and wash them before chopping.

### Task_31_Keyframe_to_StepGoal_Matching_MC（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What step goal does this image most likely correspond to?
- A: It most likely corresponds to gathering a cutting board and a knife and placing them on the countertop.

### Task_32_Init_vs_Complete_Keyframe_Order（自由文本示例）

- Evidence（两张关键帧）
- Q: Which image looks like the initiation of the step?
- A: Image A shows the initiation because it captures the start of the interaction, while B reflects completion.

### Task_33_Hotspot_AffordanceType_MCQ（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What is the affordance type of the interaction hotspot?
- A: It is a pressable surface.

### Task_34_Hotspot_Mechanism_MCQ（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What mechanism makes this hotspot functional?
- A: Pressing transfers force to an internal toggle mechanism that completes the circuit.

### Task_35_Action_Phrase_MCQ（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What action is the agent performing?
- A: The agent is applying downward pressure to press the switch.

### Task_36_Patient_Identification_MCQ（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What is the primary object being acted on?
- A: The light switch.

### Task_37_Step_vs_Keyframe_CausalChain_Consistency（自由文本示例）

- Evidence（`keyframe_single`）
- Q: Are the step-level and keyframe-level causal descriptions consistent?
- A: They are consistent if they refer to the same patient and compatible actions/effects.

### Task_38_Spatial_Postcondition_Check（自由文本示例）

- Evidence（step-end keyframe）
- Q: What spatial postconditions are supported by this image?
- A: The vegetable appears placed on the countertop, which supports the on_top_of relation.

### Task_39_Affordance_Postcondition_Check（自由文本示例）

- Evidence（step-end keyframe）
- Q: What affordance/state changes are supported after this step?
- A: The switch appears turned on, implying a switched_on state.

### Task_40_Counterfactual_Outcome_MCQ（自由文本示例）

- Evidence（`keyframe_single`）
- Q: What would likely happen under the counterfactual condition?
- A: The cutting board may slide, making the action unstable unless friction is increased.

### Task_41_Recovery_Strategy_MCQ（自由文本示例）

- Evidence（`keyframe_single` 或 prefix）
- Q: What recovery strategy best addresses the failure?
- A: Stabilize the object or increase friction, depending on the failure mode.

### Task_42_Prefix_Completed_Steps_MultiSelect（自由文本示例）

- Evidence（`video_prefix`）
- Q: What has been completed so far?
- A: The first few setup steps have been completed up to the current prefix endpoint.

### Task_43_Stage2_Temporal_Localization_Check（自由文本示例）

- Evidence（`images_uniform_scene`）
- Q: Where does step 4 start and end in the sampled frame indices?
- A: It begins around when washing starts and ends after placing vegetables on the countertop.

### Task_44_Next_Step_After_Recovery（自由文本示例）

- Evidence（`video_prefix`）
- Q: After recovering from the failure, what should happen next?
- A: The agent should resume the intended step that was previously unstable, such as continuing slicing after stabilizing the cutting board.
