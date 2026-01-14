# ECCV/chat 文档核验（最终版）& Three-Stage TaskList（以 `ECCV/three_stage/prompts.py` 为唯一真值）

最后更新：2026-01-14  
适用范围：`ECCV/three_stage/` 三阶段数据生成管线 + `ECCV/chat/` 下所有任务/评测/笔记文档（用于你即将进行的大规模批量生成前的最终核验）。

> 底线原则（请务必严格遵守）
>
> 1) **Prompt 不改**：`ECCV/three_stage/prompts.py` 的内容不再做任何改动；所有一致性问题只能在“代码/校验器/任务构造”侧解决。  
> 2) **Schema 唯一真值**：三阶段最终真值字段以 `ECCV/three_stage/prompts.py`（Stage1/2/3）为准，并由 `ECCV/three_stage/common.py:normalize_*` + `ECCV/three_stage/validate_three_stage_output.py` 强制执行。  
> 3) **旧口径文档只能当理念参考**：`ECCV/chat/*.md` 中凡是与真值 schema 不一致的字段名/结构，一律不得直接用于任务生成与数据清洗（必须先做字段迁移或改写任务构造逻辑）。

---

## 0. 一句话结论（避免“混产/错字段”灾难）

- 你接下来要跑的大规模数据生成，**真值字段以 `ECCV/three_stage/prompts.py` 为准**；任何仍在使用旧字段（如 `affordance_hotspot/tool_and_material_usage/failure_handling/action_description/state_change_description/keyframe_image_path`）的任务构造脚本或文档，都必须先迁移到新 schema，否则会出现**标注字段错位/任务取值为空/质量不可控**。

---

## 1. Three-Stage 真值输出 Schema（来自 `ECCV/three_stage/prompts.py`）

### 1.1 Stage1：`stage1/draft_plan.json`（只允许 step-level；严禁 keyframe 字段）

文件：`<video_out>/stage1/draft_plan.json`

顶层：
- `high_level_goal: str`
- `steps: List[StepDraft]`

`StepDraft`（每步）：
- `step_id: int`（脚本会归一化为 `1..N`）
- `step_goal: str`
- `rationale: str`
- `causal_chain: CausalChain`（见下）
- `counterfactual_challenge_question: str`
- `expected_challenge_outcome: str`
- `failure_reflecting: {reason: str, recovery_strategy: str}`

Stage1 **禁止项**（任何层级任何位置都不允许出现）：
- `critical_frames`, `frame_index`, `interaction`, `keyframe_image_path`
- 文本中禁止出现 “Frame 12 / Image 12 / t=3.2s …” 这类帧引用

### 1.2 Stage2：定位边界（模型只输出 index；脚本再切片）

模型输出（落盘 `localization_raw.json`，schema 由 prompt 强制）：
- 顶层只允许：`{"steps":[...]}`
- 每个 entry 只允许：`{step_id, start_frame_index, end_frame_index}`

脚本输出（落盘 `step_segments.json`）：
- 在上述 index 基础上补齐 `start_sec/end_sec/clip_relpath/...` 并生成 `stage2/step_clips/*.mp4`

关键语义（必须牢记，后续所有校验/任务都依赖它）：
- Stage2 的 `end_frame_index` 是 **exclusive 边界**
- Stage2 的 index 作用域：**full video** 的 `stage1/frame_manifest.json`（1-based）

### 1.3 Stage3：最终产物 `causal_plan_with_keyframes.json`（step-level + 2 个 keyframes）

最终文件：`<video_out>/causal_plan_with_keyframes.json`

顶层：
- `high_level_goal: str`
- `steps: List[StepFinal]`

`StepFinal`（每步）：
- `step_id: int`
- `step_goal: str`（必须与 Stage1 draft 完全一致；不允许改）
- `rationale: str`
- `causal_chain: CausalChain`
- `counterfactual_challenge_question: str`
- `expected_challenge_outcome: str`
- `failure_reflecting: {reason: str, recovery_strategy: str}`
- `critical_frames: List[CriticalFrame]`（**长度必须为 2**，且 `frame_index` 递增）

`CriticalFrame`（每个关键帧）：
- `frame_index: int`（1-based；仅对**当前 step clip 的帧池**有效）
- `action_state_change_description: str`
- `causal_chain: CausalChain`
- `interaction: {tools: List[str], materials: List[str], hotspot: Hotspot}`

`Hotspot`：
- `description: str`
- `affordance_type: str`（建议 `snake_case`）
- `mechanism: str`

`CausalChain`（Stage1/Stage3 共用形状）：
- `agent: str`
- `action: str`
- `patient: str`
- `causal_precondition_on_spatial: List[{relation: str, objects: List[str], truth: bool}]`（非空）
- `causal_precondition_on_affordance: List[{object_name: str, affordance_types: List[str], reasons: str}]`（非空）
- `causal_effect_on_spatial: List[{relation: str, objects: List[str], truth: bool}]`（非空）
- `causal_effect_on_affordance: List[{object_name: str, affordance_types: List[str], reasons: str}]`（非空）

Stage3 **禁止项/硬约束**（`normalize_stage3_step_output()` 会强制）：
- JSON 不允许出现任何额外 key（包括 `keyframe_image_path`）
- 文本不得包含 “Frame/Image + 数字” 这类帧引用
- `critical_frames` 必须恰好 2 个、且时间推进（`frame_index` 与 timestamps 单调）

---

## 2. 关键帧图片如何定位（非常关键：新 schema 不写 `keyframe_image_path`）

Stage3 会把关键帧 JPEG 复制到每个 step 目录根下：
- `<video_out>/{step_id:02d}_<slug>/frame_{frame_index:03d}_ts_{timestamp:.2f}s.jpg`

其中：
- `frame_index` 来自 `StepFinal.critical_frames[*].frame_index`
- `timestamp` 来自 `<video_out>/{step_id:02d}_<slug>/frame_manifest.json` 里同 index 的 `timestamp_sec`
- **timestamp 使用原视频时间轴（global timestamp）**（Stage3 写 manifest 时已加上 `clip_start_sec` 偏移）

因此任务构造/清洗时，严禁依赖 `keyframe_image_path`（旧版本曾注入绝对路径，跨机器不可复现）。

---

## 3. `ECCV/chat/` 全文件核验（逐文件给出“是否可直接用”结论）

> 判定口径：是否以 **新 schema**（上文 §1）为准；若混用旧字段，则必须先迁移/重写任务构造逻辑。

### 3.1 文件清单与结论

- `ECCV/chat/final_master_summary.md`：**旧字段为主**（大量 `affordance_hotspot/.../failure_handling/...`），只能当历史总结/思路参考。
- `ECCV/chat/mani_longvideo_tasks_plan_final_alignment_and_qa_strategy.md`：**旧字段**；若要复用任务框架，需要按 §4 的映射改写取值字段。
- `ECCV/chat/mani_longvideo_tasks_plan_final_causal_planning_failure_reflecting_audit.md`：**旧字段**（尽管标题包含 “failure_reflecting”）；建议只保留“失败闭环任务设计思想”，字段取值需迁移。
- `ECCV/chat/mani_longvideo_tasks_plan_final_master_summary.md`：**旧字段**；同上。
- `ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`：**旧字段**；可复用“strict scoring 题型模板”，但字段必须改为新 schema。
- `ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`：**旧字段**；同上。
- `ECCV/chat/useful.md`：**旧字段笔记**；建议仅作备忘，不作为自动任务生成依据。
- `ECCV/chat/mani_longvideo_tasks_plan_final.md`：**混合口径**（新字段与旧字段同时出现）；其中关于 `counterfactual_challenge_question / failure_reflecting / interaction / action_state_change_description / causal_precondition_on_*` 的内容与当前 prompts 基本一致，但文内仍夹杂旧字段示例，使用时必须以 §1 真值为准。

---

## 4. 旧字段 → 新字段 的迁移映射（只用于“迁移任务”，不改 prompt）

旧字段（多出现在上述旧文档/旧任务描述中）与新字段的关系如下：

| 旧字段（旧口径） | 新字段（真值 schema） | 迁移说明 |
|---|---|---|
| `failure_handling.reason` | `failure_reflecting.reason` | 字段名替换；语义一致 |
| `failure_handling.recovery_strategy` | `failure_reflecting.recovery_strategy` | 同上 |
| `causal_challenge_question` | `counterfactual_challenge_question` | 新字段更强调 counterfactual；任务题面可继续用“what if …”形式 |
| `affordance_hotspot.{description,affordance_type,mechanism}` | `interaction.hotspot.{description,affordance_type,mechanism}` | hotspot 入口从 keyframe-level 的 `affordance_hotspot` 改为 `interaction.hotspot` |
| `tool_and_material_usage.tools/materials` | `interaction.tools/materials` | 新 schema 在 **keyframe-level** 给 tools/materials；若旧任务需要 step-level，可对 2 个 keyframe 做 union 作为弱标签 |
| `action_description` + `state_change_description` | `action_state_change_description` | 可用 `"action -> state_change"` 拼接做迁移（用于题面构造） |
| `spatial_preconditions` | `causal_chain.causal_precondition_on_spatial` | 新 schema 用 causal_chain 统一承载 preconditions |
| `affordance_preconditions` | `causal_chain.causal_precondition_on_affordance` | 同上 |
| `spatial_postconditions_detail` | `causal_chain.causal_effect_on_spatial` | 旧的 step-level 后置空间关系，对应新 causal effect |
| `affordance_postconditions_detail` | `causal_chain.causal_effect_on_affordance` | 同上 |
| `keyframe_image_path` | （不再存在） | 新 schema 不写路径；请用 §2 的文件系统解析 |
| `predicted_next_actions` | （不再存在） | 新 schema 未提供；若旧任务依赖它，需要删除该任务或改成基于 `step_goal`/`causal_chain` 的可验证任务 |

---

## 5. 最终推荐 TaskList（以新 schema 可直接落地；优先 strict scoring）

下面给出一版“**可直接用新 schema 生成数据**”的 TaskList。按证据形态分层，默认优先 **strict scoring / 客观题**（便于规模化、可复现、可自动验收）。

### 5.1 Tier-0（强烈建议：先跑这些，确保字段/证据链可用）

1) **Keyframe Image Existence Check（自动验收）**  
   - 证据：`<step_folder>/frame_###_ts_*.jpg`  
   - 标签来源：`steps[i].critical_frames[*].frame_index` + `<step_folder>/frame_manifest.json`  
   - 目标：保证“关键帧图像可解析”，避免后续任务全空。

2) **Step ID/Goal Alignment Check（自动验收）**  
   - 标签来源：`stage1/draft_plan.json` vs `causal_plan_with_keyframes.json`  
   - 目标：保证 Stage3 没改 `step_goal`（否则任务会串步）。

### 5.2 Tier-1（images/keyframe 单图可评分任务：建议作为第一批主力）

| 任务名（建议） | 证据 | 真值字段（JSONPath） | 题型建议 |
|---|---|---|---|
| **SP-Pre Spatial Precondition Check** | `keyframe_single`（initiation） | `steps[i].critical_frames[0].causal_chain.causal_precondition_on_spatial[*]` | Yes/No 或 MCQ（给 relation+objects，问是否成立） |
| **SP-Post Spatial Effect Check** | `keyframe_single`（completion） | `steps[i].critical_frames[1].causal_chain.causal_effect_on_spatial[*]` | Yes/No 或 MCQ |
| **AF-Pre Affordance Precondition Check** | `keyframe_single`（initiation） | `steps[i].critical_frames[0].causal_chain.causal_precondition_on_affordance[*]` | Yes/No（问 object 是否具备 affordance） |
| **AF-Post Affordance Effect Check** | `keyframe_single`（completion） | `steps[i].critical_frames[1].causal_chain.causal_effect_on_affordance[*]` | Yes/No |
| **HS Affordance Type MCQ** | `keyframe_single` | `steps[i].critical_frames[j].interaction.hotspot.affordance_type` | 4 选 1（干扰来自同 item/全局词表） |
| **Tool/Material Role MCQ** | `keyframe_single` | `steps[i].critical_frames[j].interaction.tools/materials` | 给候选 x，问 x 属于 tool/material/none |
| **Agent/Patient Role MCQ** | `keyframe_single` | `steps[i].causal_chain.agent/patient` | 给候选 x，问 x 是 agent/patient/none |

### 5.3 Tier-2（step clip 主力任务：更贴近“过程理解”）

证据优先用：`<video_out>/stage2/step_clips/*.mp4`

| 任务名（建议） | 证据 | 真值字段 | 题型建议 |
|---|---|---|---|
| **Step Goal Matching (video)** | `video_clip` | `steps[i].step_goal` | 4 选 1（干扰来自同 video 其他 step_goal） |
| **Action-State Change Presence (video)** | `video_clip` | `steps[i].critical_frames[*].action_state_change_description` | Yes/No（给 statement，问 clip 是否发生） |
| **Causal Triple Presence (video)** | `video_clip` | `steps[i].causal_chain.agent/action/patient` | Yes/No 或 MCQ |

### 5.4 Tier-3（full-video 抽帧任务：辅助全局目标/步骤排序）

证据优先用：`<video_out>/stage1/sampled_frames/`（或根目录 compat `sampled_frames/`）

| 任务名（建议） | 证据 | 真值字段 | 题型建议 |
|---|---|---|---|
| **High-level Goal MCQ** | `images_uniform_scene` | `high_level_goal` | 4 选 1（干扰来自同域其他视频） |
| **Step Order AB** | `keyframe_pair`（两步 completion） | `critical_frames[*]` 的 `ts_XX.XXs`（从文件名/manifest） | A/B 谁先发生（strict scoring） |

---

## 6. 跑批前最后 Checklist（强烈建议照着勾）

- [ ] `python3 -m py_compile` 通过：`ECCV/three_stage/*.py`
- [ ] `python3 ECCV/three_stage/validate_three_stage_output.py --self-test` 通过
- [ ] 小样本 smoke：选 1–3 个视频跑 `python3 ECCV/three_stage/pipeline.py --post-validate ...`
- [ ] 输出目录不混产：不同源视频不要共享同名 `video_id`（pipeline 已做 collision guard，但建议你也检查）
- [ ] 关键帧 jpg 可解析：每步目录存在 2 张 `frame_###_ts_*.jpg`
- [ ] `causal_plan_with_keyframes.json` 中无 `keyframe_image_path`、且 `critical_frames` 长度=2

