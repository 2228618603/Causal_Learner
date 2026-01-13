# mani_longvideo_tasks_plan_final 与 causal_plan_with_keyframes.json 的一致性审视 & QA 数据生成策略

## 摘要（对应本次对话回复）

- **是否符合**：当前 `causal_plan_with_keyframes.json` 在字段层面基本覆盖 `ECCV/mani_longvideo_tasks_plan_final.md` 的 Task_01–Task_30 所需信息；但存在一个关键不一致：主规范多处引用 `affordance_hotspot.mechanism`，而实际产物更常把“机制”写在 `causal_chain.causal_affordance_focus_detail`，建议在生成器里做映射与可追溯记录。
- **四类证据如何选**：瞬时空间关系/热点 → `keyframe_single`；全局场景锚点/高阶目标 → `images_uniform_scene`；动作过程/边界/计划-执行对齐 → `video_clip`；进度/下一步预测/规划题 → `video_prefix`。
- **训练数据格式**：建议统一 ShareGPT JSON(L)（`image`/`video` + `<image>/<video>` tag + `conversations`），在 `meta` 里强制记录 `evidence_type/source/files/options/answer_format/neg_sample` 以便回放与评测。
- **任务形态选择**：把可稳定自动标注的任务做成**客观题**（Yes/No、AB、ABCD、排序、match/mismatch）作为主力训练与回归；把需要解释/总结的任务作为**主观题**，用模型打分（并标记 `weak_supervision` 或用 LLM rubric）。
- **路径与媒体可用性**：`keyframe_image_path` 可能是旧机器的绝对路径，生成器必须做 resolve + glob fallback；`video_clip/prefix` 若没有现成 mp4，要么生成（ffmpeg/三阶段），要么用“完整视频 + start/end 秒”在加载器侧裁剪。
- **下一步建议**：按本文策略把 `causal_plan_with_keyframes.json` 批量转成两套数据（强监督客观题/主观生成题），输出 JSONL 直接喂给多模态大模型 finetune。

本文档面向以下目标：

1) 评估 `ECCV/mani_longvideo_tasks_plan_final.md`（Task_01–Task_30）与当前可产出的 `causal_plan_with_keyframes.json` 是否一致、哪里需要补齐/回退。  
2) 给出一套**可落地**的“从 `causal_plan_with_keyframes.json` → 多模态 QA JSONL”的生成策略，确保最终数据能直接用于多模态大模型（含图像/视频）训练。  
3) 对每个任务，选择四类多模态证据中的**最合适**类型，并给出更推荐的 QA 形态（客观题 vs 主观题）与可自动标注/可模型打分的实现方式。

---

## 1. 两份规范的角色关系（先把边界讲清楚）

- `ECCV/mani_longvideo_tasks_plan_final.md`：**任务定义规范**。它定义了 Task_01~Task_30 的字段来源、证据形态、采样规则、以及 QA 样例（见 `ECCV/mani_longvideo_tasks_plan_final.md:388`）。
- `causal_plan_with_keyframes.json`：**单视频 item 的结构化中间产物**。它提供：
  - 计划层文本（`high_level_goal`、每步 `step_goal/rationale/preconditions/expected_effects/...`）
  - 关键帧层标注（`critical_frames[*]`，含 `keyframe_image_path` + `action/state_change/spatial_preconditions/affordance_preconditions/causal_chain/affordance_hotspot`）
  - 关键帧文件名里通常包含 `ts_XX.XXs`，可作为**全局时间轴**锚点（主规范也强调此点，见 `ECCV/mani_longvideo_tasks_plan_final.md:66`）。

结论（重要）：
- **Task 的文本字段 supervision**：大多数可直接由 JSON 字段生成（“自描述式 pseudo GT”）。
- **真正“多模态证据”**（图片/视频）不完全来自 JSON：JSON 只存路径/时间锚；样本还需要在 item_dir 中存在相应媒体文件（关键帧 jpg、全局 sampled_frames、mp4 clip/prefix 等）。

---

## 2. 一致性检查：当前 JSON 是否覆盖 Task_01–Task_30 所需字段？

### 2.1 JSON 顶层/step/critical_frame 字段是否匹配主规范

主规范里给出的 schema（见 `ECCV/mani_longvideo_tasks_plan_final.md:43`）与当前 `P01_01_part*` 的实际 JSON 基本一致：

- 顶层：`high_level_goal`, `steps`
- step：`step_id/step_goal/rationale/preconditions/expected_effects/predicted_next_actions/tool_and_material_usage/.../critical_frames`
- critical_frame：`frame_index/keyframe_image_path/action_description/state_change_description/spatial_preconditions/affordance_preconditions/causal_chain/affordance_hotspot`

### 2.2 一个明确的不一致点：`affordance_hotspot.mechanism`

主规范的多个任务（例如 Task_03/06/17）会引用 `critical_frames[*].affordance_hotspot.mechanism`（见 `ECCV/mani_longvideo_tasks_plan_final.md:445`、`ECCV/mani_longvideo_tasks_plan_final.md:497`、`ECCV/mani_longvideo_tasks_plan_final.md:701`）。

但当前实际 JSON（示例：`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json`）里：
- `affordance_hotspot` 常见字段为：`description/affordance_type/causal_role`
- 机制类文本通常出现在：`causal_chain.causal_affordance_focus_detail`

推荐的**规范化映射（生成侧做，不要求改 JSON 产物）**：
- `mechanism := affordance_hotspot.mechanism` 若存在  
- 否则 `mechanism := causal_chain.causal_affordance_focus_detail`  
并在 `meta.schema_fallbacks` 里记录 `"mechanism_source": "causal_chain.causal_affordance_focus_detail"`，便于追溯。

### 2.3 路径一致性风险：`keyframe_image_path` 可能是机器相关的绝对路径

实际 JSON 的 `keyframe_image_path` 可能是旧机器的绝对路径（例如 `/e2e-data/...`），但 item_dir 内仍有同名 jpg。

推荐做法（生成器必须具备）：
1) 若 `keyframe_image_path` 存在且可读 → 直接用  
2) 否则按 `<item_dir>/<step_id>_<step_slug>/frame_{frame_index:03d}_ts_*s.jpg` 进行 glob 回退  
3) 若仍失败 → 跳过该样本（或 `meta.missing_media=true` 且不用于训练/评分）

可参考 `ECCV/extract_last_frame_segments.py` 的 `find_keyframe_path()` 逻辑。

---

## 3. 四类多模态证据（从训练视角重新定义“最合适”）

主规范的证据类型枚举更细（见 `ECCV/mani_longvideo_tasks_plan_final.md:82`），但从“任务选择”角度，你可以把可选多模态证据压缩为 4 类（最常用、最可控）：

1) **keyframe_single**：单张关键帧（`critical_frames[*].keyframe_image_path`）  
2) **images_uniform_scene**：全局均匀抽帧的多图（`sampled_frames/` 等距取 4–8 张）  
3) **video_clip**：局部片段（相邻 step 尾帧之间、或 step 执行片段、或任意裁剪 clip）  
4) **video_prefix**：前缀累积片段（从开头到 step 尾帧）

选择原则（强烈建议）：
- 只需要“瞬时状态/空间关系/热点区域” → `keyframe_single` 最稳（信息密度高、歧义最小、成本最低）。
- 需要“场景锚点/全局目标/环境布局” → `images_uniform_scene`（避免只看局部视角）。
- 需要“动作过程/转折/一致性对齐” → `video_clip`（单帧很容易缺失过程信息）。
- 需要“进度/下一步预测/未来规划” → `video_prefix`（必须给到前缀累积信息）。

当训练框架不支持 `video`：
- 用 `video_clip/video_prefix` 抽帧成 `images_uniform_clip`（属于实现细节，不改变“证据类别选择”）。

---

## 4. 统一的训练数据格式（直接兼容 Qwen-VL finetune）

建议沿用 ShareGPT 风格（`Qwen-PC/qwen-vl-finetune/README.md:46`），并把证据类型写进 meta：

```json
{
  "id": "uuid",
  "image": ["rel/or/abs/a.jpg", "rel/or/abs/b.jpg"],
  "video": "rel/or/abs/clip.mp4",
  "conversations": [
    {"from": "human", "value": "<image>\\n<image>\\n<Question...>"},
    {"from": "gpt", "value": "<Answer...>"}
  ],
  "meta": {
    "task_name": "Task_27_Visual_Spatial_Relation_Check",
    "evidence_type": "keyframe_single|images_uniform_scene|video_clip|video_prefix",
    "evidence_source": "keyframes|sampled_frames|last_frame_segments|cumulative_last_frame_segments|source_video",
    "evidence_files": ["..."],
    "source_path": "<item_dir>/causal_plan_with_keyframes.json",
    "step_id": 3,
    "frame_index": 2,
    "ts_sec": 68.39,
    "answer_format": "free|YesNo|AB|ABCD|ordered_list|structured",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "neg_sample": false
  }
}
```

约定：
- `image` / `video` 二选一或同时存在，但同一个样本的问句要与证据一致（不要“有 video 却用 <image>”）。
- 客观题务必在 prompt 中写清“只输出标签”，例如：`Only output A, B, C, or D.`（训练时也能减少发散）。

---

## 5. 从 JSON 生成 QA 的总体流水线（可落地）

### 5.1 解析与规范化

对每个 item_dir：
1) 读 `<item_dir>/causal_plan_with_keyframes.json`  
2) steps 按 `step_id` 排序  
3) 对每个 critical_frame：
   - 解析 `ts_sec`（优先从文件名 `ts_XX.XXs`）
   - resolve `keyframe_image_path`（见 2.3）
   - 生成 `mechanism`（见 2.2）

### 5.2 构建证据池（四类）

- keyframe_single：直接使用 resolved keyframe path
- images_uniform_scene：
  - 优先 `<item_dir>/sampled_frames/sample_*.jpg`（如果你没有它，建议补齐；否则全局任务会弱很多）
  - fallback：用“每步最早关键帧集合”做代理，再等距采样到 4–8 张
- video_clip：
  - 优先 `last_frame_segments/*.mp4` 或三阶段 `stage2/step_clips/`
  - fallback：仅存完整视频 + `video_start_sec/video_end_sec`（裁剪窗口来自关键帧 ts）
- video_prefix：
  - 优先 `cumulative_last_frame_segments/*.mp4`
  - fallback：完整视频 + `video_end_sec = step_last_ts`

### 5.3 样本构造策略：强监督（客观题）优先

理由：  
客观题（Yes/No/AB/ABCD/排序）可以稳定自动评测，且更利于训练 early stage 的对齐能力；主观题更适合后续 instruction-tuning 或用作“能力扩展”。

推荐做法：
- 每个 Task 至少保留 1 个“客观题变体”（即使原规范给的是自由文本示例）。
- 主观题变体保留，但数量控制 + 用 LLM 打磨/打分（避免噪声过大）。

---

## 6. 逐任务审视：推荐证据类型 & 推荐 QA 形态（训练优先）

下面给出每个 Task 的“最推荐”配置（可作为默认），并指出哪些任务**强监督**、哪些任务更适合**模型打分**。

记号：
- ✅ = 可稳定自动标注 & 适合做客观题
- ⚠️ = 适合做主观题（可生成参考答案，但更建议模型打分）

### Task_01_Macro_Anchor_Extraction
- **证据**：`images_uniform_scene`
- **推荐 QA**：⚠️（开放式列举）或 ✅（多选/多轮 Yes-No）
- **标注来源**：聚合 JSON 中 `tools/materials/objects/object_name/agent/patient`
- **训练建议**：
  - 若要客观化：把候选对象池限制在 8–12 个，做 “Select all that are anchors” + 输出如 `A,C,E`（否则集合评测复杂）。

### Task_02_Transient_Geometric_Verification
- **证据**：`keyframe_single`
- **推荐 QA**：✅（关系判别）
- **标注来源**：`spatial_preconditions[*].relation/objects/truth`
- **训练建议**：
  - 直接转成 Yes/No（其实就是 Task_27 的近亲），比“自由复述 relation”更稳定。

### Task_03_Micro_Affordance_Visual_Semantics
- **证据**：`keyframe_single`
- **推荐 QA**：✅（affordance_type MCQ）+ ⚠️（补充一句机制解释）
- **标注来源**：`affordance_hotspot.affordance_type`；机制从 2.2 的映射获得
- **训练建议**：
  - MCQ 选项从全局 affordance_type 词表采样 3 个干扰项（难度可控）。

### Task_04_Entity_Role_Identification
- **证据**：`keyframe_single`（step 最早关键帧）
- **推荐 QA**：✅（工具/材料二分类或 Yes/No）
- **标注来源**：`tool_and_material_usage.tools/materials`
- **训练建议**：
  - 用“可见对象集合”与 tools/materials 取交集，避免问到画面里根本不可见的名词导致退化为纯文本背诵。

### Task_05_State_Evolution_Description
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）或 ✅（caption matching 4 选 1）
- **标注来源**：`action_description/state_change_description`
- **训练建议**：
  - 最推荐做成 4 选 1：正确描述 + 3 个同域干扰（来自同 item 其他关键帧），能显著增强“看图才能选对”。

### Task_06_Holistic_Causal_Chain_Analysis
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（双段生成 + LLM 打分/打磨）
- **标注来源**：`causal_chain.*` + （可选）`spatial/affordance_preconditions` + mechanism
- **训练建议**：
  - 训练目标明确约束：不得引入 JSON 外实体；结构固定为两段（主规范已建议）。

### Task_07_Scene_Goal_Derivation
- **证据**：`images_uniform_scene`
- **推荐 QA**：✅（high_level_goal MCQ）
- **标注来源**：`high_level_goal`
- **训练建议**：
  - 干扰项来自其他 item 的 high_level_goal（同场景域更难）。

### Task_08_Strategic_Rationale_Justification
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）或 ✅（rationale 4 选 1）
- **标注来源**：`steps[i].rationale`
- **训练建议**：
  - 客观化时注意：干扰 rationale 必须与 step_goal 同域，否则太容易。

### Task_09_Precondition_Statement
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）
- **标注来源**：`steps[i].preconditions`
- **训练建议**：
  - 若要客观化：把 preconditions 拆成多条 Yes/No（“该条件是否必须？”），但容易退化为文本任务。

### Task_10_Step_Execution_Statement
- **证据**：优先 `video_clip`（step 执行片段），否则 `keyframe_single`
- **推荐 QA**：✅（step_goal MCQ）或 ✅（严格复述 step_goal）
- **标注来源**：`steps[i].step_goal`
- **训练建议**：
  - 更推荐 MCQ：同 item 的其他 step_goal 做干扰项（视觉对齐更强）。

### Task_11_Expected_Physical_Effects
- **证据**：`keyframe_single`（step 尾关键帧）
- **推荐 QA**：⚠️（自由文本）
- **标注来源**：`steps[i].expected_effects`
- **训练建议**：
  - 它本质是“计划监督”，视觉依赖弱；适合做 SFT 补充，不适合作为主要评测指标。

### Task_12_Inter_Step_Dependency_Analysis
- **证据**：`keyframe_single`（建议 step i 尾帧）
- **推荐 QA**：⚠️（自由文本）或 ✅（依赖点定位）
- **标注来源**：`expected_effects` 与下一步 `preconditions` 的词项重合/匹配（规则生成）
- **训练建议**：
  - 若客观化：问“哪个 precondition 被上一步效果直接满足？”提供 4 选 1（从 preconditions 中抽 1 个 gold + 3 个非重合干扰）。

### Task_13_Next_Action_Prediction
- **证据**：`keyframe_single`（计划版）或（更推荐）用 Task_24 的 `video_prefix`
- **推荐 QA**：✅（next step_goal MCQ）
- **标注来源**：`steps[i+1].step_goal`

### Task_14_Counterfactual_Prediction
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）
- **标注来源**：`causal_challenge_question/expected_challenge_outcome`

### Task_15_Failure_Recovery_Protocol
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）
- **标注来源**：`failure_handling.reason/recovery_strategy`

### Task_16_Physical_Feasibility_Verification
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）或 ✅（match/mismatch 可行性二分类）
- **标注来源**：
  - 原始自由文本：`spatial_preconditions/affordance_preconditions`
  - 客观化建议：构造“错配 step_goal + keyframe”负样本（同 item 跨 step 交换），标签 `Yes`(匹配) / `No`(不匹配)

### Task_17_Holistic_Step_Synthesis_Why_How
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（双段生成）
- **标注来源**：Why=`rationale`，How=`mechanism/causal_chain`

### Task_18_Visual_Precondition_Check
- **证据**：`video_prefix`（强烈推荐），无则退化 `images_uniform_scene`
- **推荐 QA**：⚠️（三态输出：satisfied / not satisfied / not directly observable）
- **标注来源**：难以完全自动化（preconditions 是自然语言，且未提供可观测性标注）
- **训练建议（让它“可训练”）**：
  - 使用“可观测性启发式”生成参考答案：包含 `inside/refrigerator contains/...` 之类内部不可见 → `not directly observable`；环境光照/位置类 → `satisfied`（若 prefix_end_step ≥ 对应完成 step）。
  - 同时把该任务标记为 `meta.weak_supervision=true`，避免与强监督任务混淆。

### Task_19_Visual_Effect_Check
- **证据**：`keyframe_single`（step 尾帧）+（可选）`video_clip`（执行片段）
- **推荐 QA**：⚠️（支持/不确定）
- **标注来源**：同 Task_18，强监督不足
- **训练建议**：同样标记 `weak_supervision`，或只保留“明显可见”的 effect（如 “object on countertop”）。

### Task_20_Step_Boundary_Localization
- **证据**：`video_clip`（相邻 step 尾帧之间）
- **推荐 QA**：⚠️（自由描述）或 ✅（边界识别 MCQ）
- **标注来源**：
  - 自由描述：LLM 打分
  - MCQ：clip 对应的边界 `(i→i+1)` 作为 gold（可自动标注）

### Task_21_Keyframe_Justification
- **证据**：`keyframe_single`
- **推荐 QA**：⚠️（自由文本）
- **标注来源**：`action/state_change/spatial/affordance/mechanism`

### Task_22_Plan_Execution_Alignment
- **证据**：`video_clip`
- **推荐 QA**：✅（match / mismatch / partial 三分类）
- **标注来源**：可通过“证据-步骤”配对规则自动生成：
  - match：step i 的执行片段 + step i 的 step_goal
  - mismatch：step i 片段 + step k 的 step_goal（k≠i）
  - partial：可用相邻 step_goal 作为 partial（可选，需规则固定）

### Task_23_Goal_Recognition_From_Prefix
- **证据**：`video_prefix`
- **推荐 QA**：✅（high_level_goal MCQ）
- **标注来源**：`high_level_goal`

### Task_24_Next_Step_Goal_Prediction_From_Prefix
- **证据**：`video_prefix`
- **推荐 QA**：✅（next step_goal MCQ/或严格复述）
- **标注来源**：`steps[i+1].step_goal`

### Task_25_Progress_Summary_From_Prefix
- **证据**：`video_prefix`
- **推荐 QA**：⚠️（自由文本总结）
- **标注来源**：弱监督参考为 `steps[0..i].step_goal`（仅写 meta，不写 prompt）
- **训练建议**：用 LLM 打分（覆盖度、是否引入未观察对象、是否与 meta 一致）。

### Task_26_Temporal_Order_Check
- **证据**：优先 `video_prefix`，无则 `images_uniform_scene`
- **推荐 QA**：✅（A/B）
- **标注来源**：关键帧文件名 `ts_XX.XXs` 比较（主规范也明确，见 `ECCV/mani_longvideo_tasks_plan_final.md:863`）

### Task_27_Visual_Spatial_Relation_Check
- **证据**：`keyframe_single`
- **推荐 QA**：✅（Yes/No）
- **标注来源**：`spatial_preconditions[*].truth`（可构造弱负样本）

### Task_28_Failed_Planning_Flaw_Pointing
- **证据**：优先 `video_prefix`，无则 `images_uniform_scene`
- **推荐 QA**：✅（结构化输出：`FlawStep=...; FlawType=...; Reason=...`）
- **标注来源**：通过扰动算子生成 bad plan，并记录 flaw_step/flaw_type（主规范已给出，见 `ECCV/mani_longvideo_tasks_plan_final.md:902`）

### Task_29_Next_K_Steps_Reordering_From_Prefix
- **证据**：`video_prefix`
- **推荐 QA**：✅（K 步排序，严格 1..K 输出）
- **标注来源**：`steps[i+1:i+K].step_goal` 的 gold 顺序

### Task_30_Middle_Steps_Infill_From_Head_Tail
- **证据**：`images_uniform_scene`（head+tail）
- **推荐 QA**：✅（中间步骤序列补全，严格 1..M 输出）
- **标注来源**：`steps[1:-1].step_goal`（或任意两锚点之间的中间序列）

---

## 7. 一个实操建议：训练集怎么配比更合理？

建议分两条线构建数据（同一 item 可同时产两类样本）：

1) **强监督客观题（主力训练/可回归）**：
   - Task_02/03(MC)/04/05(MC)/07(MC)/10(MC)/13(MC)/22/23(MC)/24(MC)/26/27/28/29/30
   - 这些任务能稳定自动标注，且更能迫使模型“看图/看视频”完成选择。

2) **主观生成题（能力扩展 + 模型打分）**：
   - Task_06/08/11/12/14/15/16/17/18/19/20(描述版)/21/25
   - 用 meta.fields 做裁判依据，统一用 LLM 评分（覆盖度、一致性、是否幻觉）。

这样做的好处：
- 训练收敛更稳（客观题提供明确监督信号）
- 同时保留复杂生成能力（主观题补充）

---

## 8. 最关键的“训练可用性”检查清单（生成前必须过）

对每条样本：
- `image`/`video` 路径可读（否则丢弃）
- prompt 中 `<image>`/`<video>` tag 数量与媒体数量一致
- 客观题 `answer_format` 与输出严格一致（训练时不要混用 “Answer: (B)” 和 “B”）
- `meta` 中记录足够的追溯信息：`source_path/step_id/frame_index/ts_sec/evidence_files/options/neg_sample`
- 对弱监督任务（18/19/25 等）显式标记 `meta.weak_supervision=true`

---

## 9. 下一步：把规范真正变成“可训练数据”的实现路线（建议）

如果你准备开始落地生成数据，推荐按“先强后弱”的工程顺序推进（避免一上来就被主观题质量拖垮）：

1) **先做强监督客观题（可自动打分）**：优先实现 Task_27、Task_26、Task_03(MC)、Task_04、Task_05(MC)、Task_07(MC)、Task_10(MC)、Task_22、Task_24、Task_28、Task_29、Task_30。  
2) **再做弱监督/主观题（模型打分）**：逐步加入 Task_06/08/11/12/14/15/16/17/18/19/20/21/25，并统一用 `meta.fields` + rubric 评分，或者先做数据打磨再混入训练。  
3) **统一数据接口**：输出 JSONL，保证与 `Qwen-PC/qwen-vl-finetune` 的数据格式一致（`image`/`video`、`<image>/<video>` tag、`conversations`）。
4) **质量控制**：对每个任务至少加 3 类 hard filter：
   - 媒体缺失/不可读（硬丢弃）
   - 文本不可判别（过于抽象/泛化，硬丢弃）
   - 负样本/扰动样本（必须 `meta.neg_sample=true`，并保证扰动算子是“单一可解释”的）

