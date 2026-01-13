# mani_longvideo_tasks_plan_strict_scoring_mp4_video（严格可自动评分 · 支持完整视频/clip 的 MP4 证据）

本文档是 `mani_longvideo_tasks_plan_final.md`（Task_01–Task_30 体系）的**补充**：专门面向“**严格可自动评分（strict scoring）**”的 QA/监督样本构造，并允许证据直接来自 **MP4 视频**（可以是**完整视频**或任意**视频片段 clip**）。目标是让评测/回归尽量“客观可对齐”，同时利用视频输入覆盖**动作过程/时序信息**，减少仅靠单帧带来的歧义。

适用场景：
- 支持视频输入的 VLM/VLA 评测或训练（可直接喂 `.mp4`）。
- 需要大规模自动评测/回归（label 可 exact match），同时希望题目更依赖“看视频”而不是“背文本”。

---

## 0. 范围与前置条件（默认与主规范一致）

- 本文档默认你已经有单视频 item 目录（例如 `causal_spafa_plan_dataset_long/<video_id>/`），至少包含：
  - `causal_plan_with_keyframes.json`
  - 能解析到关键帧时间戳的 `keyframe_image_path`（通常文件名形如 `frame_###_ts_XX.XXs.jpg`；图片文件本身可缺失，但**字符串里必须有 `ts_`** 才能稳定对齐时间轴）
- 你需要至少具备一种视频证据来源（MP4）：
  1) **完整源视频**：`run_summary.json` 内可解析的 `config_planning.VIDEO_PATH` 或 `source_video`（或你自己额外提供的 mp4 路径）
  2) **Step 间片段**：`last_frame_segments/*.mp4`（可用 `python ECCV/extract_last_frame_segments.py --video-output-dir <item_dir>` 生成）
  3) **累积前缀片段**：`cumulative_last_frame_segments/*.mp4`（可用 `python ECCV/extract_cumulative_last_frame_segments.py --video-output-dir <item_dir>` 生成）
  4) （可选）**三阶段 step clip**：`three_stage` 的 `stage2/step_clips/`（若你保留了三阶段中间产物）
- 严格评分任务统一要求输出**短标签**（`Yes/No`、`A/B`、`A/B/C/D`），避免解释文本导致评分歧义。

> 重要：三阶段产物中 `critical_frames[*].frame_index` 是“step clip 的局部帧池索引”，跨 step 不可直接比较；凡是需要时间顺序/对齐，一律以 `keyframe_image_path` 文件名中的 `ts_XX.XXs`（全局时间轴）为准。

---

## 1. 证据（MP4）与裁剪/选段规则（优先级 + fallback）

本文件尽量不新增主规范的 `meta.evidence_type` 枚举：视频证据统一落在 `video_clip` 或 `video_prefix`，其余用 `meta.evidence_subtype` 做区分。

### 1.1 `video_clip`（单个 mp4：可为完整视频或任意 clip）

- **证据文件**：一个 `.mp4` 路径（完整视频或裁剪后的 clip 均可）。
- **建议的 meta**：
  - `meta.evidence_type="video_clip"`
  - `meta.evidence_subtype ∈ {"full_video","step_clip","keyframe_window","two_event_window"}`
  - `meta.evidence_source ∈ {"source_video","last_frame_segments","stage2_step_clips","custom_clip"}`
  - `meta.evidence_files=[".../something.mp4"]`
  - （若 clip 来自完整视频裁剪）`meta.video_start_sec` / `meta.video_end_sec`（全局时间轴，单位秒）
- **裁剪策略（推荐）**：
  - **关键帧窗口**（对齐某个关键帧时刻 `ts`）：`[max(ts - pre, 0), ts + post]`
    - 推荐 `pre=1.0s`，`post=0.05s`（让关键帧时刻尽量落在 clip 尾部，便于问“clip 末帧”）
  - **两事件窗口**（覆盖两件事 A/B：`ts_a`、`ts_b`）：`[max(min(ts_a,ts_b)-pre,0), max(ts_a,ts_b)+post]`
    - 推荐 `pre=1.0s`，`post=1.0s`（保证两事件都有上下文）
- **实现方式（两选一）**：
  1) **预生成 clip mp4 文件**：用 ffmpeg 或三阶段 pipeline 产出的 clip（最兼容）。
  2) **运行时裁剪**：保留完整视频 mp4，并在数据/loader 层记录 `video_start_sec`/`video_end_sec`（若你的视频读取工具支持按时间取段，例如 qwen-vl-utils 支持 `video_start`/`video_end`）。

> 注意：如果你用 `-c copy` 方式裁剪，clip 起点可能不在关键帧导致部分解码器读取异常；评测/训练稳定性优先时建议 re-encode（或使用三阶段 pipeline 的 `cut-mode reencode`）。

### 1.2 `video_prefix`（累积前缀 mp4：从开头到某步尾帧）

- **证据文件**：`cumulative_last_frame_segments/segment_start_to_step{step_id:02d}_last.mp4`
- **语义**：从视频开始到 step `{step_id}` 尾关键帧时刻（强对齐“已完成进度”）。
- **建议的 meta**：
  - `meta.evidence_type="video_prefix"`
  - `meta.evidence_source="cumulative_last_frame_segments"`
  - `meta.segment_label="segment_start_to_stepXX_last"`

### 1.3 Step clip 的可靠来源（推荐顺序）

当你需要“某一步发生了什么”的视频证据（用于 SS04/SS05/SS06/SS07 等 step 级严格评分），推荐按如下优先级选择：

1) 三阶段产物（若存在）：`<item_dir>/stage2/step_clips/<step_id>.mp4`（或相近命名）
2) `last_frame_segments/` 的**尾帧间片段**（无需三阶段也可生成）：
   - step 1：`last_frame_segments/segment_start_to_step01.mp4`
   - step s（s>1）：`last_frame_segments/segment_step{(s-1):02d}_to_step{s:02d}.mp4`
3) 若以上都缺失：退化为完整视频 `video_clip` + `video_start_sec/video_end_sec`（用关键帧时间戳裁剪出近似 step 片段）

### 1.4 路径归一化（强烈建议）

- 若视频路径是相对路径：以 `<item_dir>` 作为基准 resolve。
- 为了可移植性：建议在最终数据集中存**相对 item_dir 的路径**（同时可在 `meta.abs_path_debug` 里保留绝对路径用于本地调试）。

---

## 2. 严格评分输出协议（强制）

- 模型输出必须是**单行**，且严格匹配以下之一（大小写敏感）：
  - 二分类：`Yes` 或 `No`
  - 二选一：`A` 或 `B`
  - 四选一：`A` / `B` / `C` / `D`
- 不允许输出解释、标点、额外空行、JSON、markdown。
- 评分方式建议：对输出做 `strip()` 后 exact match。

---

## 3. 样本 JSON 结构（建议沿用主规范的 meta）

延续主规范推荐的 ShareGPT 风格，并强化 `meta` 可追溯性（只填 `video`，`image` 可省略或置空）：

```json
{
  "id": "uuid",
  "video": "/abs/or/rel/path/clip_or_full.mp4",
  "conversations": [
    {"from": "human", "value": "<English question with options if any>"},
    {"from": "gpt", "value": "A"}
  ],
  "meta": {
    "task_name": "SS03_Temporal_Order_Check_AB",
    "source_path": "<item_dir>/causal_plan_with_keyframes.json",
    "evidence_type": "video_clip",
    "evidence_subtype": "two_event_window",
    "evidence_source": "source_video",
    "evidence_files": ["/abs/or/rel/path/clip_or_full.mp4"],
    "video_start_sec": 12.30,
    "video_end_sec": 83.10,
    "answer_format": "AB",
    "step_id_a": 2,
    "step_id_b": 5,
    "ts_a": 18.42,
    "ts_b": 77.95,
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "neg_sample": false
  }
}
```

说明：
- `step_id` 推荐使用源 JSON 的 `step_id`（1-based，避免 0/1-based 混乱）。
- `video_start_sec/video_end_sec` 用于“可复现定位”；如果你直接喂 clip mp4 文件，也建议记录其在完整视频上的全局区间。
- `options` 仅在多选题（ABCD）时需要；用于复现与 debug。

---

## 4. 任务清单（Strict Scoring · MP4 Video Input）

下面任务均可从 `causal_plan_with_keyframes.json` + MP4 视频证据自动构造并自动评分。括号中给出与主任务的对应关系（便于复用生成逻辑）。

### SS01_Visual_Spatial_Relation_Check（空间关系真假核验；对应 Task_27_Visual_Spatial_Relation_Check）

- **证据**：`video_clip`（`keyframe_window` 或 step clip）
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[f].spatial_preconditions[k].relation`
  - `steps[s].critical_frames[f].spatial_preconditions[k].objects`
  - `steps[s].critical_frames[f].spatial_preconditions[k].truth`
- **标签**：`truth=true → Yes`；`truth=false → No`
- **样本构造（推荐）**：
  1) 采样一个关键帧（任意 step / 任意 critical_frame）。
  2) 解析该关键帧的全局时间戳 `ts`（从 `keyframe_image_path` 文件名）。
  3) 构造 `keyframe_window` clip：`[ts-1.0s, ts+0.05s]`（clamp 到 `>=0`）。
  4) 从该关键帧的 `spatial_preconditions` 中抽 1 条关系（避免关系爆炸）。
  5) 问题强制对齐到“clip 末帧/末时刻”，避免模型在 clip 内任意帧自由选择导致评分噪声。
  6) 若你希望平衡正负样本：建议显式构造弱负样本，并写入 `meta.neg_sample=true`：
     - 方式 A：把 `truth` 取反（label 取反）
     - 方式 B：在 `objects` 内替换 1 个 object（从同关键帧的其他对象集合中抽一个），并将 label 设为 `No`
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: At the END of this clip, is the following spatial statement true? Relation: `<relation>`. Objects: `<obj1>, <obj2>, ...`.
  - A (label): `Yes` / `No`
- **过滤建议**：
  - 优先关系：`contacting/holding/inside/on_top_of/open/closed` 等视觉信号强的关系
  - 若对象列表过多/关系文本过长导致歧义：建议丢弃该样本

### SS02_Affordance_Hotspot_Type_MC（热点 affordance_type 四选一；对应 Task_03 的 label-only 变体）

- **证据**：`video_clip`（`keyframe_window`；让热点出现在 clip 尾部）
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[f].affordance_hotspot.affordance_type`（gold）
  - （可选，用于出题不那么 trivial）`...affordance_hotspot.description`
- **标签**：四选一 `A/B/C/D`
- **样本构造（推荐）**：
  1) 采样一个关键帧，取 `affordance_type` 作为正确答案。
  2) 解析关键帧时间戳 `ts`，构造 `keyframe_window` clip。
  3) 从“全局 affordance_type 词表”（跨 item/跨 step 统计）中抽 3 个干扰项：
     - 必须与正确项不同且互不重复
     - 建议优先选“同大类看起来可能”的类型（更能测分辨能力）
  4) 随机打乱选项顺序，并在 `meta.options` 中落盘 `{A:...,B:...,C:...,D:...}` 以便复现。
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: Which affordance type best matches the visually active hotspot region at the END of this clip?
    - A) `<type_a>`
    - B) `<type_b>`
    - C) `<type_c>`
    - D) `<type_d>`
  - A (label): `A` / `B` / `C` / `D`
- **注意**：
  - 若全局词表不足以构造 4 个不同选项：跳过该样本（不要用重复选项凑数）。

### SS03_Temporal_Order_Check_AB（两事件先后判别；对应 Task_26 的更严格可评分版本）

- **证据**：`video_clip`（`two_event_window`；一个 clip 同时覆盖 A 与 B）
- **字段来源**：
  - 时间戳：两张关键帧文件名中的 `ts_XX.XXs`
  - 事件文本（用于写题面）：`action_description` / `state_change_description`
- **标签**：`A`（A 更早）或 `B`（B 更早）
- **样本构造（推荐）**：
  1) 选两条关键帧记录（建议来自不同 step，且 `ts_a != ts_b`；若相等则丢弃/重采样）。
  2) 用 `[min(ts_a,ts_b)-1.0s, max(ts_a,ts_b)+1.0s]` 裁剪出单个 `two_event_window` clip（clamp 到 `>=0`）。
  3) 为每个事件生成一个短描述（A/B），并随机打乱呈现顺序；label 仅来自 `ts` 比较。
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: Which event happens earlier in this clip, A or B?
    - A) `<event_desc_a>`
    - B) `<event_desc_b>`
  - A (label): `A` / `B`
- **清洗建议**：
  - 若两事件描述高度相似（几乎同义），容易变成猜测：丢弃。
  - 若裁剪后 clip 过长（例如 >60s），建议换事件对或缩小窗口。

### SS04_Tool_vs_Material_Check（工具/材料角色判别；对应 Task_04 的二分类变体）

- **证据**：`video_clip`（step clip；优先覆盖该 step 执行过程）
- **字段来源（JSONPath）**：
  - `steps[s].tool_and_material_usage.tools`
  - `steps[s].tool_and_material_usage.materials`
- **标签**：`Yes/No`
- **样本构造（推荐）**：
  1) 采样一个 step。
  2) 取该 step 的一个工具/材料候选名词 `x`，并尽量让问题“依赖看视频”：
     - 正样本：`x` 取自 `tools`（问 tool）或 `materials`（问 material）
     - 弱负样本：`x` 取自同 item 的其他 step（或交换 tool/material 身份），并写 `meta.neg_sample=true`
  3) 问法随机二选一：
     - “Is x a tool used by the agent in this clip?” → label = `x ∈ tools`
     - “Is x a material being acted on in this clip?” → label = `x ∈ materials`
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: In this clip, is `<x>` a tool used by the agent?
  - A (label): `Yes` / `No`
- **注意**：
  - 当 `tools` 为空时，建议在数据构造时补上 `hands`（与主规范一致），否则该任务会退化。

### SS05_Step_Goal_Matching_MC（clip 对应 step_goal 四选一；对应 Task_10 的严格可评分变体）

- **证据**：`video_clip`（step clip）
- **字段来源（JSONPath）**：
  - 正确项：`steps[s].step_goal`
  - 干扰项：同一 item 内其他 step 的 `step_goal`
- **标签**：四选一 `A/B/C/D`
- **样本构造（推荐）**：
  1) 采样一个 step s，取其 `step_goal` 为正确项，并取其 step clip 作为证据。
  2) 从同 item 的其他步骤中抽 3 条不同的 `step_goal` 作为干扰项（避免重复或高度同义）。
  3) 打乱选项顺序并写入 `meta.options`，label 为正确选项字母。
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: Which step goal best matches what is happening in this clip?
    - A) `<goal_a>`
    - B) `<goal_b>`
    - C) `<goal_c>`
    - D) `<goal_d>`
  - A (label): `A` / `B` / `C` / `D`

### SS06_Action_Presence_Check（动作存在核验；更依赖视频过程）

- **证据**：`video_clip`（step clip）
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[*].action_description`
- **标签**：`Yes/No`
- **样本构造（推荐）**：
  1) 对某个 step 取其 step clip 作为证据。
  2) 正样本：从该 step 的关键帧里抽一条 `action_description` 作为题面 → label `Yes`。
  3) 负样本：从同 item 的其他 step 抽一条 `action_description` 替换题面 → label `No`，并写 `meta.neg_sample=true`。
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: Does this clip show the agent performing the following action? `<action_description>`
  - A (label): `Yes` / `No`
- **过滤建议**：
  - 过于抽象/泛化（如 “do something”, “continue”）的描述会降低可判别性：丢弃。

### SS07_State_Change_Presence_Check（状态变化存在核验；更依赖视频前后对比）

- **证据**：`video_clip`（step clip，尽量覆盖变化前后）
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[*].state_change_description`
- **标签**：`Yes/No`
- **样本构造（推荐）**：
  1) 对某个 step 取其 step clip 作为证据。
  2) 正样本：抽一条 `state_change_description`（来自该 step 关键帧）→ label `Yes`。
  3) 负样本：用其他 step 的 `state_change_description` 替换 → label `No`，并写 `meta.neg_sample=true`。
- **问答模板（示例）**：
  - Evidence: `<VIDEO_CLIP_PATH>`
  - Q: Does the following state change happen in this clip? `<state_change_description>`
  - A (label): `Yes` / `No`

---

## 5. 常见失败模式与清洗建议（严格评分 · 视频专用）

- **视频缺失/不可读**：mp4 路径不存在、解码失败、权限不足 → 直接跳过该样本（或 `meta.missing_media=true` 且不参与评分）。
- **时间戳无法解析**：`keyframe_image_path` 不含 `ts_XX.XXs` → 该样本无法稳定对齐；建议丢弃或回退到 manifest（仅当你确实保留并可解析）。
- **裁剪边界 off-by-one**：不同视频读取器对 end time 的包含关系可能略有差异；建议统一给 `video_end_sec = ts + 0.05` 这类小余量。
- **clip 过长**：视频模型上下文受限，长 clip 会稀释关键信号 → 优先用 `keyframe_window/two_event_window` 或 step clip，必要时降低 fps 或限制最大帧数。
- **负样本未标注**：凡是通过“跨 step 替换描述/取反 truth/扰动 objects”等方式构造的负样本，都必须写 `meta.neg_sample=true`。
- **描述不可判别**：action/state-change 文本太抽象、空间关系太细微 → 丢弃（否则变成主观题）。

---

## 6. 与主规范的关系

- 本文件定义的是主规范任务体系的“严格可自动评分 + MP4 视频输入”子集（并给出更可落地的 clip 选择与 meta 细则）。
- 若你的训练/推理模型不支持 `video`：可把 mp4 预抽帧成 `image` 序列，并将 `meta.evidence_type` 设为 `images_uniform_clip`（主规范已有定义）。
