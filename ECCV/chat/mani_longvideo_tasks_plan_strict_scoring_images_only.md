# mani_longvideo_tasks_plan_strict_scoring_images_only（严格可自动评分 · 仅图像证据）

本文档是 `mani_longvideo_tasks_plan_final.md`（Task_01–Task_30 体系）的**补充**：专门面向“**严格可自动评分（strict scoring）**”的 QA/监督样本构造，并强制限定证据只来自**图像**（images only）。目标是让评测/回归尽量“客观可对齐”，避免自由文本难以评分带来的噪声。

适用场景：
- 仅允许图片输入的 VLM 评测（不传视频、不传 mp4 clip）。
- 需要做大规模自动评测/回归（label 可 exact match）。

---

## 0. 范围与前置条件（默认与主规范一致）

- 本文档默认你已经有单视频 item 目录（例如 `causal_spafa_plan_dataset_long/<video_id>/`），至少包含：
  - `causal_plan_with_keyframes.json`
  - 关键帧图片：`steps[*].critical_frames[*].keyframe_image_path` 指向的文件（通常落在各 step 子目录内，形如 `frame_###_ts_XX.XXs.jpg`）
  - （强烈建议）全局均匀抽帧：`<item_dir>/sampled_frames/sample_*.jpg`（默认 50 帧；若不存在会走 fallback）
- **不使用**（即使存在也不引用）：`last_frame_segments/`、`cumulative_last_frame_segments/`、任何 `.mp4` 证据。
- 严格评分任务统一要求输出**短标签**（`Yes/No`、`A/B`、`A/B/C/D`），避免解释文本导致评分歧义。

> 重要：三阶段产物中 `critical_frames[*].frame_index` 是“step clip 的局部帧池索引”，跨 step 不可直接比较；严格评分若需要时间顺序，一律以关键帧文件名中的 `ts_XX.XXs`（全局时间轴）为准。

---

## 1. 证据（images only）与取图规则（与主规范一致的优先级 + fallback）

### 1.1 `keyframe_single`（单关键帧）

- 证据文件：`steps[s].critical_frames[f].keyframe_image_path`
- 取帧建议（与主规范一致）：
  - step 级任务（如工具/材料角色）：优先用该 step 的最早关键帧 `critical_frames[0]`
  - “完成态/效果类”：优先用该 step 的最后关键帧 `critical_frames[-1]`
  - 瞬时物理/空间核验类：用对应的 `critical_frames[f]`

### 1.2 `keyframe_pair`（两关键帧，仅用于时间顺序类严格评分）

- 证据文件：两张关键帧图片路径（来自同一 item；可同一步或跨步）
- 时间戳来源（推荐顺序）：
  1) 从关键帧文件名解析 `ts_XX.XXs`（全局时间轴；最稳）
  2) 若文件名不含 `ts_`：可回退到生成阶段的 manifest（仅当你本地保留且可解析）
- 为兼容主规范的 `meta.evidence_type` 取值集合：若你希望避免新增 evidence_type，可设置：
  - `meta.evidence_type="keyframe_single"` + `meta.evidence_subtype="pair"`，并把两张图放进 `image=[...2 paths...]`

### 1.3 `images_uniform_scene`（全局均匀抽帧，多图）

- 证据来源优先级（与主规范一致）：
  1) `<item_dir>/sampled_frames/sample_*.jpg`
  2) 若缺失：`<item_dir>/stage1/sampled_frames/sample_*.jpg`（部分三阶段中间产物路径）
  3) 若仍缺失：用“每步最早关键帧集合”做代理（取 `steps[i].critical_frames[0].keyframe_image_path`，再做等距采样）
- 建议固定取图张数 `k=8`（或 4–8），并使用确定性等距采样：
  - `idx = round(linspace(1, N, k))`（1-based）→ 映射到路径列表
- 目的：覆盖全局环境与阶段变化，避免只给“某一步的局部视角”。

### 1.4 路径归一化（强烈建议）

- 如果 `keyframe_image_path` 是相对路径：以 `<item_dir>` 作为基准 resolve。
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

延续主规范推荐的 ShareGPT 风格，并强化 `meta` 可追溯性（只用 `image`，不填 `video`）：

```json
{
  "id": "uuid",
  "image": ["/abs/or/rel/path/a.jpg", "/abs/or/rel/path/b.jpg"],
  "conversations": [
    {"from": "human", "value": "<English question with options if any>"},
    {"from": "gpt", "value": "A"}
  ],
  "meta": {
    "task_name": "SS03_Temporal_Order_Check_AB",
    "source_path": "<item_dir>/causal_plan_with_keyframes.json",
    "evidence_type": "keyframe_single",
    "evidence_subtype": "pair",
    "evidence_source": "keyframes",
    "evidence_files": ["...a.jpg", "...b.jpg"],
    "answer_format": "AB",
    "step_id_a": 1,
    "step_id_b": 3,
    "frame_index_a": 2,
    "frame_index_b": 1,
    "ts_a": 3.59,
    "ts_b": 68.39,
    "neg_sample": false,
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
  }
}
```

说明：
- `step_id` 推荐使用源 JSON 的 `step_id`（1-based，避免 0/1-based 混乱）。
- `frame_index_*` 若来自关键帧字段，则等于 `critical_frames[f].frame_index`（仅作回溯；不要跨 step 做数值比较）。
- `options` 仅在多选题（ABCD）时需要；用于复现与 debug。

---

## 4. 任务清单（Strict Scoring · Images Only）

下面任务均可从 `causal_plan_with_keyframes.json` + 图像证据自动构造并自动评分。括号中给出与主任务的对应关系（便于复用生成逻辑）。

### SS01_Visual_Spatial_Relation_Check（空间关系真假核验；对应 Task_27_Visual_Spatial_Relation_Check）

- **证据**：`keyframe_single`
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[f].spatial_preconditions[k].relation`
  - `steps[s].critical_frames[f].spatial_preconditions[k].objects`
  - `steps[s].critical_frames[f].spatial_preconditions[k].truth`
- **标签**：`truth=true → Yes`；`truth=false → No`
- **样本构造（推荐）**：
  1) 采样一个关键帧（任意 step / 任意 critical_frame）。
  2) 从该关键帧的 `spatial_preconditions` 中抽 1 条关系（避免关系爆炸）。
  3) 若你希望平衡正负样本：建议显式构造弱负样本，并写入 `meta.neg_sample=true`：
     - 方式 A：把 `truth` 取反（label 取反）
     - 方式 B：在 `objects` 内替换 1 个 object（从同关键帧的其他对象集合中抽一个），并将 label 设为 `No`
- **问答模板（示例）**：
  - Evidence: `<IMAGE_PATH>`
  - Q: In this image, is the following spatial statement true? Relation: `<relation>`. Objects: `<obj1>, <obj2>, ...`.
  - A (label): `Yes` / `No`
- **过滤建议（让评分更“像看图”，而不是背答案）**：
  - 优先关系：`contacting/holding/inside/on_top_of/open/closed` 等视觉信号强的关系
  - 若 `relation` 极长、或对象列表过多导致歧义：建议丢弃该样本

### SS02_Affordance_Hotspot_Type_MC（热点 affordance_type 四选一；对应 Task_03 的 label-only 变体）

- **证据**：`keyframe_single`
- **字段来源（JSONPath）**：
  - `steps[s].critical_frames[f].affordance_hotspot.affordance_type`（gold）
  - （可选，用于出题不那么 trivial）`...affordance_hotspot.description`
- **标签**：四选一 `A/B/C/D`
- **样本构造（推荐）**：
  1) 采样一个关键帧，取 `affordance_type` 作为正确答案。
  2) 从“全局 affordance_type 词表”（跨 item/跨 step 统计）中抽 3 个干扰项：
     - 必须与正确项不同且互不重复
     - 建议优先选“同大类看起来可能”的类型（更能测分辨能力）
  3) 随机打乱选项顺序，并在 `meta.options` 中落盘 `{A:...,B:...,C:...,D:...}` 以便复现。
- **问答模板（示例）**：
  - Evidence: `<IMAGE_PATH>`
  - Q: Which affordance type best matches the visually active hotspot region?
    - A) `<type_a>`
    - B) `<type_b>`
    - C) `<type_c>`
    - D) `<type_d>`
  - A (label): `A` / `B` / `C` / `D`
- **注意**：
  - 若全局词表不足以构造 4 个不同选项：跳过该样本（不要用重复选项凑数）。

### SS03_Temporal_Order_Check_AB（两事件先后判别；对应 Task_26 的更严格可评分版本）

- **证据**：`keyframe_pair`
- **字段来源**：
  - 时间戳：两张关键帧文件名中的 `ts_XX.XXs`
  - 事件文本（用于写题面）：`action_description` / `state_change_description`
- **标签**：`A`（A 更早）或 `B`（B 更早）
- **样本构造（推荐）**：
  1) 选两张关键帧图片（建议来自不同 step，或来自同 step 的 `critical_frames[0]` 与 `critical_frames[-1]`）。
  2) 解析 `ts_a`、`ts_b`，并保证 `ts_a != ts_b`；若相等则丢弃/重采样（重复帧常见于短视频或抽帧失败回填）。
  3) 为每张图生成一个短事件描述（A/B），并随机打乱呈现顺序；label 仅来自 `ts` 比较。
- **问答模板（示例）**：
  - Evidence A: `<IMAGE_PATH_A>`
  - Evidence B: `<IMAGE_PATH_B>`
  - Q: Which event happens earlier in the video, A or B?
  - A (label): `A` / `B`

### SS04_Tool_vs_Material_Check（工具/材料角色判别；对应 Task_04 的二分类变体）

- **证据**：`keyframe_single`（优先该 step 的最早关键帧 `critical_frames[0]`）
- **字段来源（JSONPath）**：
  - `steps[s].tool_and_material_usage.tools`
  - `steps[s].tool_and_material_usage.materials`
- **标签**：`Yes/No`
- **样本构造（推荐）**：
  1) 采样一个 step。
  2) 选一个候选名词 `x`，建议优先从该关键帧中“可见对象集合”抽取：
     - 可见对象集合可取并集：`spatial_preconditions[*].objects` + `affordance_preconditions[*].object_name`
     - 再与 `(tools ∪ materials)` 取交集，避免选到根本不在画面中的实体导致任务退化为纯文本背诵
  3) 随机决定问法：
     - “Is x a tool used in this step?” → label = `x ∈ tools`
     - 或 “Is x a material being acted on in this step?” → label = `x ∈ materials`
  4) 可选 hard negative：从同 item 的其他 step 抽一个工具/材料作为干扰（并在 `meta.neg_sample=true` 标记）。
- **问答模板（示例）**：
  - Evidence: `<IMAGE_PATH>`
  - Q: In this step, is `<x>` a tool used by the agent?
  - A (label): `Yes` / `No`
- **注意**：
  - 当 `tools` 为空时，建议在数据构造时补上 `hands`（与主规范一致），否则该任务会退化。

### SS05_Step_Goal_Matching_MC（关键帧对应 step_goal 四选一；对应 Task_10 的严格可评分变体）

- **证据**：`keyframe_single`（优先 `critical_frames[0]`）
- **字段来源（JSONPath）**：
  - 正确项：`steps[s].step_goal`
  - 干扰项：同一 item 内其他 step 的 `step_goal`
- **标签**：四选一 `A/B/C/D`
- **样本构造（推荐）**：
  1) 采样一个 step s，取其 `step_goal` 为正确项。
  2) 从同 item 的其他步骤中抽 3 条不同的 `step_goal` 作为干扰项（避免重复或高度同义）。
  3) 打乱选项顺序并写入 `meta.options`，label 为正确选项字母。
- **问答模板（示例）**：
  - Evidence: `<IMAGE_PATH>`
  - Q: Which step goal best matches what is happening in this image?
    - A) `<goal_a>`
    - B) `<goal_b>`
    - C) `<goal_c>`
    - D) `<goal_d>`
  - A (label): `A` / `B` / `C` / `D`

---

## 5. 常见失败模式与清洗建议（严格评分专用）

- **媒体缺失**：`keyframe_image_path` 指向文件不存在、或 `sampled_frames/` 空 → 直接跳过该样本（或在 `meta.missing_media=true` 并不参与评分）。
- **重复时间戳**：`SS03` 必须剔除 `ts_a == ts_b`；必要时在构造阶段强制重采样 keyframe 对。
- **弱负样本未标注**：凡是通过“取反 truth / 扰动 objects / 跨 step 借用工具材料”等方式构造的负样本，都必须写 `meta.neg_sample=true`（防止把弱负样本当强监督）。
- **对象命名不一致**：尽量沿用源 JSON 中的 `snake_case`（例如 `cutting_board`），生成 distractor 时也要统一命名风格，否则选项对齐会崩。
- **空间关系过难**：优先使用明显接触/支撑/包含关系；姿态类（角度、微小距离）容易引入“看图也不确定”的噪声，建议丢弃。

---

## 6. 与主规范的关系

- 本文件定义的是主规范任务体系的“严格可自动评分 + images only”子集（并给出更可落地的 meta 与取图细则）。
- 完整任务定义、证据类型全集、以及视频/片段证据任务请以 `mani_longvideo_tasks_plan_final.md` 为准。
