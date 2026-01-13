# 三阶段长视频数据生成管线（完整规范）

本文档是 `ECCV/three_stage/` 三阶段管线的**设计规范 + 操作手册**，目标是用 VLM 在严格 JSON/schema 约束下，稳定生成可追溯、可复现的长视频因果计划与关键帧标注数据。

三阶段总览：

1) **Stage 1：Draft 计划生成**（仅 step-level，不生成 keyframes）  
2) **Stage 2：步骤时间段定位 + clip 切片**  
3) **Stage 3：基于 clip 的精修 + 关键帧选择**

> 说明：`ECCV/` 下原有两阶段脚本保持不变，本目录是独立实现。

## 设计原则（为什么这样做）

- **50 图像输入预算**：Stage1/2 共用 full video 的 50 帧池；Stage3 对每个 clip 再采样 50 帧池，保证每次模型调用都在常见 VLM 限制内。
- **严格 JSON only**：所有阶段要求“只输出 JSON”，并对 schema 做强校验；失败自动重试并把错误列表塞回下一轮 prompt。
- **语义可对齐**：Stage3 输出对齐 ECCV 既有 long-video `causal_plan_with_keyframes.json` 的字段与结构。
- **强可追溯性**：每阶段落盘 system/user prompt、raw response、frame manifest、run summary。
- **断点续跑优先**：默认复用已生成且通过校验的产物；需要重新生成时用 `--overwrite`。

## 约定（Conventions）

- 下面的路径与命令默认你在 `ECCV/` 目录运行；如果在 repo 根目录运行，需要在路径前加 `ECCV/`。
- prompt 定义在 `three_stage/prompts.py`，每次运行会把“实际使用的 prompt”写入输出目录，保证可复现实验。

## 配置与依赖

### 依赖

- Python 3
- `ffmpeg` 在 `PATH`（或通过 `--ffmpeg-bin` 指定）
- Python 包：

```bash
pip install openai opencv-python
```

### API 配置

可通过命令行参数或环境变量传入（推荐环境变量，避免泄漏/硬编码）：

- `API_KEY`
- `API_BASE_URL`
- `MODEL_PROVIDER_ID`
- `MODEL_NAME`

### 关键运行参数

- `--max-frames`（默认 50）：建议保持 `<= 50`
- `--max-tokens` / `MAX_TOKENS`：模型输出 token 上限（过大可能变慢，过小可能截断 JSON）
- `--no-embed-index`：不在 Stage2/Stage3 的输入图上叠加 “Frame XX”（Stage1 始终避免任何 index artifact）

## 输出目录结构（Directory Layout）

默认输出根目录：`ECCV/three_stage/causal_spafa_plan_dataset_long/<video_id>/`

```text
<video_id>/
  sampled_frames/                 # 兼容输出：full video 50 帧（默认链接到 stage1/sampled_frames）
  frame_manifest.json             # 兼容输出：full video manifest（复制自 stage1/frame_manifest.json）
  stage1/
    frame_manifest.json
    sampled_frames/
    draft_plan.json
    stage1_system_prompt.txt
    stage1_user_prompt.txt
    stage1_raw_response.txt
  stage2/
    localization_raw.json
    step_segments.json
    step_clips/
    stage2_system_prompt.txt
    stage2_user_prompt.txt
    stage2_raw_response.txt
  01_<slug>/
    frame_manifest.json
    sampled_frames/
    step_final.json
    frame_###_ts_XX.XXs.jpg
    stage3_system_prompt.txt
    stage3_user_prompt.txt
    stage3_raw_response.txt
    step_meta.json
  ...
  causal_plan_with_keyframes.json
  run_summary.json
```

说明：

- `<slug>` 由 `step_goal` 经 `sanitize_filename()` 归一化得到（小写、下划线、截断过长）。
- `keyframe_image_path` 会被填为关键帧 JPEG 的**绝对路径**（与既有 ECCV 数据兼容）。

## 帧池与索引语义（非常重要）

### 两类帧池

- **full video 帧池（Stage1/Stage2 共用）**
  - 存在 `stage1/frame_manifest.json`
- **step clip 帧池（Stage3 每步单独）**
  - 存在 `<step_folder>/frame_manifest.json`

### manifest 字段

每个 `frame_manifest.json` 都包含：

- `frame_index_1based`：1-based 索引（prompt/模型输出使用这个）
- `timestamp_sec`：该帧对应的秒级时间戳（脚本侧用于换算与命名）
- `image_relpath`：用于人工检查的 JPEG 相对路径

> 均匀采样为了保证固定帧数，可能出现**重复帧/相同 timestamp**（尤其是视频很短或解码不稳定时）。

### Stage2 索引语义（exclusive end）

Stage2 输出每步边界索引：

- `start_frame_index`：inclusive（从该边界开始）
- `end_frame_index`：exclusive（该步结束后的第一个边界）

脚本侧会用 manifest 的 `timestamp_sec` 做秒级换算并切片。

### Stage3 索引语义（每步独立帧池）

Stage3 输出 `critical_frames[*].frame_index`：

- 1-based
- **基于该 step clip 自己的 50 帧池**
- 与 Stage2 的 full video 帧池无直接可比性

## Stage 1 — Draft 计划生成（不含 keyframes）

脚本：`three_stage/stage1_generate_draft.py`

输入：

- 原视频 → 均匀采样（默认 50 帧）
- Stage1 prompt（严格 schema）

输出：

- `stage1/draft_plan.json`

硬约束（校验失败会自动重试）：

- 严禁输出任何 keyframe 字段：`critical_frames` / `frame_index` / `keyframe_image_path`
- 文本字段禁止出现 “Frame 12 / Image 12 …” 等帧引用
- `step_id` 必须是 `1..N` 且递增；`step_goal` 不能重复
- 步数硬约束 4–9（推荐 5–8）
- 关键字段非空（例如 `predicted_next_actions` 必须 2–4 条）

断点续跑：

- 若 `stage1/draft_plan.json` 与 `stage1/frame_manifest.json` 存在且通过严格校验，会直接复用（除非 `--overwrite`）。

## Stage 2 — 步骤定位 + 切片

脚本：`three_stage/stage2_localize_and_cut.py`

输入：

- 原视频
- `stage1/draft_plan.json`（只读，不能改 steps）
- `stage1/frame_manifest.json`（full video 帧池）

### 模型输出 schema（`stage2/localization_raw.json`）

模型只输出索引，不输出任何解释性字段：

```json
{
  "steps": [
    {"step_id": 1, "start_frame_index": 1, "end_frame_index": 8},
    {"step_id": 2, "start_frame_index": 8, "end_frame_index": 20}
  ]
}
```

硬约束：

- top-level 只能有 `steps`，且必须覆盖所有 draft `step_id`，不允许多/少/重复
- 每条 entry 只能包含 3 个键：`step_id`, `start_frame_index`, `end_frame_index`
- 索引范围：`1..num_frames`
- `end_frame_index` 不允许输出 `num_frames + 1`；最后一步通常用 `end_last == num_frames`
- 正时长：`start_frame_index < end_frame_index`
- 单调不重叠：对相邻步骤要求 `end_i <= start_{i+1}`
- 利用 timestamp 做额外校验：若选到重复帧导致 `start_sec == end_sec` 会判为无效并要求重选

### 脚本输出（`stage2/step_segments.json` + clips）

脚本侧把索引换算成秒并切片：

- clip 视作半开区间：`[t(start_frame_index), t(end_frame_index))`
- `start_sec = t(start_frame_index)`
- `end_sec = t(end_frame_index)`

为增强鲁棒性：如果由于重复采样等极端情况导致 `end_sec <= start_sec`，脚本可能仅在**秒级裁剪**时轻微调整 `end_sec` 以保证正时长（索引不变）。

切片模式：

- `--cut-mode reencode`（默认）：混合 seek + 重编码，边界更准但更慢
- `--cut-mode copy`：直接 stream copy，更快但边界可能对齐到关键帧
- 重编码调参：`--seek-slop-sec`, `--crf`, `--preset`, `--keep-audio`

断点续跑：

- 若 `stage2/step_segments.json` 存在、clips 非空且 segments 与当前 Stage1 的 `step_id/step_goal` 完全一致，会复用（除非 `--overwrite`）。

## Stage 3 — clip 精修 + 关键帧选择

脚本：`three_stage/stage3_refine_and_keyframes.py`

输入（每个 step）：

- `stage2/step_clips/stepXX_<slug>.mp4`
- 对该 clip 采样得到的 50 帧池（`<step_folder>/frame_manifest.json`）
- draft step JSON（`step_id/step_goal` 只读）+ 全局 outline（保证跨步一致性）

输出（每个 step）：

- `<step_folder>/step_final.json`（包含 `critical_frames`）
- `<step_folder>/frame_###_ts_XX.XXs.jpg`（脚本复制关键帧 JPEG）
- `keyframe_image_path` 由脚本填入（绝对路径）
  - 其中 `ts_XX.XXs` 使用**原视频时间轴**（global timestamp），便于后续用 `extract_last_frame_segments.py` 等工具在源视频上裁剪片段

关键硬约束（校验失败自动重试）：

- 禁止修改 `step_id` 与 `step_goal`
- 必须输出 1–2 个 `critical_frames`
- `critical_frames[*].frame_index` 必须在 `1..num_frames` 且严格递增
- 文本字段禁止出现 “Frame 12 / Image 12 …” 等帧引用
- 若输出 2 个关键帧，必须体现真实时间推进：若两帧映射到相同 timestamp 会被拒绝（提示改选或只输出 1 帧）
- `affordance_hotspot` 使用字段 `mechanism`（自然语言描述物理机制）；不允许输出 `causal_role`

断点续跑：

- 若 `causal_plan_with_keyframes.json` 已存在且非空，会直接返回（除非 `--overwrite`）。
- 否则：逐步复用 `<step_folder>/step_final.json`（需通过严格检查且关键帧图片存在）。

## 最终合并产物与 schema 对齐

最终 `causal_plan_with_keyframes.json` 会合并所有步骤，字段结构对齐 ECCV 既有 long-video 输出：

- step-level：`rationale`, `preconditions`, `expected_effects`, `spatial_postconditions_detail`,
  `affordance_postconditions_detail`, `predicted_next_actions`, `tool_and_material_usage`, `failure_handling`
- keyframe-level：`critical_frames[*]` 中包含
  - `frame_index`, `action_description`, `state_change_description`
  - `spatial_preconditions`, `affordance_preconditions`
  - `causal_chain`（`agent`, `action`, `patient`, `causal_effect_on_patient`, `causal_effect_on_environment`）
  - `affordance_hotspot`（`description`, `affordance_type`, `mechanism`）
  - `keyframe_image_path`（脚本填入绝对路径）

## 运行方式

一键跑通（在 `ECCV/` 目录）：

```bash
python3 three_stage/pipeline.py \
  --input-video-dir /abs/path/to/videos \
  --output-root three_stage/causal_spafa_plan_dataset_long \
  --api-key "$API_KEY" --api-base "$API_BASE_URL" \
  --provider "$MODEL_PROVIDER_ID" --model "$MODEL_NAME"
```

只跑部分阶段：

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 1,2
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 3
```

## 调试与质量检查

- prompt 与 raw response 都在输出目录里（见上文目录结构）
- 查某个索引对应的图：
  - Stage2：看 `stage1/frame_manifest.json` + `stage1/sampled_frames/`
  - Stage3：看 `<step_folder>/frame_manifest.json` + `<step_folder>/sampled_frames/`
- 常见失败原因（会触发重试）：
  - JSON 非法 / 多余键
  - text 字段包含帧引用（Frame/Image + 数字）
  - Stage2 选到重复帧导致 `start_sec == end_sec`
  - Stage3 选到两个 timestamp 相同的关键帧
