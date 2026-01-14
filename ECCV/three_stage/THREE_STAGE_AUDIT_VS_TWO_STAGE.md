# 三阶段 vs 两阶段（`mani_longvideo.py`）对齐审计：Prompt/校验/产物兼容性

本文是对 `ECCV/three_stage/` 三阶段管线与既有两阶段（以 `ECCV/mani_longvideo.py` 为代表）在**prompt 约束、代码校验、最终产物 schema 与下游工具链兼容性**上的对齐审计说明。

结论先行：

- 三阶段最终产物 `causal_plan_with_keyframes.json` 的 schema **以 `ECCV/three_stage/prompts.py` 为准**：`interaction.hotspot.mechanism`（必填）+ `causal_chain`（含 `causal_precondition_on_*` / `causal_effect_on_*` 列表）+ `critical_frames`（长度=2）。
- 三阶段关键帧图片文件名中的 `ts_XX.XXs` 已明确为**原视频时间轴（global timestamp）**，可直接被 `extract_last_frame_segments.py` / `extract_cumulative_last_frame_segments.py` 用于源视频裁剪。
- 三阶段同时生成 `<video_id>/sampled_frames/` 与 `<video_id>/frame_manifest.json` 兼容路径（默认指向/复制自 Stage1），便于复用既有“按根目录 sampled_frames”组织的工具链与数据规约。

---

## 1. 两阶段管线（`mani_longvideo.py`）关键特征（作为对照基准）

两阶段的核心结构：

1) Stage1（Planning）：在 full video 的 50 帧池上生成包含 `critical_frames` 的计划 JSON（关键帧内容是文本描述，不含 `frame_index`）。
2) Stage2（Frame Selection）：在同一 full video 50 帧池上，为每个 `critical_frames[*]` 选择 `frame_index`，再把关键帧图片保存为 `frame_{idx}_ts_{timestamp}.jpg` 并写回 `keyframe_image_path`。

对下游生态“最关键的事实”：

- `affordance_hotspot.mechanism` 被大量下游任务 prompt/脚本直接消费（例如 `generate_mani_longvideo_qa_api.py` 的 Task_03/06/17 等）。
- `causal_chain` 的 canonical 字段是：`agent/action/patient/causal_effect_on_patient/causal_effect_on_environment`（两阶段 dataclass 解析依赖这一形状）。
- 关键帧图片文件名中的 `ts_*.s` 来自**原视频时间轴**（`timestamp_sec = original_frame_index / fps`），后续可以基于它在源视频上裁剪片段。
- 根目录存在 `sampled_frames/`（full video 抽帧）是历史数据与不少工具链默认假设。

---

## 2. 三阶段管线的设计差异（为什么不等价、但要“对齐可用”）

三阶段的核心结构：

1) Stage1（Draft）：仅 step-level（不产出关键帧字段），输出 `stage1/draft_plan.json`。
2) Stage2（Localize/Cut）：在 full video 50 帧池上定位每步边界（`start_frame_index/end_frame_index`），并在原视频上裁剪 step clip。
3) Stage3（Refine+Keyframes）：在每个 step clip 的 50 帧池上精修该步并选 **2 个**关键帧，输出最终 `causal_plan_with_keyframes.json`。

不可避免的差异点：

- **索引空间不再唯一**：两阶段只有一个 full-video 帧池；三阶段存在 full-video 帧池（Stage1/2）与 per-clip 帧池（Stage3）。因此三阶段的 `critical_frames[*].frame_index` **按设计是 per-clip 的 1-based**。

为了与既有生态“对齐可用”，三阶段做了关键对齐：

- 下游主要依赖的是关键帧 jpg 文件（`frame_###_ts_XX.XXs.jpg`）及其 `ts_XX.XXs`（可裁剪的全局时间），而不是把 `frame_index` 直接当作 full-video `sampled_frames/` 的索引。
- 三阶段将关键帧图片的 timestamp 改为 **global timestamp**（通过 Stage2 的 `start_sec` 偏移），从而可继续用旧的裁剪工具与证据形态（segment mp4）。

---

## 3. Prompt 与代码校验：逐阶段对齐检查（含两阶段参考点）

### 3.1 Stage1（Draft）——与两阶段 Stage1 的主要差异与对齐

两阶段 Stage1：

- 允许输出 `critical_frames`（但不允许输出 `frame_index/keyframe_image_path`）。

三阶段 Stage1（设计差异）：

- **完全禁止**输出任何 keyframe-level 字段（`critical_frames/frame_index/interaction/keyframe_image_path`），原因是关键帧选择被推迟到 Stage3（clip 级别更可靠）。

对齐保障（prompt ⇄ 代码）：

- Prompt 明确声明禁止 keyframe 字段：`three_stage/prompts.py:build_stage1_user_prompt()`
- 代码对原始模型输出做递归 forbidden keys 扫描 + 白名单 schema 校验：`three_stage/stage1_generate_draft.py:_stage1_raw_schema_errors()`
- 代码对核心字段做硬约束（非空、列表长度、无帧引用等）：`three_stage/stage1_generate_draft.py:_draft_hard_errors()`

### 3.2 Stage2（Localize/Cut）——参考两阶段“严格 JSON only + 强一致性”策略

两阶段 Stage2（Frame Selection）：

- 在同一个帧池内为 `critical_frames` 选择 `frame_index`，并可能输出额外检查字段（confidence/checks）。

三阶段 Stage2（Localize）：

- 只输出 `{step_id, start_frame_index, end_frame_index}`（不允许任何额外字段），并强制 1-based、end exclusive、单调不重叠、正时长。

对齐保障（prompt ⇄ 代码）：

- Prompt 只允许一个 top-level `steps` 且每条 entry 只有 3 个键：`three_stage/prompts.py:build_stage2_user_prompt()`
- 代码强校验并在失败时把 error list 注入下一轮重试：`three_stage/common.py:validate_stage2_localization()` + `build_retry_prefix()`
- 切片采用 ffmpeg，默认 reencode（更接近两阶段“时间对齐优先”的目标）：`three_stage/common.py:cut_video_segment_ffmpeg()`

### 3.3 Stage3（Refine+Keyframes）——schema 差异与可用性对齐

两阶段最终产物的关键 schema（下游常见依赖）：

- `critical_frames[*].affordance_hotspot.mechanism`
- `critical_frames[*].causal_chain`（5 字段 canonical）
- `keyframe_image_path`

三阶段 Stage3（当前版本，以 `ECCV/three_stage/prompts.py` 为准）：

- `critical_frames` 长度必须为 2，且每个包含 `interaction.hotspot.mechanism`
- `causal_chain` 使用 `causal_precondition_on_*` / `causal_effect_on_*` 列表（不再是 5 字段 canonical）
- JSON 不包含 `keyframe_image_path`；关键帧 jpg 以 `frame_###_ts_XX.XXs.jpg` 落盘（global timestamp）

对齐保障（prompt ⇄ 代码）：

- Prompt 严格 schema：`three_stage/prompts.py:build_stage3_user_prompt()`
- 归一化/强校验：`three_stage/common.py:normalize_stage3_step_output()`
- 关键帧 jpg 落盘：`three_stage/common.py:save_keyframe_images_from_manifest()` + `three_stage/stage3_refine_and_keyframes.py`
- **时间戳语义对齐两阶段**：Stage3 将 clip-local timestamp 转为 global timestamp 后写入 `<step_folder>/frame_manifest.json` 并用于 keyframe 命名

---

## 4. 下游工具链兼容性核验（两阶段生态的“硬依赖点”）

### 4.1 关键帧裁剪工具（必须用 global timestamp）

以下脚本通过解析 `frame_*_ts_XX.XXs.jpg` 的时间戳来切源视频：

- `ECCV/extract_last_frame_segments.py`
- `ECCV/extract_cumulative_last_frame_segments.py`

三阶段已保证：

- 每个 step 目录根下存在关键帧图片 `frame_###_ts_XX.XXs.jpg`（JSON 不包含 `keyframe_image_path`）
- 图片文件名里的 `ts_XX.XXs` 是原视频时间轴（global timestamp）

### 4.2 QA 生成脚本（依赖 mechanism 与 causal_chain）

例如 `ECCV/generate_mani_longvideo_qa_api.py` 会读取：

- `interaction.hotspot.mechanism`
- `causal_chain`（含 `causal_precondition_on_*` / `causal_effect_on_*` 列表）

三阶段已保证：

- `mechanism` 必填且非占位符
- `causal_chain` 形状严格按 `ECCV/three_stage/prompts.py`；若下游仍假设“两阶段 canonical 5 字段”，需要同步更新解析逻辑

### 4.3 根目录 `sampled_frames/` 兼容路径

两阶段输出默认在 `<video_dir>/sampled_frames/` 放 full video 抽帧。

三阶段已提供兼容路径：

- `<video_id>/sampled_frames/`（默认创建为指向 `stage1/sampled_frames` 的相对 symlink；失败则复制）
- `<video_id>/frame_manifest.json`（复制自 `stage1/frame_manifest.json`）

这样可最大化复用“按根目录 sampled_frames”写死路径的工具/规约。

---

## 5. 建议的使用方式（保证高质量与高可复现性）

优先一键跑通（默认断点续跑）：

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4
```

建议默认参数：

- `--max-frames 50`（与两阶段一致）
- Stage2 `--cut-mode reencode`（推荐，避免 copy 模式的关键帧吸附导致时间漂移）

强制重跑（用于 prompt/代码更新后的再生成）：

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --overwrite
```

只做“旧产物升级/对齐修复”（通常不会触发模型调用）：

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 3
```
