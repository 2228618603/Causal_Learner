# 三阶段长视频数据生成管线（Draft → 定位/切片 → 精修 + 关键帧）

本目录提供一套**独立**的三阶段长视频数据生成管线（VLM + 严格 JSON 校验）。

完整规范与细节说明见：[`THREE_STAGE_PIPELINE.md`](THREE_STAGE_PIPELINE.md)。
从零跑通指南（含 prompt/校验对齐核验清单）见：[`THREE_STAGE_DATA_GENERATION_GUIDE.md`](THREE_STAGE_DATA_GENERATION_GUIDE.md)。

## 这套管线解决什么问题？

长视频直接“一次性看全+输出全量标注”往往会遇到：

- VLM 输入图片上限（通常 ≤ 50 张）导致难以覆盖全视频
- 单次输出过长、格式容易跑偏，难以保证 JSON 与 schema 稳定
- 关键帧选择缺乏约束，导致低质量或不可复现

三阶段的核心思路是：先生成**粗计划** → 再定位每步时间段并切 clip → 最后在 clip 上做**精修 + 关键帧**，并在每一步用严格校验与自动重试保证质量与可行性。

## 中文速览（最常用、最容易踩坑的点）

- 从 `ECCV/` 目录运行：`python3 three_stage/pipeline.py --input-video /abs/path/video.mp4`
- 默认输出目录：`ECCV/three_stage/causal_spafa_plan_dataset_long/<video_id>/`
- Stage2 的索引：基于 full video 的 50 帧池，**1-based**；`end_frame_index` 是 **exclusive 边界**
- Stage2 的索引范围：`start_frame_index ∈ 1..num_frames`，`end_frame_index ∈ 2..num_frames+1`（最后一步通常 `end_last == num_frames+1` 覆盖到视频末尾）
- Stage3 的 `critical_frames[*].frame_index`：基于**每个 step clip 自己的** 50 帧池，**1-based**
- 每个 stage 都会落盘：prompt、raw response、manifest、run_summary，方便追溯与 Debug

## 三个阶段概览

1) **Stage 1 — Draft（只做 step-level，不做 keyframes）**
- 输入：原视频 → 均匀采样（默认 50 帧）+ prompt
- 输出：`stage1/draft_plan.json`（严格禁止 `critical_frames/frame_index/interaction/keyframe_image_path`）
- 步数：推荐 4–7；硬约束 3–8（校验失败自动重试）

2) **Stage 2 — 步骤定位 + 切片**
- 输入：原视频 + `stage1/draft_plan.json` + `stage1/frame_manifest.json`
- 模型输出：每步 `{start_frame_index, end_frame_index}`（1-based，`end` 为 exclusive）
- 脚本输出：
  - `stage2/step_segments.json`
  - `stage2/step_clips/stepXX_<slug>.mp4`
- 约束：单调不重叠（`end_i <= start_{i+1}`）且正时长（`start < end`）

3) **Stage 3 — clip 精修 + 关键帧**
- 输入：每步 clip → 再采样（默认 50 帧/clip）+ draft step（只读）
- 输出：最终 `causal_plan_with_keyframes.json`（三阶段 schema 以 `three_stage/prompts.py` 为准）
  - `critical_frames[*].frame_index`：1-based on **step-clip** 帧池
  - 关键帧 JPEG 会被保存到每个 step 目录根下：`frame_{idx:03d}_ts_{timestamp:.2f}s.jpg`（JSON 不包含 `keyframe_image_path`）

## 快速开始

### 依赖

- Python 3
- `ffmpeg` 在 `PATH`（或通过 `--ffmpeg-bin` 指定）
- Python 包：

```bash
pip install openai opencv-python
```

### API 配置

可通过命令行参数或环境变量传入（强烈建议用环境变量，不要写死在代码里）：

- `API_KEY`
- `API_BASE_URL`
- `MODEL_PROVIDER_ID`
- `MODEL_NAME`

### 一键跑通（推荐）

在 `ECCV/` 目录下：

```bash
python3 three_stage/pipeline.py \
  --input-video-dir /abs/path/to/videos \
  --output-root three_stage/causal_spafa_plan_dataset_long \
  --api-key "$API_KEY" --api-base "$API_BASE_URL" \
  --provider "$MODEL_PROVIDER_ID" --model "$MODEL_NAME"
```

在 repo 根目录（等价写法）：

```bash
python3 ECCV/three_stage/pipeline.py \
  --input-video-dir /abs/path/to/videos \
  --output-root ECCV/three_stage/causal_spafa_plan_dataset_long \
  --api-key "$API_KEY" --api-base "$API_BASE_URL" \
  --provider "$MODEL_PROVIDER_ID" --model "$MODEL_NAME"
```

### 只跑部分阶段

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 1,2
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 3
```

### 单独跑某一阶段

```bash
python3 three_stage/stage1_generate_draft.py --input-video /abs/path/video.mp4
python3 three_stage/stage2_localize_and_cut.py --input-video /abs/path/video.mp4
python3 three_stage/stage3_refine_and_keyframes.py --input-video /abs/path/video.mp4
```

## 常用参数（影响质量/速度/可复现性）

- `--overwrite`：强制重跑（默认会“断点续跑”，只有当缓存产物通过严格检查才会跳过模型调用）
- `--continue-on-error`：批量跑目录时，单个视频失败不阻塞后续视频（失败会写入该视频的 `run_summary.json`）
- `--stages 1,2,3`：选择运行阶段子集
- `--post-validate`：在 Stage3 成功后自动对 `<video_id>/` 做一次输出校验（等价于运行 `validate_three_stage_output.py`）
- `--max-frames`：默认 50；建议保持 `<= 50`（与常见 VLM 图片限制匹配）
- `--temperature`：采样温度（默认 0.2；更低通常更稳、更易输出严格 JSON）
- `--api-call-retries/--api-call-retry-backoff-sec`：单次模型请求的重试次数与退避（用于应对偶发网络/服务错误）
- `--cut-mode reencode|copy`：Stage2 切片模式（默认 `reencode` 更准，`copy` 更快但边界可能 keyframe-snapped）
- `--no-embed-index`：不在 Stage2/Stage3 输入图上叠加 “Frame XX”（Stage1 始终避免 index artifact）
- `--stage1-retries/--stage2-retries/--stage3-retries`：每阶段最大重试次数

## 质量与可追溯性（你该看哪些文件）

- prompt 与 raw response：
  - Stage1：`stage1/stage1_system_prompt.txt`, `stage1/stage1_user_prompt.txt`, `stage1/stage1_raw_response.txt`
  - Stage2：`stage2/stage2_system_prompt.txt`, `stage2/stage2_user_prompt.txt`, `stage2/stage2_raw_response.txt`
  - Stage3：每步目录下 `stage3_*`
- 索引与时间映射：
  - full video：`stage1/frame_manifest.json`
  - step clip：`<step_folder>/frame_manifest.json`
- Step 级元信息（用于断点续跑一致性与校验）：
  - `step_meta.json`（clip 路径 + start/end 秒 + 帧池信息）
- 运行元信息：`run_summary.json`

## 输出核验（推荐跑完后做一次）

```bash
python3 three_stage/validate_three_stage_output.py --video-output-dir /abs/path/to/<video_id> --check-deps
```

## 关于产物

默认输出目录下会生成大量 JSON/图片/视频片段，它们是**运行产物**而不是源码，一般不建议提交到 Git。可在需要发布数据集时单独处理。 
