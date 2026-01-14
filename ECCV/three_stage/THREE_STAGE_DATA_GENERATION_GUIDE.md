# 三阶段长视频数据生成：代码/Prompt 核验要点 + 从零跑通指南

本文面向 `ECCV/three_stage/` 目录下的三阶段管线（Stage1 Draft → Stage2 定位/切片 → Stage3 精修+关键帧），目标是：

- 让你能**从零跑通**并稳定产出高质量 `causal_plan_with_keyframes.json`
- 明确每一阶段的**输入/输出/索引语义**
- 把 Prompt 的约束与代码里的**解析/归一化/强校验**逐条对齐，便于排错与质量把关

> 完整规范见：`three_stage/THREE_STAGE_PIPELINE.md`。本文更偏“核验清单 + Runbook（操作手册）”。
> 与两阶段（`mani_longvideo.py`）的对齐审计见：`three_stage/THREE_STAGE_AUDIT_VS_TWO_STAGE.md`。

---

## 0. 你最终会得到什么（产物与可追溯性）

对每个视频 `<video_id>`，默认输出到：

`ECCV/three_stage/causal_spafa_plan_dataset_long/<video_id>/`

核心产物：

- `causal_plan_with_keyframes.json`：最终计划 + 关键帧标注（用于下游 QA/监督生成）
- `run_summary.json`：运行元信息（源视频路径、模型信息、每阶段状态、提示词落盘路径等）

可追溯性产物（每阶段都会落盘）：

- `stage*_system_prompt.txt` / `stage*_user_prompt.txt`：当次实际使用提示词（可复现实验）
- `stage*_raw_response.txt`：原始模型输出（便于排查 JSON/schema 偏差）
- `frame_manifest.json`：该阶段使用的帧池索引与时间戳映射（**索引空间的权威来源**）

兼容性产物（对齐既有两阶段工具链的默认路径）：

- `<video_id>/sampled_frames/`：full video 的 50 帧（默认创建为到 `stage1/sampled_frames` 的相对 symlink；失败则复制）
- `<video_id>/frame_manifest.json`：full video manifest（复制自 `stage1/frame_manifest.json`）

---

## 1. 三阶段目录结构（你应该看到的文件）

单视频目录（示例）：

```text
<video_id>/
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

---

## 2. 索引空间与语义（最关键、最容易踩坑）

### 2.1 两类帧池（两个 index space）

1) **full video 帧池（Stage1/Stage2 共用）**

- manifest：`stage1/frame_manifest.json`
- 帧图像：`stage1/sampled_frames/sample_###_ts_XX.XXs.jpg`
- 这个帧池用于：Stage1 生成 Draft；Stage2 定位 step 边界

2) **step clip 帧池（Stage3 每个 step 单独一套）**

- manifest：`<step_folder>/frame_manifest.json`
- 帧图像：`<step_folder>/sampled_frames/sample_###_ts_XX.XXs.jpg`
- 这个帧池用于：Stage3 选择关键帧 `critical_frames[*].frame_index`

> 重要：`stage1/frame_manifest.json` 与 `<step_folder>/frame_manifest.json` 是**不同的索引空间**，不要混用。

### 2.2 Stage2 边界索引（exclusive end）

Stage2 只输出边界索引：

- `start_frame_index`：inclusive
- `end_frame_index`：exclusive（半开区间右边界）

脚本切片语义：

- clip = `[t(start_frame_index), t(end_frame_index))`
- `end_frame_index` 仍被约束在 `1..num_frames`（不允许 `num_frames+1`）

对应代码：`three_stage/common.py:validate_stage2_localization()` + `three_stage/stage2_localize_and_cut.py`

### 2.3 Stage3 关键帧索引（per-clip 1-based）

Stage3 输出：

- `critical_frames[*].frame_index`：**1-based**，且**只对当前 step clip 的帧池**有效

对应代码：

- Prompt：`three_stage/prompts.py:build_stage3_user_prompt()`
- 校验/归一化：`three_stage/common.py:normalize_stage3_step_output()`
- 关键帧图片落盘：`three_stage/common.py:save_keyframe_images_from_manifest()`

---

## 3. Prompt 与代码强校验：对齐核验清单（逐阶段）

### 3.1 Stage1（Draft，无关键帧字段）

Prompt 定义：

- system：`three_stage/prompts.py:SYSTEM_PROMPT_ANALYST`
- user：`three_stage/prompts.py:build_stage1_user_prompt()`

关键约束（Prompt）→ 对应实现（代码）：

- **只允许 strict JSON 输出** → `three_stage/common.py:extract_json_from_response()` + `json.loads()`
- **禁止 `critical_frames/frame_index/interaction/keyframe_image_path`**（任何位置） → `three_stage/stage1_generate_draft.py:_stage1_raw_schema_errors()`（递归扫描 forbidden keys）
- **禁止额外字段（top-level/step/nested）** → `three_stage/stage1_generate_draft.py:_stage1_raw_schema_errors()`（allowed-key 白名单）
- **步数 3–8（推荐 4–7）** → `three_stage/stage1_generate_draft.py:_draft_hard_errors()`（硬约束）+ warnings（软提醒）
- **字段非空、列表非空（`causal_chain` 四个 list + `counterfactual_*` + `failure_reflecting`）** → `three_stage/stage1_generate_draft.py:_draft_hard_errors()`
- **文本不得出现 “Frame 12 / Image 12 …”** → `three_stage/common.py:_contains_frame_ref()`（正则）+ `_draft_hard_errors()`

断点续跑一致性：

- `three_stage/stage1_generate_draft.py:_can_resume_stage1()` 会检查缓存产物存在且通过严格校验才复用

### 3.2 Stage2（定位边界 + 切片）

Prompt 定义：

- user：`three_stage/prompts.py:build_stage2_user_prompt()`

关键约束（Prompt）→ 对应实现（代码）：

- **top-level 只能有 `steps`** → `three_stage/common.py:validate_stage2_localization()`（extra_top 判错）
- **每条 entry 只能有 `step_id/start_frame_index/end_frame_index`** → 同上（allowed_entry_keys）
- **索引范围 `1..num_frames`；禁止 `num_frames+1`** → 同上（range 判错）
- **正时长 `start < end`、单调不重叠 `end_i <= start_{i+1}`** → 同上（index + timestamp 双重校验）
- **重复帧/重复 timestamp 导致的“零时长”会被拒绝并触发重试** → 同上（timestamp 校验）

切片实现：

- `three_stage/stage2_localize_and_cut.py` 将索引映射到秒并用 ffmpeg 裁剪
- 推荐默认 `--cut-mode reencode` 以减少关键帧吸附导致的边界漂移

### 3.3 Stage3（精修 + 关键帧）

Prompt 定义：

- user：`three_stage/prompts.py:build_stage3_user_prompt()`

关键约束（Prompt）→ 对应实现（代码）：

- **`step_id/step_goal` 不可改** → `three_stage/common.py:normalize_stage3_step_output()`（强一致校验）
- **`critical_frames` 必须恰好 2 个，且 `frame_index` 递增** → 同上（长度与递增校验）
- **所有文本字段禁止出现 “Frame 12 …”** → 同上（`_contains_frame_ref` 全字段扫描）
- **`interaction.hotspot.description/affordance_type/mechanism` 必须非空** → 同上（必填判错）
- **`causal_chain` 形状以 `three_stage/prompts.py` 为准**（含 `causal_precondition_on_*` / `causal_effect_on_*` 列表） → 同上（allowed keys + list 非空判错）
- **不允许输出 `keyframe_image_path`**（脚本从文件系统解析关键帧 JPEG） → 同上（extra_cf 判错）

关键帧图片落盘（不写回 JSON）：

- `three_stage/stage3_refine_and_keyframes.py` 会把选中的帧复制为 `frame_###_ts_XX.XXs.jpg`
- 关键帧 JPEG 的定位由 `<step_folder>/frame_manifest.json` + `frame_index` 决定，JSON 不包含 `keyframe_image_path`

旧产物轻量升级（无模型调用）：

- 若历史产物里出现 `keyframe_image_path`（旧版本脚本注入的路径），Stage3 会在断点续跑检查时自动清理并按新 schema 重新落盘

---

## 4. 从零跑通：最小可用流程（推荐）

### 4.1 安装依赖

- Python 3（命令使用 `python3`）
- `ffmpeg` 在 `PATH`（或用 `--ffmpeg-bin` 指定）
- Python 包：

```bash
pip install openai opencv-python
```

### 4.2 配置模型 API（推荐用环境变量）

```bash
export API_KEY="..."
export API_BASE_URL="http://model.mify.ai.srv/v1"
export MODEL_PROVIDER_ID="vertex_ai"
export MODEL_NAME="gemini-3-pro-preview"
export MAX_TOKENS="30000"
```

### 4.3 一键跑全流程（单视频/多视频）

在 `ECCV/` 目录下：

```bash
# 单视频
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4

# 多视频（目录）
python3 three_stage/pipeline.py --input-video-dir /abs/path/to/videos
```

常用参数：

- `--output-root three_stage/causal_spafa_plan_dataset_long`
- `--max-frames 50`（建议保持 ≤ 50）
- `--overwrite`（强制重跑，默认断点续跑）
- `--stages 1,2,3`（只跑子集，如 `--stages 1,2`）
- `--cut-mode reencode|copy`（Stage2 切片）

---

## 5. 质量检查（强烈建议按顺序做）

### 5.1 先看 `run_summary.json`

确认：

- `source_video` 是否正确可访问
- `stage1/stage2/stage3` 是否 `completed`
- 是否存在 warnings（Stage1 可能提示 step 数量、goal 过长等）

### 5.2 Stage1：抽帧是否合理

检查：

- `stage1/frame_manifest.json` 里 `num_frames == --max-frames`
- `stage1/sampled_frames/` 是否有对应数量的 jpg
- 若视频很短导致大量重复帧：Stage2/3 更容易失败，需要人工抽检或换更合理的切片策略

### 5.3 Stage2：定位边界是否单调、clip 是否非空

检查：

- `stage2/localization_raw.json` 是否只包含 `steps`
- `stage2/step_segments.json` 中每个 segment 是否满足：
  - `start_frame_index < end_frame_index`
  - `start_sec < end_sec`
- `stage2/step_clips/*.mp4` 是否存在且文件大小 > 0

### 5.4 Stage3：关键帧与机制字段是否齐全

检查：

- 每个 step 目录的 `step_final.json`：
  - `critical_frames` 长度必须为 2
  - 每个 `critical_frames[*].frame_index` 都应在 step 目录根下找到对应关键帧 jpg：`frame_{idx:03d}_ts_{timestamp:.2f}s.jpg`
  - 关键帧 jpg 文件名里的 `ts_XX.XXs` 是**原视频时间轴**（global timestamp），可直接用于后续在源视频上裁剪片段
  - `interaction.hotspot.mechanism` 非空（否则机制解释缺失）
  - `causal_chain` 形状以 `three_stage/prompts.py` 为准（含 `causal_precondition_on_*` / `causal_effect_on_*` 列表）

---

## 6. 常见失败模式与处理建议

### 6.1 Stage1 一直重试失败

高频原因：

- 模型输出带了多余字段/markdown/解释文字
- 输出包含 forbidden keys（`critical_frames`/`frame_index`/`interaction`/`keyframe_image_path`）
- 必填字段为空或列表为空

排查：

- 看 `stage1/stage1_raw_response.txt` 与 `stage1_*prompt.txt`
- 错误会写入下一轮 prompt 的前缀（`build_retry_prefix`）

### 6.2 Stage2 定位不稳定 / 报“timestamp 相同”

高频原因：

- full video 抽帧重复（短视频/解码失败导致 padding）

建议：

- 优先保证视频本身可正常解码（cv2/ffmpeg）
- 必要时降低 `--max-frames` 或换更合适的视频源

### 6.3 Stage3 报 “mechanism 为空 / extra keys”

建议：

- 这是刻意的强约束：为了保证下游数据可用且 schema 稳定
- 看 `stage3_raw_response.txt`，一般是模型未严格按 schema 输出；重试会自动纠正

---

## 7. 旧三阶段产物轻量升级（尽量不重跑模型）

如果你之前用旧版本三阶段输出过 `keyframe_image_path`（脚本注入的路径）：

在 `ECCV/` 下对目标视频重新跑一次 Stage3（通常不会触发模型调用，会直接复用并升级）：

```bash
python3 three_stage/pipeline.py --input-video /abs/path/video.mp4 --stages 3
```

升级会重写（清理 `keyframe_image_path` 等旧字段注入）：

- `<step_folder>/step_final.json`
- `<video_id>/causal_plan_with_keyframes.json`

---

## 8. 下游使用提示（QA/监督生成）

若你使用 `generate_mani_longvideo_qa_api.py`：

- 下游通常会读取 `steps[*].critical_frames[*].interaction.hotspot.mechanism`
- 因此 Stage3 必须保证 `mechanism` 非空且为自然语言机制描述（本管线已用强校验保证）
