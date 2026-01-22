# CoT 标注/生成指南（Task_28–Task_42，API-only）

本文用于把 `ECCV/three_stage/` 三阶段产物（`causal_plan_with_keyframes.json`）通过 OpenAI-compatible API **直接生成** planning 强相关（Task_28–Task_42） 的 CoT(JSONL) 数据集。

说明：

- 本目录只保留 **API 生成版**（不包含任何基于规则/模板的 CoT 提取逻辑）。
- 生成的 `conversations`（human/gpt）默认使用英文提示与回答（见 `ECCV/cot/generate_cot_dataset_api.py` 的系统提示词）。

## 1) 输入目录（per video）

每个视频 item 的关键结构如下（仅列出 CoT 生成会用到的文件/目录）：

```
<input_root>/<video_id>/
  causal_plan_with_keyframes.json
  stage1/sampled_frames/               # 可选：全局均匀抽帧（来自 Stage1）
  cumulative_last_frame_segments/      # 可选：prefix mp4（若开启 prefix 任务）
  01_<slug>/ 02_<slug>/ ...            # step folders
    step_meta.json
    frame_###_ts_XX.XXs.jpg            # step clip 的关键帧图（对应 critical_frames[*].frame_index）
```

要求：

- `causal_plan_with_keyframes.json` 必须严格符合 `ECCV/three_stage/prompts.py` 的 schema，否则该 video 会被跳过（避免生成低质量 CoT）。

## 2) 生成命令（API-only）

建议先对三阶段产物做一次先验校验（强建议）：

```bash
python3 ECCV/three_stage/validate_three_stage_output.py --video-output-dir <input_root>/<video_id>
```

然后生成 CoT(JSONL)：

```bash
python3 ECCV/cot/generate_cot_dataset_api.py \
  --input-root <input_root> \
  --output-dir <cot_root> \
  --api-base http://model.mify.ai.srv/v1 \
  --provider <MODEL_PROVIDER_ID> \
  --model <MODEL_NAME> \
  --api-key <API_KEY> \
  --seed 42 \
  --post-validate
```

常用可选参数：

- `--tasks`：逗号分隔的 Task 名称子集（默认 Task_28–Task_42 全部生成）
- `--require-video-prefix`：如果没有 `cumulative_last_frame_segments/*.mp4`，则跳过 prefix 相关样本（更保守、更一致）
- `--abs-paths`：输出 `image/video/source_path` 为绝对路径
- `--max-sample-attempts`：单条样本 API 重试次数（输出不满足格式/泄漏约束会自动重试）
- `--max-tokens`：单条样本最大输出 tokens（CoT 被约束为单段落，通常不需要很大）

## 3) 硬约束（数据质量）

- `conversations[*].value` 文本禁止泄漏任何索引/路径信息：`frame_### / sample_### / ts_... / .jpg/.mp4 / Frame 12 / Image 12` 等。
- 证据路径只允许出现在条目字段 `image` / `video` / `meta.evidence_files` 中。
- `conversations[0].value`（human）必须是**单行**自然问句：不包含任何 `fields.*` 上下文行；不包含 options/candidates 等列表；不在 Q 中加入如何生成 CoT 的引导语。
- `conversations[1].value`（assistant）必须是 `<think>...</think>` + 原始答案，并满足以下要求：
  - `<think>...</think>` 内必须是**一段**自然语言（禁止换行；禁止 bullet list）。
  - CoT 内容必须按**因果规划**顺序组织：先分析 spatial/affordance 的 preconditions，再分析 spatial/affordance 的 effects，最后做 failure_reflecting（失败原因 + 恢复策略），并自然过渡到“为什么答案成立”。
  - `</think>` 之后必须紧跟**原始答案文本**（不得添加 `Answer:` 前缀；不得改写答案）。
  - 生成器会从 `causal_plan_with_keyframes.json` 的 `causal_chain.*` 与 `failure_reflecting.*` 自动抽取若干句“必须逐字包含”的锚点句，用于强约束 grounded 的 planning+reflecting（输出不满足会自动重试；仍失败则丢弃该样本）。

## 4) 质量策略（重试/丢弃规则）

- 对每个样本：生成器会解析 API 返回的 JSON，并做严格校验（字段、`<think>` 单段落、禁止泄漏、必须包含锚点句、答案原样拷贝等）。
- 若输出不满足约束：自动重试（`--max-sample-attempts` 控制上限）；达到上限仍失败则**丢弃该样本**（避免混入低质量/不一致数据）。
- 若输入三阶段产物 schema 不合格：直接跳过该 video（避免“垃圾输入 → 垃圾输出”）。

## 5) 校验（推荐）

生成后建议跑严格校验器：

```bash
python3 ECCV/cot/validate_cot_dataset.py \
  --input-root <input_root> \
  --cot-root <cot_root> \
  --strict
```
