# 三阶段数据生成管线（详细说明 + 产物示例）

本文完整介绍 `ECCV/three_stage/` 目录下的三阶段数据生成管线：**Stage1 Draft → Stage2 Localize/Cut → Stage3 Refine+Keyframes**。

目标：

- 稳定生成高质量、可追溯的 `causal_plan_with_keyframes.json`
- 最终 JSON **严格对齐** `ECCV/three_stage/prompts.py` 中定义的 schema（字段白名单 + 强校验 + 自动重试）

说明：为便于阅读，下面示例会把帧池规模缩小到 6–8 帧；实际默认通常为 `--max-frames 50`，结构与字段完全一致。

---

## 0. 一句话总览

1) **Stage1**：看 full video 的均匀抽帧 → 生成“只含 step-level 字段”的草案计划（不允许任何关键帧字段）。  
2) **Stage2**：在同一 full video 帧池上，为每个 step 预测时间边界索引 → 生成 step 片段元数据并用 ffmpeg 切出每个 step clip。  
3) **Stage3**：对每个 step clip 再抽帧 → 精修该 step 的完整标注，并选择 **2 个关键帧**（仅输出 `frame_index`，关键帧 JPEG 由脚本落盘）。  

---

## 1. 如何运行（入口）

三阶段总入口：

```bash
python3 ECCV/three_stage/pipeline.py --input-video /abs/path/video.mp4
```

常用参数（与 `pipeline.py` 一致）：

- `--output-root`：输出根目录（默认：`ECCV/three_stage/causal_spafa_plan_dataset_long`）
- `--max-frames`：抽帧上限（默认 50）
- `--stages 1,2,3`：只跑某些阶段（例如 `--stages 1,2`）
- `--overwrite`：强制重跑（默认会复用“已存在且通过严格校验”的缓存产物）

---

## 2. 输出目录结构（单视频）

对每个视频 `<video_id>`，输出目录为：

`<output_root>/<video_id>/`

典型结构如下：

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
      step01_<slug>.mp4
      step02_<slug>.mp4
      ...
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
  02_<slug>/
    ...
  causal_plan_with_keyframes.json
  run_summary.json
```

其中：

- `stage1/`：full video 帧池与草案计划（只含 step-level 字段）
- `stage2/`：step 边界索引（模型输出）+ step clip（ffmpeg 切片）+ step 片段元数据
- `01_<slug>/` 等 step 目录：clip 帧池、该步最终 JSON、关键帧 JPEG、该步元信息
- `causal_plan_with_keyframes.json`：最终合并产物（**schema 以 `prompts.py` 为准**）

---

## 3. Stage1（Draft）：生成 step-level 草案计划

### 3.1 Stage1 做什么？

输入：

- 原始视频
- 从视频均匀采样得到的 full video 帧池（默认最多 50 帧）
- `ECCV/three_stage/prompts.py:build_stage1_user_prompt()` 生成的提示词

模型任务：

- 输出**整个视频**的高层目标 `high_level_goal`
- 输出 `steps` 列表（通常 3–8 步）
- 为每个 step 填写 step-level 的因果链与挑战问题等字段

硬约束（最关键的）：

- **严禁**输出任何关键帧相关字段：`critical_frames` / `frame_index` / `interaction` / `keyframe_image_path`
- 输出必须是严格 JSON（只允许 schema 中出现的 key）

产出（Stage1 输出文件）：

- `stage1/frame_manifest.json`：full video 帧池索引与时间戳映射（权威索引空间）
- `stage1/sampled_frames/sample_###_ts_XX.XXs.jpg`：抽帧图像
- `stage1/draft_plan.json`：草案计划（严格 step-level schema）
- `stage1/stage1_*prompt.txt`、`stage1/stage1_raw_response.txt`：可追溯性文件

### 3.2 Stage1 产物示例

#### (1) `stage1/frame_manifest.json`（示例）

```json
{
  "num_frames": 8,
  "note": "frame_index_1based is the 1-based index used in prompts and model outputs for this frame pool.",
  "frames": [
    {
      "frame_index_1based": 1,
      "timestamp_sec": 0.0,
      "original_frame_index": 0,
      "image_relpath": "sampled_frames/sample_001_ts_0.00s.jpg"
    },
    {
      "frame_index_1based": 2,
      "timestamp_sec": 1.0,
      "original_frame_index": 30,
      "image_relpath": "sampled_frames/sample_002_ts_1.00s.jpg"
    },
    {
      "frame_index_1based": 3,
      "timestamp_sec": 2.0,
      "original_frame_index": 60,
      "image_relpath": "sampled_frames/sample_003_ts_2.00s.jpg"
    },
    {
      "frame_index_1based": 4,
      "timestamp_sec": 3.0,
      "original_frame_index": 90,
      "image_relpath": "sampled_frames/sample_004_ts_3.00s.jpg"
    },
    {
      "frame_index_1based": 5,
      "timestamp_sec": 4.0,
      "original_frame_index": 120,
      "image_relpath": "sampled_frames/sample_005_ts_4.00s.jpg"
    },
    {
      "frame_index_1based": 6,
      "timestamp_sec": 5.0,
      "original_frame_index": 150,
      "image_relpath": "sampled_frames/sample_006_ts_5.00s.jpg"
    },
    {
      "frame_index_1based": 7,
      "timestamp_sec": 6.0,
      "original_frame_index": 180,
      "image_relpath": "sampled_frames/sample_007_ts_6.00s.jpg"
    },
    {
      "frame_index_1based": 8,
      "timestamp_sec": 7.0,
      "original_frame_index": 210,
      "image_relpath": "sampled_frames/sample_008_ts_7.00s.jpg"
    }
  ]
}
```

#### (2) `stage1/draft_plan.json`（示例）

```json
{
  "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
  "steps": [
    {
      "step_id": 1,
      "step_goal": "Loosen and remove the jar lid.",
      "rationale": "Removing the lid eliminates the physical barrier that seals the jar, enabling access to the contents in later steps.",
      "causal_chain": {
        "agent": "hands",
        "action": "apply torque to loosen",
        "patient": "jar_lid",
        "causal_precondition_on_spatial": [
          {
            "relation": "contacting",
            "objects": ["hands", "jar_lid"],
            "truth": true
          }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar_lid",
            "affordance_types": ["graspable"],
            "reasons": "The lid edge is exposed and can be firmly gripped to transmit torque."
          }
        ],
        "causal_effect_on_spatial": [
          {
            "relation": "separated_from",
            "objects": ["jar_lid", "jar"],
            "truth": true
          }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["open"],
            "reasons": "With the lid removed, the jar opening becomes accessible for pouring."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the jar lid is wet and too slippery to maintain a stable grip?",
      "expected_challenge_outcome": "The applied torque would not transfer effectively, preventing the lid from loosening and requiring a higher-friction grip strategy.",
      "failure_reflecting": {
        "reason": "The lid cannot be loosened due to insufficient grip friction.",
        "recovery_strategy": "Increase friction using a dry cloth or adjust hand placement to improve grip."
      }
    },
    {
      "step_id": 2,
      "step_goal": "Position the open jar above the bowl.",
      "rationale": "Aligning the jar over the bowl ensures gravity can move the contents into the target container without spilling elsewhere.",
      "causal_chain": {
        "agent": "hands",
        "action": "translate and align",
        "patient": "jar",
        "causal_precondition_on_spatial": [
          {
            "relation": "holding",
            "objects": ["hands", "jar"],
            "truth": true
          }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["portable"],
            "reasons": "The jar can be lifted and repositioned by hand."
          }
        ],
        "causal_effect_on_spatial": [
          {
            "relation": "above",
            "objects": ["jar", "bowl"],
            "truth": true
          }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["pourable"],
            "reasons": "Once aligned above the bowl, tilting can route contents into the bowl opening."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the bowl is unstable and shifts when the jar approaches?",
      "expected_challenge_outcome": "Misalignment would increase the chance of missing the bowl opening and cause spillage unless the bowl is stabilized first.",
      "failure_reflecting": {
        "reason": "The bowl moves, breaking alignment between jar and bowl.",
        "recovery_strategy": "Stabilize the bowl by holding it or placing it on a non-slip surface before aligning the jar."
      }
    },
    {
      "step_id": 3,
      "step_goal": "Tilt the jar to pour the contents into the bowl.",
      "rationale": "Tilting changes the jar orientation so gravity drives the contents out of the jar and into the bowl, achieving the intended transfer.",
      "causal_chain": {
        "agent": "hands",
        "action": "tilt to pour",
        "patient": "jar",
        "causal_precondition_on_spatial": [
          {
            "relation": "above",
            "objects": ["jar", "bowl"],
            "truth": true
          }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["open"],
            "reasons": "The jar opening must be unobstructed so contents can exit when tilted."
          }
        ],
        "causal_effect_on_spatial": [
          {
            "relation": "inside",
            "objects": ["jar_contents", "bowl"],
            "truth": true
          }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "bowl",
            "affordance_types": ["filled"],
            "reasons": "After pouring, the contents accumulate in the bowl interior."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the jar opening is partially blocked by residue that restricts flow?",
      "expected_challenge_outcome": "Flow would be reduced or stopped, requiring agitation or clearing the blockage before the contents can be transferred.",
      "failure_reflecting": {
        "reason": "The contents do not flow due to blockage at the opening.",
        "recovery_strategy": "Gently shake or rotate the jar and clear the opening to restore a continuous flow."
      }
    }
  ]
}
```

---

## 4. Stage2（Localize/Cut）：给每个 step 找时间边界并切出 clip

### 4.1 Stage2 做什么？

输入：

- `stage1/draft_plan.json`（只读：step_id/step_goal 已固定）
- `stage1/frame_manifest.json`（full video 帧池）
- full video 抽帧图像（供模型做时间定位）

模型任务：

- 对每个 step 输出：
  - `start_frame_index`：inclusive 起点（1-based，落在 full video 帧池上）
  - `end_frame_index`：exclusive 终点（1-based 边界；允许取 `num_frames + 1` 表示“最后一帧之后的边界”）

脚本任务：

- 将边界索引映射到时间秒（`frame_manifest.json` 的 `timestamp_sec`）
- 用 ffmpeg 按 `[start_sec, end_sec)` 切出每个 step 的视频 clip
- 写出 `step_segments.json`（包含每个 clip 的路径与时间范围等元数据）

产出（Stage2 输出文件）：

- `stage2/localization_raw.json`：模型输出的边界索引（严格 JSON）
- `stage2/step_segments.json`：每步 segment 元数据（含 start/end 秒与 clip 路径）
- `stage2/step_clips/*.mp4`：每步 clip

### 4.2 Stage2 产物示例

#### (1) `stage2/localization_raw.json`（示例）

```json
{
  "steps": [
    { "step_id": 1, "start_frame_index": 1, "end_frame_index": 4 },
    { "step_id": 2, "start_frame_index": 4, "end_frame_index": 7 },
    { "step_id": 3, "start_frame_index": 7, "end_frame_index": 9 }
  ]
}
```

#### (2) `stage2/step_segments.json`（示例）

```json
{
  "source_video": "/abs/path/video.mp4",
  "video_id": "video",
  "generated_at_utc": "2026-01-22T10:00:00Z",
  "num_frames": 8,
  "cut": {
    "mode": "reencode",
    "seek_slop_sec": 1.0,
    "crf": 18,
    "preset": "veryfast",
    "keep_audio": false,
    "ffmpeg_bin": "ffmpeg"
  },
  "segments": [
    {
      "step_id": 1,
      "step_goal": "Loosen and remove the jar lid.",
      "start_frame_index": 1,
      "end_frame_index": 4,
      "start_sec": 0.0,
      "end_sec": 3.0,
      "start_image_relpath": "sampled_frames/sample_001_ts_0.00s.jpg",
      "end_image_relpath": "sampled_frames/sample_004_ts_3.00s.jpg",
      "clip_relpath": "step_clips/step01_loosen_and_remove_the_jar_lid.mp4"
    },
    {
      "step_id": 2,
      "step_goal": "Position the open jar above the bowl.",
      "start_frame_index": 4,
      "end_frame_index": 7,
      "start_sec": 3.0,
      "end_sec": 6.0,
      "start_image_relpath": "sampled_frames/sample_004_ts_3.00s.jpg",
      "end_image_relpath": "sampled_frames/sample_007_ts_6.00s.jpg",
      "clip_relpath": "step_clips/step02_position_the_open_jar_above_the_bowl.mp4"
    },
    {
      "step_id": 3,
      "step_goal": "Tilt the jar to pour the contents into the bowl.",
      "start_frame_index": 7,
      "end_frame_index": 9,
      "start_sec": 6.0,
      "end_sec": 8.0,
      "start_image_relpath": "sampled_frames/sample_007_ts_6.00s.jpg",
      "end_image_relpath": "sampled_frames/sample_008_ts_7.00s.jpg",
      "clip_relpath": "step_clips/step03_tilt_the_jar_to_pour_the_contents_into_the_bowl.mp4"
    }
  ]
}
```

---

## 5. Stage3（Refine+Keyframes）：在 clip 上精修并输出 2 个关键帧标注

### 5.1 Stage3 做什么？

输入（对每个 step 单独运行一次）：

- `stage2/step_clips/stepXX_<slug>.mp4`（该步视频片段）
- 从该 clip 均匀采样得到的帧池（默认最多 50 帧）
- Stage1 草案 step（只读：`step_id` 与 `step_goal` 不允许改）
- `ECCV/three_stage/prompts.py:build_stage3_user_prompt()` 生成的提示词

模型任务：

- 输出一个**完整 step JSON**（字段严格等于 `prompts.py` 的 Stage3 schema）
- 选择并输出恰好 **2 个**关键帧：
  - `critical_frames[*].frame_index`：1-based，落在该 step clip 的帧池上
  - 每个关键帧都必须包含 `causal_chain` 与 `interaction`（同样遵循 schema）

脚本任务：

- 将模型输出严格解析为 JSON，并做强校验/归一化
- 根据 `frame_manifest.json` + `frame_index` 把关键帧 JPEG 落盘到 step 目录（JSON 不包含任何图像路径字段）
- 合并所有 step 的 `step_final.json` → 写出 `causal_plan_with_keyframes.json`

产出（Stage3 输出文件）：

- `<step_folder>/frame_manifest.json`：该 step clip 帧池（**注意：这是独立索引空间**）
- `<step_folder>/step_final.json`：该 step 最终 JSON（严格 schema）
- `<step_folder>/frame_###_ts_XX.XXs.jpg`：关键帧 JPEG（由脚本保存）
- `<video_id>/causal_plan_with_keyframes.json`：最终合并 JSON

### 5.2 Stage3 产物示例

#### (1) `<step_folder>/frame_manifest.json`（示例）

```json
{
  "num_frames": 6,
  "note": "frame_index_1based is the 1-based index used in prompts and model outputs for this frame pool.",
  "frames": [
    {
      "frame_index_1based": 1,
      "timestamp_sec": 0.2,
      "original_frame_index": 6,
      "image_relpath": "sampled_frames/sample_001_ts_0.20s.jpg"
    },
    {
      "frame_index_1based": 2,
      "timestamp_sec": 0.7,
      "original_frame_index": 21,
      "image_relpath": "sampled_frames/sample_002_ts_0.70s.jpg"
    },
    {
      "frame_index_1based": 3,
      "timestamp_sec": 1.2,
      "original_frame_index": 36,
      "image_relpath": "sampled_frames/sample_003_ts_1.20s.jpg"
    },
    {
      "frame_index_1based": 4,
      "timestamp_sec": 1.7,
      "original_frame_index": 51,
      "image_relpath": "sampled_frames/sample_004_ts_1.70s.jpg"
    },
    {
      "frame_index_1based": 5,
      "timestamp_sec": 2.2,
      "original_frame_index": 66,
      "image_relpath": "sampled_frames/sample_005_ts_2.20s.jpg"
    },
    {
      "frame_index_1based": 6,
      "timestamp_sec": 2.7,
      "original_frame_index": 81,
      "image_relpath": "sampled_frames/sample_006_ts_2.70s.jpg"
    }
  ]
}
```

#### (2) `<step_folder>/step_meta.json`（示例）

`step_meta.json` 用于记录该步 clip 的来源与时间范围，便于断点续跑的一致性检查与验收。

```json
{
  "step_id": 1,
  "step_goal": "Loosen and remove the jar lid.",
  "clip_path": "../stage2/step_clips/step01_loosen_and_remove_the_jar_lid.mp4",
  "clip_start_sec": 0.0,
  "clip_end_sec": 3.0,
  "num_frames": 6,
  "generated_at_utc": "2026-01-22T10:00:00Z",
  "manifest_path": "frame_manifest.json"
}
```

#### (2) `<step_folder>/step_final.json`（示例：step_id=1）

> 注意：以下 JSON 字段结构严格对应 `prompts.py` 的 Stage3 schema；`critical_frames` 必须恰好 2 个；并且 step-level 与关键帧的 `causal_chain.agent/action/patient` 必须一致。

```json
{
  "step_id": 1,
  "step_goal": "Loosen and remove the jar lid.",
  "rationale": "Removing the lid breaks the sealing constraint and exposes the opening so the contents can be accessed and transferred later.",
  "causal_chain": {
    "agent": "hands",
    "action": "apply torque to loosen",
    "patient": "jar_lid",
    "causal_precondition_on_spatial": [
      { "relation": "contacting", "objects": ["hands", "jar_lid"], "truth": true }
    ],
    "causal_precondition_on_affordance": [
      {
        "object_name": "jar_lid",
        "affordance_types": ["graspable"],
        "reasons": "The lid edge is exposed and supports a stable grip needed to transmit twisting force."
      }
    ],
    "causal_effect_on_spatial": [
      { "relation": "separated_from", "objects": ["jar_lid", "jar"], "truth": true }
    ],
    "causal_effect_on_affordance": [
      {
        "object_name": "jar",
        "affordance_types": ["open"],
        "reasons": "Once the lid is removed, the opening is unobstructed and can receive or release contents."
      }
    ]
  },
  "counterfactual_challenge_question": "What if the lid is stuck due to high friction at the threads?",
  "expected_challenge_outcome": "The applied torque would be insufficient to break static friction, so the lid would not rotate unless friction is reduced or leverage is increased.",
  "failure_reflecting": {
    "reason": "The lid does not rotate and remains engaged with the jar threads.",
    "recovery_strategy": "Increase grip friction and apply a steadier, higher torque while stabilizing the jar body."
  },
  "critical_frames": [
    {
      "frame_index": 2,
      "action_state_change_description": "Initiation: hands establish a firm grip on the lid and begin twisting relative to the jar body.",
      "causal_chain": {
        "agent": "hands",
        "action": "apply torque to loosen",
        "patient": "jar_lid",
        "causal_precondition_on_spatial": [
          { "relation": "contacting", "objects": ["hands", "jar_lid"], "truth": true }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar_lid",
            "affordance_types": ["graspable"],
            "reasons": "The exposed rim provides a surface that can be pinched and held without slipping."
          }
        ],
        "causal_effect_on_spatial": [
          { "relation": "rotating_relative_to", "objects": ["jar_lid", "jar"], "truth": true }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar_lid",
            "affordance_types": ["loosened"],
            "reasons": "Twisting reduces the thread engagement, making the lid easier to lift off."
          }
        ]
      },
      "interaction": {
        "tools": ["hands"],
        "materials": ["jar_lid"],
        "hotspot": {
          "description": "The ridged rim of the lid where fingers can press to resist slipping.",
          "affordance_type": "grasp_point",
          "mechanism": "Finger pressure and friction generate a tangential force that creates torque about the lid axis."
        }
      }
    },
    {
      "frame_index": 5,
      "action_state_change_description": "Completion: the lid is fully disengaged from the jar and no longer constrains the opening.",
      "causal_chain": {
        "agent": "hands",
        "action": "apply torque to loosen",
        "patient": "jar_lid",
        "causal_precondition_on_spatial": [
          { "relation": "rotating_relative_to", "objects": ["jar_lid", "jar"], "truth": true }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar_lid",
            "affordance_types": ["removable"],
            "reasons": "Once thread engagement is sufficiently reduced, the lid can be lifted away without resistance."
          }
        ],
        "causal_effect_on_spatial": [
          { "relation": "separated_from", "objects": ["jar_lid", "jar"], "truth": true }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["open"],
            "reasons": "The opening is exposed after the lid is removed, enabling pouring or scooping."
          }
        ]
      },
      "interaction": {
        "tools": ["hands"],
        "materials": ["jar_lid"],
        "hotspot": {
          "description": "The lid edge used as the contact surface for lifting after loosening.",
          "affordance_type": "contact_surface",
          "mechanism": "Upward force overcomes remaining thread contact and separates the lid from the jar mouth."
        }
      }
    }
  ]
}
```

#### (3) `<video_id>/causal_plan_with_keyframes.json`（示例：合并后的最终产物）

该文件结构为：

- 顶层：`high_level_goal` + `steps`（**只允许这两个顶层字段**）
- `steps[*]`：每个元素就是一个 Stage3 `step_final.json` 对象（字段完全相同）

下面给出一个“3 步完整示例”。其中 `step_id=1` 与上面的 `step_final.json` 一致；`step_id=2/3` 为同结构的完整对象：

```json
{
  "high_level_goal": "Open a jar and transfer its contents into a bowl so the contents are ready to be used.",
  "steps": [
    {
      "step_id": 1,
      "step_goal": "Loosen and remove the jar lid.",
      "rationale": "Removing the lid breaks the sealing constraint and exposes the opening so the contents can be accessed and transferred later.",
      "causal_chain": {
        "agent": "hands",
        "action": "apply torque to loosen",
        "patient": "jar_lid",
        "causal_precondition_on_spatial": [
          { "relation": "contacting", "objects": ["hands", "jar_lid"], "truth": true }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar_lid",
            "affordance_types": ["graspable"],
            "reasons": "The lid edge is exposed and supports a stable grip needed to transmit twisting force."
          }
        ],
        "causal_effect_on_spatial": [
          { "relation": "separated_from", "objects": ["jar_lid", "jar"], "truth": true }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["open"],
            "reasons": "Once the lid is removed, the opening is unobstructed and can receive or release contents."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the lid is stuck due to high friction at the threads?",
      "expected_challenge_outcome": "The applied torque would be insufficient to break static friction, so the lid would not rotate unless friction is reduced or leverage is increased.",
      "failure_reflecting": {
        "reason": "The lid does not rotate and remains engaged with the jar threads.",
        "recovery_strategy": "Increase grip friction and apply a steadier, higher torque while stabilizing the jar body."
      },
      "critical_frames": [
        {
          "frame_index": 2,
          "action_state_change_description": "Initiation: hands establish a firm grip on the lid and begin twisting relative to the jar body.",
          "causal_chain": {
            "agent": "hands",
            "action": "apply torque to loosen",
            "patient": "jar_lid",
            "causal_precondition_on_spatial": [
              { "relation": "contacting", "objects": ["hands", "jar_lid"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "jar_lid",
                "affordance_types": ["graspable"],
                "reasons": "The exposed rim provides a surface that can be pinched and held without slipping."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "rotating_relative_to", "objects": ["jar_lid", "jar"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "jar_lid",
                "affordance_types": ["loosened"],
                "reasons": "Twisting reduces the thread engagement, making the lid easier to lift off."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar_lid"],
            "hotspot": {
              "description": "The ridged rim of the lid where fingers can press to resist slipping.",
              "affordance_type": "grasp_point",
              "mechanism": "Finger pressure and friction generate a tangential force that creates torque about the lid axis."
            }
          }
        },
        {
          "frame_index": 5,
          "action_state_change_description": "Completion: the lid is fully disengaged from the jar and no longer constrains the opening.",
          "causal_chain": {
            "agent": "hands",
            "action": "apply torque to loosen",
            "patient": "jar_lid",
            "causal_precondition_on_spatial": [
              { "relation": "rotating_relative_to", "objects": ["jar_lid", "jar"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "jar_lid",
                "affordance_types": ["removable"],
                "reasons": "Once thread engagement is sufficiently reduced, the lid can be lifted away without resistance."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "separated_from", "objects": ["jar_lid", "jar"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "jar",
                "affordance_types": ["open"],
                "reasons": "The opening is exposed after the lid is removed, enabling pouring or scooping."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar_lid"],
            "hotspot": {
              "description": "The lid edge used as the contact surface for lifting after loosening.",
              "affordance_type": "contact_surface",
              "mechanism": "Upward force overcomes remaining thread contact and separates the lid from the jar mouth."
            }
          }
        }
      ]
    },
    {
      "step_id": 2,
      "step_goal": "Position the open jar above the bowl.",
      "rationale": "Aligning the jar above the bowl sets the geometry needed for gravity-driven transfer while reducing the risk of spillage.",
      "causal_chain": {
        "agent": "hands",
        "action": "translate and align",
        "patient": "jar",
        "causal_precondition_on_spatial": [
          { "relation": "holding", "objects": ["hands", "jar"], "truth": true }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["portable"],
            "reasons": "The jar can be lifted and repositioned to a target location by hand."
          }
        ],
        "causal_effect_on_spatial": [
          { "relation": "above", "objects": ["jar", "bowl"], "truth": true }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["pourable"],
            "reasons": "Once the jar is above the bowl, tilting can route the contents into the bowl opening."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the bowl opening is partially occluded by another object?",
      "expected_challenge_outcome": "Even if the jar is above the bowl, the occlusion would block the intended landing area, increasing spill risk unless the obstruction is removed.",
      "failure_reflecting": {
        "reason": "The jar cannot be aligned cleanly over the bowl opening.",
        "recovery_strategy": "Clear the area around the bowl opening and re-align the jar while maintaining a stable hold."
      },
      "critical_frames": [
        {
          "frame_index": 1,
          "action_state_change_description": "Initiation: hands lift the jar and start moving it toward the bowl location.",
          "causal_chain": {
            "agent": "hands",
            "action": "translate and align",
            "patient": "jar",
            "causal_precondition_on_spatial": [
              { "relation": "holding", "objects": ["hands", "jar"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "jar",
                "affordance_types": ["portable"],
                "reasons": "A stable grip allows controlled translation without dropping the jar."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "moving_toward", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "jar",
                "affordance_types": ["position_changing"],
                "reasons": "Hand motion changes the jar position to approach the target alignment."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar", "bowl"],
            "hotspot": {
              "description": "The jar body area where fingers wrap to control translation and orientation.",
              "affordance_type": "grasp_point",
              "mechanism": "A firm grip supplies support force and friction, enabling controlled motion without slipping."
            }
          }
        },
        {
          "frame_index": 5,
          "action_state_change_description": "Completion: the jar is positioned directly above the bowl with a stable alignment.",
          "causal_chain": {
            "agent": "hands",
            "action": "translate and align",
            "patient": "jar",
            "causal_precondition_on_spatial": [
              { "relation": "moving_toward", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "bowl",
                "affordance_types": ["available"],
                "reasons": "A clear receiving area is needed so alignment has a meaningful target."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "above", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "jar",
                "affordance_types": ["pourable"],
                "reasons": "With the jar over the bowl, tilting can direct contents into the bowl rather than elsewhere."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar", "bowl"],
            "hotspot": {
              "description": "The jar mouth region whose pose determines whether the opening is aligned over the bowl.",
              "affordance_type": "contact_surface",
              "mechanism": "Controlling orientation keeps the opening positioned so gravity-driven flow can be captured by the bowl."
            }
          }
        }
      ]
    },
    {
      "step_id": 3,
      "step_goal": "Tilt the jar to pour the contents into the bowl.",
      "rationale": "Tilting reorients the jar opening so gravity moves the contents out of the jar and into the bowl, completing the transfer.",
      "causal_chain": {
        "agent": "hands",
        "action": "tilt to pour",
        "patient": "jar",
        "causal_precondition_on_spatial": [
          { "relation": "above", "objects": ["jar", "bowl"], "truth": true }
        ],
        "causal_precondition_on_affordance": [
          {
            "object_name": "jar",
            "affordance_types": ["open"],
            "reasons": "The opening must be unobstructed so contents can exit when the jar is tilted."
          }
        ],
        "causal_effect_on_spatial": [
          { "relation": "inside", "objects": ["jar_contents", "bowl"], "truth": true }
        ],
        "causal_effect_on_affordance": [
          {
            "object_name": "bowl",
            "affordance_types": ["filled"],
            "reasons": "After pouring, the contents accumulate within the bowl interior."
          }
        ]
      },
      "counterfactual_challenge_question": "What if the jar is tilted but the contents adhere to the jar interior and do not move?",
      "expected_challenge_outcome": "Without relative motion, little or no material would leave the jar, requiring agitation or a larger tilt to overcome adhesion and friction.",
      "failure_reflecting": {
        "reason": "The contents remain stuck and do not flow out despite tilting.",
        "recovery_strategy": "Increase the tilt angle and gently agitate the jar to initiate flow."
      },
      "critical_frames": [
        {
          "frame_index": 2,
          "action_state_change_description": "Initiation: hands begin rotating the jar so the opening faces toward the bowl.",
          "causal_chain": {
            "agent": "hands",
            "action": "tilt to pour",
            "patient": "jar",
            "causal_precondition_on_spatial": [
              { "relation": "above", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "jar",
                "affordance_types": ["open"],
                "reasons": "An open mouth provides an exit path once the jar orientation is changed."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "tilted_toward", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "jar_contents",
                "affordance_types": ["in_motion"],
                "reasons": "Tilting changes the effective gravity direction relative to the contents, initiating movement."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar", "bowl"],
            "hotspot": {
              "description": "The jar body surfaces where hands apply rotational control to change orientation.",
              "affordance_type": "grasp_point",
              "mechanism": "Applied torque rotates the jar so gravity can drive contents toward the opening."
            }
          }
        },
        {
          "frame_index": 6,
          "action_state_change_description": "Completion: the contents have exited the jar and are collected inside the bowl.",
          "causal_chain": {
            "agent": "hands",
            "action": "tilt to pour",
            "patient": "jar",
            "causal_precondition_on_spatial": [
              { "relation": "tilted_toward", "objects": ["jar", "bowl"], "truth": true }
            ],
            "causal_precondition_on_affordance": [
              {
                "object_name": "bowl",
                "affordance_types": ["receiving"],
                "reasons": "A stable container is needed to retain the poured contents without overflowing."
              }
            ],
            "causal_effect_on_spatial": [
              { "relation": "inside", "objects": ["jar_contents", "bowl"], "truth": true }
            ],
            "causal_effect_on_affordance": [
              {
                "object_name": "bowl",
                "affordance_types": ["filled"],
                "reasons": "After transfer, the contents occupy the bowl interior as the new resting location."
              }
            ]
          },
          "interaction": {
            "tools": ["hands"],
            "materials": ["jar", "bowl"],
            "hotspot": {
              "description": "The jar mouth rim that defines the exit path for the contents during pouring.",
              "affordance_type": "contact_surface",
              "mechanism": "The opening geometry and orientation constrain the flow path so contents leave the jar into the bowl."
            }
          }
        }
      ]
    }
  ]
}
```

#### (4) `<video_id>/run_summary.json`（示例片段）

`run_summary.json` 记录每次运行的源视频、阶段状态、提示词落盘路径与部分配置，便于追溯与排错：

```json
{
  "source_video": "/abs/path/video.mp4",
  "video_id": "video",
  "output_root": "/abs/path/to/ECCV/three_stage/causal_spafa_plan_dataset_long",
  "updated_at_utc": "2026-01-22T10:00:00Z",
  "stage1": { "status": "completed", "generated_at_utc": "2026-01-22T10:00:00Z" },
  "stage2": { "status": "completed", "generated_at_utc": "2026-01-22T10:00:00Z" },
  "stage3": { "status": "completed", "generated_at_utc": "2026-01-22T10:00:00Z" }
}
```

---

## 6. 质量与一致性保障（为什么能稳产出高质量 JSON）

### 6.1 强校验（schema 对齐）

每个阶段都遵循同一原则：**只接受严格 JSON 且字段必须在白名单内**。

- Stage1：严格禁止 keyframe 字段；同时对必填/列表非空/文本不允许出现索引或时间引用等做硬校验。
- Stage2：只接受 `{step_id,start_frame_index,end_frame_index}`；要求单调、非重叠、正时长（边界索引合法）。
- Stage3：按 `prompts.py` 的 step schema 做强校验：字段白名单、必填非空、关键帧数量=2、关键帧索引递增、`causal_chain` 的一致性与 `interaction` 的约束等。

### 6.2 自动重试（把错误反馈给模型）

当某次模型输出不满足校验时，脚本会：

1) 从模型返回内容中提取 JSON  
2) 输出错误列表（例如缺字段、空字段、多余字段、类型不对）  
3) 将错误列表作为下一轮提示词前缀，触发重试  

这样能显著提高最终 `json` 严格对齐 schema 的概率。

### 6.3 运行后验收（推荐）

对某个 `<video_id>` 输出目录做结构与 schema 核验：

```bash
python3 ECCV/three_stage/validate_three_stage_output.py --video-output-dir /abs/path/to/<video_id> --check-deps
```

它会检查：

- Stage1/Stage2/Stage3 必要文件是否齐全
- `causal_plan_with_keyframes.json` 顶层字段是否严格为 `high_level_goal` + `steps`
- 每个 step 文件与合并后的 step 是否一致（同一套 schema-normalized 结果）
- 关键帧 JPEG 是否存在、是否与 manifest 对应的 `frame_index` 一致
