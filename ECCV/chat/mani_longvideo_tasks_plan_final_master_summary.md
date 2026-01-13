# mani_longvideo 任务体系最终整合版（Task_01–Task_30 × 四类证据 × Strict Scoring × 可训练 QA）

本文档是对以下材料的**统一整合与最终落地方案**，目标是：在真实 item（以 `ECCV/causal_spafa_plan_dataset_long/P01_01_part1` 为例）的约束下，把 `ECCV/mani_longvideo_tasks_plan_final.md` 中定义的 Task_01–Task_30 转换为**可以直接用于多模态大模型训练**的 QA 数据，并给出严格可自动评分（strict scoring）的子集与视频/图像两种证据模式。

## 参考材料（必须一致）

- 主任务规范：`ECCV/mani_longvideo_tasks_plan_final.md`
- 严格评分（仅图像证据）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`
- 严格评分（支持 MP4 完整视频/clip）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`
- 对齐与 QA 策略（已整理）：`ECCV/chat/mani_longvideo_tasks_plan_final_alignment_and_qa_strategy.md`
- 真实 item（样例）：`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/`
- 真实 JSON 产物：`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json`

---

## 0. 最终目标与核心结论（先给可执行结论）

### 0.1 四类多模态证据（最终统一口径）

结合主规范的 evidence_type 枚举（更细）与本项目训练落地的可控性，最终建议把可用多模态证据压缩为 4 类（其余如 `images_uniform_clip` 视作工程实现细节）：

1) `keyframe_single`：单张关键帧图像（最稳、成本最低、最易 strict scoring）  
2) `images_uniform_scene`：全局均匀抽帧的多图（覆盖环境/阶段变化，支持“全局目标/锚点”类任务）  
3) `video_clip`：局部片段（动作过程/边界/对齐最依赖）  
4) `video_prefix`：累积前缀片段（进度/下一步预测/规划重排最依赖）

### 0.2 任务训练落地的“强监督/弱监督”分层（必须做）

为了让训练稳定、可回归、可评测，Task_01–Task_30 必须分两条线构建数据：

- **强监督（客观题，可自动打分）**：用 Yes/No、A/B、A/B/C/D、排序、match/mismatch 等形式输出，exact match 评分。  
  - 主要覆盖：空间关系核验、可供性类型识别、工具/材料角色、step_goal 匹配、时间顺序、计划-执行对齐、错误计划诊断、未来步骤重排/补全。
- **弱监督/主观题（模型打分）**：需要解释/总结/机制描述的任务，必须通过 judge 模型评分或 rubric 评分，并显式标记 `meta.weak_supervision=true`。  
  - 主要覆盖：因果链解释、动机/必要性、前置/效果核验（可观测性问题）、边界现象描述、进度总结、失败恢复等。

### 0.3 关于 strict scoring：必须有“评测子集”

严格评分任务用于回归测试、对齐迭代、以及评测集构建：

- **Images-only strict scoring**：适用于只给图片的模型或评测（SS01–SS05）。  
- **MP4 strict scoring**：适用于支持视频输入的模型（SS01–SS07），能显著增强“动作过程”判别能力。  

严格评分的输出协议必须统一：只输出 `Yes/No`、`A/B`、`A/B/C/D`（单行，exact match），详见两份 strict scoring 文档。

---

## 1. 真实 item 审计：P01_01_part1 能提供什么？缺什么？

本节把“规范能做什么”落回到“现实数据有什么”，否则任务设计会出现不可生成/不可训练的问题。

### 1.1 P01_01_part1 目录结构与媒体资产现状

`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/` 当前包含：

- `causal_plan_with_keyframes.json`（核心监督来源）
- 6 个 step 子目录（每个目录内有关键帧 jpg；总计 9 张关键帧）
- `run_summary.json`（记录源视频路径，但 item 内没有 mp4）
- `last_frame_segments/`（空目录，未生成 mp4）
- `cumulative_last_frame_segments/`（空目录，未生成 mp4）
- **缺失**：`sampled_frames/`（全局均匀抽帧）
- **缺失**：源视频文件（`P01_01_part1.MP4` 不在 item 内）

这意味着：

- 目前可直接生成的任务：以 `keyframe_single` 为主（Task_02/03/04/05/06/08/09/10/11/12/13/14/15/16/17/21/27 等，以及 images-only strict scoring 的大部分）。
- 需要 `images_uniform_scene` 的任务（Task_01/07/30 等）只能用“每步最早关键帧集合”作弱替代，质量会下降。
- 需要 `video_clip/video_prefix` 的任务（Task_18/20/22/23/24/25/26/28/29 等）在当前 item 中**无法用 mp4 直接训练**，必须补齐视频资产或用运行时裁剪方案。

### 1.2 源视频可追溯性（run_summary.json）

`run_summary.json` 中记录：

- `source_video`: `P01_01_part1.MP4`
- `config_planning.VIDEO_PATH`: `/e2e-data/embodied-research-data/luzheng/kitchen/long/P01_01_part1.MP4`

最佳实践（强烈建议）：

- 将源视频复制/软链到 item 内，例如 `<item_dir>/source_video.mp4`（保证跨机器可复现）。
- 在后续生成脚本里优先使用 item 内相对路径；无法找到再回退到 `run_summary.json` 的绝对路径。

### 1.3 Step 概览（来自 causal_plan_with_keyframes.json 的真实字段）

`high_level_goal`（真实文本）：

> Prepare ingredients for cooking by entering a kitchen, gathering vegetables and tools, washing the vegetables, and chopping them on a cutting board.

Step 表（来自 JSON，ts 由关键帧文件名解析）：

| step_id | step_goal | #keyframes | ts_list | tools | materials |
|---:|---|---:|---|---|---|
| 1 | Enter the kitchen, illuminate the space, and retrieve a cucumber and a carrot from the refrigerator. | 2 | 3.59, 25.19 | light_switch, refrigerator | cucumber, carrot |
| 2 | Assemble the preparation workspace by placing vegetables on the counter, then retrieving a cutting board and knife. | 2 | 39.59, 53.99 | kitchen_cupboard, cutlery_drawer, cutting_board, knife | cucumber, carrot |
| 3 | Wash the cucumber and carrot in the sink under running water. | 1 | 68.39 | sink, faucet | cucumber, carrot, water |
| 4 | Retrieve a frying pan from a low cupboard and place it on the stovetop. | 1 | 97.20 | frying_pan, stovetop |  |
| 5 | Slice the entire washed zucchini into circular pieces. | 1 | 136.79 | knife, cutting_board | zucchini |
| 6 | Dice a portion of the zucchini slices into small cubes. | 2 | 154.79, 162.00 | knife, cutting_board | zucchini_slices |

补充：item 内可直接用作 distractor/候选池的词表（从 JSON 聚合得到）：

- tools（去重）：`light_switch, refrigerator, kitchen_cupboard, cutlery_drawer, cutting_board, knife, sink, faucet, frying_pan, stovetop`
- materials（去重）：`cucumber, carrot, water, zucchini, zucchini_slices`
- affordance_type（去重）：`pressable_surface, support_surface, grabbable_handle, cutting_edge, stackable_and_cuttable_group, ...`（详见 JSON）

---

## 2. JSON schema 与主任务规范是否一致？（对齐要点）

### 2.1 字段覆盖结论

以 `P01_01_part1/causal_plan_with_keyframes.json` 为例，主规范中 Task_01–Task_30 所需的关键字段基本都存在：

- 顶层：`high_level_goal`
- step：`step_goal/rationale/preconditions/expected_effects/predicted_next_actions/tool_and_material_usage/...`
- critical_frame：`keyframe_image_path/action_description/state_change_description/spatial_preconditions/affordance_preconditions/causal_chain/affordance_hotspot`

这意味着：从“字段可取到”的角度，Task 体系是可生成的。

### 2.2 一个需要显式处理的不一致：`affordance_hotspot.mechanism`

主规范在 Task_03/06/17 等处会引用 `affordance_hotspot.mechanism`；但当前真实 JSON 的 `affordance_hotspot` 只有：

- `description`
- `affordance_type`
- `causal_role`

而“机制细节”更常写在 `causal_chain.causal_affordance_focus_detail`。

最终建议（生成侧统一映射，不要求修改 JSON 产物）：

- `mechanism = critical_frames[*].affordance_hotspot.mechanism`（若存在）
- 否则 `mechanism = critical_frames[*].causal_chain.causal_affordance_focus_detail`
- 并在 `meta.schema_fallbacks.mechanism_source` 记录来源，避免追溯困难。

### 2.3 路径可移植性：keyframe_image_path 可能是旧机器绝对路径

在 P01_01_part1 中，JSON 记录的关键帧路径是 `/e2e-data/...` 绝对路径，但 item 内存在同名 jpg。

生成器必须实现“路径归一化 + 回退查找”：

1) 若 `keyframe_image_path` 可读 → 直接用  
2) 否则：以 basename 在 `<item_dir>` 下 `rglob` 查找  
3) 再否则：按 `<item_dir>/{step_id:02d}_*/frame_{frame_index:03d}_ts_*s.jpg` glob  
4) 仍失败：丢弃该样本或标记 `meta.missing_media=true`（不用于训练/评分）

---

## 3. 四类证据：生成、回退、以及“最适合哪些任务”

本节给出**最终可执行**的证据选择规则。它必须同时兼容：

- 主规范的 evidence_type 约定（用于 meta 可追溯）
- images-only strict scoring（禁止视频）
- mp4 strict scoring（允许完整视频或 clip）
- 真实 item 可能缺少部分媒体资产的情况（必须有 fallback）

### 3.1 `keyframe_single`（单关键帧）

- **来源**：`steps[i].critical_frames[j].keyframe_image_path`（路径需归一化）
- **适用任务（强烈推荐）**：
  - 空间关系核验：Task_27 / SS01
  - 可供性类型：Task_03 / SS02
  - 工具/材料角色：Task_04 / SS04
  - step_goal 匹配：Task_10 / SS05
  - 单帧动作/状态描述核验：Task_05（建议做 MCQ 版）
  - 因果链/机制解释：Task_06/17（主观）
- **优势**：最稳、最便宜、最容易做严格评分；适合作为训练主力和回归评测基础。
- **风险**：动作过程缺失；部分任务（边界、进度）无法仅靠单帧完成。

### 3.2 `images_uniform_scene`（全局均匀抽帧，多图）

- **标准来源**：`<item_dir>/sampled_frames/sample_*.jpg`（等距取 4–8 张）
- **回退策略**：
  1) 若无 `sampled_frames/`：用“每步最早关键帧集合”做代理（再等距采样到 4–8 张）
  2) 若关键帧也不足：丢弃该样本
- **适用任务**：
  - Task_01（场景锚点/关键对象）
  - Task_07（高阶目标）
  - Task_30（head-tail infill 的 head/tail 证据）
  - 作为无视频 fallback：Task_26/28 等
- **关键建议**：若你希望这些任务“真的像看全局”，必须补齐 `sampled_frames/`；否则会退化成“看关键帧猜全局”。

### 3.3 `video_clip`（局部片段）

允许两种工程实现：

1) **离线 clip 文件**：生成并保存 `.mp4`（最兼容训练管线）
2) **运行时裁剪**：只保存完整视频路径，并记录 `video_start_sec/video_end_sec`（依赖读取器支持按时间取段）

推荐来源优先级：

- 三阶段 step clip（若存在）：`<item_dir>/stage2/step_clips/...`
- `last_frame_segments/*.mp4`（相邻 step 尾帧间片段）
- 完整视频 `source_video.mp4` + `[start,end]` 裁剪窗口（来自关键帧 `ts`）

适用任务：

- Task_20（step 边界定位/转折）
- Task_22（计划-执行一致性 match/mismatch）
- mp4 strict scoring 的 SS06/SS07（动作/状态变化存在核验）

### 3.4 `video_prefix`（累积前缀）

同样支持离线/运行时两种实现：

- 离线：`cumulative_last_frame_segments/segment_start_to_stepXX_last.mp4`
- 运行时：完整视频 + `[0, end_ts]`

适用任务：

- Task_18（视觉前置条件核验，弱监督）
- Task_23（前缀识别高阶目标，推荐做 MCQ）
- Task_24（前缀预测下一步，推荐做 MCQ）
- Task_25（进度总结，主观+打分）
- Task_26（时间顺序判别 A/B）
- Task_28/29（错误计划诊断/未来 K 步重排）

---

## 3.5 关键帧/片段选取策略（与主规范一致的“最小共识”）

为避免“同一任务不同人取证据差异巨大”导致训练噪声，建议把证据选取固化为以下规则（与主规范推荐一致）：

### 3.5.1 keyframe_single 的选帧规则

- **描述/计划类**（例如 Task_04/08/09/10/14/15）：优先用该 step 的最早关键帧 `critical_frames[0]`（更像“开始执行该步”）
- **效果/完成状态类**（例如 Task_11/19）：优先用该 step 的最后关键帧 `critical_frames[-1]`（更像“完成态”）
- **瞬时物理/机制/空间核验类**（例如 Task_02/03/05/06/16/17/21/27）：用对应的 `critical_frames[j]`（若无更细对齐信息，可默认用 `critical_frames[0]`）

### 3.5.2 video_clip 的“可落地片段定义”

主规范给出了一套弱监督但可落地的 step clip 定义（基于尾帧片段）：

- 若已生成 `last_frame_segments/`：
  - step 1：`segment_start_to_step01.mp4`（从视频开始到 step1 尾关键帧）
  - step i（i>1）：`segment_step{i-1:02d}_to_step{i:02d}.mp4`（从 step{i-1} 尾帧到 step{i} 尾帧）

注意：这是“近似 step 执行片段”，并不等价于精确的 step 边界，但足够支撑 Task_10(video)/22/SS06/SS07 等任务的落地。

### 3.5.3 video_prefix 的前缀定义

- 若已生成 `cumulative_last_frame_segments/`：
  - 前缀到 step i：`segment_start_to_step{i:02d}_last.mp4`
- 若未生成 prefix 文件：
  - 用完整视频 + `video_end_sec = step_i_last_ts` 运行时裁剪（依赖读取器支持 `video_start/video_end`）

---

## 4. 训练数据 JSON 统一格式（最终版，必须遵守）

以 `Qwen-PC/qwen-vl-finetune` 支持的数据结构为准（`image`/`video` + `<image>/<video>` tag + `conversations`），并强化 meta 可追溯性。

### 4.1 基础结构（ShareGPT 风格）

```json
{
  "id": "uuid",
  "image": ["rel/or/abs/a.jpg", "rel/or/abs/b.jpg"],
  "video": "rel/or/abs/clip.mp4",
  "conversations": [
    {"from": "human", "value": "<image>\\n<image>\\n<question>"},
    {"from": "gpt", "value": "<answer>"}
  ],
  "meta": {
    "task_name": "Task_27_Visual_Spatial_Relation_Check",
    "evidence_type": "keyframe_single|images_uniform_scene|video_clip|video_prefix",
    "evidence_source": "keyframes|sampled_frames|last_frame_segments|cumulative_last_frame_segments|source_video",
    "evidence_files": ["..."],
    "source_path": "<item_dir>/causal_plan_with_keyframes.json",
    "step_id": 1,
    "frame_index": 2,
    "ts_sec": 3.59,
    "answer_format": "YesNo|AB|ABCD|ordered_list|free|structured",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "neg_sample": false,
    "weak_supervision": false,
    "schema_fallbacks": {}
  }
}
```

说明：

- `image` 与 `video` 允许二选一或同时存在，但问句中的 `<image>`/`<video>` tag 必须与媒体一致。
- 客观题必须在问句最后加硬约束：例如 `Only output A, B, C, or D.`，并保证答案字段只含标签。
- `options` 必须落盘到 meta（便于复现），但不要把 meta 注入 prompt（防止泄露）。

---

## 4.2 题面与输出格式：严格可训练的约束模板（强烈建议统一）

为了避免训练时输出发散、评测时难以自动评分，建议把“题面格式”与“答案输出格式”固化为以下模板：

### 4.2.1 Yes/No（二分类）

- 题面结尾强约束：`Only output Yes or No.`
- 参考答案：只输出 `Yes` 或 `No`（单行）

### 4.2.2 A/B（两选一）

- 题面结尾强约束：`Only output A or B.`
- 参考答案：只输出 `A` 或 `B`

### 4.2.3 A/B/C/D（四选一）

- 题面包含选项段：`Options:\n(A) ...\n(B) ...\n(C) ...\n(D) ...`
- 题面结尾强约束：`Only output A, B, C, or D.`
- 参考答案：只输出 `A/B/C/D`

### 4.2.4 ordered_list（严格序列输出，用于 Task_29/30）

- 题面明确：`Output the ordered steps as a numbered list from 1 to K.`
- 参考答案格式（示例）：
  - `1) ...`
  - `2) ...`
  - `3) ...`
- 评分建议：对每行做 normalize（去掉多余空格/编号符号），再与 gold 顺序逐项对齐评分。

### 4.2.5 structured（结构化单行输出，用于 Task_28）

- 题面明确：`Output in a single line: FlawStep=<int>; FlawType=<type>; Reason=<one sentence>.`
- 评分建议：用正则抽取 `FlawStep/FlawType` 做客观评分；`Reason` 可选做 judge 模型打分。

---

## 5. Task_01–Task_30：最终推荐的证据选择 + QA 题型 + 评分方式（总表）

说明：

- “默认证据”只从四类中选：`keyframe_single/images_uniform_scene/video_clip/video_prefix`
- “QA 题型”给出训练时最推荐的版本（必要时附一个可选变体）
- “评分”区分客观（exact match）与主观（judge model/rubric）
- “P01_01_part1 可用性”基于当前 item 资产：仅 keyframes 已齐，scene/video 需补齐

| Task | 核心能力 | 默认证据（四类） | 推荐 QA（训练版） | 评分/监督来源 | 负样本/扰动 | P01_01_part1 当前可用性 |
|---|---|---|---|---|---|---|
| 01 | 场景锚点/关键对象 | images_uniform_scene | 多选/列表（建议转多轮 Yes/No 或多选标签） | 由 JSON 聚合对象词表（弱+规则） | distractor 来自全局对象池 | 需 `sampled_frames/`；否则弱回退 |
| 02 | 空间关系抽取 | keyframe_single | 建议并入 27（Yes/No） | `spatial_preconditions[*].truth` | 取反 truth / 替换 objects | 可用 |
| 03 | 可供性热点类型 | keyframe_single | ABCD（affordance_type MCQ） | `affordance_hotspot.affordance_type` | 干扰项来自全局词表 | 可用 |
| 04 | 工具/材料角色 | keyframe_single | Yes/No（tool? / material?） | `tool_and_material_usage` | 跨 step 替换实体 | 可用 |
| 05 | 动作/状态描述 | keyframe_single | ABCD（caption matching） | `action_description/state_change_description` | 干扰来自其他关键帧 | 可用 |
| 06 | 因果链解释 | keyframe_single | 双段主观解释 | `causal_chain.*`（强文本） | 不做自动负样本 | 可用（主观） |
| 07 | 高阶目标识别 | images_uniform_scene | ABCD（high_level_goal MCQ） | `high_level_goal` | 干扰来自其他 item goal | 需 `sampled_frames/`；否则弱回退 |
| 08 | 步骤动机/必要性 | keyframe_single | 主观解释 / 或 rationale MCQ | `steps[i].rationale` | rationale 干扰需同域 | 可用（主观为主） |
| 09 | 前置条件陈述 | keyframe_single | 主观列举（建议降权） | `steps[i].preconditions` | 可做 Yes/No 拆解但易退化 | 可用（偏文本） |
| 10 | step_goal 对齐 | keyframe_single / video_clip | ABCD（step_goal matching） | `steps[i].step_goal` | 干扰来自同 item 其他 step_goal | keyframe 可用；video 需补齐 |
| 11 | 期望效果 | keyframe_single | 主观列举（建议降权） | `expected_effects` | 不建议做负样本 | 可用（偏文本） |
| 12 | 跨步依赖 | keyframe_single | MCQ：哪个 precondition 被满足 | effects↔preconditions 重合规则 | 干扰为非重合 preconditions | 可用（规则生成） |
| 13 | 下一步（计划版） | keyframe_single | ABCD（next step_goal MCQ） | `steps[i+1].step_goal` | 干扰来自同 item 其他 step_goal | 可用 |
| 14 | 反事实挑战 | keyframe_single | 主观回答 | `causal_challenge_question/outcome` | 不建议做负样本 | 可用（主观） |
| 15 | 失败与恢复 | keyframe_single | 主观回答 | `failure_handling.*` | 不建议做负样本 | 可用（主观） |
| 16 | 可行性核验 | keyframe_single | match/mismatch（二分类） | 证据-步骤配对规则 | 跨 step 错配构造 No | 可用（客观化推荐） |
| 17 | Why/How 综合 | keyframe_single | 双段主观解释 | Why=rationale；How=mechanism/causal_chain | 不做自动负样本 | 可用（主观） |
| 18 | 视觉前置核验 | video_prefix | 三态：satisfied/not/not observable | 弱监督（可观测性启发式） | 不建议强负样本 | 需视频/前缀 |
| 19 | 视觉效果核验 | keyframe_single (+clip) | 三态：supported/uncertain/not | 弱监督（可见性） | 不建议强负样本 | keyframe 可用（弱监督） |
| 20 | step 边界/转折 | video_clip | 边界识别 MCQ 或主观描述 | clip 与边界 i→i+1 自动标注 | mismatch 边界为负样本 | 需视频/clip |
| 21 | 关键帧理由 | keyframe_single | 主观解释 | 关键帧字段集合 | 不建议负样本 | 可用（主观） |
| 22 | 计划-执行一致性 | video_clip | match/partial/mismatch（三分类） | step-clip 与 step_goal 配对规则 | 跨 step 错配 | 需视频/clip |
| 23 | 前缀识别目标 | video_prefix | ABCD（high_level_goal MCQ） | `high_level_goal` | 干扰来自其他 item | 需视频/前缀 |
| 24 | 前缀预测下一步 | video_prefix | ABCD（next step_goal MCQ） | `steps[i+1].step_goal` | 干扰来自同 item 其他 step_goal | 需视频/前缀 |
| 25 | 前缀进度总结 | video_prefix | 主观总结 + judge 打分 | 参考 steps[0..i]（仅 meta） | 不建议负样本 | 需视频/前缀 |
| 26 | 时间顺序判别 | video_prefix / images_uniform_scene | A/B（谁更早） | ts 比较（来自 keyframe 文件名） | 事件描述 cross-step 采样 | 需视频或 scene；但可做 keyframe_pair（images-only） |
| 27 | 空间关系真假核验 | keyframe_single | Yes/No | `spatial_preconditions[*].truth` | 取反 truth/扰动 objects | 可用（强推荐做评测） |
| 28 | 错误计划诊断 | video_prefix / images_uniform_scene | 结构化输出（FlawStep/Type） | bad_plan 由扰动算子生成 | 单一扰动算子（order/tool/delete） | 需视频或 scene；可弱回退 |
| 29 | 未来 K 步重排 | video_prefix | 严格 1..K 输出 | gold 顺序来自 JSON | shuffle 作为输入扰动 | 需视频/前缀 |
| 30 | head-tail 补全中间步骤 | images_uniform_scene | 严格 1..M 输出 | middle steps 来自 JSON | 随机锚点增强难度 | 需 `sampled_frames/`；否则弱回退 |

---

## 5.1 每个任务的“样本数量建议”（避免关系爆炸/文本堆叠）

为了让数据集规模与质量可控，建议在生成时为每个任务设置“每个 item/step/keyframe 的上限”：

- Task_27 / SS01：每个关键帧最多抽 1–2 条 `spatial_preconditions`（否则关系爆炸）
- Task_03 / SS02：每个关键帧最多 1 条（热点类型本身就是 1 个）
- Task_04 / SS04：每个 step 1–2 条（问 tool 或问 material；再配 0–1 条 hard negative）
- Task_10 / SS05：每个 step 1 条（同 item 内其他 step_goal 做干扰）
- Task_26 / SS03：每个 item 采样 3–8 对事件（跨 step 优先；必须 `ts_a != ts_b`）
- Task_22：每个 step 1 条 match + 1 条 mismatch（mismatch 从同 item 其他 step_goal 抽）
- Task_28：每个 prefix_end_step 采样 1 个 bad_plan（每条样本只注入 1 个错误）
- Task_29：每个 prefix_end_step 采样 1 个 K（K=3–6）重排任务
- Task_30：每个 item 1 条（或按锚点采样 2–3 条增强）

---

## 6. strict scoring 子集：最终可评测套件（两种模式统一解释）

strict scoring 的目的不是覆盖所有任务，而是提供一套**稳定、自动化、可回归**的评测指标，并且可直接用于训练（强监督）。

### 6.1 Images-only strict scoring（SS01–SS05）

来源：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`

| SS 任务 | 对应主任务 | 证据（四类） | 输出 | 标签来源 |
|---|---|---|---|---|
| SS01_Visual_Spatial_Relation_Check | Task_27 | keyframe_single | Yes/No | `truth` |
| SS02_Affordance_Hotspot_Type_MC | Task_03 | keyframe_single | A/B/C/D | `affordance_type` |
| SS03_Temporal_Order_Check_AB | Task_26 | keyframe_pair（两张图） | A/B | `ts` 比较 |
| SS04_Tool_vs_Material_Check | Task_04 | keyframe_single | Yes/No | tools/materials 集合 |
| SS05_Step_Goal_Matching_MC | Task_10 | keyframe_single | A/B/C/D | step_goal |

对 P01_01_part1：

- 由于关键帧齐全，上述 SS01/SS02/SS04/SS05 可直接生成。
- SS03 可从不同 step 的关键帧中选两张图，用文件名 `ts_XX.XXs` 比较生成标签。

### 6.2 MP4 strict scoring（SS01–SS07）

来源：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`

相对 images-only 的关键增益：

- SS01/SS02/SS03 在视频模式下可以构造 `keyframe_window`/`two_event_window` clip，让问题对齐到 clip 尾部，减少单帧歧义。
- 新增 SS06/SS07：把 JSON 中的 `action_description/state_change_description` 变成“视频过程存在性核验”，更接近真实执行理解。

对 P01_01_part1 的现实约束：

- item 内没有 mp4；`last_frame_segments/` 与 `cumulative_last_frame_segments/` 为空；必须先补齐源视频与片段资产，或使用“完整视频 + start/end 秒”运行时裁剪方案。

---

## 7. 结合 P01_01_part1 的“可直接生成”示例（用真实字段证明可落地）

本节只展示少量代表性样例，强调“从 JSON + 现有关键帧”即可生成训练数据；更多示例可参考主规范第 9/10 节。

### 7.1 Task_27 / SS01（Yes/No 空间关系核验，强监督）

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_illuminate_the_space_and_retrieve_a_cucumber_and_a_carrot_from_the_refrigerator/frame_002_ts_3.59s.jpg`
- 字段来源：`spatial_preconditions[0] = {"relation": "hand is in close proximity to light switch", "objects": ["hand","light_switch"], "truth": true}`
- 训练样本（JSONL 一条）要点：
  - prompt 用 `<image>` tag
  - 问句强制输出 `Yes/No`
  - 答案只输出 `Yes`

### 7.2 Task_03 / SS02（ABCD 可供性类型，强监督）

- 证据（keyframe_single）：同上或其他关键帧
- gold：`affordance_hotspot.affordance_type = "pressable_surface"`
- 干扰项：从全局 affordance_type 词表采样 3 个（必须不同）
- 训练样本要点：meta 里写入 `options`，答案只输出 A/B/C/D

### 7.3 Task_26 / SS03（A/B 时间顺序，强监督）

从 P01_01_part1 抽两张关键帧：

- A：Step 1 `frame_002_ts_3.59s.jpg`
- B：Step 3 `frame_020_ts_68.39s.jpg`

标签来自 `3.59 < 68.39`：A 更早。

若使用 images-only：直接给两张图。若使用 mp4：更推荐裁剪 `two_event_window` clip（需要视频资产）。

### 7.4 Task_04 / SS04（工具/材料角色，Yes/No，强监督）

选取 Step 2 的最早关键帧（更可能看到刀/案板/台面）：

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/02_assemble_the_preparation_workspace_by_placing_vegetables_on_the_counter_then_retrieving_a_cutting_board_and_knife/frame_012_ts_39.59s.jpg`
- gold 字段（来自 Step 2）：  
  `tools = ["kitchen_cupboard","cutlery_drawer","cutting_board","knife"]`  
  `materials = ["cucumber","carrot"]`

可构造的严格评分问句示例：

- Q（tool 判别）：`<image>\nIn this step, is knife a tool used by the agent? Only output Yes or No.`
- A：`Yes`

注意：为了避免退化为纯文本背诵，建议 `x`（这里是 knife）尽量来自该关键帧可见对象集合（或来自 `spatial_preconditions/affordance_preconditions` 里出现的 object_name/objects）。

### 7.5 Task_10 / SS05（step_goal 匹配，ABCD，强监督）

选取 Step 3 的关键帧（洗菜动作很明显）：

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/03_wash_the_cucumber_and_carrot_in_the_sink_under_running_water/frame_020_ts_68.39s.jpg`
- 正确 step_goal（来自 Step 3）：`Wash the cucumber and carrot in the sink under running water.`
- 干扰 step_goal（来自同 item 其他 step）：Step 1/2/4/5/6 中抽 3 条

ABCD 题面示例（仅示意，选项顺序应随机并写入 meta.options）：

```
<image>
Which step goal best matches what is happening in this image?
Options:
(A) Wash the cucumber and carrot in the sink under running water.
(B) Retrieve a frying pan from a low cupboard and place it on the stovetop.
(C) Dice a portion of the zucchini slices into small cubes.
(D) Enter the kitchen, illuminate the space, and retrieve a cucumber and a carrot from the refrigerator.
Only output A, B, C, or D.
```

答案：`A`

---

## 8. 最终训练配方（建议作为“最终版本”的工程落地默认值）

### 8.1 训练集任务配比（建议）

为了让模型既能稳定对齐（客观题），又能扩展解释能力（主观题），建议按以下配比构建训练集（可按需求调整）：

- 60%：强监督 strict scoring/客观题（SS01–SS05 为核心；视频可用时加入 SS06–SS07）
- 25%：规划/时序任务（Task_22/23/24/26/28/29/30 的客观化版本）
- 15%：主观解释任务（Task_06/17/20/25 等，必须 judge 打分或经过文本打磨）

### 8.2 评测集（回归）建议

强烈建议把评测集锁定在 strict scoring 任务上：

- 图像模型：SS01–SS05（images-only）
- 视频模型：SS01–SS07（mp4）

这样你可以做到：

- 每次改动生成器/模型/数据后，快速得到可比的准确率指标
- 避免主观题评测波动

### 8.3 数据质量硬约束（生成器必须 enforce）

对每条样本：

1) 媒体必须可读：图片存在/视频存在（否则丢弃，不要留在训练集）
2) prompt 的 `<image>/<video>` tag 数量与媒体数量一致
3) 客观题答案必须是单行标签（严格匹配 Yes/No 或 A/B 或 A/B/C/D）
4) `meta` 必须记录：`task_name/evidence_type/evidence_files/source_path/step_id/frame_index/ts/options/neg_sample`
5) 弱监督任务必须标记 `meta.weak_supervision=true`，避免与强监督混在一起做准确率回归

---

## 9. 资产补齐建议（让 Task_01/07/18/20/22/23/24/25/28/29/30 真正可用）

结合 P01_01_part1 的缺失项，若要完整覆盖四类证据与主规范任务，建议按以下顺序补齐：

1) **补齐源视频到 item 内**：`<item_dir>/source_video.mp4`
2) **生成 `sampled_frames/`**：全局均匀抽 50 帧（或主规范建议的数量），用于 `images_uniform_scene`
3) **生成 `last_frame_segments/`**：相邻 step 尾帧间片段（用于 `video_clip`）
4) **生成 `cumulative_last_frame_segments/`**：从开头到每步尾帧（用于 `video_prefix`）

备注：

- `ECCV/extract_last_frame_segments.py` 与 `ECCV/extract_cumulative_last_frame_segments.py` 需要 ffmpeg；如果环境没有 ffmpeg，可走“运行时裁剪”方案（但训练管线要支持 `video_start/video_end`）。

---

## 10. 最终建议：把“主规范 Task_01–Task_30”真正变成可训练体系的取舍

最后给出一个“务实”的结论：不要追求每个 Task 都以同等比例进入训练集。

- **优先级最高（强烈建议做主力训练 + 回归评测）**：Task_03/04/10/26/27/22/28/29/30（其中 03/04/10/26/27 可直接落 strict scoring）
- **中优先级（补齐视频/scene 资产后再上）**：Task_01/07/18/20/23/24/25
- **低优先级（偏文本监督或主观解释，建议降权）**：Task_08/09/11/14/15/16(主观版)/17/21

原因：

- 强监督客观题能提供稳定梯度与可回归指标。
- 需要“可观测性判定”的任务（18/19）在没有额外标注时天然弱监督，必须降权并明确区分。
- 纯文本复述类任务容易让多模态训练退化成语言建模，必须通过 MCQ/mismatch 等方式把任务“拽回视觉/视频证据”。

---

## 11. 主观题的 judge 打分建议（让弱监督也可用）

对于 Task_06/08/17/20(描述版)/21/25 等主观题，建议使用一个 judge 模型按 rubric 打分，并把分数写入 `meta.judge_score`（或单独保存评测文件）。

推荐 rubric（每项 0–2 分，总分 10 分）：

1) **Fidelity（忠实性）**：是否只使用 `meta.fields` 与证据可支持的信息；是否出现明显幻觉  
2) **Grounding（可落地）**：是否引用了证据中可见的对象/关系/动作；是否泛泛而谈  
3) **Coverage（覆盖度）**：是否覆盖题面要求的字段（例如因果链五元组、Why/How 双段）  
4) **Clarity（清晰性）**：是否表达清晰、无严重语法/逻辑断裂  
5) **Constraint（格式约束）**：是否满足结构要求（如双段/单段）与禁用格式（如不使用 bullet）等

使用建议：

- 低分样本（例如 <6）不要进入训练集，或只进入低权重 bucket。
- judge prompt 必须包含 `meta.fields`（作为“唯一允许事实来源”），并明确禁止引入外部信息。

