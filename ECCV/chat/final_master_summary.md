# mani_longvideo 任务体系 Final Master Summary（Task_01–Task_30 × 四类证据 × Strict Scoring × Causal Planning & Failure Reflecting）

本文档是对以下两份“最终方案文档”的**合并与再收敛版本**，目标是生成一个**单文件、可执行、可评测、紧扣研究主题**的 master summary：

- `ECCV/chat/mani_longvideo_tasks_plan_final_master_summary.md`（工程落地总方案：证据、schema、QA 格式、Task 总表、strict scoring、训练配方）
- `ECCV/chat/mani_longvideo_tasks_plan_final_causal_planning_failure_reflecting_audit.md`（面向主题的审计与重构建议：聚焦因果规划与失败反思、去冗余、新增更贴题且可评分的任务）

最终核心严格落在：

- **因果规划（causal planning）**：显式状态/约束（spatial+affordance）→ 可行性 → 因果后果 → 跨步依赖 → 长时序规划（prefix/reorder/infill）
- **失败反思（failure reflecting）**：不一致检测 → 缺陷类型定位（flaw type）→ 恢复策略选择 → 失败驱动重规划（replanning）

> 本文件不替代主规范 `ECCV/mani_longvideo_tasks_plan_final.md` 的 Task 定义细节；它提供“如何把规范变成可训练/可评测数据”的最终口径与取舍。

---

## 参考材料（必须一致）

- 主任务规范：`ECCV/mani_longvideo_tasks_plan_final.md`
- 严格评分（仅图像证据）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`
- 严格评分（支持 MP4 完整视频/clip）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`
- 对齐与 QA 策略：`ECCV/chat/mani_longvideo_tasks_plan_final_alignment_and_qa_strategy.md`
- 真实 item（样例）：`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/`

---

## 0. 最终目标与可执行结论（先给工程与研究都能用的结论）

### 0.1 四类证据（最终统一口径）

为兼容主规范的 `evidence_type` 枚举与训练落地的可控性，最终建议把证据形态压缩为 4 类（其余如 `images_uniform_clip` 视作工程实现细节）：

1) `keyframe_single`：单张关键帧（最稳、成本最低、最易 strict scoring）  
2) `images_uniform_scene`：全局均匀抽帧的多图（覆盖环境/阶段变化）  
3) `video_clip`：局部片段（动作过程/边界/对齐最依赖）  
4) `video_prefix`：累积前缀片段（进度/下一步预测/重排最依赖）

### 0.2 主题聚焦：把 30 个任务“分层使用”，而不是平均用

如果论文/系统主线严格是 **causal planning + failure reflecting**，建议把 Task_01–Task_30 变成“三层使用策略”（不必修改编号；先在数据配比与评测指标层面收敛）：

**核心 · 因果规划（Causal Planning）**  
- `Task_06/12/16/23/24/26/29/30`
  - 局部因果链与机制（06）
  - 跨步依赖（12）
  - 物理可行性（16）
  - 长视频前缀规划（23/24）
  - 时序一致性（26）
  - 长时序规划（29/30）

**核心 · 失败反思（Failure Reflecting）**  
- `Task_22/28` + `Task_14/15` 的**客观化变体**（见第 6 节新增/变体任务）
  - 执行一致性检测（22）
  - 错误计划缺陷定位（28）
  - 反事实 outcome / recovery 策略选择（14/15 的 MCQ/标签版）

**支撑 · 因果落地的感知对齐（Grounding，比例要小）**  
- `Task_03/04/05/10/27`（`Task_01/07` 视资产情况作为补充）

> 支撑任务的目的：让后续规划/反思必须“落到对象/空间关系/可供性/状态变化”上，而不是纯语言空转。

### 0.3 去冗余与降权（避免训练信号被重复任务稀释）

优先级最高的重复/低增益点：

- `Task_02` 与 `Task_27` 高度重叠：默认以 `Task_27`（Yes/No 核验）为主，`Task_02` 仅在你明确需要“开放式关系抽取”时保留。
- `Task_08` 与 `Task_17` 重叠：`Task_17` 已覆盖 Why/How（含机制/因果），建议将 `Task_08` 合并为 `Task_17` 子变体或显著降权。
- `Task_07` 与 `Task_23` 功能相近：有 `video_prefix` 时优先 `Task_23`；`Task_07` 作为无视频 fallback。
- `Task_11` 与 `Task_19` 都围绕“效果”：若强调证据闭环与失败反思，优先 `Task_19`；`Task_11` 更偏计划字段复述，不宜做核心指标。
- `Task_20/21/25` 更偏辅助能力（边界/关键帧理由/进度总结），不建议进入核心指标或占比过高。

### 0.4 必须补强的缺口：可评分的“后置状态（postcondition）核验”

现有体系里 precondition/effect 很多是自由文本（Task_09/11/18/19），强监督不足。  
但 `causal_plan_with_keyframes.json` 实际提供了结构化后置状态：

- `steps[i].spatial_postconditions_detail[*].truth`
- `steps[i].affordance_postconditions_detail`

建议新增两类 strict scoring 任务（见第 6 节），它们更贴“因果后果/可执行性”的核心主题，且可稳定自动评分。

---

## 1. 真实 item 审计：P01_01_part1 现状（决定哪些任务能直接落地）

`ECCV/causal_spafa_plan_dataset_long/P01_01_part1/` 当前包含：

- `causal_plan_with_keyframes.json`（核心监督来源）
- step 子目录与关键帧 jpg（可直接支持 `keyframe_single` 类任务）
- `run_summary.json`（可追溯源视频路径）
- `last_frame_segments/`（空目录，未生成 mp4）
- `cumulative_last_frame_segments/`（空目录，未生成 mp4）
- 缺失：`sampled_frames/`（全局均匀抽帧）
- 缺失：源视频文件（mp4 不在 item 内）

结论：

- **可直接生成**：以 `keyframe_single` 为证据的任务 + images-only strict scoring 的大部分任务。
- **需要补资产才能“真多模态”**：
  - `images_uniform_scene`：最好补齐 `sampled_frames/`
  - `video_clip/video_prefix`：必须补齐源视频与 clip/prefix（或用运行时裁剪方案）

资产补齐建议见第 9 节。

---

## 2. JSON schema 与主规范对齐要点（生成器必须处理的两个坑）

### 2.1 `keyframe_image_path` 可移植性（绝对路径问题）

真实 JSON 常写入生成机器相关的绝对路径（如 `/e2e-data/...`）；换环境不可读，但 item 内往往存在同名 jpg。生成器应做 resolve：

1) 若 `keyframe_image_path` 可读 → 直接用  
2) 否则回退到 `<item_dir>/` 下同名文件（可用 glob）  
3) 仍失败 → 丢弃样本（不要留训练集）

建议在最终数据集中存 **相对 item_dir 的路径**，本地调试时在 `meta.abs_path_debug` 保留绝对路径。

### 2.2 `mechanism` 字段不一致（affordance_hotspot vs causal_chain）

主规范的 Task_03/06/17 会引用 `affordance_hotspot.mechanism`，但真实 JSON 中常见：

- `affordance_hotspot` 只有 `{description, affordance_type, causal_role}`  
- 机制细节更常出现在 `causal_chain.causal_affordance_focus_detail`

推荐统一派生：

- `mechanism := affordance_hotspot.mechanism` 若存在  
- 否则 `mechanism := causal_chain.causal_affordance_focus_detail`  
- 并在 `meta.schema_fallbacks.mechanism_source` 记录来源

---

## 3. 四类证据：生成、回退、以及最适合的任务

### 3.1 `keyframe_single`

- 来源：`steps[i].critical_frames[j].keyframe_image_path`（需路径归一化）
- 适用（强推荐）：
  - `Task_27/SS01` 空间关系核验（Yes/No）
  - `Task_03/SS02` 可供性类型（MCQ）
  - `Task_04/SS04` 工具/材料角色（Yes/No）
  - `Task_10/SS05` step_goal 对齐（MCQ）
  - `Task_26/SS03` 时间顺序（images-only：两关键帧 A/B）
  - `Task_06/17` 因果链/Why-How（主观+judge）
- 优势：最稳、最便宜、最易 strict scoring；作为训练主力与回归评测基础。
- 风险：动作过程缺失；边界/进度/对齐等任务不适合只用单帧。

### 3.2 `images_uniform_scene`

- 标准来源：`<item_dir>/sampled_frames/sample_*.jpg`（等距取 4–8 张）
- 回退策略：
  1) 若缺失：用“每步最早关键帧集合”做代理（再等距采样到 4–8 张）
  2) 仍不足：丢弃样本
- 适用：
  - `Task_01` 锚点对象
  - `Task_07` 高阶目标（fallback）
  - `Task_30` head-tail infill 的 head/tail 证据
  - 无视频 fallback：`Task_26/28` 等

### 3.3 `video_clip`

两种工程实现：

1) 离线保存 clip `.mp4`（最兼容训练）  
2) 运行时裁剪：完整视频 + `video_start_sec/video_end_sec`（依赖读取器支持）

推荐来源优先级：

- 三阶段 step clip（若存在）：`<item_dir>/stage2/step_clips/...`
- `last_frame_segments/*.mp4`
- 完整视频 `source_video.mp4` + `[start,end]`（来自关键帧 `ts`）

适用：

- `Task_20` step 边界/转折
- `Task_22` 计划-执行一致性（match/mismatch）
- mp4 strict scoring 的 SS06/SS07（过程存在性核验）

### 3.4 `video_prefix`

- 离线：`cumulative_last_frame_segments/segment_start_to_stepXX_last.mp4`
- 或运行时：完整视频 + `[0, end_ts]`

适用：

- `Task_23/24` prefix→goal/next step（强推荐做 MCQ）
- `Task_26` 时间顺序（可用更长上下文）
- `Task_28/29` 错误计划诊断/未来 K 步重排
- `Task_18/25`（弱监督：可观测性/进度总结）

---

## 4. 训练数据 JSON 统一格式与输出协议（必须统一，否则无法回归评测）

### 4.1 样本结构（ShareGPT 风格，强化 meta 可追溯）

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

### 4.2 strict scoring 输出协议（强烈建议统一）

- Yes/No：题面结尾强约束 `Only output Yes or No.`；答案只输出 `Yes` 或 `No`
- A/B：题面结尾强约束 `Only output A or B.`；答案只输出 `A` 或 `B`
- A/B/C/D：题面含 Options；题面结尾强约束 `Only output A, B, C, or D.`；答案只输出字母
- ordered_list（Task_29/30）：题面要求严格编号 1..K；答案逐行 `1) ...`
- structured（Task_28）：单行 `FlawStep=<int>; FlawType=<type>; Reason=<one sentence>.`

> 客观题 `options` 必须落盘在 `meta.options`，但不要把 meta 注入 prompt（防止泄露）。

---

## 5. Task_01–Task_30：最终推荐的证据选择 + QA 题型 + 评分方式（总表）

说明：

- “默认证据”只从四类中选：`keyframe_single/images_uniform_scene/video_clip/video_prefix`
- “推荐 QA（训练版）”优先 strict scoring（可回归评测）
- “P01_01_part1 可用性”基于当前 item 资产：仅 keyframes 已齐，scene/video 需补齐

| Task | 核心能力 | 默认证据（四类） | 推荐 QA（训练版） | 评分/监督来源 | 负样本/扰动 |
|---|---|---|---|---|---|
| 01 | 场景锚点/关键对象 | images_uniform_scene | 多选/列表（建议转多轮 Yes/No 或多选标签） | 由 JSON 聚合对象词表（弱+规则） | distractor 来自全局对象池 |
| 02 | 空间关系抽取 | keyframe_single | 建议并入 27（Yes/No） | `spatial_preconditions[*].truth` | 取反 truth / 替换 objects |
| 03 | 可供性热点类型 | keyframe_single | ABCD（affordance_type MCQ） | `affordance_hotspot.affordance_type` | 干扰项来自全局词表 |
| 04 | 工具/材料角色 | keyframe_single | Yes/No（tool? / material?） | `tool_and_material_usage` | 跨 step 替换实体 |
| 05 | 动作/状态描述 | keyframe_single | ABCD（caption matching） | `action_description/state_change_description` | 干扰来自其他关键帧 |
| 06 | 因果链解释 | keyframe_single | 双段主观解释（judge） | `causal_chain.*` + mechanism | 不做自动负样本 |
| 07 | 高阶目标识别 | images_uniform_scene | ABCD（high_level_goal MCQ） | `high_level_goal` | 干扰来自其他 item goal |
| 08 | 步骤动机/必要性 | keyframe_single | 主观解释 / rationale MCQ | `steps[i].rationale` | rationale 干扰需同域 |
| 09 | 前置条件陈述 | keyframe_single | 主观列举（建议降权） | `steps[i].preconditions` | 不建议做强负样本 |
| 10 | step_goal 对齐 | keyframe_single / video_clip | ABCD（step_goal matching） | `steps[i].step_goal` | 干扰来自同 item 其他 step_goal |
| 11 | 期望效果 | keyframe_single | 主观列举（建议降权） | `expected_effects` | 不建议做负样本 |
| 12 | 跨步依赖 | keyframe_single | MCQ：哪个 precondition 被满足 | effects↔preconditions 重合规则 | 干扰为非重合 preconditions |
| 13 | 下一步（计划版） | keyframe_single | ABCD（next step_goal MCQ） | `steps[i+1].step_goal` | 干扰来自同 item 其他 step_goal |
| 14 | 反事实挑战 | keyframe_single | **ABCD（outcome MCQ，推荐）** / 主观回答 | `expected_challenge_outcome` | outcome 干扰来自其他 step/item |
| 15 | 失败与恢复 | keyframe_single | **ABCD（recovery MCQ，推荐）** / 主观回答 | `failure_handling.recovery_strategy` | recovery 干扰来自其他 step/item |
| 16 | 可行性核验 | keyframe_single | match/mismatch（二分类） | 证据-步骤配对规则 | 跨 step 错配构造 No |
| 17 | Why/How 综合 | keyframe_single | 双段主观解释（judge） | Why=rationale；How=mechanism/causal_chain | 不做自动负样本 |
| 18 | 视觉前置核验 | video_prefix | 三态：satisfied/not/not observable（弱） | 可观测性启发式 | 不建议强负样本 |
| 19 | 视觉效果核验 | keyframe_single (+clip) | 三态：supported/uncertain/not（弱） | 可见性启发式 | 不建议强负样本 |
| 20 | step 边界/转折 | video_clip | 边界识别 MCQ 或主观描述 | clip 与边界 i→i+1 自动标注 | mismatch 边界为负样本 |
| 21 | 关键帧理由 | keyframe_single | 主观解释（judge） | 关键帧字段集合 | 不建议负样本 |
| 22 | 计划-执行一致性 | video_clip | match/partial/mismatch（三分类） | step-clip 与 step_goal 配对规则 | 跨 step 错配 |
| 23 | 前缀识别目标 | video_prefix | ABCD（high_level_goal MCQ） | `high_level_goal` | 干扰来自其他 item |
| 24 | 前缀预测下一步 | video_prefix | ABCD（next step_goal MCQ） | `steps[i+1].step_goal` | 干扰来自同 item 其他 step_goal |
| 25 | 前缀进度总结 | video_prefix | 主观总结 + judge 打分 | steps[0..i] 仅写 meta | 不建议负样本 |
| 26 | 时间顺序判别 | video_prefix / images_uniform_scene | A/B（谁更早） | ts 比较（来自关键帧文件名） | 事件对采样（需 ts 不等） |
| 27 | 空间关系真假核验 | keyframe_single | Yes/No | `spatial_preconditions[*].truth` | 取反 truth/扰动 objects |
| 28 | 错误计划诊断 | video_prefix / images_uniform_scene | 结构化输出（FlawStep/Type） | bad_plan 扰动算子 | 单样本只注入 1 个错误 |
| 29 | 未来 K 步重排 | video_prefix | 严格 1..K 输出 | gold 顺序来自 JSON | shuffle 作为输入扰动 |
| 30 | head-tail 补全中间步骤 | images_uniform_scene | 严格 1..M 输出 | middle steps 来自 JSON | 随机锚点增强难度 |

> `Task_14/15` 在“主题聚焦”下建议以 MCQ/标签版作为核心评测；自由文本版仅用于 SFT 小比例补充。

---

## 6. 新增/变体任务（强烈推荐）：补上 postcondition 核验 + 失败驱动重规划闭环

这一节给出最关键的“主题增强补丁”：仅使用现有 `causal_plan_with_keyframes.json` 字段即可自动构造，且适合 strict scoring 回归评测。

### 6.1 新增任务 A：Spatial Postcondition Check（空间后置状态核验，Yes/No）

- 字段来源：`steps[i].spatial_postconditions_detail[*].relation/objects/truth`
- 证据：step i 尾关键帧（优先）或 step clip 抽帧
- 输出：`Yes/No`（exact match）
- 价值：把“动作→空间状态成立/不成立”的因果后果变成可评分监督（强于自由文本 `expected_effects`）

### 6.2 新增任务 B：Affordance Postcondition Check（可供性后置状态核验，Yes/No 或 MCQ）

- 字段来源：`steps[i].affordance_postconditions_detail[*].object_name/affordance_types/reasons`
- 证据：step i 尾关键帧
- 输出：
  - Yes/No：给定 object+affordance_type 判别是否成立
  - 或 ABCD：给定 object 选正确 affordance_type
- 价值：直接监督“动作使对象获得/保持某种可操作性”（可执行规划的核心）

### 6.3 新增任务 C：Failure-Driven Replanning / Recovery Insertion（失败驱动重规划，建议拆成两段客观题）

仅复述 `recovery_strategy` 不足以形成 failure reflecting 闭环；建议拆成两段可评分任务：

1) **Recovery Strategy Selection（ABCD）**  
   - 输入：失败描述 + 证据  
   - 输出：A/B/C/D（选择正确 `failure_handling.recovery_strategy`）

2) **Next Step After Recovery（ABCD 或排序）**  
   - 输入：已选 recovery + 证据（或仅失败描述）  
   - 输出：下一步 step_goal（或候选集合排序）  

> 自由文本版（插入恢复步骤并继续规划）可作为 SFT 小比例扩展，但不建议作为回归指标。

### 6.4 Task_14/15 的推荐客观化变体（使其成为“可回归”的失败反思指标）

- `Task_14`：Counterfactual Outcome MCQ（ABCD）  
  - gold：`expected_challenge_outcome`
  - 干扰：来自其他 step/item 的 outcome
- `Task_15`：Recovery Strategy MCQ（ABCD）  
  - gold：`failure_handling.recovery_strategy`
  - 干扰：来自其他 step/item 的 recovery

### 6.5 Task_28（错误计划诊断）建议的 flaw_type 枚举与落盘字段（保证可回归）

为保证 failure reflecting 的训练信号稳定，建议将 Task_28 的错误类型固定在一个可控枚举集合（允许多标签但要慎用）：

- `order_violation`
- `precondition_missing`
- `spatial_infeasible`
- `affordance_incompatible`
- `tool_mismatch`

生成侧建议：

- 每条 bad_plan **只注入 1 个错误**（便于定位与评分）
- 在 `meta` 落盘：`flaw_type`, `flaw_step_id`, `bad_plan_text`, `gold_plan_text`（可选），以及可选 `fix_hint`（例如 recovery_strategy 或“需要先完成的前置步骤”）

---

## 7. strict scoring 子集：最终可评测套件（建议扩展加入 postcondition）

### 7.1 Images-only strict scoring（SS01–SS05）

来源：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`

- SS01 ↔ Task_27（空间关系 Yes/No）
- SS02 ↔ Task_03（可供性类型 ABCD）
- SS03 ↔ Task_26（时间顺序 A/B）
- SS04 ↔ Task_04（工具/材料 Yes/No）
- SS05 ↔ Task_10（step_goal 匹配 ABCD）

### 7.2 MP4 strict scoring（SS01–SS07）

来源：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`

相对 images-only 的增益：

- 可构造 `keyframe_window/two_event_window` clip，把问题对齐到 clip 尾部，减少单帧歧义
- 新增 SS06/SS07：把 `action_description/state_change_description` 变成“过程存在性核验”

### 7.3 建议加入的扩展 strict scoring（本文件新增）

- **SP01_Spatial_Postcondition_Check**（新增任务 A）  
- **SP02_Affordance_Postcondition_Check**（新增任务 B）  
- **FR01_Recovery_Strategy_MC**（Task_15 变体）  
- **FR02_Counterfactual_Outcome_MC**（Task_14 变体）

它们使评测套件更贴“因果后果 + 失败纠错”的主题主线。

---

## 8. 最终训练配方（建议作为工程落地默认值）

### 8.1 训练集任务配比（建议）

建议按“强监督回归指标优先”的原则配比：

- 55–65%：strict scoring/客观题（SS01–SS05 为核心；视频可用时加入 SS06–SS07；并加入新增 SP01/SP02/FR01/FR02）
- 20–30%：规划主线任务（Task_22/23/24/26/28/29/30 的客观化版本）
- 10–15%：主观解释任务（Task_06/17 等，必须 judge 打分或文本打磨）

### 8.2 评测集（回归）建议

评测集尽量锁定在 strict scoring：

- 图像模型：SS01–SS05 + SP01/SP02 + FR01/FR02
- 视频模型：SS01–SS07 + SP01/SP02 + FR01/FR02

### 8.3 数据质量硬约束（生成器必须 enforce）

对每条样本：

1) 媒体必须可读（否则丢弃）  
2) `<image>/<video>` tag 数量与媒体数量一致  
3) 客观题答案必须是单行标签（Yes/No 或 A/B 或 A/B/C/D）  
4) `meta` 必须记录：`task_name/evidence_type/evidence_files/source_path/step_id/frame_index/ts/options/neg_sample/schema_fallbacks`  
5) 弱监督任务必须标记 `meta.weak_supervision=true`，不要与强监督一起做准确率回归  

---

## 9. 资产补齐建议（让 prefix/clip/scene 任务真正可用）

按优先级补齐：

1) **补齐源视频到 item 内**：`<item_dir>/source_video.mp4`（确保跨机器可复现）
2) **生成 `sampled_frames/`**：全局均匀抽帧（用于 `images_uniform_scene`）
3) **生成 `last_frame_segments/`**：相邻 step 尾帧间片段（用于 `video_clip`）
4) **生成 `cumulative_last_frame_segments/`**：从开头到每步尾帧（用于 `video_prefix`）

工具：

- `python ECCV/extract_last_frame_segments.py --video-output-dir <item_dir>`
- `python ECCV/extract_cumulative_last_frame_segments.py --video-output-dir <item_dir>`

> 若环境无 ffmpeg，可走“运行时裁剪”（完整视频 + start/end 秒），但训练/loader 需支持按时间取段。

---

## 10. 最终建议：把 Task_01–Task_30 真正变成“贴题”的可训练体系

### 10.1 以主题为导向的最终优先级（推荐）

- **最高优先级（强烈建议做主力训练 + 回归评测）**：  
  `Task_03/04/10/16/22/23/24/26/27/28/29/30` + 新增 `SP01/SP02` + `FR01/FR02`
- **中优先级（补齐视频/scene 资产后再上）**：  
  `Task_01/07/18/20/25`（以及 22/23/24/28/29 在视频模式下的增强版本）
- **低优先级（偏主观解释或弱监督，建议降权）**：  
  `Task_06/17/21`（主观解释可保留但要 judge）、以及所有“纯字段复述型”的自由文本版本

### 10.2 一句话原则

- **核心任务要么能严格评分，要么能显式输出约束/缺陷类型**；否则就不要当主指标。  
- **失败反思必须形成闭环**：诊断（22/28）→ 选择修复（FR01）→ 继续规划（新增 C 的第二段）。

---

## 11. 生成配额建议（避免关系爆炸/文本堆叠）

为了让数据集规模与质量可控，建议在生成时为关键任务设置“每个 item/step/keyframe 的上限”：

- `Task_27/SS01`：每个关键帧抽 1–2 条 `spatial_preconditions`（避免关系爆炸）
- `Task_03/SS02`：每个关键帧 1 条（热点通常只有 1 个）
- `Task_04/SS04`：每个 step 1–2 条（问 tool 或问 material；再配 0–1 条 hard negative）
- `Task_10/SS05`：每个 step 1 条（同 item 内其他 step_goal 做干扰）
- `Task_26/SS03`：每个 item 采样 3–8 对事件（跨 step 优先；必须 `ts_a != ts_b`）
- `Task_22`：每个 step 1 条 match + 1 条 mismatch（mismatch 从同 item 其他 step_goal 抽）
- `Task_28`：每个 prefix_end_step 采样 1 个 bad_plan（每条样本只注入 1 个错误）
- `Task_29`：每个 prefix_end_step 采样 1 个（K=3–6）重排任务
- `Task_30`：每个 item 1 条（或按锚点采样 2–3 条增强）
- 新增 `SP01/SP02`：每个 step 抽 1–2 条 postcondition（优先视觉可核验的关系/affordance）

---

## 12. 结合 P01_01_part1 的“可直接生成”样例（用真实字段证明可落地）

本节展示少量代表性样例，强调“从 JSON + 已有关键帧”即可生成训练数据；更大规模样例可参考主规范与 strict scoring 文档。

### 12.1 Task_27 / SS01（Yes/No 空间关系核验，强监督）

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_illuminate_the_space_and_retrieve_a_cucumber_and_a_carrot_from_the_refrigerator/frame_002_ts_3.59s.jpg`
- 字段来源：`spatial_preconditions[0] = {"relation": "hand is in close proximity to light switch", "objects": ["hand","light_switch"], "truth": true}`
- 题面关键约束：`Only output Yes or No.`
- 答案：`Yes`

### 12.2 Task_03 / SS02（ABCD 可供性类型，强监督）

- 证据：同上或其他关键帧  
- gold：`affordance_hotspot.affordance_type = "pressable_surface"`
- 干扰项：从全局 affordance_type 词表采样 3 个（必须不同）
- 输出：只输出 `A/B/C/D`（exact match）

### 12.3 Task_26 / SS03（A/B 时间顺序，强监督）

从 P01_01_part1 抽两张关键帧：

- A：`.../frame_002_ts_3.59s.jpg`
- B：`.../frame_020_ts_68.39s.jpg`

标签来自 `3.59 < 68.39`：A 更早。  
images-only 直接给两张图；mp4 模式更推荐裁剪 `two_event_window` clip（需要视频资产）。

### 12.4 Task_04 / SS04（工具/材料角色，Yes/No，强监督）

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/02_assemble_the_preparation_workspace_by_placing_vegetables_on_the_counter_then_retrieving_a_cutting_board_and_knife/frame_012_ts_39.59s.jpg`
- gold 字段（来自该 step）：`tools/materials`
- 示例问句：`In this step, is knife a tool used by the agent? Only output Yes or No.`
- 答案：`Yes`

### 12.5 Task_10 / SS05（step_goal 匹配，ABCD，强监督）

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/03_wash_the_cucumber_and_carrot_in_the_sink_under_running_water/frame_020_ts_68.39s.jpg`
- 正确项：该 step 的 `step_goal`
- 干扰项：同 item 其他 3 个 `step_goal`
- 输出：只输出 `A/B/C/D`

---

## 13. 主观题的 judge 打分建议（让弱监督也可用）

对于 `Task_06/17/20(描述版)/21/25` 等主观题，建议使用 judge 模型按 rubric 打分，并把分数写入 `meta.judge_score`（或单独保存评测文件）。

推荐 rubric（每项 0–2 分，总分 10 分）：

1) **Fidelity（忠实性）**：是否只使用 `meta.fields` 与证据可支持的信息；是否出现明显幻觉  
2) **Grounding（可落地）**：是否引用证据中可见的对象/关系/动作；是否泛泛而谈  
3) **Coverage（覆盖度）**：是否覆盖题面要求的字段（例如因果链五元组、Why/How 双段）  
4) **Clarity（清晰性）**：是否表达清晰、无严重语法/逻辑断裂  
5) **Constraint（格式约束）**：是否满足结构要求（如双段/单段）与禁用格式（如不使用 bullet）等

使用建议：

- 低分样本（例如 <6）不要进入训练集，或只进入低权重 bucket。  
- judge prompt 必须包含 `meta.fields`（作为“唯一允许事实来源”），并明确禁止引入外部信息。  

