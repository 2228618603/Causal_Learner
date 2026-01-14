最终核心严格落在：

- **因果规划（causal planning）**：显式状态/约束（spatial+affordance）→ 可行性 → 因果后果 → 跨步依赖 → 长时序规划（prefix/reorder/infill）
- **失败反思（failure reflecting）**：不一致检测 → 缺陷类型定位（flaw type）→ 恢复策略选择 → 失败驱动重规划（replanning）

### 0.1 四类证据（最终统一口径）

为兼容主规范的 `evidence_type` 枚举与训练落地的可控性，最终建议把证据形态压缩为 4 类（其余如 `images_uniform_clip` 视作工程实现细节）：

1) `keyframe_single`：单张关键帧
2) `images_uniform_scene`：全局视频均匀抽帧的多图
3) `video_clip`：两个step之间的局部视频片段
4) `video_prefix`：累积前缀片段

优先级最高的重复/低增益点：

- `Task_08` 与 `Task_17` 重叠：`Task_17` 已覆盖 Why/How（含机制/因果），建议将 `Task_08` 合并为 `Task_17` 子变体或显著降权。我的想法是删除task17，另外再看有没有别的任何使用跟how相关（mechanism）

- `Task_07` 与 `Task_23` 功能相近：有 `video_prefix` 时优先 `Task_23`；`Task_07` 作为无视频 fallback。删除任务7，修改任务23为直接给完整视频，推测high_level_goal。

- `Task_11` 与 `Task_19` 都围绕“效果”：若强调证据闭环与失败反思，优先 `Task_19`；`Task_11` 更偏计划字段复述，不宜做核心指标。删除任务19。

- `Task_20/21/25` 更偏辅助能力（边界/关键帧理由/进度总结），不建议进入核心指标或占比过高。删除任务20，21，25

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

### 6.4 Task_14/15 的推荐客观化变体（使其成为“可回归”的失败反思指标）

- `Task_14`：Counterfactual Outcome MCQ（ABCD）  
  - gold：`expected_challenge_outcome`
  - 干扰：来自其他 step/item 的 outcome
- `Task_15`：Recovery Strategy MCQ（ABCD）  
  - gold：`failure_handling.recovery_strategy`
  - 干扰：来自其他 step/item 的 recovery

task28可以用原始的多个step的plan，随机修改其中的一个step plan为一个错误的sub-plan

### 12.5 Task_10 / SS05（step_goal 匹配，ABCD，强监督）

- 证据（keyframe_single）：  
  `ECCV/causal_spafa_plan_dataset_long/P01_01_part1/03_wash_the_cucumber_and_carrot_in_the_sink_under_running_water/frame_020_ts_68.39s.jpg`
- 正确项：该 step 的 `step_goal`
- 干扰项：同 item 其他 3 个 `step_goal`
- 输出：只输出 `A/B/C/D`

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
