# mani_longvideo_tasks_plan_final（Task_01–Task_30）任务体系审计与重构建议（聚焦 Causal Planning & Failure Reflecting）

本文档在 `ECCV/mani_longvideo_tasks_plan_final.md` 的 Task_01–Task_30 基础上，做一次**面向研究主题**的系统性审计与重构建议：  
最终核心严格落在 **因果规划（causal planning）** 与 **失败反思（failure reflecting）** 上，优先保证任务集合：

1) 能从现有 `causal_plan_with_keyframes.json` **稳定自动构建**；  
2) 尽可能形成“**证据 → 约束/因果 → 计划/纠错**”的闭环；  
3) 优先可做成 **客观可评分（strict scoring）** 的监督形式；  
4) 最小化冗余与“字段复述型”任务比例，避免训练信号被纯文本背诵淹没。

相关规范/补充：
- 主规范：`ECCV/mani_longvideo_tasks_plan_final.md`
- 一致性与生成策略：`ECCV/chat/mani_longvideo_tasks_plan_final_alignment_and_qa_strategy.md`
- 严格评分（images only）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_images_only.md`
- 严格评分（mp4 video）：`ECCV/chat/mani_longvideo_tasks_plan_strict_scoring_mp4_video.md`

---

## 0. 结论先行（建议怎么改）

如果你们最终论文/系统的“主线”就是 **causal planning + failure reflecting**，建议把 Task_01–Task_30 变成“三层使用策略”（不必修改编号；先在数据配比与评测指标层面收敛）：

### 0.1 核心任务（推荐作为主训练/主评测指标）

**Causal Planning（规划主线）**  
- `Task_06`（关键帧因果链/机制解释）
- `Task_12`（跨步依赖：effect → next precondition）
- `Task_16`（物理可行性：空间/可供性约束）
- `Task_23/24`（prefix → goal / next step）
- `Task_26`（长时序先后判别）
- `Task_29/30`（prefix reordering / head-tail infill）

**Failure Reflecting（失败反思主线）**  
- `Task_22`（计划-执行一致性：match/partial/mismatch）
- `Task_28`（错误计划缺陷定位：flaw type + reason）
- `Task_14/15`（反事实挑战 + 失败原因/恢复策略；建议做成 MCQ/标签变体以便回归评测）

### 0.2 支撑任务（为“因果/反思”提供落地锚点；比例要小）

- `Task_03`（可供性热点语义）
- `Task_04`（工具/材料角色）
- `Task_05`（动作-状态变化）
- `Task_27`（空间关系真假核验）
- `Task_10`（step_goal 与视觉证据对齐；强烈建议用 MCQ 形式）

> 这些任务的价值是：让后续“因果解释/纠错”能引用可检验的对象、关系、可供性与状态变化，减少空泛 reasoning。

### 0.3 建议合并/降权/可删的优先级（直接减少冗余）

- **合并/替代**：`Task_02` ≈ `Task_27`（空间关系核验）  
  - 若追求稳定训练/评测，建议以 `Task_27`（Yes/No）为主；`Task_02` 仅在需要“开放式抽取 relation 文本”时保留。
- **合并/降权**：`Task_08` 与 `Task_17`（Why/How）重叠  
  - `Task_17` 已覆盖 Why/How，建议将 `Task_08` 并入 `Task_17` 的一个子变体或显著降权。
- **降权**：`Task_07`（scene goal derivation）与 `Task_23`（prefix goal）功能相近  
  - 有 `video_prefix` 时优先用 `Task_23`；`Task_07` 作为“无视频，仅抽帧”fallback。
- **降权/择一**：`Task_11`（expected effects）与 `Task_19`（visual effect check）  
  - 若强调“证据闭环 + 失败反思”，优先 `Task_19`；`Task_11` 更偏“计划字段复述”，不建议做核心指标。
- **可删/低配比**：`Task_20/21/25`（边界定位/关键帧理由/进度总结）  
  - 对主题主线的直接增益较低，可作为扩展能力但不宜占比高。

### 0.4 明确缺口：缺少“后置状态（postcondition）的可评分核验任务”

当前体系里 “precondition/effect” 多以自由文本出现（Task_09/11/18/19），但可评分、可对齐的“状态真值”监督相对弱。  
而 `causal_plan_with_keyframes.json` **实际上提供了结构化后置状态**：

- `steps[i].spatial_postconditions_detail[*].truth`
- `steps[i].affordance_postconditions_detail`

建议新增两类任务（强烈推荐做成 strict scoring），见第 3 节。

---

## 1. 主题定义与设计原则（保证任务不跑偏）

### 1.1 主题 1：因果规划（Causal Planning）

我们希望模型具备的不仅是“下一句文本”，而是能用**可执行约束**组织计划：

- **状态表示**：当前场景/对象的可操作性与空间关系是什么（可从关键帧/视频前缀感知）。
- **动作可行性**：动作是否满足空间/可供性前置条件（feasibility）。
- **因果后果**：动作会导致对象/环境状态如何变化（effect/postcondition）。
- **跨步依赖**：上一步效果如何满足下一步前置条件（dependency）。
- **长时序规划**：给定部分观察（prefix / head-tail），能预测后续步骤顺序/缺失步骤（reorder/infill）。

### 1.2 主题 2：失败反思（Failure Reflecting）

失败反思应包含两部分闭环：

1) **诊断**：发现不一致/不可行，并定位是哪类约束被违反（order/precondition/tool/affordance/spatial）。  
2) **修复**：给出合理恢复策略（recovery），并把后续计划调整到可执行（replanning）。

### 1.3 设计原则（用于审计任务“是否值得留”）

对每个任务，我们用以下标准判断是否与主题强相关、是否值得保留/做核心：

- **P1 证据闭环**：输出是否能被图像/视频证据支撑（而不是纯文本背诵）。
- **P2 因果/约束显式化**：是否要求模型显式引用 precondition/effect/affordance/spatial 等约束。
- **P3 可自动标注/可自动评分**：优先 Yes/No、A/B、A/B/C/D、排序等形式（利于回归评测与稳定训练）。
- **P4 负样本可控**：是否能稳定构造高质量负样本（避免“弱负样本”过多导致噪声）。
- **P5 不重复**：是否与其他任务产出高度同质监督信号。

---

## 2. Task_01–Task_30 系统性审计（逐类给结论）

下面把 30 个任务按“主题相关性/多模态依赖/监督强度/建议动作”做一次结构化审计。  
说明：
- **主题相关性**：与 causal planning + failure reflecting 的直接关联强弱。
- **多模态依赖**：任务是否必须“看图/看视频”才能完成。
- **监督强度**：是否容易做成 strict scoring（客观可评测）。
- **建议动作**：Core（核心）、Support（支撑）、Aux（辅助/低比例）、Merge（合并/替代）、Drop（建议删除）。

| Task | 核心内容（1 句话） | 主题相关性 | 多模态依赖 | 监督强度 | 建议动作 | 关键备注（如何改才更贴题） |
|---|---|---:|---:|---:|---|---|
| 01 | 场景锚点对象抽取 | 中 | 中 | 中 | Support/Aux | 建议客观化：多选/Yes-No；否则易退化为“列名词” |
| 02 | 开放式空间关系描述 | 低-中 | 中 | 低 | Merge→27 | 与 27 重叠，且自由文本难评测 |
| 03 | 热点可供性语义+机制 | 中-高 | 高 | 中 | Support | 强烈建议加 MCQ 变体（affordance_type） |
| 04 | 工具 vs 材料角色 | 中 | 中 | 高 | Support | 适合 Yes/No 或 MCQ，注意“可见性”过滤 |
| 05 | 动作+状态变化描述 | 中 | 中 | 中 | Support/Aux | 建议做 MCQ（caption matching）增强视觉依赖 |
| 06 | 关键帧因果链/机制解释 | 高 | 中-高 | 低-中 | Core | 可配结构化输出或分段输出；用于因果主线 |
| 07 | 全局目标识别（scene） | 中 | 中 | 高 | Aux | 有 prefix 时被 23 替代；作为 fallback |
| 08 | 步骤动机/必要性 | 中 | 低-中 | 中 | Merge→17 | 与 17 重叠，建议并入 Why/How |
| 09 | 前置条件复述 | 中 | 低 | 低 | Aux/Drop | 容易纯文本背诵；若保留应做“可观测性”标注或拆 Yes/No |
| 10 | 步骤执行动作描述/对齐 | 中-高 | 中 | 高 | Support/Core? | 建议固定成 MCQ：关键帧/clip → step_goal |
| 11 | 期望效果（文本） | 中 | 低 | 低 | Aux | 更像计划字段监督，不宜做核心指标 |
| 12 | effect → next precondition 依赖 | 高 | 低-中 | 中 | Core | 建议增加客观化变体：依赖点定位 MCQ |
| 13 | 下一步/下一动作预测（计划版） | 中 | 低 | 中 | Aux | 更推荐用 24（视觉 prefix）做核心 |
| 14 | 反事实挑战与结果 | 高（反思） | 低-中 | 中 | Core（变体） | 强烈建议做 MCQ（outcome）以便回归评测 |
| 15 | 失败原因与恢复策略 | 高（反思） | 低-中 | 中 | Core（变体） | 强烈建议做 MCQ（recovery）+ “replanning”扩展 |
| 16 | 物理可行性核验 | 高 | 中 | 中 | Core | 建议提供可控负样本：错配 step_goal/关键帧 |
| 17 | Why/How 综合（步级） | 高 | 中 | 低-中 | Core/Aux | 适合 SFT/解释能力；客观化较难但很贴题 |
| 18 | 视觉前置条件核验 | 中-高 | 高 | 低 | Aux | 强监督困难；若保留需“可观测性启发式”+ weak_supervision 标记 |
| 19 | 视觉效果核验 | 中-高 | 高 | 低 | Aux | 同 18；建议只保留明显可见的 effect 或改用新增 postcondition check |
| 20 | step 边界定位/转折 | 中 | 高 | 中 | Aux | 可做 MCQ（该 clip 对应哪个边界）提高可评测性 |
| 21 | 关键帧选择理由 | 中 | 中 | 低 | Aux | 对主题非刚需；可少量保留做解释能力 |
| 22 | 计划-执行一致性判别 | 高（反思） | 高 | 高 | Core | 易做 match/mismatch/partial；负样本构造稳定 |
| 23 | prefix → high_level_goal | 高 | 高 | 高 | Core | 强烈建议用 MCQ（跨 item 干扰项） |
| 24 | prefix → next step_goal | 高 | 高 | 高 | Core | 规划主线任务；MCQ 或严格复述 |
| 25 | prefix 进度总结 | 中 | 高 | 低 | Aux | 主观题，适合后期扩展；不做核心指标 |
| 26 | 两事件先后判别 | 高 | 中-高 | 高 | Core | label 来自 `ts_`，非常适合作为时序一致性回归 |
| 27 | 空间关系真假核验 | 中-高 | 高 | 高 | Support | 与 02 重叠；建议作为“空间约束 grounding”核心支撑 |
| 28 | 错误计划缺陷定位 | 高（反思） | 中-高 | 中-高 | Core | 关键是“扰动算子”+ flaw_type 标签要稳定 |
| 29 | prefix → 后续 K 步重排 | 高 | 高 | 高 | Core | 适合作为长时序规划回归任务 |
| 30 | head-tail → 中间步骤补全 | 高 | 中 | 中-高 | Core | 适合测试“宏观规划补全”；注意减少纯语言猜测 |

### 2.1 冗余对（最优先处理）

- `Task_02` vs `Task_27`：同为“空间关系”，但 27 有明确 truth label，更适合核心训练/评测。  
- `Task_08` vs `Task_17`：同为 “Why”；17 还额外要求 How（机制/因果），更贴 “因果规划”。  
- `Task_07` vs `Task_23`：同为 “goal recognition”；23 更贴长视频前缀规划主线。  
- `Task_11` vs `Task_19`：都围绕 effect；19 更贴“证据闭环”，但两者都难强监督，建议用第 3 节新增的 postcondition check 替代/补强。

### 2.2 “看起来相关但风险高”的任务（不建议做核心指标）

这些任务确实属于 planning/reflecting，但容易退化成纯文本背诵或弱监督噪声：

- `Task_09/11/13/14/15`：如果问答形式是“直接复述字段”，模型很可能在训练中学到“按模板输出”，而不是从视觉证据出发做因果/反思。  
  **建议**：优先改成 MCQ/标签式变体，或者把这些任务当作 SFT 小比例补充。
- `Task_18/19/25`：强监督难，因为 preconditions/effects 多为自然语言且未标注“可观测性”。  
  **建议**：显式标记 `weak_supervision=true`，并只保留“明显可见”的子集（或用新增结构化 postcondition check 取代）。

---

## 3. 基于 `causal_plan_with_keyframes.json` 的新增任务（更贴主题、且更可评分）

这一节只提出 **不依赖额外人工标注**、能从现有 JSON 字段稳定构造的任务，且严格围绕 causal planning / failure reflecting。

### 3.1 新增任务 A：Spatial Postcondition Check（空间后置状态核验）

**动机**：  
“动作是否导致目标空间状态成立”是最基础的因果后果验证，但目前体系缺少可评分的“后置真值”任务；而 JSON 已提供结构化 `truth`。

**字段来源（JSONPath）**：  
- `steps[i].spatial_postconditions_detail[*].relation/objects/truth`

**证据建议**：  
- 优先 `keyframe_single`：step i 的最后关键帧 `steps[i].critical_frames[-1].keyframe_image_path`  
- 备选 `images_uniform_clip`：若有 step 内 clip（或关键帧窗口 clip）可抽帧

**任务形式（strict scoring）**：  
输入：关键帧 + 一条 postcondition 陈述（relation + objects）  
输出：`Yes/No`（对应 `truth`）

**负样本构造**：  
- 弱负样本 A：对同一 postcondition 将 truth 取反（标记 `meta.neg_sample=true`）  
- 弱负样本 B：替换 objects（从同 step 的对象集合中抽取），label=No（同样标记 `meta.neg_sample=true`）

**质量过滤**：  
- relation 文本过长/对象过多导致歧义 → 丢弃  
- 若关键帧中对象不可见概率高（例如 “inside fridge”）→ 丢弃或标记不可观测（避免噪声）

### 3.2 新增任务 B：Affordance Postcondition Check（可供性后置状态核验）

**动机**：  
规划的核心之一是“动作使对象获得某种可操作性”（例如 cutting_board 变成可切割表面 ready）。这比自由文本 effect 更贴“可执行因果”。

**字段来源**：  
- `steps[i].affordance_postconditions_detail[*].object_name/affordance_types/reasons`

**证据建议**：  
- `keyframe_single`：step i 尾关键帧  

**任务形式（两种可选）**：  
1) Yes/No：给定 object + affordance_type，问“该 affordance 是否在此步后成立？”  
2) MCQ：给定 object，四选一选择正确 affordance_type（干扰项来自全局 affordance_type 词表或同 item 其他条目）

**负样本构造**：  
- 从其他 step 的 affordance_types 抽取干扰（更难、更像真实混淆）

### 3.3 新增任务 C：Failure-Driven Replanning / Recovery Insertion（失败驱动重规划）

**动机**：  
仅让模型“复述 recovery_strategy”不足以形成反思闭环；我们需要模型在失败发生后**插入恢复动作**并继续执行可行计划。

**字段来源**：  
- `steps[i].failure_handling.reason`  
- `steps[i].failure_handling.recovery_strategy`  
- 原计划后续：`steps[i+1:].step_goal`

**证据建议**：  
- `video_prefix`：到 step i 的前缀（或 “失败发生点”的关键帧窗口）  
- fallback：`images_uniform_scene` 或 step i 的关键帧

**任务形式（推荐拆成两段客观题，保证可评分）**：

1) **Recovery Strategy Selection（MCQ）**  
   - 输入：失败描述 + 证据  
   - 输出：A/B/C/D（选正确 recovery_strategy）  
   - 干扰项：来自其他 step/item 的 recovery_strategy（同域更难）

2) **Next Step After Recovery（MCQ 或排序）**  
   - 输入：已选择 recovery + 证据（或仅失败描述）  
   - 输出：下一步 step_goal（或在候选集合中排序）  
   - gold：通常为原计划的 `steps[i+1].step_goal`，但可加“插入 recovery 后再继续”的约束说明

> 如果要做自由文本版（插入恢复步骤并继续规划），建议作为 SFT 小比例，并用上述两段任务作为评测回归指标。

### 3.4 新增任务 D：Counterfactual Outcome MCQ（Task_14 的可评分变体）

**字段来源**：`steps[i].causal_challenge_question` + `steps[i].expected_challenge_outcome`  
**任务形式**：四选一选择正确 outcome；干扰项来自其他 step/item。  
**价值**：把“反事实推演”变成可回归的标签任务，服务 failure reflecting 主线。

---

## 4. 任务重构后的“推荐落地版本”（如何变成可训练/可评测）

### 4.1 强监督（建议作为主训练/回归评测）

优先选择能做 strict scoring 的任务形态（Yes/No、A/B、A/B/C/D、排序）：

- 空间关系核验：`Task_27`（Yes/No）
- 空间/可供性后置核验：新增任务 A/B（Yes/No 或 MCQ）
- step_goal 对齐：`Task_10`（MCQ）
- prefix→goal/next：`Task_23/24`（MCQ）
- 时序一致性：`Task_26`（A/B）
- 执行一致性：`Task_22`（match/partial/mismatch）
- 错误计划诊断：`Task_28`（结构化标签：flaw_type + flaw_step）
- 长时序规划：`Task_29/30`（排序 / 严格编号序列输出）
- 失败反思（客观化）：`Task_14/15` 的 outcome/recovery MCQ 变体 + 新增任务 C 的两段评测

### 4.2 弱监督/生成题（建议小比例 SFT，用模型打分/规则打分）

- `Task_06/17`（因果解释/机制）
- `Task_12`（依赖解释自由文本版）
- `Task_18/19/25`（可观测性/效果/进度总结）

原则：这些任务不宜承担“主指标”，更适合扩展能力；需要显式 `meta.weak_supervision=true`，并配一致性检查（是否引入不存在实体/是否与 meta.fields 冲突）。

---

## 5. 生成与质量控制要点（避免数据噪声破坏主题）

### 5.1 两个必须处理的 schema/路径问题（否则任务会崩）

1) **`keyframe_image_path` 的可移植性**  
很多产物会写入生成机器相关的绝对路径；换环境不可读。生成侧必须做 resolve：  
优先原路径可读，否则回退到 `<item_dir>` 下同名文件（glob），失败则跳过样本。

2) **`mechanism` 字段不一致**  
部分产物 `affordance_hotspot` 不含 `mechanism`，机制解释常在 `causal_chain.causal_affordance_focus_detail`。  
建议统一派生：  
`mechanism := affordance_hotspot.mechanism` 若存在，否则 `mechanism := causal_chain.causal_affordance_focus_detail`，并在 `meta.schema_fallbacks` 记录来源。

### 5.2 防止“纯文本背诵”的三条硬约束

- **硬约束 1：涉及视觉对齐的任务尽量用 MCQ/核验题**（而不是“请复述字段”）。  
- **硬约束 2：负样本必须可控**（错配 step_goal/关键帧、扰动 causal_chain 字段、替换对象等），并写清 `meta.neg_sample=true`。  
- **硬约束 3：证据缺失就降级或跳过**（不要强行生成导致噪声扩散）：  
  - 没有 mp4 prefix/clip → 用抽帧序列（images_uniform_scene/clip）  
  - 仍没有可用图像 → 丢弃样本

### 5.3 对“失败反思”的特别建议：把错误类型做成固定集合

为保证 Task_28 / 新增任务 C 的训练信号稳定，建议 flaw_type 使用固定枚举（可多标签但要可控）：

- `order_violation`
- `precondition_missing`
- `spatial_infeasible`
- `affordance_incompatible`
- `tool_mismatch`

并在生成时记录：
- `meta.flaw_type`
- `meta.flaw_step_id`
- `meta.fix_hint`（如果有：例如 recovery_strategy 或建议插入的恢复动作）

---

## 6. 推荐的下一步（如果你要把它变成一套可跑的 pipeline）

1) 先把 “强监督 strict scoring” 的核心任务做成回归评测集（最小闭环）：  
   - `Task_27` + 新增 A/B（postcondition） + `Task_10` + `Task_23/24` + `Task_26` + `Task_22` + `Task_28`
2) 再补 “失败驱动重规划（新增 C）” 作为 failure reflecting 的核心指标。  
3) 最后用 `Task_06/17` 等生成题做能力扩展（SFT 小比例 + 模型打分），避免噪声主导训练。

