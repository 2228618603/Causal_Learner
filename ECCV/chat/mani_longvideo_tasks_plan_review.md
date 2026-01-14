# mani_longvideo 任务集全景审查与增补建议（基于 three_stage Schema）

本文档目标：结合 `ECCV/three_stage/prompts.py` 的字段/输出约束（Schema 与 hard constraints）以及 `ECCV/chat/mani_longvideo_tasks_plan_final.md`（Task_01~Task_30 任务定义），对当前任务集进行系统性剖析：

- 识别冗余/重复/低增益/高噪声的任务，并给出删改/合并建议；
- 基于现有 Schema 字段（尤其是可被客观化的字段）提出可新增的高价值任务；
- 给出落地时的反“标签泄漏”与工程注意事项，避免训练与评测被路径/文件名等旁路信息污染。

> 约定：本文所说 “Schema” 以 `three_stage/prompts.py` 中 Stage1/2/3 的输出格式为准；实际落盘 JSON 可能会被脚本补充额外字段（如 `keyframe_image_path`），但**任务构造不应依赖**这些非 schema 字段。

---

## 0. 一句话结论（可直接执行）

- **优先保留**：以“核验/判别/选择题”为核心的 grounded 任务（例如 Task_18/19/22/24/27/28/29/30），并将 “纯复述类” 任务（如 Task_02/08/09/11/13/21/25）显著降权或合并。
- **优先新增**：把 `causal_chain.action`、`patient`、`interaction.hotspot.affordance_type/mechanism`、`causal_effect_on_*` 的 `truth/affordance_types` 做成 **MCQ/Yes-No/一致性判别** 的客观题，补齐“可评测、低噪声、强负样本”的训练信号。
- **必须规避**：不要把 `step_slug`、`ts_XX.XXs`、文件路径等写进模型可见输入；它们会强泄漏 `step_goal/时间顺序`，导致任务被“读文件名”捷径击穿。

---

## 1. 输入材料与边界

### 1.1 参考文件

- Schema/约束：`ECCV/three_stage/prompts.py`
- 任务集定义：`ECCV/chat/mani_longvideo_tasks_plan_final.md`

### 1.2 三阶段索引空间（容易踩坑）

three_stage 体系里存在 **两个不同的 index space**：

1. **全视频均匀抽帧池（Stage2）**：`start_frame_index/end_frame_index` 是对 “FULL video 的 sampled frames（1..N）” 的边界索引（`end` 为 exclusive）。
2. **step-clip 内重采样帧池（Stage3）**：`critical_frames[*].frame_index` 是对 “某个 step clip 的帧池（1..M）” 的局部索引。

因此：

- **不能**把 Stage3 的 `frame_index` 当作 `sampled_frames/` 的序号使用；
- 跨 step 的时间顺序/对齐应以 **step 目录关键帧文件名中的 `ts_XX.XXs`**（或 Stage2 的边界）作为依据，但 `ts` 不应进入模型可见输入（见第 6 节）。

---

## 2. Schema 硬约束（决定“哪些任务天然不稳/不该做”）

### 2.1 Stage 1：禁止关键帧字段与任何帧号/时间引用

Stage1 输出必须是 step-level 的结构化 JSON，并且 **不能出现**：

- `critical_frames` / `frame_index` / `interaction` / `keyframe_image_path`
- 文本里也不能出现 “Frame 12 / t=3.2s” 等引用

推论：任何以 “关键帧选择/关键帧解释” 为核心监督的任务，都不应绑定 Stage1 产物；应绑定 Stage3。

### 2.2 Stage 3：必须输出 2 张关键帧 + 完整的 grounded 机制字段

Stage3 强制要求：

- 每步输出 **恰好 2 个** `critical_frames`（且 `frame_index` 递增）；
- 文本字段里不能出现帧号；
- `interaction.tools/materials` 至少一项非空（无外部工具时可用 “hands”）；
- 关键帧中 `interaction.hotspot.description/affordance_type/mechanism` 必填且非空；
- `causal_chain` 的 precondition/effect 列表均为非空列表（可被任务客观化利用）。

推论：Stage3 提供了大量可被 “核验/一致性/选择题” 使用的强监督信号，应该成为任务集的主体信息源。

---

## 3. 任务设计的评价标准（用于判断“冗余/无意义/高噪声”）

建议用以下维度给每个任务打分（高优先级从上到下）：

1. **Groundable（可落到证据）**：问题的答案是否能被给定的图像/视频证据直接支持？
2. **Objective（可客观评分）**：输出是否能用自动规则/对齐指标稳定评分（Yes/No/MCQ/分类/结构化输出优于自由文本长答）？
3. **Non-leaky（不被旁路信息击穿）**：是否会被目录名/文件名/时间戳/slug 等泄漏直接“抄答案”？
4. **Coverage（覆盖核心能力）**：是否直接服务于 “因果规划 + 失败反思 + 长时序” 的目标，而不是泛化闲聊能力？
5. **Cost（构造与推理成本）**：证据准备、负样本生成、prompt 复杂度是否可控？

---

## 4. Task_01~Task_30 全景归类（面向“保留/合并/删除”决策）

下面按功能域归类，而不是按编号顺序：

### 4.1 Grounded 感知与实体/机制（更适合做客观题/核验）

- Task_01：场景锚点/关键对象（适合作为候选集合构造器）
- Task_03：hotspot 语义（建议客观化为 MCQ/填空）
- Task_04：工具/材料角色（可做多选/分类）
- Task_05：状态变化描述（容易变成复述，但可用于事件抽取）
- Task_27：空间关系真假核验（强、可构造负样本）
- Task_02：空间关系自然语言复述（与 27 重叠，低增益）

### 4.2 因果规划（步级/跨步）

- Task_06：关键帧级因果链分析（信息密度高，但自由文本易漂移）
- Task_12：跨步依赖（强，建议保持）
- Task_17：Why/How 综合（覆盖 rationale + 机制）
- Task_09：前置条件陈述（偏复述，易泄漏/低增益）
- Task_11：期望效果复述（偏复述，易不可观测导致噪声）

### 4.3 失败反思

- Task_14：反事实挑战与结果（可客观化为 MCQ）
- Task_15：失败原因与恢复策略（可客观化为 MCQ）
- Task_28：失败规划不合理点定位（强负样本任务，建议保留）

### 4.4 视觉对齐与时序（长视频）

- Task_18：前置条件核验（三态：satisfied/not/not directly observable）
- Task_19：后置效果核验（三态/多条 effect）
- Task_22：计划-执行一致性（match/partial/mismatch，且可造负样本）
- Task_23：前缀目标识别
- Task_24：前缀预测下一步目标（强）
- Task_26：事件先后判别（需严格避免 ts 泄漏）
- Task_29：后续 K 步重排（强）
- Task_30：头尾补全中间步骤（强）
- Task_20：Step 间边界“过渡现象描述”（辅助任务）
- Task_25：前缀进度总结（辅助任务，开放式总结噪声较大）

---

## 5. 冗余/重复/低增益任务：删改与合并建议

本节给出 “建议动作 + 原因 + 替代项”。

### 5.1 强重复：建议直接合并或仅保留一个

#### A) Task_02 与 Task_27

- **现象**：都围绕 `causal_precondition_on_spatial.relation/objects/truth`。
- **问题**：Task_02 是开放式复述，评分难；Task_27 是真假核验，可造负样本、可自动评分。
- **建议**：
  - 训练/评测主线：**保留 Task_27，删除或极低比例保留 Task_02**（作为 paraphrase 辅助即可）。

#### B) Task_08 与 Task_17

- **现象**：Task_08（rationale）与 Task_17（Why/How）本质同域。
- **问题**：分开会导致同一信号被重复计权；Why/How 的联合更贴合规划。
- **建议**：
  - **合并**：把 Task_08 作为 Task_17 的 `variant=why_only`（或直接降权/移除）。

#### C) Task_09 与 Task_18

- **现象**：都聚焦 precondition，但 Task_18 是视觉核验（三态）。
- **问题**：Task_09 容易变成“复述 JSON 列表”，不要求证据闭环。
- **建议**：
  - 主线：**保留 Task_18**（三态核验更 grounded）。
  - Task_09：降权/删除；若保留，务必要求 “仅输出可观测的前置条件 + 不可观测必须标注 not directly observable”。

#### D) Task_11 与 Task_19

- **现象**：都围绕 effect/postcondition。
- **问题**：Task_11 偏“期望复述”，而 effect 往往不可见（clean/ready 等），噪声大。
- **建议**：
  - **保留 Task_19**（证据核验），Task_11 降权/删除。
  - 若要保留 Task_11：建议仅保留 spatial 类 effect（如 on_top_of/inside/open/closed），减少 affordance/属性类不可观测 effect。

#### E) Task_13 与 Task_24

- **现象**：都做 “下一步 step_goal 预测”。
- **问题**：Task_13 使用 keyframe_single + 计划文本上下文，容易退化为“读 step_goal 列表的语言模型任务”。
- **建议**：
  - 多模态长视频主线：**保留 Task_24（prefix 证据）**；
  - Task_13 仅作为 text-only baseline 或直接删除。

### 5.2 低增益/辅助型：建议降权

#### Task_21（关键帧选择理由）

- **问题**：容易变成复述 `action_state_change_description`；与 Task_05/06/17 信息高度重叠且评分困难。
- **建议**：删除或极低比例保留（仅用于 instruction-following 风格训练，不用于核心评测）。

#### Task_20（边界过渡描述）、Task_25（进度总结）

- **问题**：偏摘要/叙述，难客观评分；对“因果规划+失败反思”核心贡献较弱。
- **建议**：作为补充能力可少量保留，但不作为核心任务集。

### 5.3 需要“客观化改造”的任务（不建议保持纯自由文本）

#### Task_06（Holistic Causal Chain）

- **问题**：信息密度很高，但自由文本长答易漂移、难评分。
- **建议**：保留字段来源，但将输出改造为：
  - 结构化 JSON（固定 key）或
  - MCQ/填空（见第 6 节新增任务：Action/Patient/Mechanism）。

#### Task_16（Physical Feasibility）

- **问题**：容易受主观影响；且与 Task_18（precondition 核验）语义近。
- **建议**：要么并入 Task_18 作为 “feasibility label=feasible/not/uncertain”，要么加负样本与三态输出并严格定义“不可观测”。

---

## 5.4 推荐“核心任务子集”（两种目标口径）

### 口径 A：聚焦“因果规划 + 失败反思”（推荐）

建议高权重：

- Grounded 核验：Task_18、Task_19、Task_22、Task_27
- 规划与长时序：Task_23、Task_24、Task_29、Task_30、Task_12、Task_28
- 失败反思：Task_14、Task_15（建议做成 MCQ 变体）

建议低权重/可删：

- Task_02、Task_08、Task_09、Task_11、Task_13、Task_20、Task_21、Task_25、Task_16（未客观化前）

### 口径 B：更全面的多模态理解（可选）

在口径 A 基础上，少量加入：

- Task_01（对象锚点抽取，辅助构造候选集与负样本）
- Task_03/04/05（hotspot/角色/状态变化，帮助 grounded 表达）
- Task_26（事件先后判别，严格防止时间戳泄漏）

---

## 6. 基于 Schema 的新增高价值任务（建议优先做“客观题”）

下面给出一组建议新增任务（可编号为 Task_31+，也可作为现有任务的 `variant`）。每个任务都给出：字段来源、证据形态、构造规则、输出格式与负样本策略。

> 统一强约束：prompt 中不要包含任何文件名/路径/slug/ts；这些只能在 `meta` 中记录。

### Task_31_Keyframe_to_StepGoal_Matching（关键帧→步骤匹配，MCQ/分类）

- **动机**：把“关键帧语义”与 “step_goal” 强绑定，且天然可做强负样本。
- **字段来源（JSONPath）**：
  - 正例：`steps[i].step_goal`
  - 候选池：同 item 的 `steps[*].step_goal`（或跨 item 扩展）
- **证据**：`keyframe_single`（选 `steps[i].critical_frames[0]` 或 `[-1]`）
- **构造规则**：
  - 输入：一张关键帧图 + K 个候选 step_goal（K=4/6）。
  - 输出：选项字母或 `step_id`（固定格式便于评分）。
- **负样本**：
  - 同 item 的其它 step_goal（hard negatives），或同场景同类动作的跨 item step_goal（harder）。

### Task_32_Init_vs_Complete_Keyframe_Order（同一步两关键帧阶段判别）

- **动机**：利用 Stage3 “两关键帧递增顺序” 的结构化监督，形成强时序/状态变化学习信号。
- **字段来源**：`steps[i].critical_frames[0]` 与 `[1]` 的图像（标签：initiation vs completion）。
- **证据**：`keyframe_pair`（两张关键帧图，顺序可打乱）
- **输出**：`A_is_initiation` / `B_is_initiation`（二分类）
- **注意**：不要暴露 `frame_index`。

### Task_33_Hotspot_AffordanceType_MCQ（热点可供性类别选择题）

- **字段来源**：`steps[i].critical_frames[j].interaction.hotspot.affordance_type`
- **证据**：`keyframe_single`
- **输出**：从 4 选 1 的 `affordance_type` 里选正确项
- **负样本构造**：
  - 来自其它关键帧的 affordance_type；或从同关键帧 hotspot.mechanism 语义近但类别不同的项做 hard negative。

### Task_34_Hotspot_Mechanism_MCQ（热点物理机制选择题）

- **字段来源**：`steps[i].critical_frames[j].interaction.hotspot.mechanism`
- **证据**：`keyframe_single`
- **输出**：4 选 1（机制句子，建议做短句标准化）
- **工程建议**：先把 mechanism 做轻量模板化（如 “force_transfer/friction/leverage/fluid_flow/heat_transfer/…” + 一句解释），否则自由文本难当 label。

### Task_35_Action_Phrase_MCQ（`causal_chain.action` 选择题）

- **字段来源**：
  - keyframe-level：`steps[i].critical_frames[j].causal_chain.action`
  - step-level：`steps[i].causal_chain.action`
- **证据**：`keyframe_single`（更 grounded）
- **输出**：4 选 1（动作短语）
- **价值**：补齐当前任务集中对 `action` 字段利用不足的问题。

### Task_36_Patient_Identification_MCQ（受事对象选择题）

- **字段来源**：`steps[i].critical_frames[j].causal_chain.patient`
- **证据**：`keyframe_single`
- **候选池**：来自 Task_01 的对象锚点集合（同 item 去重后的 objects/tools/materials）
- **输出**：多选或单选（取决于 patient 是否单实体）

### Task_37_Step_vs_Keyframe_CausalChain_Consistency（步级↔关键帧因果链一致性判别）

- **动机**：直接训练 “同一步的 step-level 因果链” 与 “关键帧级因果链” 的一致性，构造强负样本。
- **字段来源**：
  - 正例：`steps[i].causal_chain` 与 `steps[i].critical_frames[j].causal_chain`
  - 负例：把 `critical_frames[j].causal_chain` 替换为其它 step 的 causal_chain（或只替换 patient/action）
- **证据**：`keyframe_single` +（可选）step_goal
- **输出**：`consistent / inconsistent / not directly observable`
- **评分**：自动评分；负样本可控。

### Task_38_Spatial_Postcondition_Check（空间后置状态核验，Yes/No/Uncertain）

- **字段来源**：`steps[i].causal_chain.causal_effect_on_spatial[*].relation/objects/truth`
- **证据**：step i 的最后关键帧（或 step clip 抽帧）
- **输出**：对每条 effect 输出 `supported / contradicted / not observable`
- **价值**：比 Task_11 的“复述效果”更 grounded、更可评测。

### Task_39_Affordance_Postcondition_Check（可供性后置状态核验）

- **字段来源**：`steps[i].causal_chain.causal_effect_on_affordance[*].object_name/affordance_types`
- **证据**：step-end 关键帧（注意很多 affordance 不可见）
- **输出**：同 Task_38 三态
- **建议**：优先选择“可见状态类 affordance”（如 open/closed/inserted/holding），避免纯功能性不可见标签。

### Task_40_Counterfactual_Outcome_MCQ（反事实结果选择题）

- **字段来源**：`steps[i].counterfactual_challenge_question` 与 `expected_challenge_outcome`
- **证据**：关键帧（或 step clip 抽帧）
- **输出**：4 选 1 outcome
- **负样本**：来自其它 step 的 outcome，或相同工具/材料但机制不同的 outcome（hard negative）。

### Task_41_Recovery_Strategy_MCQ（失败恢复策略选择题）

- **字段来源**：`steps[i].failure_reflecting.recovery_strategy`
- **证据**：关键帧（或 prefix 到该步）
- **输出**：4 选 1 strategy
- **价值**：把失败反思从自由文本变成客观题，更可评测、更稳定。

### Task_42_Prefix_Completed_Steps_MultiSelect（前缀已完成步骤多选）

- **动机**：替代 Task_25 的开放式总结，用可评分的多选形式衡量“长时序进度理解”。
- **字段来源**：`steps[0..i].step_goal`（作为 gold；不要塞进输入）
- **证据**：`video_prefix`（到 step i 的尾帧）
- **输出**：从候选 step_goal 列表中选出“已完成”的集合（或输出最大已完成 step_id）。
- **负样本**：候选中混入未来 step_goal。

### Task_43_Stage2_Temporal_Localization_Check（可选：基于 Stage2 的客观时间定位）

仅当 item 内存在可读的 Stage2 产物（例如 stage2 的 JSON/manifest 可取）时启用：

- **字段来源**：`stage2` 预测的 `start_frame_index/end_frame_index`
- **证据**：全视频 `sampled_frames/`（或其抽帧子集）
- **输出**：预测/核验某一步的边界索引（更严格、更客观）

---

## 7. 反泄漏与工程注意事项（强建议写进数据生成器）

### 7.1 严禁在模型可见输入中出现的泄漏源

- step 目录名常包含 `step_slug`（几乎等价 `step_goal`）；
- 关键帧文件名常包含 `ts_XX.XXs`（等价时间顺序标签）；
- `frame_###` 或 `sample_###` 等序号也可能在某些任务中直接泄漏目标。

**建议做法**：

- 模型输入只给 “图片/视频内容本身” +（必要时）候选文本（候选列表必须来自 JSON 字段而非文件名）。
- 文件路径、ts、step_slug 全部放到 `meta.evidence_files` 等字段中，不进入 prompt。

### 7.2 “不可观测”必须成为显式标签

在 precondition/effect/affordance 类任务里，很多命题本质上不可从单帧或短片段可靠判断（如 “clean”、“ready”、“container contains X” 等）。

- 对这类命题，建议统一使用三态：`supported / contradicted / not directly observable`（或 satisfied/not/not directly observable）。
- 数据生成器应对字段做筛选：优先选 “可视觉核验” 的 relation/affordance。

### 7.3 负样本要“单一扰动”，避免不可控噪声

像 Task_22/27/28/37/38/39 这类可造负样本的任务，建议每条样本只注入 1 个错误（swap/replace/flip truth），以便：

- 自动评分更稳定；
- 模型学习目标更清晰；
- 便于错误归因与 ablation。

---

## 8. 推荐落地路线图（从“可跑”到“可评测”）

1. **先裁剪任务集**：按第 5 节把重复/复述类任务降权或移除，保留核验/前缀/重排/纠错主线。
2. **把关键字段客观化**：优先落地 Task_31/33/35/37/38/40/41/42（都是可自动评分或强约束输出）。
3. **加入严格反泄漏处理**：在数据构造时剥离所有路径与文件名文本，只保留视觉内容。
4. **做小规模 sanity check**：随机抽取 50~200 条样本人工检查 “是否可由证据回答”“是否泄漏”“负样本是否合理”。
5. **再扩展难度**：跨 item hard negatives、prefix curriculum、K 步重排窗口扩大等。

---

## 附录 A：Schema 字段速查（用于写任务的 JSONPath）

顶层：

- `high_level_goal: str`
- `steps: List[Step]`

Step：

- `step_id: int`
- `step_goal: str`
- `rationale: str`
- `causal_chain: CausalChain`
- `counterfactual_challenge_question: str`
- `expected_challenge_outcome: str`
- `failure_reflecting.reason: str`
- `failure_reflecting.recovery_strategy: str`
- `critical_frames: List[CriticalFrame]`（Stage3）

CausalChain：

- `agent: str`
- `action: str`
- `patient: str`
- `causal_precondition_on_spatial: List[{relation:str, objects:[str], truth:bool}]`
- `causal_precondition_on_affordance: List[{object_name:str, affordance_types:[str], reasons:str}]`
- `causal_effect_on_spatial: List[{relation:str, objects:[str], truth:bool}]`
- `causal_effect_on_affordance: List[{object_name:str, affordance_types:[str], reasons:str}]`

CriticalFrame：

- `frame_index: int`（step clip 局部索引）
- `action_state_change_description: str`
- `causal_chain: CausalChain`
- `interaction.tools: List[str]`
- `interaction.materials: List[str]`
- `interaction.hotspot.description: str`
- `interaction.hotspot.affordance_type: str`
- `interaction.hotspot.mechanism: str`

