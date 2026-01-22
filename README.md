# Causal_Learner

本仓库是一个多模块工作区，围绕“长视频 → 结构化步骤/因果计划数据 → 多模态模型训练/研究”组织，主要包含两部分：

- `ECCV/`：面向长视频/多模态监督的数据生成与三阶段流水线（脚本、规范文档、校验工具与产出目录）。
- `Qwen-PC/`：多模态训练工作区，包含 Qwen3‑VL 相关代码、微调框架，以及点云方向的 `PointLLM` 子项目。

更细的开发/贡献约定见根目录 `AGENTS.md`，以及子目录 `ECCV/AGENTS.md`、`Qwen-PC/AGENTS.md`。

## ECCV/：数据生成与三阶段流水线

`ECCV/` 目录以脚本为主，目标是把单个长视频加工为可用于训练/分析的结构化数据（步骤计划、片段边界、关键帧等）。流水线采用“三阶段”方式，避免在全视频上直接做密集关键帧标注，从而降低计算与标注难度，并保持索引空间清晰可校验。

**核心结构**

- `ECCV/three_stage/`：三阶段流水线实现与校验工具（推荐从这里理解整体流程）。
  - `pipeline.py`：串联 Stage 1–3 的统一入口。
  - `stage1_generate_draft.py`：在全视频帧池上生成草案计划（不包含任何关键帧字段）。
  - `stage2_localize_and_cut.py`：在同一全视频帧池上为每一步预测 `{start_frame_index, end_frame_index}` 并导出 step clips。
  - `stage3_refine_and_keyframes.py`：对每个 step clip 重新采样帧并补全关键帧/证据等字段。
  - `validate_three_stage_output.py`：输出一致性校验（尤其是不同 frame manifest 的索引空间）。
  - `common.py` / `prompts.py`：公共逻辑与提示词模板。
- `ECCV/two_satge_old/mani_*video*.py`、`ECCV/two_satge_old/nav_*`、`ECCV/two_satge_old/generate_*api*.py`：旧版（两阶段）相关脚本入口。
- `ECCV/two_satge_old/mani_longvideo_tasks_plan_final.md`：旧版（两阶段）任务定义与数据规则说明文档。

**产出目录（默认视为运行产物）**

- `ECCV/causal_spafa_plan_dataset*/`、`ECCV/generated_plans_output_*`：生成的数据、帧、片段与中间日志；除非明确发布数据集，一般不建议将大规模产出物长期纳入版本控制。

## Qwen-PC/：多模态训练与点云子项目

`Qwen-PC/` 聚合了多模态模型训练相关代码，核心用途是：对齐多模态样本的输入/预处理方式，并提供视觉-语言与点云-语言的训练/适配实现，便于把 `ECCV/` 侧产生的数据用于模型训练或进一步研究。

**核心结构**

- `Qwen-PC/qwen-vl-finetune/`：训练框架与脚本配置，训练代码主要在 `Qwen-PC/qwen-vl-finetune/qwenvl/train/`。
  - 覆盖图像/视频 + 文本训练，以及点云 + 文本训练（点云相关建模与适配组件也位于该子目录内）。
- `Qwen-PC/qwen-vl-utils/`：可安装的 Python 工具包（`Qwen-PC/qwen-vl-utils/src/qwen_vl_utils/`），用于图像/视频等多媒体预处理与通用辅助函数。
- `Qwen-PC/PointLLM/`：点云 LLM 子项目（模型、数据处理、训练与推理相关代码）。
- `Qwen-PC/Qwen3-VL/`：上游/参考代码与资源（用于对齐基线能力与目录组织）。

**本地目录（默认不入库）**

- `Qwen-PC/outputs/`、`Qwen-PC/checkpoints/`：训练输出与权重缓存；仓库已在 `.gitignore` 中忽略这些路径，避免误提交大文件。

## 大文件说明（GitHub 限制）

GitHub 对单文件大小有 100MB 的硬性限制。仓库中个别示例资源会以压缩形式提交；例如点云示例以 `*.ply.gz` 形式存放，恢复方法见 `Qwen-PC/qwen-vl-finetune/demo/points/README.md`。

> 说明：本 README 只描述模块定位与结构，不展开具体的 demo / eval 细节；可分别查看子目录的 `README.md`/`AGENTS.md` 获取更具体的使用方式。
