# CoT 标注/生成（Task_28–Task_42）

本目录汇总 **CoT（Chain-of-Thought）标注/生成** 相关脚本与文档，用于把 `ECCV/three_stage/` 三阶段产物（`causal_plan_with_keyframes.json`）通过 OpenAI-compatible API 生成 CoT(JSONL) 数据集（Task_28–Task_42）。

## 文档

- 最终样例 + 数据格式契约 + 生成/校验流程（唯一文档入口）：`COT_API_GENERATION_WORKFLOW.md`

## 脚本

- API CoT 生成：`generate_cot_dataset_api.py`
- 严格校验器：`validate_cot_dataset.py`

> 说明：这些脚本依赖 `ECCV/three_stage/common.py` 中的通用工具与 API 调用封装。
