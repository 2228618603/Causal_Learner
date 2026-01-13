# Causal_Learner

本仓库是一个多模块工作区，主要包含两部分：

- `ECCV/`：面向长视频/多模态监督的数据生成与三阶段流水线（包含脚本、规范文档与校验工具）。
- `Qwen-PC/`：Qwen3‑VL 相关的 demo、微调框架、以及点云方向的 `PointLLM` 子项目。

更细的开发/贡献约定见根目录 `AGENTS.md`，以及子目录 `ECCV/AGENTS.md`、`Qwen-PC/AGENTS.md`。

## 目录结构

- `ECCV/three_stage/`：三阶段流水线实现与文档（`THREE_STAGE_PIPELINE.md` 等）。
- `ECCV/causal_spafa_plan_dataset*/`、`ECCV/generated_plans_output_*`：生成物/数据产出（如需发布数据集再纳入版本管理）。
- `Qwen-PC/qwen-vl-utils/`：可安装的 Python 包（`src/qwen_vl_utils/`），含 Ruff 格式化/检查配置。
- `Qwen-PC/qwen-vl-finetune/`：训练框架与脚本（`qwenvl/train/`、`scripts/`、`demo/`）。
- `Qwen-PC/PointLLM/`：点云 LLM 子项目（依赖较重，建议按需使用）。

## 快速开始（常用命令）

### ECCV：三阶段流水线

```bash
python ECCV/three_stage/pipeline.py --input-video /abs/path/video.mp4
```

### Qwen3‑VL：Web Demo

```bash
python Qwen-PC/web_demo_mm.py --backend hf -c <checkpoint-or-hf-id>
```

### 代码风格（qwen-vl-utils）

```bash
cd Qwen-PC/qwen-vl-utils
pip install -e .
ruff format .
ruff check .
```

## 大文件与产物说明

- `Qwen-PC/outputs/`、`Qwen-PC/checkpoints/` 等本地训练产物已在 `.gitignore` 中忽略，避免误提交。
- GitHub 对单文件大小有硬性限制（100MB）。示例点云文件以压缩形式提交：`Qwen-PC/qwen-vl-finetune/demo/points/scene0000_01_vh_clean.ply.gz`，解压方式见 `Qwen-PC/qwen-vl-finetune/demo/points/README.md`。

