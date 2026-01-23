#!/usr/bin/env sh
set -eu

# 三阶段长视频数据生成管线一键脚本（固定进入指定 ECCV 目录）
#
# 运行前：请在环境变量中提供 API Key（不要写进脚本/仓库文件）
#   export OPENAI_API_KEY="你的key"   # 或 export API_KEY="你的key"
#
# 本脚本会“强制”使用以下模型配置（忽略你外部设置的同名环境变量）：
#   API_BASE_URL="http://model.mify.ai.srv/v1"
#   MODEL_PROVIDER_ID="volcengine_maas"
#   MODEL_NAME="doubao-seed-1-8-251228"
#
# 可选覆盖（不改脚本）：
#   ECCV_ROOT=/abs/path/to/ECCV \
#   INPUT_VIDEO_DIR=/abs/path/to/videos \
#   OUTPUT_ROOT=/abs/path/to/out \
#   OVERWRITE=1 \
#   sh run_three_stage_pipeline.sh

# 默认进入脚本所在目录（应为 <repo>/ECCV）
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ECCV_ROOT="${ECCV_ROOT:-$SCRIPT_DIR}"
if [ ! -d "$ECCV_ROOT" ]; then
  echo "Error: ECCV_ROOT 目录不存在：$ECCV_ROOT" >&2
  echo "请确认该机器上的 ECCV 路径，或通过环境变量 ECCV_ROOT 覆盖。" >&2
  exit 1
fi
cd "$ECCV_ROOT"

INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/long_three_stage_out}"
STAGES="${STAGES:-1,2,3}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

# 强制 API 配置（按你的要求写死）

API_BASE_URL="http://model.mify.ai.srv/v1"
MODEL_PROVIDER_ID="volcengine_maas"
MODEL_NAME="doubao-seed-1-8-251228"
export API_BASE_URL MODEL_PROVIDER_ID MODEL_NAME

# 只从环境变量读取 key（支持 OPENAI_API_KEY -> API_KEY 映射）
if [ -n "${OPENAI_API_KEY:-}" ]; then
  API_KEY="${OPENAI_API_KEY}"
fi
if [ -z "${API_KEY:-}" ]; then
  echo "Error: 未检测到 API key，请先设置 OPENAI_API_KEY 或 API_KEY。" >&2
  exit 1
fi
export API_KEY

OVERWRITE="${OVERWRITE:-0}"
POST_VALIDATE="${POST_VALIDATE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [ ! -d "$INPUT_VIDEO_DIR" ]; then
  echo "Error: 输入视频目录不存在：$INPUT_VIDEO_DIR" >&2
  exit 1
fi

set -- \
  python3 three_stage/pipeline.py \
  --input-video-dir "$INPUT_VIDEO_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --stages "$STAGES" \
  --ffmpeg-bin "$FFMPEG_BIN"

if [ "$OVERWRITE" = "1" ]; then
  set -- "$@" --overwrite
fi
if [ "$POST_VALIDATE" = "1" ]; then
  set -- "$@" --post-validate
fi
if [ "$CONTINUE_ON_ERROR" = "1" ]; then
  set -- "$@" --continue-on-error
fi

echo "[three-stage] eccv_root=$ECCV_ROOT"
echo "[three-stage] input_video_dir=$INPUT_VIDEO_DIR"
echo "[three-stage] output_root=$OUTPUT_ROOT"
echo "[three-stage] stages=$STAGES overwrite=$OVERWRITE post_validate=$POST_VALIDATE continue_on_error=$CONTINUE_ON_ERROR"
echo "[three-stage] api_base_url=$API_BASE_URL provider=$MODEL_PROVIDER_ID model=$MODEL_NAME"

# 注意：不要打印/回显 API_KEY
exec "$@"
