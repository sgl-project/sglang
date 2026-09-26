#!/bin/bash
# 唯一global CPU汇总入口；日期由job启动时固定，不在重试中刷新。
set -euo pipefail
: "${TREND_AS_OF_DATE:?必须指定UTC逻辑日期}"
: "${TREND_TOOL_SHA:?必须指定受信任工具完整SHA}"
: "${TREND_OUTPUT_DIR:?必须指定候选artifact目录}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${HERE}/trend_publish.py" derived
