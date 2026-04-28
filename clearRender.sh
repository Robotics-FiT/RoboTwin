#!/usr/bin/env bash
# clearRender.sh
# -----------------------------------------------------------------------------
# Clear only the *rendered* outputs of a task under data/<task>/<task_config>/
# so that re-running collectData.sh will skip the (expensive) simulation/seed
# phase and go straight to re-rendering from the saved trajectories.
#
# Directory layout (produced by script/collect_data.py):
#   data/<task>/<task_config>/
#     seed.txt          <- seeds (physics pass output)  [KEEP]
#     _traj_data/       <- per-episode trajectory pkl   [KEEP]
#     data/             <- per-episode .hdf5            [CLEAR]
#     video/            <- per-episode .mp4             [CLEAR]
#     scene_info.json   <- per-episode metadata         [CLEAR]
#     .cache/           <- transient per-frame pkl      [CLEAR]
#
# Usage:
#   ./clearRender.sh                                # defaults below
#   ./clearRender.sh <task_name> <task_config>
# -----------------------------------------------------------------------------

set -euo pipefail

TASK_NAME="${1:-random_dance}"
TASK_CONFIG="${2:-random_dance}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${BASE_DIR}/data/${TASK_NAME}/${TASK_CONFIG}"

if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "[clearRender] nothing to do: ${TARGET_DIR} does not exist."
    exit 0
fi

echo "[clearRender] target: ${TARGET_DIR}"

# Each item is cleared *only if it exists*, so the script is idempotent.
for sub in "data" "video" ".cache"; do
    path="${TARGET_DIR}/${sub}"
    if [[ -e "${path}" ]]; then
        echo "[clearRender] removing ${path}"
        rm -rf "${path}"
    fi
done

# scene_info.json is rebuilt by collect_data.py on every render pass, so wipe
# it to avoid stale episode entries.
info_json="${TARGET_DIR}/scene_info.json"
if [[ -f "${info_json}" ]]; then
    echo "[clearRender] removing ${info_json}"
    rm -f "${info_json}"
fi

echo "[clearRender] done. seed.txt and _traj_data/ preserved; "\
"next collectData.sh run will skip the simulation phase and only re-render."
