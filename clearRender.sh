#!/usr/bin/env bash
# clearRender.sh
# -----------------------------------------------------------------------------
# Clear rendered outputs of a task under data/<task>/<task_config>/ so the
# next collectData.sh run either
#   (a) re-renders from saved trajectories     (default, fast)
#   (b) re-runs the whole seed-search pipeline  (--full, needed when you
#       changed anything in the simulation / trajectory generation itself)
#
# Directory layout (produced by script/collect_data.py):
#   data/<task>/<task_config>/
#     seed.txt          <- seeds (physics pass output)
#     _traj_data/       <- per-episode trajectory pkl
#     data/             <- per-episode .hdf5
#     video/            <- per-episode .mp4
#     scene_info.json   <- per-episode metadata
#     .cache/           <- transient per-frame pkl
#
# Render-only mode (default): clears {data, video, .cache, scene_info.json}.
#                             Keeps seed.txt + _traj_data/ so collect_data.py
#                             skips physics-only Stage 1 and goes straight
#                             to re-rendering the same trajectories.
#
# Full mode (--full / -f):    ALSO clears seed.txt + _traj_data/. Use this
#                             after changing anything in the simulation
#                             logic (envs/*.py keyframe generation, IK,
#                             etc.) so the next run actually regenerates
#                             the trajectories.
#
# Usage:
#   ./clearRender.sh                                      # default task, render-only
#   ./clearRender.sh --full                               # default task, full wipe
#   ./clearRender.sh <task_name> <task_config>            # explicit task, render-only
#   ./clearRender.sh --full <task_name> <task_config>     # explicit task, full wipe
# -----------------------------------------------------------------------------

set -euo pipefail

FULL=0
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --full|-f)
            FULL=1
            ;;
        -h|--help)
            sed -n '2,/^# -----/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            POSITIONAL+=("${arg}")
            ;;
    esac
done

TASK_NAME="${POSITIONAL[0]:-random_dance}"
TASK_CONFIG="${POSITIONAL[1]:-random_dance}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${BASE_DIR}/data/${TASK_NAME}/${TASK_CONFIG}"

if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "[clearRender] nothing to do: ${TARGET_DIR} does not exist."
    exit 0
fi

echo "[clearRender] target: ${TARGET_DIR}"
echo "[clearRender] mode  : $([[ ${FULL} -eq 1 ]] && echo 'FULL (wipe seeds + trajectories)' || echo 'render-only')"

# Always clear these -- they are pure render artifacts / transient caches.
for sub in "data" "video" ".cache"; do
    path="${TARGET_DIR}/${sub}"
    if [[ -e "${path}" ]]; then
        echo "[clearRender] removing ${path}"
        rm -rf "${path}"
    fi
done

info_json="${TARGET_DIR}/scene_info.json"
if [[ -f "${info_json}" ]]; then
    echo "[clearRender] removing ${info_json}"
    rm -f "${info_json}"
fi

# In full mode also wipe seeds + trajectories so Stage 1 runs again.
if [[ ${FULL} -eq 1 ]]; then
    for p in "${TARGET_DIR}/seed.txt" "${TARGET_DIR}/_traj_data"; do
        if [[ -e "${p}" ]]; then
            echo "[clearRender] removing ${p}"
            rm -rf "${p}"
        fi
    done
fi

if [[ ${FULL} -eq 1 ]]; then
    echo "[clearRender] done. next collectData.sh run will re-run the full pipeline (seed search + render)."
else
    echo "[clearRender] done. seed.txt and _traj_data/ preserved; next collectData.sh run will skip Stage 1 and only re-render."
fi
