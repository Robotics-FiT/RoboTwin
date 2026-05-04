#!/usr/bin/env bash
# continuousCollect.sh
# -----------------------------------------------------------------------------
# Keep producing data in batches forever. Each iteration raises the target
# ``episode_num`` in the task config by BATCH_SIZE and invokes the
# existing ``collect_data.sh`` pipeline, which will auto-resume (it reads
# seed.txt and _traj_data/ and only fills in what's missing).
#
# Within one iteration the pipeline still does
#     (1) generate trajectories for the new batch  (Stage 1)
#  -> (2) render video + save hdf5 for the new batch (Stage 2)
#
# because that is how script/collect_data.py is structured. Across
# iterations, state is preserved: seed.txt and _traj_data/ just grow.
#
# Stop the script with Ctrl-C at any time. A trap restores the original
# ``episode_num`` value in the yaml before exit so the file doesn't get
# left in a mutated state.
#
# Usage:
#   ./continuousCollect.sh                         # default task, batch=50, gpu=1
#   ./continuousCollect.sh <task> <config>         # explicit task
#   ./continuousCollect.sh <task> <config> <size>  # explicit batch size
#   ./continuousCollect.sh <task> <config> <size> <gpu>
#
# Env overrides:
#   TASK_NAME / TASK_CONFIG / BATCH_SIZE / GPU_ID / MAX_ITERATIONS
# (MAX_ITERATIONS=0, the default, means "loop forever".)
# -----------------------------------------------------------------------------

set -euo pipefail

# Show the leading comment block as help when the user asks.
for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            # Print every line at the top of the file that starts with "# "
            # (the usage banner), up to the first blank line after it.
            awk '/^# /{print substr($0, 3); next} NF==0 && seen{exit} /./{seen=1}' "$0"
            exit 0
            ;;
    esac
done

TASK_NAME="${1:-${TASK_NAME:-random_dance}}"
TASK_CONFIG="${2:-${TASK_CONFIG:-random_dance}}"
BATCH_SIZE="${3:-${BATCH_SIZE:-50}}"
GPU_ID="${4:-${GPU_ID:-1}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_PATH="${BASE_DIR}/task_config/${TASK_CONFIG}.yml"

if [[ ! -f "${YAML_PATH}" ]]; then
    echo "[continuous] task config not found: ${YAML_PATH}" >&2
    exit 1
fi

# Read the original episode_num so we can (a) bump it and (b) restore it.
ORIG_EP_NUM="$(grep -E '^episode_num:' "${YAML_PATH}" | head -n 1 | awk '{print $2}')"
if ! [[ "${ORIG_EP_NUM}" =~ ^[0-9]+$ ]]; then
    echo "[continuous] could not parse 'episode_num:' from ${YAML_PATH}" >&2
    exit 1
fi

restore_yaml() {
    # Put episode_num back the way we found it, no matter how we exit.
    if [[ -f "${YAML_PATH}" ]] && [[ -n "${ORIG_EP_NUM:-}" ]]; then
        sed -i -E "s/^episode_num: .*/episode_num: ${ORIG_EP_NUM}/" "${YAML_PATH}"
        echo "[continuous] restored episode_num=${ORIG_EP_NUM} in ${YAML_PATH}"
    fi
}
trap restore_yaml EXIT INT TERM

count_current_episodes() {
    # How many episodes are already fully collected? collect_data.py
    # considers an episode done when its hdf5 exists under data/.
    local dir="${BASE_DIR}/data/${TASK_NAME}/${TASK_CONFIG}/data"
    if [[ -d "${dir}" ]]; then
        ls -1 "${dir}"/episode*.hdf5 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

set_episode_num() {
    local new_val="$1"
    sed -i -E "s/^episode_num: .*/episode_num: ${new_val}/" "${YAML_PATH}"
}

iteration=0
start_ts=$(date +%s)

while true; do
    iteration=$((iteration + 1))
    if [[ "${MAX_ITERATIONS}" -gt 0 && "${iteration}" -gt "${MAX_ITERATIONS}" ]]; then
        echo "[continuous] reached MAX_ITERATIONS=${MAX_ITERATIONS}, stopping."
        break
    fi

    iter_start=$(date +%s)
    current="$(count_current_episodes)"
    target=$((current + BATCH_SIZE))

    echo ""
    echo "=============================================================="
    echo "[continuous] iter ${iteration}  batch_size=${BATCH_SIZE}  task=${TASK_NAME}/${TASK_CONFIG}  gpu=${GPU_ID}"
    echo "[continuous] already collected : ${current}"
    echo "[continuous] this batch target : ${target}"
    echo "=============================================================="

    set_episode_num "${target}"

    # Run one full pipeline pass (Stage 1 seeds + Stage 2 render).
    # collect_data.sh already ``cd``s into the right paths; run in a
    # subshell to keep our own cwd intact.
    (
        cd "${BASE_DIR}"
        bash collect_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}"
    )

    after="$(count_current_episodes)"
    iter_elapsed=$(( $(date +%s) - iter_start ))
    total_elapsed=$(( $(date +%s) - start_ts ))
    produced=$(( after - current ))
    echo "[continuous] iter ${iteration} done: +${produced} episodes  " \
         "(iter ${iter_elapsed}s  total ${total_elapsed}s  now have ${after})"

    # Safety: if an iteration produced nothing (e.g. all seeds failed) we
    # would otherwise loop forever setting target to the same value. Break
    # with a warning so the operator can intervene.
    if [[ "${produced}" -eq 0 ]]; then
        echo "[continuous] iteration produced 0 episodes; stopping to avoid an infinite loop." >&2
        break
    fi
done

echo "[continuous] all done."
