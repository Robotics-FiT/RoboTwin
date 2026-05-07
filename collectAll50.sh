#!/usr/bin/env bash
# collectAll50.sh
# -----------------------------------------------------------------------------
# Run data collection for every "original" task in the repo (i.e. every task
# under envs/ except the custom random_dance one), producing N episodes each.
#
# Uses a shared task_config (default: demo_clean) for every task. The script
# temporarily rewrites episode_num in that yaml to EPISODES_PER_TASK, runs
# collect_data.sh, and restores the yaml on exit.
#
# Because collect_data.py auto-resumes (it counts existing hdf5 files under
# data/<task>/<config>/data/), re-running this script will only top up the
# tasks that still have fewer than EPISODES_PER_TASK episodes.
#
# Usage:
#   ./collectAll50.sh                          # defaults: config=demo_clean, N=10, gpu=1
#   ./collectAll50.sh <config> <N> <gpu>
#
# Env overrides:
#   TASK_CONFIG / EPISODES_PER_TASK / GPU_ID / SKIP_TASKS / ONLY_TASKS
#
# SKIP_TASKS / ONLY_TASKS are space-separated lists of task names, e.g.
#   SKIP_TASKS="adjust_bottle click_bell" ./collectAll50.sh
#   ONLY_TASKS="lift_pot turn_switch"     ./collectAll50.sh
# -----------------------------------------------------------------------------

set -uo pipefail

TASK_CONFIG="${1:-${TASK_CONFIG:-demo_clean}}"
EPISODES_PER_TASK="${2:-${EPISODES_PER_TASK:-10}}"
GPU_ID="${3:-${GPU_ID:-1}}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_PATH="${BASE_DIR}/task_config/${TASK_CONFIG}.yml"
ENVS_DIR="${BASE_DIR}/envs"

if [[ ! -f "${YAML_PATH}" ]]; then
    echo "[all50] task config not found: ${YAML_PATH}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Discover tasks: every envs/*.py except the known non-task files and
# random_dance (which is the user's custom task).
# ---------------------------------------------------------------------------
EXCLUDE_REGEX='^(_base_task|_GLOBAL_CONFIGS|__init__|random_dance)$'

ALL_TASKS=()
while IFS= read -r f; do
    name="$(basename "${f}" .py)"
    if [[ "${name}" =~ ${EXCLUDE_REGEX} ]]; then
        continue
    fi
    ALL_TASKS+=("${name}")
done < <(ls "${ENVS_DIR}"/*.py | sort)

# Apply ONLY_TASKS / SKIP_TASKS filters.
ONLY_TASKS="${ONLY_TASKS:-}"
SKIP_TASKS="${SKIP_TASKS:-}"

TASKS=()
for t in "${ALL_TASKS[@]}"; do
    if [[ -n "${ONLY_TASKS}" ]]; then
        # whitespace-delimited membership check
        if [[ " ${ONLY_TASKS} " != *" ${t} "* ]]; then
            continue
        fi
    fi
    if [[ -n "${SKIP_TASKS}" ]] && [[ " ${SKIP_TASKS} " == *" ${t} "* ]]; then
        continue
    fi
    TASKS+=("${t}")
done

echo "=============================================================="
echo "[all50] task_config       : ${TASK_CONFIG}"
echo "[all50] episodes per task : ${EPISODES_PER_TASK}"
echo "[all50] gpu               : ${GPU_ID}"
echo "[all50] total tasks       : ${#TASKS[@]} (of ${#ALL_TASKS[@]} discovered)"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Mutate yaml: set episode_num to EPISODES_PER_TASK and restore on exit.
# ---------------------------------------------------------------------------
ORIG_EP_NUM="$(grep -E '^episode_num:' "${YAML_PATH}" | head -n 1 | awk '{print $2}')"
if ! [[ "${ORIG_EP_NUM}" =~ ^[0-9]+$ ]]; then
    echo "[all50] could not parse 'episode_num:' from ${YAML_PATH}" >&2
    exit 1
fi

restore_yaml() {
    if [[ -f "${YAML_PATH}" ]] && [[ -n "${ORIG_EP_NUM:-}" ]]; then
        sed -i -E "s/^episode_num: .*/episode_num: ${ORIG_EP_NUM}/" "${YAML_PATH}"
        echo "[all50] restored episode_num=${ORIG_EP_NUM} in ${YAML_PATH}"
    fi
}
trap restore_yaml EXIT INT TERM

sed -i -E "s/^episode_num: .*/episode_num: ${EPISODES_PER_TASK}/" "${YAML_PATH}"
echo "[all50] set episode_num=${EPISODES_PER_TASK} in ${YAML_PATH}"

# ---------------------------------------------------------------------------
# Run each task. We do NOT `set -e` around the inner call so one failing task
# doesn't abort the whole sweep; we just log it and move on.
# ---------------------------------------------------------------------------
count_episodes() {
    local task="$1"
    local dir="${BASE_DIR}/data/${task}/${TASK_CONFIG}/data"
    if [[ -d "${dir}" ]]; then
        ls -1 "${dir}"/episode*.hdf5 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

start_ts=$(date +%s)
declare -a OK_TASKS=()
declare -a FAIL_TASKS=()
declare -a SKIP_DONE=()

idx=0
for task in "${TASKS[@]}"; do
    idx=$((idx + 1))
    have="$(count_episodes "${task}")"

    echo ""
    echo "--------------------------------------------------------------"
    echo "[all50] (${idx}/${#TASKS[@]}) task=${task}  already_have=${have}  target=${EPISODES_PER_TASK}"
    echo "--------------------------------------------------------------"

    if [[ "${have}" -ge "${EPISODES_PER_TASK}" ]]; then
        echo "[all50] already has >= ${EPISODES_PER_TASK} episodes, skipping."
        SKIP_DONE+=("${task}")
        continue
    fi

    t0=$(date +%s)
    if (
        cd "${BASE_DIR}"
        bash collect_data.sh "${task}" "${TASK_CONFIG}" "${GPU_ID}"
    ); then
        after="$(count_episodes "${task}")"
        dt=$(( $(date +%s) - t0 ))
        echo "[all50] ${task} done: have=${after}  (+$((after - have)) in ${dt}s)"
        OK_TASKS+=("${task}")
    else
        dt=$(( $(date +%s) - t0 ))
        echo "[all50] ${task} FAILED after ${dt}s" >&2
        FAIL_TASKS+=("${task}")
    fi
done

total=$(( $(date +%s) - start_ts ))
echo ""
echo "=============================================================="
echo "[all50] finished in ${total}s"
echo "[all50] ok        : ${#OK_TASKS[@]}"
echo "[all50] skipped   : ${#SKIP_DONE[@]} (already had enough episodes)"
echo "[all50] failed    : ${#FAIL_TASKS[@]}"
if [[ ${#FAIL_TASKS[@]} -gt 0 ]]; then
    echo "[all50] failed tasks:"
    for t in "${FAIL_TASKS[@]}"; do echo "          - ${t}"; done
fi
echo "=============================================================="
