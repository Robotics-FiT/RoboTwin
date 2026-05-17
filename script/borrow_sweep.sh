#!/usr/bin/env bash
# borrow_sweep.sh
# -----------------------------------------------------------------------------
# Run the ``random_dance`` task once per entry in BORROW_TASKS, switching
# ``random_dance.borrow_actors_from`` in the task yaml between runs so each
# iteration spawns a different task's tabletop layout under random_dance's
# motion logic.
#
# Output is automatically routed by collect_data.py to
#   data/<borrow_task>/<task_config>/
# so each entry in the list ends up in its own folder. collect_data.py
# auto-resumes (it counts existing hdf5 / image files), so re-running this
# script will only top up the entries that still have fewer than the
# yaml's ``episode_num`` episodes.
#
# Usage:
#   ./script/borrow_sweep.sh                         # config=random_dance, gpu=1
#   ./script/borrow_sweep.sh <task_config>           # explicit yaml under task_config/
#   ./script/borrow_sweep.sh <task_config> <gpu>
#
# Env overrides:
#   TASK_CONFIG / GPU_ID / ONLY_TASKS / SKIP_TASKS
#
# ONLY_TASKS / SKIP_TASKS are space-separated lists of task names, e.g.
#   ONLY_TASKS="beat_block_hammer lift_pot" ./script/borrow_sweep.sh
#   SKIP_TASKS="turn_switch"                ./script/borrow_sweep.sh
#
# Configure the sweep list by editing BORROW_TASKS below. Each entry must
# be a task name that exists as ``envs/<name>.py`` (the same string you
# would put in ``borrow_actors_from`` by hand).
# -----------------------------------------------------------------------------

set -uo pipefail

# =============================================================================
# vvv  EDIT THIS LIST  vvv
# =============================================================================
BORROW_TASKS=(
    # 5 tasks chosen for richer arm <-> scene interaction (articulated
    # objects, button-press, stacking, hanging on a hook):
    turn_switch        # flip an articulated wall switch
    open_microwave     # swing a microwave's hinged door open
    click_bell         # press a desk bell's plunger button
    stack_blocks_three # stack three cubes on top of each other
    hanging_mug        # hang a mug by its handle onto a hook
)
# =============================================================================
# ^^^  EDIT THIS LIST  ^^^
# =============================================================================

TASK_CONFIG="${1:-${TASK_CONFIG:-random_dance}}"
GPU_ID="${2:-${GPU_ID:-1}}"

# random_dance is the only entry point that supports borrow_actors_from.
TASK_NAME="random_dance"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAML_PATH="${BASE_DIR}/task_config/${TASK_CONFIG}.yml"

if [[ ! -f "${YAML_PATH}" ]]; then
    echo "[borrow_sweep] task config not found: ${YAML_PATH}" >&2
    exit 1
fi

if [[ ${#BORROW_TASKS[@]} -eq 0 ]]; then
    echo "[borrow_sweep] BORROW_TASKS is empty. Edit ${BASH_SOURCE[0]} and" >&2
    echo "               fill in the task names you want to sweep over." >&2
    exit 1
fi

# Apply ONLY_TASKS / SKIP_TASKS filters.
ONLY_TASKS="${ONLY_TASKS:-}"
SKIP_TASKS="${SKIP_TASKS:-}"
TASKS=()
for t in "${BORROW_TASKS[@]}"; do
    # tolerate empty / commented lines defensively
    [[ -z "${t// }" ]] && continue
    if [[ -n "${ONLY_TASKS}" ]] && [[ " ${ONLY_TASKS} " != *" ${t} "* ]]; then
        continue
    fi
    if [[ -n "${SKIP_TASKS}" ]] && [[ " ${SKIP_TASKS} " == *" ${t} "* ]]; then
        continue
    fi
    # sanity-check: the entry must correspond to envs/<t>.py
    if [[ ! -f "${BASE_DIR}/envs/${t}.py" ]]; then
        echo "[borrow_sweep] WARNING: envs/${t}.py does not exist; skipping '${t}'." >&2
        continue
    fi
    TASKS+=("${t}")
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "[borrow_sweep] no tasks left after filtering. Aborting." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Capture the yaml's current ``borrow_actors_from`` line so we can restore it
# on exit. The line is nested under ``random_dance:`` so it has leading
# whitespace; we match that explicitly. Also remember whether the yaml had
# the line at all (it might be commented out or missing entirely).
# ---------------------------------------------------------------------------
ORIG_BORROW_LINE_NO="$(grep -nE '^[[:space:]]*borrow_actors_from:' "${YAML_PATH}" | head -n 1 | cut -d: -f1)"
if [[ -z "${ORIG_BORROW_LINE_NO}" ]]; then
    echo "[borrow_sweep] could not find an active 'borrow_actors_from:' line" >&2
    echo "               under ${YAML_PATH}. Add one (any value) and retry." >&2
    exit 1
fi
ORIG_BORROW_LINE="$(sed -n "${ORIG_BORROW_LINE_NO}p" "${YAML_PATH}")"

restore_yaml() {
    if [[ -f "${YAML_PATH}" ]] && [[ -n "${ORIG_BORROW_LINE:-}" ]]; then
        # Replace whatever line is currently at borrow_actors_from with the
        # exact text we captured at startup (preserves indentation + value
        # + trailing comment).
        local esc
        esc="$(printf '%s\n' "${ORIG_BORROW_LINE}" | sed -e 's/[\/&]/\\&/g')"
        sed -i -E "0,/^[[:space:]]*borrow_actors_from:.*/s//${esc}/" "${YAML_PATH}"
        echo "[borrow_sweep] restored borrow_actors_from line in ${YAML_PATH}"
    fi
}
trap restore_yaml EXIT INT TERM

set_borrow() {
    # Rewrite the (first) borrow_actors_from line to point at $1, preserving
    # the original indentation (2 spaces, but read it from the captured line
    # rather than hard-coding).
    local new_val="$1"
    local indent
    indent="$(printf '%s' "${ORIG_BORROW_LINE}" | sed -E 's/^([[:space:]]*).*/\1/')"
    # Use the first match only, in case (somehow) multiple lines exist.
    sed -i -E "0,/^[[:space:]]*borrow_actors_from:.*/s//${indent}borrow_actors_from: ${new_val}/" "${YAML_PATH}"
}

count_episodes() {
    # collect_data.py routes random_dance + borrow_actors_from output to
    # data/<borrow>/<task_config>/. With ``generate_pic: true`` the
    # per-episode artefact is the head png pair, otherwise it's hdf5.
    local borrow="$1"
    local data_dir="${BASE_DIR}/data/${borrow}/${TASK_CONFIG}/data"
    local img_dir="${BASE_DIR}/data/${borrow}/${TASK_CONFIG}/images"
    local n_hdf5=0
    local n_png=0
    if [[ -d "${data_dir}" ]]; then
        n_hdf5="$(ls -1 "${data_dir}"/episode*.hdf5 2>/dev/null | wc -l | tr -d ' ')"
    fi
    if [[ -d "${img_dir}" ]]; then
        n_png="$(ls -1 "${img_dir}"/episode*_head.png 2>/dev/null | wc -l | tr -d ' ')"
    fi
    # Whichever mode the yaml is in, the larger of the two is the right
    # "how many episodes do we already have" answer.
    if [[ "${n_hdf5}" -ge "${n_png}" ]]; then
        echo "${n_hdf5}"
    else
        echo "${n_png}"
    fi
}

echo "=============================================================="
echo "[borrow_sweep] task_config : ${TASK_CONFIG}"
echo "[borrow_sweep] gpu         : ${GPU_ID}"
echo "[borrow_sweep] sweep size  : ${#TASKS[@]} (of ${#BORROW_TASKS[@]} configured)"
echo "[borrow_sweep] tasks       : ${TASKS[*]}"
echo "=============================================================="

start_ts=$(date +%s)
declare -a OK_TASKS=()
declare -a FAIL_TASKS=()

idx=0
for borrow in "${TASKS[@]}"; do
    idx=$((idx + 1))
    have="$(count_episodes "${borrow}")"

    echo ""
    echo "--------------------------------------------------------------"
    echo "[borrow_sweep] (${idx}/${#TASKS[@]}) borrow=${borrow}  already_have=${have}"
    echo "--------------------------------------------------------------"

    set_borrow "${borrow}"

    t0=$(date +%s)
    if (
        cd "${BASE_DIR}"
        bash collect_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}"
    ); then
        after="$(count_episodes "${borrow}")"
        dt=$(( $(date +%s) - t0 ))
        echo "[borrow_sweep] ${borrow} done: have=${after}  (+$((after - have)) in ${dt}s)"
        OK_TASKS+=("${borrow}")
    else
        dt=$(( $(date +%s) - t0 ))
        echo "[borrow_sweep] ${borrow} FAILED after ${dt}s" >&2
        FAIL_TASKS+=("${borrow}")
    fi
done

total=$(( $(date +%s) - start_ts ))
echo ""
echo "=============================================================="
echo "[borrow_sweep] finished in ${total}s"
echo "[borrow_sweep] ok     : ${#OK_TASKS[@]}"
echo "[borrow_sweep] failed : ${#FAIL_TASKS[@]}"
if [[ ${#FAIL_TASKS[@]} -gt 0 ]]; then
    echo "[borrow_sweep] failed tasks:"
    for t in "${FAIL_TASKS[@]}"; do echo "          - ${t}"; done
fi
echo "=============================================================="
