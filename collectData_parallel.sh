#!/bin/bash
# Parallel data collection on a single GPU.
#
# Usage:
#   bash collectData_parallel.sh <task_name> <task_config> <gpu_id> <num_workers>
#
# Example:
#   bash collectData_parallel.sh random_dance random_dance 0 3
#
# The script works by temporarily splitting the requested ``episode_num`` in
# task_config/<task_config>.yml into N equal chunks, spawning N ``collect_data.py``
# workers on the same GPU, each pointed at a per-worker copy of the yaml whose
# ``save_path`` is a unique subdirectory. After all workers exit, data is
# merged back into the original save_path so downstream tooling sees a single
# dataset.
#
# Notes:
# * 3-4 workers per 24 GB GPU (RTX 4090) is typically safe. SAPIEN's path
#   tracer allocates ~2-3 GB per scene with our RT settings; monitor with
#   ``nvidia-smi``.
# * Workers are independent processes, so the `seed.txt` handshake from the
#   single-process script does not apply here. Each worker collects its own
#   chunk of episodes end-to-end.

set -e

task_name=${1}
task_config=${2}
gpu_id=${3}
num_workers=${4:-3}

if [ -z "${task_name}" ] || [ -z "${task_config}" ] || [ -z "${gpu_id}" ]; then
    echo "Usage: bash collectData_parallel.sh <task_name> <task_config> <gpu_id> [num_workers]"
    exit 1
fi

./script/.update_path.sh > /dev/null 2>&1 || true

ROOT_CFG="./task_config/${task_config}.yml"
if [ ! -f "${ROOT_CFG}" ]; then
    echo "Config not found: ${ROOT_CFG}"
    exit 1
fi

# Parse original episode_num (simple grep; the yaml key is always at the top level).
ORIG_EPS=$(grep -E "^episode_num:" "${ROOT_CFG}" | head -1 | awk '{print $2}')
if ! [[ "${ORIG_EPS}" =~ ^[0-9]+$ ]]; then
    echo "Could not read episode_num from ${ROOT_CFG} (got '${ORIG_EPS}')"
    exit 1
fi

ORIG_SAVE=$(grep -E "^save_path:" "${ROOT_CFG}" | head -1 | awk '{print $2}')
ORIG_SAVE="${ORIG_SAVE:-./data}"

echo "============================================================"
echo "[parallel] task=${task_name}  config=${task_config}"
echo "[parallel] gpu=${gpu_id}  workers=${num_workers}  episodes=${ORIG_EPS}"
echo "[parallel] orig save_path=${ORIG_SAVE}"
echo "============================================================"

# Per-worker chunk size (ceil division so last worker never runs a larger chunk).
PER=$(( (ORIG_EPS + num_workers - 1) / num_workers ))

TMP_DIR=$(mktemp -d -t robotwin_parallel_XXXXXX)
echo "[parallel] temp dir: ${TMP_DIR}"

PIDS=()
WORKER_SAVES=()
for ((w=0; w<num_workers; w++)); do
    START=$(( w * PER ))
    END=$(( START + PER ))
    if [ ${END} -gt ${ORIG_EPS} ]; then
        END=${ORIG_EPS}
    fi
    CHUNK=$(( END - START ))
    if [ ${CHUNK} -le 0 ]; then
        continue
    fi
    WCFG_NAME="${task_config}_w${w}"
    WCFG_PATH="./task_config/${WCFG_NAME}.yml"
    WSAVE="${TMP_DIR}/worker${w}"
    WORKER_SAVES+=("${WSAVE}")

    # Clone the config and rewrite episode_num + save_path + offset knob.
    cp "${ROOT_CFG}" "${WCFG_PATH}"
    # sed portable form (works on GNU sed).
    sed -i -E "s|^episode_num:.*|episode_num: ${CHUNK}|" "${WCFG_PATH}"
    sed -i -E "s|^save_path:.*|save_path: ${WSAVE}|" "${WCFG_PATH}"

    echo "[parallel] worker ${w}: episodes ${START}..${END}  cfg=${WCFG_PATH}  save=${WSAVE}"

    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        # Staggered start so all workers don't hammer disk/GPU at the same
        # instant during import and first-render.
        sleep $(( w * 3 ))
        PYTHONWARNINGS=ignore::UserWarning \
            python script/collect_data.py "${task_name}" "${WCFG_NAME}" \
                > "${TMP_DIR}/worker${w}.log" 2>&1
    ) &
    PIDS+=("$!")
done

# Wait for every worker; propagate failures.
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

if [ ${FAIL} -ne 0 ]; then
    echo "[parallel] at least one worker failed -- check ${TMP_DIR}/worker*.log"
    exit ${FAIL}
fi

# Merge per-worker datasets back into the original save_path.
FINAL_DIR="${ORIG_SAVE}/${task_name}/${task_config}"
mkdir -p "${FINAL_DIR}/data" "${FINAL_DIR}/video" "${FINAL_DIR}/_traj_data"

NEXT=0
for WSAVE in "${WORKER_SAVES[@]}"; do
    WDIR="${WSAVE}/${task_name}/${task_config}"
    if [ ! -d "${WDIR}/data" ]; then
        continue
    fi
    # Renumber episode files so ids are contiguous in the merged dataset.
    for f in "${WDIR}/data"/episode*.hdf5; do
        [ -f "${f}" ] || continue
        cp "${f}" "${FINAL_DIR}/data/episode${NEXT}.hdf5"
        vf="${WDIR}/video/$(basename "${f}" .hdf5).mp4"
        [ -f "${vf}" ] && cp "${vf}" "${FINAL_DIR}/video/episode${NEXT}.mp4"
        tf="${WDIR}/_traj_data/$(basename "${f}" .hdf5).pkl"
        [ -f "${tf}" ] && cp "${tf}" "${FINAL_DIR}/_traj_data/episode${NEXT}.pkl"
        NEXT=$(( NEXT + 1 ))
    done
done

echo "============================================================"
echo "[parallel] merged ${NEXT} episodes into ${FINAL_DIR}"
echo "[parallel] per-worker logs: ${TMP_DIR}/worker*.log"
echo "============================================================"

# Clean up the temporary per-worker yamls. We keep TMP_DIR around so you can
# inspect the logs and the raw per-worker output if anything looks off.
for ((w=0; w<num_workers; w++)); do
    rm -f "./task_config/${task_config}_w${w}.yml"
done
