import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import numpy as np
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    # ``random_dance`` supports a ``borrow_actors_from: <other_task>`` knob:
    # the simulation still runs random_dance's play_once / ik_debug / etc.,
    # but the tabletop layout is taken from the named other task. To keep the
    # output organised by *what is on the table* rather than *which class
    # produced the motion*, we route the saved data into
    # ``data/<borrow_target>/<task_config>/`` instead of
    # ``data/random_dance/<task_config>/``. The class loaded via
    # ``class_decorator`` above is unaffected (it's still random_dance).
    _borrow = (((args.get("random_dance") or {}).get("borrow_actors_from")) or "")
    _borrow = str(_borrow).strip()
    if task_name == "random_dance" and _borrow:
        print(f"\033[95m[random_dance] borrowing tabletop from '{_borrow}'; "
              f"output will be saved under data/{_borrow}/{task_config}/\033[0m")
        args["task_name"] = _borrow

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # Per-episode timing buckets. Populated during the two stages below and
    # printed as a summary at the end of ``run``. Times are wall-clock
    # seconds measured with ``time.perf_counter()`` so we capture whatever
    # the OS scheduler, GPU, and disk actually took.
    timings = {
        "physics_episodes": [],   # Stage 1: physics-only sim (seed search)
        "render_episodes": [],    # Stage 2: whole episode including render
        "merge_video_ep": [],     # Stage 2 sub: pkl->hdf5 + mp4 encode only
    }

    def _fmt_stats(name, series):
        if not series:
            return f"  {name:24s}  (no samples)"
        arr = np.asarray(series, dtype=np.float64)
        return (f"  {name:24s}  n={len(arr):3d}  "
                f"total={arr.sum():8.2f}s  mean={arr.mean():6.2f}s  "
                f"min={arr.min():6.2f}s  max={arr.max():6.2f}s")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            # ----- Stage 1: physics-only simulation for one episode -----
            ep_t0 = time.perf_counter()
            episode_ok = False
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                    episode_ok = True
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            # Only count time for episodes that produced a usable trajectory
            # -- failed ones are noisy (variable exception-handling time) and
            # not what we want to characterise.
            if episode_ok:
                timings["physics_episodes"].append(time.perf_counter() - ep_t0)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        # ``generate_pic`` mode: run the full simulation (so the scene, the
        # play_once / ik_debug logic, randomisation and the borrow-actors-from
        # plumbing all behave identically), but DON'T render a video and DON'T
        # build the hdf5. Instead, after each episode finishes, save just two
        # PNGs of the final frame:
        #     data/<task>/images/episode<N>_head.png
        #     data/<task>/images/episode<N>_observer.png
        # We achieve "no video / no hdf5 / no pkl cache" by forcing
        # ``save_data=False``: ``_take_picture`` and ``merge_pkl_to_hdf5_video``
        # both short-circuit when ``self.save_data`` is False (see
        # ``envs/_base_task.py``).
        generate_pic = bool(args.get("generate_pic", False))
        if generate_pic:
            args["save_data"] = False
            images_dir = os.path.join(args["save_path"], "images")
            os.makedirs(images_dir, exist_ok=True)
            print(f"\033[95m[generate_pic] enabled -- skipping video/hdf5; "
                  f"saving final-frame PNGs to {images_dir}/\033[0m")

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        def exist_pic(idx):
            head_p = os.path.join(args["save_path"], "images", f"episode{idx}_head.png")
            obs_p = os.path.join(args["save_path"], "images", f"episode{idx}_observer.png")
            return os.path.exists(head_p) and os.path.exists(obs_p)

        # In generate_pic mode, resume by looking for already-saved PNG pairs
        # instead of hdf5 files (since no hdf5 is produced).
        if generate_pic:
            while exist_pic(st_idx):
                st_idx += 1
        else:
            while exist_hdf5(st_idx):
                st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            # ----- Stage 2: trajectory replay + rendering + video encode -----
            ep_t0 = time.perf_counter()

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            # In generate_pic mode, grab the two final-frame PNGs BEFORE
            # close_env so the SAPIEN scene is still alive.
            if generate_pic:
                pic_t0 = time.perf_counter()
                TASK_ENV.save_final_frame_pic(
                    images_dir=os.path.join(args["save_path"], "images"),
                    ep_num=episode_idx,
                )
                print(f"[generate_pic] episode {episode_idx}: final-frame PNGs "
                      f"saved in {time.perf_counter() - pic_t0:.2f}s")

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))

            # Measure the hdf5/mp4 merge separately so we can tell whether the
            # bottleneck is the renderer itself or the post-processing.
            merge_t0 = time.perf_counter()
            if not generate_pic:
                TASK_ENV.merge_pkl_to_hdf5_video()
            timings["merge_video_ep"].append(time.perf_counter() - merge_t0)

            if not generate_pic:
                TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

            timings["render_episodes"].append(time.perf_counter() - ep_t0)
            print(f"[timing] episode {episode_idx}: render+save "
                  f"{timings['render_episodes'][-1]:.2f}s  "
                  f"(of which merge+mp4 {timings['merge_video_ep'][-1]:.2f}s)")

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)

    # ========== Final timing summary ==========
    print("\n\033[96m" + "=" * 70 + "\033[0m")
    print("\033[96m[timing] per-episode wall-clock summary\033[0m")
    print("\033[96m" + "=" * 70 + "\033[0m")
    print(_fmt_stats("Stage 1  physics-only",    timings["physics_episodes"]))
    print(_fmt_stats("Stage 2  render + save",   timings["render_episodes"]))
    print(_fmt_stats("  of which merge+mp4",     timings["merge_video_ep"]))
    if timings["physics_episodes"] and timings["render_episodes"]:
        p = float(np.mean(timings["physics_episodes"]))
        r = float(np.mean(timings["render_episodes"]))
        print(f"\n  render/physics ratio (mean): {r / max(p, 1e-6):.2f}x  "
              f"=> rendering is {'the' if r > p else 'not the'} bottleneck.")
    print("\033[96m" + "=" * 70 + "\033[0m\n")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
