#!/usr/bin/env python3
"""Interactive local viewer for HU_D04 retargeted motion.

Shows the HU_D04 robot playing back a PKL motion in MuJoCo's interactive viewer
(requires a display). Click-drag to rotate camera, keyboard shortcuts for
pause/frame-step.

Setup on your laptop:
    pip install mujoco joblib numpy

Usage:
    python local_viewer.py path/to/motion.pkl path/to/hu_d04.xml
"""
import sys
import time
import numpy as np
import joblib
import mujoco
import mujoco.viewer


def main():
    if len(sys.argv) != 3:
        print("usage: python local_viewer.py MOTION.pkl HU_D04.xml")
        sys.exit(1)
    pkl_path, mjcf_path = sys.argv[1], sys.argv[2]

    m = next(iter(joblib.load(pkl_path).values()))
    T = m["dof"].shape[0]
    fps = float(m.get("fps", 30))
    dt = 1.0 / fps
    print(f"Motion: {pkl_path}")
    print(f"  {T} frames @ {fps} fps = {T/fps:.1f} seconds")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as v:
        t = 0
        last = time.time()
        paused = False
        while v.is_running():
            # Drive qpos from the PKL motion
            data.qpos[:3] = m["root_trans_offset"][t]
            data.qpos[3:7] = m["root_rot"][t]     # stored as wxyz
            data.qpos[7:] = m["dof"][t]
            mujoco.mj_forward(model, data)
            v.sync()
            # advance
            now = time.time()
            if now - last >= dt:
                t = (t + 1) % T
                last = now
            else:
                time.sleep(0.005)
    print("closed")


if __name__ == "__main__":
    main()
