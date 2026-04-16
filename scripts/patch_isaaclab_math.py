#!/usr/bin/env python3
"""Patch Isaac Lab 2.1's math.py to add the missing quat_apply_inverse function.

Isaac Lab 2.1 (the latest pip version) lacks quat_apply_inverse which SONIC requires.
This script adds it after the existing quat_apply function.

Usage: python patch_isaaclab_math.py /path/to/isaaclab/utils/math.py
"""
import sys

PATCH = '''
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # conjugate: negate xyz components
    xyz = -quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

'''

TARGET = "@torch.jit.script\ndef quat_apply_yaw"

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, "r") as f:
        content = f.read()

    if "quat_apply_inverse" in content:
        print("Already patched.")
        sys.exit(0)

    if TARGET not in content:
        print(f"ERROR: Could not find insertion point in {path}")
        sys.exit(1)

    content = content.replace(TARGET, PATCH + TARGET)
    with open(path, "w") as f:
        f.write(content)
    print(f"Patched: {path}")
