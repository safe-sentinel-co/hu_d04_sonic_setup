"""
Verify IsaacLab's BFS joint/body ordering for HU_D04 by parsing the URDF directly.

IsaacLab traverses the kinematic tree in breadth-first order. This script
reproduces that traversal from the raw URDF XML and compares against hu_d04.py.
No Isaac Sim runtime needed.
"""

import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path


def parse_urdf_tree(urdf_path):
    """Parse URDF and return the kinematic tree as parent->children adjacency list."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect all joints with parent/child links
    joints = {}
    children_of = {}  # parent_link -> [(child_link, joint_name, joint_type)]

    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        joints[jname] = {"parent": parent, "child": child, "type": jtype}

        if parent not in children_of:
            children_of[parent] = []
        children_of[parent].append((child, jname, jtype))

    # Find root link (link that is never a child)
    all_children = {j["child"] for j in joints.values()}
    all_parents = {j["parent"] for j in joints.values()}
    root_links = all_parents - all_children
    assert len(root_links) == 1, f"Expected 1 root, got {root_links}"
    root_link = root_links.pop()

    return root_link, children_of, joints


def bfs_traversal(root_link, children_of):
    """BFS traversal matching IsaacLab's ordering. Returns (bodies, revolute_joints)."""
    bodies = []
    revolute_joints = []
    queue = deque([root_link])

    while queue:
        link = queue.popleft()
        bodies.append(link)
        for child_link, joint_name, joint_type in children_of.get(link, []):
            if joint_type == "revolute":
                revolute_joints.append((joint_name, child_link))
            queue.append(child_link)

    return bodies, revolute_joints


def main():
    urdf_path = Path(__file__).resolve().parent.parent / "data/assets/robot_description/urdf/hu_d04/hu_d04.urdf"
    root_link, children_of, joints = parse_urdf_tree(urdf_path)

    print(f"Root link: {root_link}")
    print(f"Total joints: {len(joints)}")

    # BFS traversal
    all_bodies, revolute_joints = bfs_traversal(root_link, children_of)

    # Filter to only actuated (revolute) body ordering
    # In IsaacLab, the body list includes ALL links (including fixed-joint links)
    # but DOFs only come from revolute joints.
    # The body ordering for the mapping arrays uses the BFS order of links
    # that have revolute joints, plus fixed-joint children appear after their parent.

    # For the mapping, we care about: root + all revolute-joint child links
    revolute_child_links = {j["child"] for j in joints.values() if j["type"] == "revolute"}
    revolute_child_links.add(root_link)  # root is always included

    # BFS body order (only bodies connected by revolute joints)
    actuated_bodies = [b for b in all_bodies if b in revolute_child_links]

    print("\n" + "=" * 60)
    print("IsaacLab BFS Body Order (revolute-joint bodies only)")
    print("=" * 60)
    for i, body in enumerate(actuated_bodies):
        print(f"  Body {i}: {body}")

    print("\n" + "=" * 60)
    print("IsaacLab BFS DOF Order")
    print("=" * 60)
    # DOF order: for each body in actuated_bodies[1:] (skip root), the joint is body_name -> joint_name
    # We need to find the revolute joint whose child is this body
    child_to_joint = {j["child"]: name for name, j in joints.items() if j["type"] == "revolute"}
    for i, body in enumerate(actuated_bodies[1:]):  # skip root
        jname = child_to_joint[body]
        print(f"  DOF {i}: {jname}")

    # Compare against expected
    EXPECTED_BODIES = [
        "base_link",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "waist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "waist_roll_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "waist_pitch_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "head_yaw_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "head_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    ]

    print("\n" + "=" * 60)
    print("Comparing BFS result against HU_D04_ISAACLAB_JOINTS in hu_d04.py")
    print("=" * 60)
    errors = 0
    for i in range(max(len(EXPECTED_BODIES), len(actuated_bodies))):
        expected = EXPECTED_BODIES[i] if i < len(EXPECTED_BODIES) else "<missing>"
        actual = actuated_bodies[i] if i < len(actuated_bodies) else "<missing>"
        ok = expected == actual
        if not ok:
            errors += 1
            print(f"  MISMATCH Body {i}: expected '{expected}', got '{actual}'")
        else:
            print(f"  OK Body {i}: {actual}")

    if errors == 0:
        print(f"\nAll {len(EXPECTED_BODIES)} body orderings match!")
    else:
        print(f"\n{errors} mismatches found!")


if __name__ == "__main__":
    main()
