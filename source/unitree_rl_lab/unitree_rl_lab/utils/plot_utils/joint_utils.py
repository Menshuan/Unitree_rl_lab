import yaml
from importlib import resources


def load_symmetric_joint_indices(asset, yaml_name="joint_map.yaml"):
    """
    return:
      names:        ["hip_roll", "hip_pitch", ...]
      left_indices:  [idx, idx, ...]
      right_indices: [idx, idx, ...]
    """

    # 1. 读 yaml
    with resources.files("unitree_rl_lab.assets").joinpath(yaml_name).open("r") as f:
        cfg = yaml.safe_load(f)

    joints_cfg = cfg["joints"]

    left_joints = []
    right_joints = []

    for k, v in joints_cfg.items():
        if k.startswith("left_"):
            left_joints.extend(v)
        elif k.startswith("right_"):
            right_joints.extend(v)

    # 2. asset joint name -> index
    name_to_index = {name: i for i, name in enumerate(asset.joint_names)}

    names = []
    left_indices = []
    right_indices = []

    for lj in left_joints:
        if not lj.startswith("left_"):
            continue

        base = lj.replace("left_", "")
        rj = "right_" + base

        if rj not in right_joints:
            continue

        if lj not in name_to_index or rj not in name_to_index:
            raise KeyError(f"Joint pair ({lj}, {rj}) not found in asset")

        semantic_name = base.replace("_joint", "")

        names.append(semantic_name)
        left_indices.append(name_to_index[lj])
        right_indices.append(name_to_index[rj])

    return names, left_indices, right_indices
