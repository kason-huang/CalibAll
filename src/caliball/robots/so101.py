"""SO101 geometry and FK from the user-supplied calibrated URDF.

State units must be explicitly confirmed before using this model for calibration.
All visual elements are merged per link, including URDF visual origin/scale.
"""
from pathlib import Path
import hashlib
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from caliball.robots._base import BaseTF
from caliball.robots._registry import register_robot

_ROOT = Path(__file__).resolve().parents[3]
_URDF = _ROOT / "third_party/urdf/so101/so101_new_calib.urdf"

def _origin(node):
    T = np.eye(4)
    if node is not None:
        T[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return T

@register_robot("so101")
class SO101TF(BaseTF):
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    TCP_NAME = "gripper_frame_link"

    def __init__(self, input_units=None, **kwargs):
        self.input_units = input_units or os.environ.get("SO101_STATE_UNITS", "unconfirmed")
        if self.input_units not in ("radians", "degrees", "unconfirmed"):
            raise ValueError("SO101_STATE_UNITS must be radians, degrees or unconfirmed")
        self.tree = ET.parse(_URDF).getroot()
        self.joints = {j.find("child").get("link"): j for j in self.tree.findall("joint")}
        self.LINK_NAMES = [l.get("name") for l in self.tree.findall("link") if l.findall("visual")]
        signature = hashlib.sha256(_URDF.read_bytes()).hexdigest()[:16]
        cache = _ROOT / "work/cache/so101" / signature
        cache.mkdir(parents=True, exist_ok=True)
        self.MESH_PATHS = []
        for link_name in self.LINK_NAMES:
            path = cache / (link_name + ".ply")
            if not path.exists():
                link = next(l for l in self.tree.findall("link") if l.get("name") == link_name)
                parts = []
                for visual in link.findall("visual"):
                    spec = visual.find("geometry/mesh")
                    if spec is None:
                        raise ValueError("SO101 adapter expects mesh visuals")
                    mesh = trimesh.load(_URDF.parent / spec.get("filename"), force="mesh")
                    mesh.apply_scale(np.fromstring(spec.get("scale", "1 1 1"), sep=" "))
                    mesh.apply_transform(_origin(visual.find("origin")))
                    parts.append(mesh)
                trimesh.util.concatenate(parts).export(path)
            self.MESH_PATHS.append(str(path))
        self.grasp_point_R_align = self._build_grasp_point_R_align(kwargs.get("grasp_point_rotation_align"))

    def _angles(self, q):
        if self.input_units == "unconfirmed":
            raise ValueError("SO101 joint units are not confirmed. Dataset preview is available, but calibration is blocked. Confirm all six joint-angle mappings (including gripper), then configure SO101_STATE_UNITS=radians or degrees. Normalized motor values cannot be used as angles.")
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("SO101 requires six finite joint angles in URDF joint order")
        return np.deg2rad(q) if self.input_units == "degrees" else q

    def _poses(self, q):
        values = dict(zip(self.JOINT_NAMES, self._angles(q)))
        poses = {"base_link": np.eye(4)}
        def pose(name):
            if name in poses: return poses[name]
            j = self.joints[name]
            T = _origin(j.find("origin"))
            if j.get("type") != "fixed":
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                axis = axis / np.linalg.norm(axis)
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * values[j.get("name")]).as_matrix()
                T = T @ turn
            poses[name] = pose(j.find("parent").get("link")) @ T
            return poses[name]
        for name in self.joints: pose(name)
        return poses

    def fkine_flange(self, q):
        # Rigid gripper attachment, before the fixed TCP offset.
        return self._poses(q)["gripper_link"][None]

    def _fkine_tcp_raw(self, q):
        return self._poses(q)[self.TCP_NAME][None]

    def fkine_all(self, q):
        poses = self._poses(q)
        return np.stack([poses[n] for n in self.LINK_NAMES])[None]
