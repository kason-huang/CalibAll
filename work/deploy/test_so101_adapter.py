import json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from roboticstoolbox.robot.Robot import Robot
from caliball.robots import build_robot, list_robots
from caliball.dataset.lerobot_v3_dataset import LeRobotV3Dataset
from caliball.web.services.dataset_builder import DatasetBuilder

root=Path("/home/kason/workspace/CalibAll")
assert "so101" in list_robots()
blocked=build_robot("so101")
try:
    blocked.fkine_flange(np.zeros(6))
except ValueError as exc:
    assert "not confirmed" in str(exc)
else:
    raise AssertionError("Unknown joint units must block calibration")
r=build_robot("so101", input_units="radians")
links,name,text,path=Robot.URDF_read(file_path="so101_new_calib.urdf", tld=str(root/"third_party/urdf/so101"))
reference=Robot(links, name=name, urdf_string=text, urdf_filepath=path)
xml=ET.parse(path).getroot()
children={j.get("name"):j.find("child").get("link") for j in xml.findall("joint")}
for sample in [np.zeros(6), np.array([.2,-.4,.6,.3,-.2,.1])]:
    mapped=np.zeros(reference.n)
    for j,v in zip(r.JOINT_NAMES,sample):
        link=next(l for l in reference.links if l.name==children[j])
        mapped[link.jindex]=v
    actual=r._poses(sample)
    for link in r.LINK_NAMES+[r.TCP_NAME]:
        np.testing.assert_allclose(actual[link],reference.fkine(mapped,end=link).A,atol=1e-8)
    assert r.fkine_all(sample).shape == (1,len(r.MESH_PATHS),4,4)
print("Independent RoboticsToolbox FK comparison passed",flush=True)
assert all(Path(p).is_file() for p in r.MESH_PATHS)
print("Merged visual meshes",len(r.MESH_PATHS),flush=True)
path="/home/kason/workspace/embodied-data-evaluator/data/svla_so101_pickplace"
for ep in [0,1]:
    ds=LeRobotV3Dataset(path,episodes=[ep],decode_video_keys=["observation.images.side"])
    sample=ds[0]
    assert sample["states"].shape[1]==6
    assert sample["videos"]["observation.images.side"].shape[0]==sample["states"].shape[0]
    print("Decoded episode",ep,sample["states"].shape,sample["videos"]["observation.images.side"].shape,flush=True)
    del sample,ds
cfg=DatasetBuilder.parse_config("so101_v3.yaml")
assert cfg["state_keys"]==["observation.state"]
ds,info=DatasetBuilder.build("so101_v3.yaml",path,episode_idx=1)
assert int(ds.episodes.iloc[0].episode_index)==1
print("Web dataset config construction passed",flush=True)
print("SO101 ADAPTER TESTS PASSED",flush=True)
