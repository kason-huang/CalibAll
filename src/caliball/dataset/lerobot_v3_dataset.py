"""Read LeRobot v3 episode metadata and shared video offsets without conversion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from caliball.dataset.state_processors import StateProcessor
from caliball.utils.video_io import decode_video_frames

class LeRobotV3Dataset:
    def __init__(self, repo_id, episodes=None, state_keys="observation.state", state_processor=None, decode_video_keys=None, video_backend="pyav"):
        self.root = Path(repo_id)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("LeRobotV3Dataset requires codebase_version v3.0")
        self.fps = self.info["fps"]
        self.robot_type = self.info.get("robot_type", "unknown")
        self.camera_keys = [k for k,v in self.info["features"].items() if v.get("dtype") == "video"]
        self.video_keys = set(self.camera_keys)
        self.decode_video_keys = decode_video_keys or self.camera_keys
        self.state_keys = [state_keys] if isinstance(state_keys, str) else list(state_keys)
        self.processor = state_processor or StateProcessor()
        self.video_backend = video_backend
        paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not paths: raise FileNotFoundError("No v3 episode metadata found")
        rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if episodes is not None:
            missing = set(episodes) - set(rows.episode_index)
            if missing: raise ValueError(f"Episodes not found: {sorted(missing)}")
            rows = rows[rows.episode_index.isin(episodes)]
        self.episodes = rows.sort_values("episode_index").reset_index(drop=True)
        self.total_episodes = len(self.episodes)

    def __len__(self): return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes.iloc[idx]
        wanted = int(ep.episode_index)
        # Episode records can cross data shards; select by episode_index across
        # all shards rather than assuming one episode per file or local row 0.
        pieces = []
        for path in sorted((self.root / "data").rglob("*.parquet")):
            part = pd.read_parquet(path, filters=[("episode_index", "=", wanted)])
            if not part.empty: pieces.append(part)
        if not pieces: raise ValueError(f"No rows for episode {wanted}")
        df = pd.concat(pieces, ignore_index=True).sort_values("frame_index")
        if len(df) != int(ep.length): raise ValueError("Episode frame count mismatch")
        if df.frame_index.duplicated().any(): raise ValueError("Duplicate frame indices")
        ts = np.asarray(df.timestamp, dtype=float)
        ts = ts - ts[0]
        videos = {}
        for camera in self.decode_video_keys:
            if camera not in self.video_keys: raise ValueError(f"Unknown camera {camera}")
            prefix = "videos/" + camera
            path = self.root / self.info["video_path"].format(video_key=camera, chunk_index=int(ep[prefix+"/chunk_index"]), file_index=int(ep[prefix+"/file_index"]))
            times = ts + float(ep[prefix+"/from_timestamp"])
            videos[camera] = decode_video_frames(str(path), times.tolist(), 1.0/self.fps, self.video_backend)
            if len(videos[camera]) != len(df): raise ValueError("Video/state frame count mismatch")
        columns = {k: np.stack(df[k].values).astype(np.float32) for k in self.state_keys}
        return {"videos": videos, "states": self.processor(columns)}

    def __iter__(self):
        for i in range(len(self)): yield self[i]
