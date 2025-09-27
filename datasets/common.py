from __future__ import annotations
import io, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict
import torch

def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf)
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(x, dtype=np.uint8)
        if rgb.size == 2764800: rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600: rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)

def quat_to_rotate6d(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def action_slice(abs_traj: torch.Tensor, rel_idx: Sequence[int] = ()) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]

    if rel_idx:
        idx = torch.as_tensor(rel_idx, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    return {"proprio": proprio, "action": action}