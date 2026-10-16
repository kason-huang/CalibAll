# CalibAll deployment — 2026-10-16

Server: 8-v100-kason, /home/kason/workspace/CalibAll
Server URL: http://127.0.0.1:8766/
Windows SSH tunnel URL: http://127.0.0.1:18766/

## Service management

systemctl --user status caliball
systemctl --user restart caliball
systemctl --user stop caliball
journalctl --user -u caliball -f

User service is enabled; user linger is enabled for persistence after logout.
Port 8765 belongs to EgoWAM and was preserved. CalibAll uses GPU 0.

## Windows tunnel

ssh -N -L 127.0.0.1:18766:127.0.0.1:8766 8-v100-kason

The current hidden tunnel was started by Codex; recreate it if the Windows machine restarts.

## Runtime

.venv is isolated for added dependencies but shares the existing torch 2.5.1+cu124
runtime and its dependencies through existing-runtime.pth, referencing:
/home/kason/workspace/EgoWAM/emimic/lib/python3.11/site-packages
Do not remove that environment while CalibAll is using it.
CUDA toolkit: /usr/local/cuda-12.4, V100 architecture 7.0.
USE_PERFLIB=0 avoids unsupported optimized attention paths on V100.
PyTorch3D is used from its source tree for pure Python transforms; compiled
PyTorch3D operators are not installed. CalibAll rendering uses NVDiffrast.
RTB 1.1.1 uses rtb-data 1.0.1, spatialgeometry 1.1.0, swift-sim 1.1.0.

Checkpoints live under ckpt/; third-party source trees under third_party/.
No project application source files were modified for this deployment.

## Checks

See smoke_models.log for GPU rasterization and SAM3 point segmentation checks.
See load-all-models.log for complete model loading, MoGe and tracking smoke checks.
A real dataset calibration accuracy run requires dataset selection and annotation;
component smoke tests do not establish calibration accuracy.
