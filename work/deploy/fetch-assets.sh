#!/bin/bash
set -eu
cd /home/kason/workspace/CalibAll
fetch_repo() {
  mkdir -p "third_party/$2"
  curl -fL --retry 3 "https://codeload.github.com/$1/tar.gz/refs/heads/main" | tar -xz --strip-components=1 -C "third_party/$2"
}
fetch_repo facebookresearch/sam3 sam3
fetch_repo facebookresearch/co-tracker co-tracker
fetch_repo facebookresearch/dinov2 dinov2
fetch_repo microsoft/MoGe moge
fetch_repo NVlabs/nvdiffrast nvdiffrast
fetch_repo facebookresearch/pytorch3d pytorch3d
mkdir -p ckpt/sam3 ckpt/cotracker ckpt/dinov2 ckpt/moge
curl -fL --retry 3 -o ckpt/sam3/sam3.pt.part https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt
mv ckpt/sam3/sam3.pt.part ckpt/sam3/sam3.pt
curl -fL --retry 3 -o ckpt/cotracker/scaled_offline.pth.part https://hf-mirror.com/facebook/cotracker3/resolve/main/scaled_offline.pth
mv ckpt/cotracker/scaled_offline.pth.part ckpt/cotracker/scaled_offline.pth
curl -fL --retry 3 -o ckpt/dinov2/dinov2_vitb14_pretrain.pth.part https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
mv ckpt/dinov2/dinov2_vitb14_pretrain.pth.part ckpt/dinov2/dinov2_vitb14_pretrain.pth
curl -fL --retry 3 -o ckpt/moge/model.pt.part https://hf-mirror.com/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt
mv ckpt/moge/model.pt.part ckpt/moge/model.pt
