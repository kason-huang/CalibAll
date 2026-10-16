import os
os.environ['USE_PERFLIB']='0'
import numpy as np
import torch
from caliball.rendering.nvdiffrast_renderer import NVDiffrastRenderer
renderer=NVDiffrastRenderer([64,64],device='cuda')
v=torch.tensor([[-.2,-.2,1],[.2,-.2,1],[0,.2,1]],device='cuda')
f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
k=torch.tensor([[60.,0,32],[0,60,32],[0,0,1]],device='cuda')
mask=renderer.render_mask(v,f,k,torch.eye(4,device='cuda'))
assert mask.sum()>0
print('GPU rasterization OK',mask.sum().item(),flush=True)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
m=build_sam3_image_model(bpe_path='third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',checkpoint_path='ckpt/sam3/sam3.pt',device='cuda',enable_inst_interactivity=True)
p=Sam3Processor(m,device='cuda')
s=p.set_image(Image.open('assets/test_img/source.png').convert('RGB'))
out=m.predict_inst(s,point_coords=np.array([[376,131]],dtype=np.float32),point_labels=np.array([1],dtype=np.int32),multimask_output=False)
print('SAM3 point segmentation OK',out[0].shape,flush=True)
