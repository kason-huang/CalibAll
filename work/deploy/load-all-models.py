import os,sys,runpy
os.environ['USE_PERFLIB']='0'
sys.path.insert(0,os.getcwd())
from caliball.robots import build_robot
import numpy as np
for name,n in [('franka',7),('ur5e',6)]:
    robot=build_robot(name)
    print('FK OK',name,robot.fkine_flange(np.zeros(n)).shape,flush=True)
app=runpy.run_path('scripts/caliball_web.py',run_name='caliball_deploy_smoke')
models=app['_load_models']('cuda','franka','third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz','ckpt/sam3/sam3.pt')
print('ALL MODELS LOADED OK',flush=True)
from PIL import Image
coarse=models[0]
K=coarse.intrinsic_estimator.estimate(Image.open('assets/test_img/source.png').convert('RGB'))[0]
print('MoGe inference OK', K,flush=True)
frames=np.repeat(np.asarray(Image.open('assets/test_img/source.png').convert('RGB').resize((256,144)))[None],8,axis=0)
tracks=coarse.point_tracker.track(video=frames,uv=(75,26),img_idx=0)
print('CoTracker inference OK', tracks[0].shape,flush=True)
print('DEPLOYMENT SMOKE PASSED',flush=True)
