import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
import argparse
from accelerate import Accelerator
import wandb

'''
image=Image.open("file.jpg")
text_prompt="a blonde woman"
learning_rate=1e-4
adam_beta1=0.9
adam_beta2=0.999
adam_weight_decay=1e-2
adam_epsilon=1e-08
max_grad_norm=1.0
epochs=1
seed=0
timesteps_per_image=30
size=256
train_batch_size=1
num_validation_images=1
noise_offset=0.0
max_grad_norm=1.0
prior_loss_weight=0.9
prior_text_prompt="a blonde woman"
prior_images=[]
'''

parser=argparse.ArgumentParser()
parser.add_argument("--learning_rate",type=float,default=0.0001)
parser.add_argument("--adam_beta1",type=float,default=0.9)
parser.add_argument("--adam_beta2",type=float,default=0.999)
parser.add_argument("--adam_weight_decay",type=float, default=0.01)
parser.add_argument("--adam_epsilon",type=float,default=0.00000001)
parser.add_argument("--max_grad_norm",type=float,default=1.0)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--timesteps_per_image",type=int,default=30)
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--train_batch_size",type=int,default=2)
parser.add_argument("--noise_offset",type=float,default=0.0)
parser.add_argument("--prior_loss_weight",type=float,default=0.5)
parser.add_argument("--dataset",type=str,default="jlbaker361/league_faces_captioned_priors")

def main(args):
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="comparison")

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)