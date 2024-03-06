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
from string_globals import *
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

chosen_one_args={
            "n_generated_img":10,
            "convergence_scale":0.95,
            "min_cluster_size":3,
            "max_iterations":3,
            "target_cluster_size":5
        }

'''

parser=argparse.ArgumentParser()
parser.add_argument("--n_generated_img",type=int,default=128,help="n image to generate for chosen one")
parser.add_argument("--convergence_scale",type=float,default=0.75,help="chosen one convergence scale")
parser.add_argument("--min_cluster_size",type=int,default=5)
parser.add_argument("--max_iterations",type=int, default=20,help="max times to do chosen one if not converging")
parser.add_argument("--target_cluster_size",type=int,default=10)
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
parser.add_argument("--retain_fraction",type=float,default=0.5)
parser.add_argument("--negative_prompt",type=str,default=NEGATIVE_PROMPT)
parser.add_argument("--target_prompt",type=str,default=LOL_SUFFIX)
parser.add_argument("--limit",type=int,default=30, help="n characters to try to do")

def main(args):
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="comparison")
    chosen_one_args={
            "n_generated_img":args.n_generated_img,
            "convergence_scale":args.convergence_scale,
            "min_cluster_size":args.min_cluster_size,
            "max_iterations":args.max_iterations,
            "target_cluster_size":args.target_cluster_size
    }

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)