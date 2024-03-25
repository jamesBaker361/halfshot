import os
import torch
if "SLURM_JOB_ID" in os.environ:
    cache_dir="/scratch/jlb638/trans_cache"
    os.environ["TRANSFORMERS_CACHE"]=cache_dir
    os.environ["HF_HOME"]=cache_dir
    os.environ["HF_HUB_CACHE"]=cache_dir

    torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
import argparse
from accelerate import Accelerator
from datasets import load_dataset
import wandb
from string_globals import *
from experiment import train_and_evaluate
from huggingface_hub import hf_hub_download, ModelCard, upload_file
from datasets import Dataset
import numpy as np
import memray
import time
import datetime
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
parser.add_argument("--n_generated_img",type=int,default=64,help="n image to generate for chosen one")
parser.add_argument("--convergence_scale",type=float,default=0.5,help="chosen one convergence scale")
parser.add_argument("--min_cluster_size",type=int,default=5)
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
parser.add_argument("--train_batch_size",type=int,default=8)
parser.add_argument("--num_validation_images",type=int,default=1)
parser.add_argument("--noise_offset",type=float,default=0.0)
parser.add_argument("--dataset",type=str,default="jlbaker361/league_faces_captioned_lite")
parser.add_argument("--retain_fraction",type=float,default=0.5)
parser.add_argument("--negative_prompt",type=str,default=NEGATIVE_PROMPT)
parser.add_argument("--target_prompt",type=str,default=LOL_SUFFIX)
parser.add_argument("--limit",type=int,default=20, help="n characters to try to do")
parser.add_argument("--training_method_suite",type=str, default=CHOSEN_SUITE)
parser.add_argument("--training_method",type=str, default=UNET)
parser.add_argument("--suffix",type=str,help="suffix to append to base text prompts",default="")
parser.add_argument("--img_type",type=str,default="tile",help="whether to use splash or tile")
parser.add_argument("--prior_loss_weight",type=float,default=0.5,help="weight for prior preservation")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/test_chosen_runner",help="destination dataset to push results")
parser.add_argument("--ip_adapter_weight_name",type=str,default="ip-adapter_sd15_light.bin")
parser.add_argument("--n_prior",type=int,default=5)
parser.add_argument("--pretrained_lora_path",type=str,default="jlbaker361/test-ddpo-runway")
parser.add_argument("--cooldown",type=float,default=10.0,help="time to sleep between training methods, maybe helps to reduce memory usage")
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/faceip")
parser.add_argument("--subfolder",type=str,help="subfolder for reward model",default="checkpoint_10")

def main(args):
    os.makedirs(args.image_dir,exist_ok=True)
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="chosen_comparison")
    chosen_one_args={
            "n_generated_img":args.n_generated_img,
            "convergence_scale":args.convergence_scale,
            "min_cluster_size":args.min_cluster_size,
            "max_iterations":args.epochs,
            "target_cluster_size":args.target_cluster_size
    }
    metric_list=["prompt_similarity","identity_consistency","negative_prompt_similarity","target_prompt_similarity","aesthetic_score"]
    columns=["method","label"]+metric_list
    data=[]
    dataset=load_dataset(args.dataset,split="train")
    src_dict={
        "label":[]
    }
    for metric in metric_list:
        src_dict[f"{metric}"]=[]
    for i,row in enumerate(dataset):
        if i>args.limit:
            break
        splash=row["splash"]
        tile=row["tile"]
        label=str(row["label"])
        src_dict["label"].append(label)
        text_prompt=row["caption"]+" "+args.suffix
        if args.img_type=="splash":
            ip_adapter_image=splash
        elif args.img_type=="tile":
            ip_adapter_image=tile
        result_dict=train_and_evaluate(
                ip_adapter_image=ip_adapter_image,
                description_prompt=text_prompt,
                accelerator=accelerator,
                learning_rate=args.learning_rate,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                adam_weight_decay=args.adam_weight_decay,
                adam_epsilon=args.adam_epsilon,
                n_image=args.n_prior,
                training_method=args.training_method,
                epochs=args.epochs,
                prior_loss_weight=args.prior_loss_weight,
                seed=args.seed,
                timesteps_per_image=args.timesteps_per_image,
                size=args.size,
                train_batch_size=args.train_batch_size,
                num_validation_images=args.num_validation_images,
                noise_offset=args.noise_offset,
                max_grad_norm=args.max_grad_norm,
                cold_prompt=args.negative_prompt,
                hot_prompt=args.target_prompt,
                retain_fraction=args.retain_fraction,
                ip_adapter_weight_name=args.ip_adapter_weight_name,
                chosen_one_args=chosen_one_args,
                pretrained_lora_path=args.pretrained_lora_path,
                label=label,
                subfolder=args.subfolder
            )
        for metric in metric_list:
            src_dict[f"{metric}"].append(result_dict[metric])
        data.append([args.training_method,label]+[result_dict[metric] for metric in metric_list])
        for i,image in enumerate(result_dict["images"]):
            os.makedirs(f"{args.image_dir}/{label}/",exist_ok=True)
            path=f"{args.image_dir}/{label}/{args.training_method}_{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}/{args.training_method}_{i}":wandb.Image(path)
            })
        del result_dict
        time.sleep(args.cooldown)
        print(src_dict)
        Dataset.from_dict(src_dict).push_to_hub(args.dest_dataset)
        model_card_content=f"""
        method: {args.training_method} \n
        num_inference_steps: {args.timesteps_per_image}\n
        """
        for metric in metric_list:
            mean=np.mean(src_dict[metric])
            model_card_content+="{metric} : {mean} \n"
        with open("tmp_prior.md","w+") as file:
            file.write(model_card_content)
        upload_file(path_or_fileobj="tmp_prior.md", 
                    path_in_repo="README.md",
                    repo_id=args.dest_dataset,
                    repo_type="dataset")
    accelerator.get_tracker("wandb").log({
        "result_table":wandb.Table(columns=columns,data=data)
        })
    print(f"pushed to {args.dest_dataset}")


if __name__=='__main__':
    args=parser.parse_args()
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    current_date_time = datetime.datetime.now()
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Formatted Date and Time:", formatted_date_time)
    print(args)
    main(args)
    print("runner totally all done :)))")