import os
os.environ["TEST_ENV"]="true"
import torch
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
import argparse
from string_globals import *
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download, ModelCard, upload_file
from datasets import Dataset
from experiment import prepare_unet_from_path
from accelerate import Accelerator

parser=argparse.ArgumentParser()
parser.add_argument("--n_img",type=int,default=100,help="how many images to generate for each class")
parser.add_argument("--flavor",type=str,default="hot",help="hot, cold or reward")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/prior")
parser.add_argument("--pretrained_lora_path",type=str, default="jlbaker361/test-ddpo-runway")


def main(args):
    print(args)
    prior_name_list=TOKEN_LIST+["character"]

    src_dict={
        prior:[] for prior in prior_name_list
    }
    accelerator=Accelerator()
    pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None)
    unet=pipeline.unet
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    if args.flavor==REWARD:
        weight_path=hf_hub_download(repo_id=args.pretrained_lora_path,filename="pytorch_lora_weights.safetensors", repo_type="model")
        unet=prepare_unet_from_path(unet,weight_path,["to_k", "to_q", "to_v", "to_out.0"])
    for model in [unet,vae,text_encoder]:
        model.requires_grad_(False)
    unet,vae,text_encoder=accelerator.prepare(unet,vae,text_encoder)
    for _ in range(args.n_img):
        for prior in prior_name_list:
            print(prior)
            prompt=prior.replace("_"," ")
            if args.flavor==COLD:
                image=pipeline(prompt,num_inference_steps=args.num_inference_steps,negative_prompt=NEGATIVE_PROMPT,safety_checker=None).images[0]
            elif args.flavor==HOT:
                image=pipeline(prompt+LOL_SUFFIX,num_inference_steps=args.num_inference_steps,safety_checker=None).images[0]
            else:
                image=pipeline(prompt, num_inference_steps=args.num_inference_steps,safety_checker=None).images[0]
            #image.save(f"{prior}.png")
            src_dict[prior].append(image)
        Dataset.from_dict(src_dict).push_to_hub(args.dest_dataset)
    model_card_content=f"""
    flavor: {args.flavor} \n
    num_inference_steps: {args.num_inference_steps}
    """
    with open("tmp_prior.md","w+") as file:
        file.write(model_card_content)
    upload_file(path_or_fileobj="tmp_prior.md", 
                path_in_repo="README.md",
                repo_id=args.dest_dataset,
                repo_type="dataset")

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
    main(args)