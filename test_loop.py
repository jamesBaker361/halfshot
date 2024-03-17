
import os
os.environ["TEST_ENV"]="true"
import torch
if "SLURM_JOB_ID" in os.environ:
    cache_dir="/scratch/jlb638/trans_cache"
    os.environ["TRANSFORMERS_CACHE"]=cache_dir
    os.environ["HF_HOME"]=cache_dir
    os.environ["HF_HUB_CACHE"]=cache_dir

    torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
    timesteps_per_image=2
    n_generated_img=10
    convergence_scale=0.95
else:
    timesteps_per_image=30
    n_generated_img=64
    convergence_scale=0.5
from adapter_training_loop import loop
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from experiment import train_and_evaluate
from accelerate import Accelerator
from string_globals import *
import unittest
import wandb

ip_adapter_image=Image.open("file.jpg")
text_prompt="a blonde woman"
learning_rate=1e-4
adam_beta1=0.9
adam_beta2=0.999
adam_weight_decay=1e-2
adam_epsilon=1e-08
max_grad_norm=1.0
epochs=1
seed=0
size=256
train_batch_size=1
num_validation_images=1
noise_offset=0.0
max_grad_norm=1.0
prior_loss_weight=0.9
prior_text_prompt="a blonde woman"
prior_images=[]
target_prompt=LOL_SUFFIX
negative_prompt=NEGATIVE_PROMPT
retain_fraction=0.5
ip_adapter_weight_name="ip-adapter-plus-face_sd15.bin"
chosen_one_args={
            "n_generated_img":n_generated_img,
            "convergence_scale":convergence_scale,
            "min_cluster_size":3,
            "max_iterations":3,
            "target_cluster_size":5
        }
n_prior=4
n_image=5
pretrained_lora_path="jlbaker361/test-ddpo-b"

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.accelerator=Accelerator(log_with="wandb")
        self.accelerator.init_trackers(project_name="test")
        wandb.define_metric("custom_step")

    def test_all(self):
        for training_method in [
            UNET,
            DB_MULTI,
            TEX_INV,
            UNET_IP,
            DB_MULTI_IP,
            TEX_INV_IP,
            CHOSEN_NEG_IP,
            CHOSEN_TARGET,
            CHOSEN_TARGET_IP,
            CHOSEN_TEX_INV,
            CHOSEN_TEX_INV_IP
        ]:
            with self.subTest(training_method=training_method):
                print(f"testing {training_method}")
                result_dict=train_and_evaluate(
                    ip_adapter_image=ip_adapter_image,
                    text_prompt=text_prompt,
                    accelerator=self.accelerator,
                    learning_rate=learning_rate,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    adam_weight_decay=adam_weight_decay,
                    adam_epsilon=adam_epsilon,
                    n_image=n_image,
                    prior_loss_weight=prior_loss_weight,
                    training_method=training_method,
                    epochs=epochs,
                    seed=seed,
                    timesteps_per_image=timesteps_per_image,
                    size=size,
                    train_batch_size=train_batch_size,
                    num_validation_images=num_validation_images,
                    noise_offset=noise_offset,
                    max_grad_norm=max_grad_norm,
                    chosen_one_args=chosen_one_args,
                    negative_prompt=negative_prompt,
                    target_prompt=target_prompt,
                    ip_adapter_weight_name=ip_adapter_weight_name,
                    retain_fraction=retain_fraction,
                    pretrained_lora_path=pretrained_lora_path
                )
                self.assertIsNotNone(result_dict)

if __name__=='__main__':
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    unittest.main()