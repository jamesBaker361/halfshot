import os
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from peft.utils import get_peft_model_state_dict
from string_globals import *
from torch.utils.data import DataLoader
from torchvision import transforms
from data_helpers import make_dataloader
import wandb
import torch
import torch.nn.functional as F

def loop(images: list,
               text_prompt_list:list,
               ip_adapter_image: Image,
               pipeline:StableDiffusionPipeline,
               start_epoch:int,
               optimizer:object,
               accelerator:object,
               use_ip_adapter:bool,
               with_prior_preservation:bool,
               prior_loss_weight:float,
               training_method:str,
               epochs:int,
               seed:int,
                timesteps_per_image:int,
                size:int,
                train_batch_size:int,
                num_validation_images:int,
                noise_offset:float,
                max_grad_norm:float,
               )->StableDiffusionPipeline:
    '''
    given images generated from text prompt, and the src_image, trains the unet lora pipeline for epochs
    using the prompt and the src_image for conditioning and returns the trained pipeline
    
    images: PIL imgs to be used for training
    ip_adapter_image: 
    text_prompt: text prompts describing character
    pipeline: should already have ip adapter loaded
    start_epoch: epoch we're starting at
    epochs: how many epochs to do this training
    optimizer: optimizer
    acceelrator: accelerator object
    size: img dim 
    train_batch_size: batch size
    with_prior_preservation: whether to use prior preservation (for dreambooth)
    noise_offset: https://www.crosslabs.org//blog/diffusion-with-offset-noise
    '''
    tracker=accelerator.get_tracker("wandb")
    for i in range(num_validation_images):
        wandb.define_metric(f"{training_method}_img_{i}",step_metric="custom_step")
    tokenizer=pipeline.tokenizer
    vae=pipeline.vae
    text_encoder=pipeline.text_encoder
    dataloader=make_dataloader(images,text_prompt_list,tokenizer,size, train_batch_size)
    unet=pipeline.unet
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters()) #optimizer should already be listening to whatever layers we're optimzing
    unet,text_encoder,vae, optimizer, dataloader= accelerator.prepare(
        unet,text_encoder,vae, optimizer, dataloader
    )
    added_cond_kwargs={}
    if use_ip_adapter:
        image_embeds = pipeline.prepare_ip_adapter_image_embeds(
                    ip_adapter_image, None, accelerator.device, train_batch_size)
        added_cond_kwargs = {"image_embeds": image_embeds}
    weight_dtype=pipeline.dtype
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps_per_image,clip_sample=False)
    global_step=0
    for e in range(start_epoch, epochs):
        unet.train() #only if unet has trainable layers
        train_loss = 0.0
        for step,batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch[IMAGES].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if noise_offset:
                    noise += noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch[TEXT_INPUT_IDS])[0]

                noise_pred = unet(noisy_latents, 
                                timesteps, 
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs).sample
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item()
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({f"{training_method}_train_loss": train_loss}, step=global_step)
                train_loss = 0.0
        if accelerator.is_main_process:
            '''save_path = os.path.join(output_dir, f"checkpoint-{e}")
            accelerator.save_state(save_path)

            unet_lora_state_dict = get_peft_model_state_dict(unet)

            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_path,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )'''

            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(seed)

            path="tmp.png"
            for i in range(num_validation_images):
                text_prompt=text_prompt_list[i]
                img=pipeline(text_prompt, num_inference_steps=timesteps_per_image, generator=generator).images[0]
                img.save(path)
                tracker.log({f"{training_method}_img_{i}": wandb.Image(path), "custom_step":e})

    return pipeline