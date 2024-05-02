from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from string_globals import *
import torchvision
from transformers import CLIPProcessor, CLIPModel
from aesthetic_reward import AestheticScorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import torch
import random
from PIL import Image

class PipelineOutput:
    def __init__(self,images):
        self.images=images

def align_generate(
        accelerator:Accelerator,
        pipeline:StableDiffusionPipeline,
        prompt:str,
        flavor:str, #one of COLD,HOT,REWARD
        truncated_backprop_rand:bool,
        truncated_backprop_min:int,
        truncated_backprop_max:int,
        truncated_backprop_timestep:int,
        guidance_scale:float,
        optimizer:object,
        target_prompt:str,
        negative_prompt:str=""
        
): #https://github.com/mihirp1998/AlignProp/blob/main/main.py#L150
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    
    def prepare_images(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        #im_pix = normalize(im_pix)
        return im_pix
    
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",do_rescale=False)
    clip_model.to(accelerator.device)
    clip_processor.to(accelerator.device)
    clip_model,clip_processor=accelerator.prepare(
        clip_model,clip_processor
    )

    if flavor==COLD:
        def loss_fn(images):
            input_ids=clip_processor(NEGATIVE_PROMPT,return_tensors="pt", padding=True).input_ids
            pixel_values=prepare_images(images)
            pixel_values.to(accelerator.device)
            input_ids.to(accelerator.device)
            clip_outputs=clip_model(input_ids=input_ids, pixel_values=pixel_values)
            return  clip_outputs.logits_per_image
    elif flavor==HOT:
        def loss_fn(images):
            input_ids=clip_processor(HOT_PROMPT,return_tensors="pt", padding=True).input_ids
            pixel_values=prepare_images(images)
            pixel_values.to(accelerator.device)
            input_ids.to(accelerator.device)
            clip_outputs=clip_model(input_ids=input_ids, pixel_values=pixel_values)
            return -1.0 * clip_outputs.logits_per_image
    elif flavor==REWARD:
        aesthetic_scorer=AestheticScorer(
            dtype=pipeline.unet.dtype,
            model_id=hf_hub_aesthetic_model_id,
            model_filename=hf_hub_aesthetic_model_filename,
            do_rescale=False
        )
        aesthetic_scorer=accelerator.prepare(aesthetic_scorer)
        def loss_fn(images):
            pixel_values=prepare_images(images)
            pixel_values.to(accelerator.device)
            return aesthetic_scorer.call_grad(pixel_values)[0]
    elif flavor==PROMPT:
        input_ids=clip_processor(prompt,return_tensors="pt", padding=True).input_ids
        pixel_values=prepare_images(images)
        pixel_values.to(accelerator.device)
        input_ids.to(accelerator.device)
        clip_outputs=clip_model(input_ids=input_ids, pixel_values=pixel_values)
        return -1.0 * clip_outputs.logits_per_image
    images=[]
    unet=pipeline.unet
    timesteps = pipeline.scheduler.timesteps
    latent = torch.randn((1, 4, 64, 64), device=accelerator.device, dtype=unet.dtype)
    unet.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    pipeline.vae.to(accelerator.device)
    prompt_ids=pipeline.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
    negative_prompt_ids=pipeline.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
    negative_prompt_embeds=pipeline.text_encoder(negative_prompt_ids)[0]
    with accelerator.accumulate(unet):
        with torch.enable_grad():
            for i,t in enumerate(timesteps):
                print("step ",i)
                t = torch.tensor([t],dtype=unet.dtype,device=latent.device)
                noise_pred_uncond = unet(latent, t, negative_prompt_embeds).sample
                noise_pred_cond = unet(latent, t, prompt_embeds).sample
                if truncated_backprop_rand:
                    timestep = random.randint(truncated_backprop_min,truncated_backprop_max)
                    print("timestep",timestep)
                    if i < timestep:
                        noise_pred_uncond = noise_pred_uncond.detach()
                        noise_pred_cond = noise_pred_cond.detach()
                elif i < truncated_backprop_timestep:
                    noise_pred_uncond = noise_pred_uncond.detach()
                    noise_pred_cond = noise_pred_cond.detach()

                grad = (noise_pred_cond - noise_pred_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * grad                
                latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                
                accelerator.free_memory()
                torch.cuda.empty_cache()
                del grad,noise_pred_uncond,noise_pred

            image= pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample.to(accelerator.device)


            loss=loss_fn(image)
            loss =  loss.sum()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            image = (image.squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
            images.append(image)
    accelerator.free_memory()
    torch.cuda.empty_cache()
    del latent
    return PipelineOutput(images)