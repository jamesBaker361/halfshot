from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch
from PIL import Image
from accelerate import Accelerator
from adapter_training_loop import loop

def get_trained_pipeline(
        pipeline:StableDiffusionPipeline,
        chosen_one:bool,
        with_prior_preservation:bool,
        use_ip_adapter:bool
)->StableDiffusionPipeline:
    loop_kwargs={}
    return None

def evaluate_pipeline(image,text_prompt,pipeline)->dict:
    return {"pipeline":pipeline}
    

def train_and_evaluate(image: Image,
                       text_prompt:str, 
                       accelerator:Accelerator,
                       learning_rate:float,
                       adam_beta1:float,
                       adam_beta2:float,
                       adam_weight_decay:float,
                       adam_epsilon:float,
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
                       )->dict:
    pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    unet=pipeline.unet
    vae=pipeline.vae
    tokenizer=pipeline.tokenizer
    text_encoder=pipeline.text_encoder
    for model in [vae,unet,text_encoder]:
        model.requires_grad_(False)
    trainable_parameters=[]
    with_prior_preservation=False
    use_ip_adapter=False
    ip_adapter_image=None
    use_chosen_one=False
    if training_method=="dreambooth":
        return
    elif training_method=="ip_adapter":
        #if trainable ip-adapter well have to do some shit here
        return
    elif training_method=="unet_lora":
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none")
        unet = get_peft_model(unet, config)
        unet.train()
        images=[image]*5
        text_prompt_list=[text_prompt]*5
    elif training_method=="textual_inversion":
        return
    for model in [vae,unet,text_encoder]:
        trainable_parameters+=[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    if not use_chosen_one:
        pipeline=loop(
            images=images,
            text_prompt_list=text_prompt_list,
            ip_adapter_image=ip_adapter_image,
            pipeline=pipeline,
            start_epoch=0,
            optimizer=optimizer,
            accelerator=accelerator,
            use_ip_adapter=use_ip_adapter,
            with_prior_preservation=with_prior_preservation,
            prior_loss_weight=prior_loss_weight,
            training_method=training_method,
            epochs=epochs,
            seed=seed,
            timesteps_per_image=timesteps_per_image,
            size=size,
            train_batch_size=train_batch_size,
            num_validation_images=num_validation_images,
            noise_offset=noise_offset,
            max_grad_norm=max_grad_norm
        )
        return evaluate_pipeline(image,text_prompt,pipeline)
