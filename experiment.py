from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch
from PIL import Image
from accelerate import Accelerator
from adapter_training_loop import loop
from string_globals import *

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
    
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

def train_and_evaluate(image: Image,
                       text_prompt:str, 
                       accelerator:Accelerator,
                       learning_rate:float,
                       adam_beta1:float,
                       adam_beta2:float,
                       adam_weight_decay:float,
                       adam_epsilon:float,
                       prior_text_prompt:str,
                       prior_images:list,
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
    prior_text_prompt_list=[]
    use_ip_adapter=False
    ip_adapter_image=None
    use_chosen_one=False
    random_text_prompt=False
    if training_method=="dreambooth":
        text_encoder_target_modules=["q_proj", "v_proj"]
        text_encoder_config=LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=text_encoder_target_modules,
            lora_dropout=0.0
        )
        text_encoder=get_peft_model(text_encoder,text_encoder_config)
        text_encoder.train()

        unet_target_modules= ["to_q", "to_v", "query", "value"]
        unet_config=LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=unet_target_modules,
            lora_dropout=0.0
        )
        unet=get_peft_model(unet,unet_config)
        unet.train()
        with_prior_preservation=True
        prior_text_prompt_list=[prior_text_prompt]*len(prior_images)
        images=[image]*len(prior_images)
        text_prompt_list=[text_prompt]*len(prior_images)
    elif training_method=="ip_adapter":
        #if trainable ip-adapter well have to do some shit here
        return
    elif training_method=="unet_lora":
        unet_target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=unet_target_modules,
            lora_dropout=0.0,
            bias="none")
        unet = get_peft_model(unet, config)
        unet.train()
        images=[image]*5
        text_prompt_list=[text_prompt]*5
    elif training_method=="textual_inversion":
        initializer_token="person"
        for token in [ "man "," woman "," boy "," girl "]:
            if text_prompt.find(token)!=-1:
                initializer_token=token
        placeholder_tokens=[NEW_TOKEN]
        tokenizer.add_tokens(placeholder_tokens)
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        initializer_token_id = token_ids[0]
        placeholder_token_ids = tokenizer.convert_tokens_to_ids()

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        text_encoder.get_input_embeddings().requires_grad_(True)
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
            random_text_prompt=random_text_prompt,
            with_prior_preservation=with_prior_preservation,
            prior_text_prompt_list=prior_text_prompt_list,
            prior_images=prior_images,
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
