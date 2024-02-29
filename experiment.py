
import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from transformers import ViTImageProcessor, ViTModel
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch
from PIL import Image
from accelerate import Accelerator
from adapter_training_loop import loop
from string_globals import *
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from numpy.linalg import norm
from clustering import get_hidden_states,get_best_cluster_kmeans

def get_trained_pipeline(
        pipeline:StableDiffusionPipeline,
        chosen_one:bool,
        with_prior_preservation:bool,
        use_ip_adapter:bool
)->StableDiffusionPipeline:
    loop_kwargs={}
    return None


evaluation_prompt_list=[
    "a photo of {} at the beach",
    "a photo of {} in the jungle",
    "a photo of {} in the snow",
    "a photo of {} in the street",
    "a photo of {} with a city in the background",
    "a photo of {} with a mountain in the background",
    "a photo of {} with the Eiffel Tower in the background",
    "a photo of {} near the Statue of Liberty",
    "a photo of {} near the Sydney Opera House",
    "a photo of {} floating on top of water",
    "a photo of {} eating a burger",
    "a photo of {} drinking a beer",
    "a photo of {} wearing a blue hat",
    "a photo of {} wearing sunglasses",
    "a photo of {} playing with a ball",
    "a photo of {} as a police officer"
]

evaluation_prompt_list=[
    "a photo of {} at the beach", #this is just for testing
    "a photo of {} in the jungle"
]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def prepare_textual_inversion(text_prompt:str, tokenizer:object,text_encoder):
    initializer_token="person"
    for token in [ "man "," woman "," boy "," girl "]:
        if text_prompt.find(token)!=-1:
            initializer_token=token
    placeholder_tokens=[NEW_TOKEN]
    tokenizer.add_tokens(placeholder_tokens)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    text_encoder.get_input_embeddings().requires_grad_(True)
    return tokenizer,text_encoder

def evaluate_pipeline(image:Image,
                      text_prompt:str,
                      entity_name:str,
                      pipeline:StableDiffusionPipeline,
                      timesteps_per_image:int,
                      use_ip_adapter:bool)->dict:
    evaluation_image_list=[]
    generator=torch.Generator(pipeline.device)
    generator.manual_seed(123)
    for evaluation_prompt in evaluation_prompt_list:
        prompt=evaluation_prompt.format(entity_name)
        if use_ip_adapter:
            eval_image=pipeline(prompt,num_inference_steps=timesteps_per_image,generator=generator,ip_adapter_image=image).images[0]
        else:
            eval_image=pipeline(prompt,num_inference_steps=timesteps_per_image,generator=generator).images[0]
        evaluation_image_list.append(eval_image)
    clip_inputs=clip_processor(text=[text_prompt], images=evaluation_image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    text_embeds=outputs.text_embeds.detach().numpy()[0]
    image_embed_list=outputs.image_embeds.detach().numpy()
    prompt_similarity_list=[]
    identity_consistency_list=[]
    for i in range(len(image_embed_list)):
        vector_i=image_embed_list[i]
        prompt_similarity_list.append(np.dot(vector_i, text_embeds)/(norm(vector_i)* norm(text_embeds)))
        for j in range(i+1, len(image_embed_list)):
            vector_j=image_embed_list[j]
            identity_consistency_list.append(np.dot(vector_j,vector_i)/(norm(vector_i)*norm(vector_j)))
    
    return {"pipeline":pipeline,"images":evaluation_image_list,
            "prompt_similarity":np.mean(prompt_similarity_list),"identity_consistency":np.mean(identity_consistency_list) }
    
imagenet_template_list = [
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
                        chosen_one_args:dict={}
                       )->dict:
    pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline("nothing",num_inference_steps=2) #if we dont do this the properties wont instantiate correctly???
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
    entity_name=text_prompt
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
        #if trainable with ip-adapter well only be training the unet
        #this particular case well not actually use b/c training images=ip image
        use_ip_adapter=True
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
        unet_target_modules= ["to_q", "to_v", "query", "value"]
        unet_config=LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=unet_target_modules,
            lora_dropout=0.0
        )
        unet=get_peft_model(unet,unet_config)
        unet.train()
        images=[image]*5
        ip_adapter_image=image
        text_prompt_list=[text_prompt]*5
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
        tokenizer,text_encoder=prepare_textual_inversion(text_prompt,tokenizer,text_encoder)
        images=[image]*5
        text_prompt_list=[imagenet_template.format(text_prompt) for imagenet_template in imagenet_template_list]
        random_text_prompt=True
        entity_name=NEW_TOKEN
    elif training_method=="chosen_one_textual_inversion":
        tokenizer,text_encoder=prepare_textual_inversion(text_prompt,tokenizer,text_encoder)
        text_prompt_list=[imagenet_template.format(text_prompt) for imagenet_template in imagenet_template_list]
        random_text_prompt=True
        use_chosen_one=True
        entity_name=NEW_TOKEN
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
    else:
        n_generated_img=chosen_one_args["n_generated_img"] # how many images to generate and then cluster
        convergence_scale=chosen_one_args["convergence_scale"] #when cluster distance < convergence * init_dist then we stop
        min_cluster_size=chosen_one_args["min_cluster_size"] #ignore clusters smaller than this aka d_min_c
        max_iterations=chosen_one_args["max_iterations"] #how many loops before we give up
        #starting_cluster=chosen_one_args["starting_cluster"] #initial images
        target_cluster_size=chosen_one_args["target_cluster_size"] #aka dsize_c
        n_clusters=n_generated_img // target_cluster_size
        #generator=
        image_list=pipeline(text_prompt,num_inference_steps=timesteps_per_image,num_images_per_prompt=n_generated_img).images
        last_hidden_states=get_hidden_states(image_list)
        init_dist=np.mean(cdist(last_hidden_states, last_hidden_states, 'euclidean'))
        pairwise_distances=init_dist
        iteration=0
        while pairwise_distances>=convergence_scale*init_dist and iteration<max_iterations:
            valid_image_list, pairwise_distances=get_best_cluster_kmeans(image_list,n_clusters, min_cluster_size)
            print(f"iteration {iteration} pairwise distances {pairwise_distances}")
            if not random_text_prompt:
                text_prompt_list=text_prompt_list*len(valid_image_list)
            pipeline=loop(
                images=valid_image_list,
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
                epochs=1,
                seed=seed,
                timesteps_per_image=timesteps_per_image,
                size=size,
                train_batch_size=train_batch_size,
                num_validation_images=num_validation_images,
                noise_offset=noise_offset,
                max_grad_norm=max_grad_norm)
            image_list=pipeline(text_prompt,num_inference_steps=timesteps_per_image,num_images_per_prompt=n_generated_img).images
            iteration+=1
    return evaluate_pipeline(image,text_prompt,entity_name,pipeline,timesteps_per_image,use_ip_adapter)
