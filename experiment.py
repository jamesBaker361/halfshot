
import os
import torch
if "SLURM_JOB_ID" in os.environ:
    cache_dir="/scratch/jlb638/trans_cache"
    os.environ["TRANSFORMERS_CACHE"]=cache_dir
    os.environ["HF_HOME"]=cache_dir
    os.environ["HF_HUB_CACHE"]=cache_dir

    torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from transformers import ViTImageProcessor, ViTModel
from huggingface_hub import hf_hub_download
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
from clustering import get_hidden_states,get_best_cluster_kmeans,get_best_cluster_sorted,get_init_dist
import time
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from safetensors import safe_open
from aesthetic_reward import get_aesthetic_scorer
import gc
from datasets import load_dataset
import random

def prepare_unet(unet,unet_target_modules=["to_k", "to_q", "to_v", "to_out.0"]):
    config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=unet_target_modules,
        lora_dropout=0.0,
        bias="none")
    unet = get_peft_model(unet, config)
    unet.train()
    print("prepare unet trainable parameters")
    unet.print_trainable_parameters()
    return unet

def prepare_unet_from_path(unet,weight_path:str,trainable_modules:list):
    unet=prepare_unet(unet)
    state_dict={}
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key]=f.get_tensor(key)
    state_dict={
        "base_model.model."+k.replace("weight","default.weight"):v for k,v in state_dict.items()
    }
    count=0
    unet.load_state_dict(state_dict,strict=False)
    for (name,param) in unet.named_parameters():
        if name in state_dict:
            count+=1
        param.requires_grad_(False)
        for module_name in trainable_modules:
            if name.find(module_name)!=-1 and name.find("default.weight")!=-1:
                param.requires_grad_(True)
    print(f"loaded {count} parameters in prepare_unet_from_path")
    unet.print_trainable_parameters()
    return unet

def get_initializer_token(text_prompt:str)->str:
    initializer_token=""
    for token in [ " man "," woman "," boy "," girl "]:
        if text_prompt.find(token)!=-1:
            initializer_token=token
    if initializer_token=="":
        initializer_token="character"
    return initializer_token.strip()

def prepare_textual_inversion(text_prompt:str, tokenizer:object,text_encoder:object):
    initializer_token=""
    for token in TOKEN_LIST:
        if text_prompt.find(token)!=-1:
            initializer_token=token
    if initializer_token=="":
        initializer_token="character"
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

def evaluate_pipeline(ip_adapter_image:Image,
                      text_prompt:str,
                      entity_name:str,
                      pipeline:StableDiffusionPipeline,
                      timesteps_per_image:int,
                      use_ip_adapter:bool,
                      negative_prompt:str,
                      target_prompt:str,
                      clip_processor:CLIPProcessor,
                       clip_model:CLIPModel )->dict:
    evaluation_image_list=[]
    generator=torch.Generator(pipeline.device)
    generator.manual_seed(123)
    aesthetic_scorer=get_aesthetic_scorer()
    evaluation_prompt_list=[
        "  {} at the beach",
        "  {} in the jungle",
        "  {} in the snow",
        "  {} in the street",
        "  {} with a city in the background",
        "  {} with a mountain in the background",
        "  {} with the Eiffel Tower in the background",
        "  {} near the Statue of Liberty",
        "  {} near the Sydney Opera House",
        "  {} floating on top of water",
        "  {} eating a burger",
        "  {} drinking a beer",
        "  {} wearing a blue hat",
        "  {} wearing sunglasses",
        "  {} playing with a ball",
        "  {} as a police officer"
    ]
    if "TEST_ENV" in os.environ:
        evaluation_prompt_list=evaluation_prompt_list[:2]
    for evaluation_prompt in evaluation_prompt_list:
        prompt=evaluation_prompt.format(entity_name)
        print(f"eval prompt {prompt}")
        if use_ip_adapter:
            eval_image=pipeline(prompt,num_inference_steps=timesteps_per_image,generator=generator,ip_adapter_image=ip_adapter_image,negative_prompt=negative_prompt).images[0]
        else:
            eval_image=pipeline(prompt,num_inference_steps=timesteps_per_image,generator=generator,negative_prompt=negative_prompt).images[0]
        evaluation_image_list.append(eval_image)
    if negative_prompt in ["", " "]:
        negative_prompt=text_prompt
    if target_prompt in ["", " "]:
        target_prompt=text_prompt
    text_list=[text_prompt, negative_prompt, target_prompt]
    clip_inputs=clip_processor(text=text_list, images=evaluation_image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    text_embeds=outputs.text_embeds.detach().numpy()[0]
    negative_text_embeds=outputs.text_embeds.detach().numpy()[1]
    target_text_embeds=outputs.text_embeds.detach().numpy()[2]
    image_embed_list=outputs.image_embeds.detach().numpy()
    prompt_similarity_list=[]
    negative_prompt_similarity_list=[]
    target_prompt_similarity_list=[]
    aesthetic_score_list=[]

    for image in evaluation_image_list:
        aesthetic_score_list.append(aesthetic_scorer(image).cpu().numpy()[0])

    identity_consistency_list=[]
    for i in range(len(image_embed_list)):
        vector_i=image_embed_list[i]
        prompt_similarity_list.append(np.dot(vector_i, text_embeds)/(norm(vector_i)* norm(text_embeds)))
        negative_prompt_similarity_list.append(np.dot(vector_i, negative_text_embeds)/(norm(vector_i)* norm(negative_text_embeds)))
        target_prompt_similarity_list.append(np.dot(vector_i, target_text_embeds)/(norm(vector_i)* norm(target_text_embeds)))
        for j in range(i+1, len(image_embed_list)):
            vector_j=image_embed_list[j]
            identity_consistency_list.append(np.dot(vector_j,vector_i)/(norm(vector_i)*norm(vector_j)))
    result_dict= {
        #"pipeline":pipeline,
        "aesthetic_score": np.mean(aesthetic_score_list),
        "images":evaluation_image_list,
            "prompt_similarity":np.mean(prompt_similarity_list),
            "identity_consistency":np.mean(identity_consistency_list),
             "negative_prompt_similarity":np.mean(negative_prompt_similarity_list),
              "target_prompt_similarity": np.mean(target_prompt_similarity_list) }
    
    if ip_adapter_image is not None:
        ip_image_similarity_list=[]
        clip_inputs=clip_processor(text=["irrelevant"], images=[ip_adapter_image], return_tensors="pt", padding=True)
        outputs = clip_model(**clip_inputs)
        ip_image_embeds=outputs.image_embeds.detach().numpy()[0]
        for i in range(len(image_embed_list)):
            vector_i=image_embed_list[i]
            ip_image_similarity_list.append(np.dot(vector_i, ip_image_embeds)/(norm(vector_i)* norm(ip_image_embeds)))
        result_dict["ip_image_similarity"]=np.mean(ip_image_similarity_list)
    else:
        result_dict["ip_image_similarity"]=-1.0
    return result_dict
    
imagenet_template_list = [
    "  a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "  a clean {}",
    "  a dirty {}",
    "a dark photo of the {}",
    "  my {}",
    "  the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "  the {}",
    "a good photo of the {}",
    "  one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "  the clean {}",
    "a rendition of a {}",
    "  a nice {}",
    "a good photo of a {}",
    "  the nice {}",
    "  the small {}",
    "  the weird {}",
    "  the large {}",
    "  a cool {}",
    "  a small {}",
]



def train_and_evaluate(ip_adapter_image:Image,
                       description_prompt:str, 
                       accelerator:Accelerator,
                       learning_rate:float,
                       adam_beta1:float,
                       adam_beta2:float,
                       adam_weight_decay:float,
                       adam_epsilon:float,
                       n_image:int,
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
                        cold_prompt:str,
                        hot_prompt:str,
                        retain_fraction:float,
                        ip_adapter_weight_name:str,
                        chosen_one_args:dict,
                        pretrained_lora_path:str,
                        label:str,
                        subfolder:str
                       )->dict:
    """
    init_image_list= the images we are starting with
    of this image and train on the same image a bunch
    text_prompt= the text prompt describing the character or whatever for ex: "a male character with a sword holding a sword and wearing a blue and black outfit"
    prior_text_prompt= for dreambooth this is the prior, (should be Man, woman, boy, girl or person)
    prior_images = images of prior text prompt
    """
    print(f"training method {training_method}")
    start=time.time()
    try:
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()
        print("cleared cache!?!?")
    except:
        print("did not clear cache")
    try:
        print('experiment torch.cuda.get_device_name()',torch.cuda.get_device_name())
        print('experiment torch.cuda.get_device_capability()',torch.cuda.get_device_capability())
        print('experiment torch.cuda.get_device_properties()',torch.cuda.get_device_properties())
    except:
        print("couldnt print cuda details")
    pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
    vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
    pipeline.safety_checker=None
    unet=pipeline.unet
    vae=pipeline.vae
    tokenizer=pipeline.tokenizer
    text_encoder=pipeline.text_encoder
    
    for model in [vae,unet,text_encoder]:
        model.requires_grad_(False)
        #set everything to not be trainable by default
    unet,text_encoder,vae,tokenizer = accelerator.prepare(
        unet,text_encoder,vae,tokenizer
    )
    #clip_processor, vit_processor, vit_model=accelerator.prepare(clip_model,clip_processor, vit_processor, vit_model)
    pipeline("nothing",num_inference_steps=2,safety_checker=None) #if we dont do this the properties wont instantiate correctly???
    trainable_parameters=[]
    with_prior_preservation=False
    prior_text_prompt_list=[]
    use_ip_adapter=False
    use_chosen_one=False
    random_text_prompt=False
    entity_name=NEW_TOKEN
    negative=True
    cluster_text_prompt=description_prompt
    prior_images=[]
    images=[]
    negative_prompt=""
    if training_method.find(COLD)!=-1 and training_method.find(BASIC)==-1:
        negative_prompt=cold_prompt
    if training_method.find(BASIC)==-1 and training_method.find(HOT) !=-1:
        description_prompt+=hot_prompt
    if training_method.find(REWARD)!=-1:
        weight_path=hf_hub_download(repo_id=pretrained_lora_path,subfolder=subfolder,filename="pytorch_lora_weights.safetensors", repo_type="model")
        trainable_modules=["to_k", "to_q", "to_v", "to_out.0"]
        if training_method in [TEX_INV_REWARD, TEX_INV_REWARD_IP]:
            trainable_modules=[]
        if training_method in [DB_MULTI_REWARD_IP, DB_MULTI_REWARD]:
            trainable_modules=["to_q","to_v"]
        unet=prepare_unet_from_path(unet, weight_path,trainable_modules)
        print(f"loaded from {pretrained_lora_path}")
    if training_method.find(IP)!=-1:
        train_batch_size=min(train_batch_size,2) #accelerate can be weird if batch size is too big with ip adapter
        use_ip_adapter=True
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_adapter_weight_name)
        images=[
            pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image, ip_adapter_image=ip_adapter_image).images[0] for _ in range(n_image)
        ]
    if training_method.find(CHOSEN)!=-1 or training_method.find(CHOSEN_TEX_INV)!=-1: #TODO all chosen AND cte should do this- might be redundant with stuff in tex inv
        use_chosen_one=True
    if training_method.find(CHOSEN)!=-1:
        chosen_one_args["n_generated_img"]=int(chosen_one_args["n_generated_img"]/retain_fraction)
    if training_method.find(DB_MULTI)!=-1:
        with_prior_preservation=True
        text_prompt_list=[NEW_TOKEN]*n_image
        entity_name=NEW_TOKEN
        validation_prompt_list=text_prompt_list
        initializer_token=get_initializer_token(description_prompt)
        prior_text_prompt_list=[initializer_token]*n_image
        prior_dataset="jlbaker361/prior-"
        for flavor in [COLD,HOT,REWARD]:
            if training_method.find(flavor)!=-1:
                prior_dataset+=flavor
                break
        if timesteps_per_image==50:
            prior_dataset+="-50"
        print("prior_dataset",prior_dataset)
        if training_method.find(REWARD)==-1: #TODO all db_multi should do this eexcept for reward
            text_encoder_target_modules=["q_proj", "v_proj"]
            text_encoder_config=LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=text_encoder_target_modules,
                lora_dropout=0.0
            )
            text_encoder=get_peft_model(text_encoder,text_encoder_config)
            text_encoder.train()
            print("text encoder parameters")
            text_encoder.print_trainable_parameters()

            unet_target_modules= ["to_q", "to_v", "query", "value"]
            unet=prepare_unet(unet,unet_target_modules=unet_target_modules)
        if training_method.find(IP)==-1:
            prior_image_mega_list=[row[initializer_token] for row in load_dataset(prior_dataset,split="train")]
        if training_method.find(IP)!=-1:
            try:
                prior_image_mega_list=load_dataset(prior_dataset+f"_{label}",split="train")[initializer_token]
            except:
                prior_image_mega_list=[
                    pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,ip_adapter_image=ip_adapter_image, num_inference_steps=timesteps_per_image).images[0] for _ in range(20)
                ]
    if training_method.find(CHOSEN)==-1 and training_method.find(CHOSEN_TEX_INV)==-1: #TODO everything but chosen should do this
        if training_method.find(IP)==-1:
            images=[
                pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image).images[0] for _ in range(n_image)
            ]
        else:
            images=[
                pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image,ip_adapter_image=ip_adapter_image).images[0] for _ in range(n_image)
            ]
    if training_method.find(UNET)!=-1: #TODO all uNet should do this except for reward
        unet=prepare_unet(unet)
        text_prompt_list=[NEW_TOKEN]*n_image
        validation_prompt_list=text_prompt_list
    if training_method.find(TEX_INV)!=-1 or training_method.find(CHOSEN)!=-1 or training_method.find(CHOSEN_TEX_INV)!=-1: #TODO all chosen,cte,and tex_inv should do this
        tokenizer,text_encoder=prepare_textual_inversion(description_prompt,tokenizer,text_encoder)
        entity_name=NEW_TOKEN
        text_prompt_list=[imagenet_template.format(entity_name) for imagenet_template in imagenet_template_list]
        random_text_prompt=True
        validation_prompt_list=[template.format(NEW_TOKEN) for template in imagenet_template_list]
    if training_method.find(CHOSEN_TEX_INV)!=-1: #TODD all cte should do this
        cluster_function=get_best_cluster_kmeans
    if training_method.find(COLD)!=-1 and training_method.find(CHOSEN)!=-1: #TODO all chosen_cold should do this (incl BASIC)
        cluster_text_prompt=cold_prompt
        cluster_function=get_best_cluster_sorted
    if training_method.find(HOT)!=-1 and training_method.find(CHOSEN)!=-1:  #TODO all chosen_hot should do this (incl BASIC)
        cluster_text_prompt=hot_prompt
        cluster_function=get_best_cluster_sorted
        negative=False
    for model in [vae,unet,text_encoder]:
        trainable_parameters+=[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    #try to reduce memory consumption
    #pipeline.enable_vae_tiling()
    #pipeline.enable_vae_slicing()
    #pipeline.enable_model_cpu_offload()

    unet,text_encoder,vae,tokenizer = accelerator.prepare(
        unet,text_encoder,vae,tokenizer
    )
    if use_ip_adapter:
        image_encoder=accelerator.prepare(pipeline.image_encoder)
        image_encoder.requires_grad_(False)

    print(f"acceleerate device {accelerator.device}")

    if use_chosen_one:
        n_generated_img=chosen_one_args["n_generated_img"] # how many images to generate and then cluster
        convergence_scale=chosen_one_args["convergence_scale"] #when cluster distance < convergence * init_dist then we stop
        min_cluster_size=chosen_one_args["min_cluster_size"] #ignore clusters smaller than this aka d_min_c
        max_iterations=chosen_one_args["max_iterations"] #how many loops before we give up
        #starting_cluster=chosen_one_args["starting_cluster"] #initial images
        target_cluster_size=chosen_one_args["target_cluster_size"] #aka dsize_c
        n_clusters=n_generated_img // target_cluster_size
        if use_ip_adapter:
            image_list=[
                pipeline(description_prompt,negative_prompt=negative_prompt,num_inference_steps=timesteps_per_image,safety_checker=None,ip_adapter_image=ip_adapter_image).images[0] for _ in range(n_generated_img)]
        else:
            image_list=[
                pipeline(description_prompt,negative_prompt=negative_prompt,num_inference_steps=timesteps_per_image,safety_checker=None).images[0] for _ in range(n_generated_img)]
        print("generated initial sets of images")
        last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
        print("last hidden staes")
        init_dist=get_init_dist(last_hidden_states)
        print("init_dist")
        pairwise_distances=init_dist
        iteration=0
        while pairwise_distances>=convergence_scale*init_dist and iteration<max_iterations:
            print("while loop")
            valid_image_list, pairwise_distances=cluster_function(image_list,
                                                                  n_clusters, 
                                                                  min_cluster_size,
                                                                  vit_processor,vit_model,
                                                                  cluster_text_prompt,
                                                                  retain_fraction,
                                                                  negative,clip_processor,clip_model)
            print(f"iteration {iteration} pairwise distances {pairwise_distances} vs target {convergence_scale*init_dist}")
            if not random_text_prompt:
                text_prompt_list=text_prompt_list*len(valid_image_list)
            pipeline=loop(
                images=valid_image_list,
                text_prompt_list=text_prompt_list,
                validation_prompt_list=validation_prompt_list,
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
            if use_ip_adapter:
                image_list=[pipeline(description_prompt,num_inference_steps=timesteps_per_image,num_images_per_prompt=1,safety_checker=None,ip_adapter_image=ip_adapter_image).images[0] for _ in range(n_generated_img)]
            else:
                image_list=[pipeline(description_prompt,num_inference_steps=timesteps_per_image,safety_checker=None,num_images_per_prompt=1).images[0] for _ in range(n_generated_img) ]
            iteration+=1
        del image_list
        del valid_image_list
    elif with_prior_preservation:
        for e in range(epochs):
            print("len(prior_image_mega_list)",len(prior_image_mega_list))
            print("n_image",n_image)
            prior_images=random.sample(prior_image_mega_list,n_image)
            pipeline=loop(
                images=images,
                text_prompt_list=text_prompt_list,
                validation_prompt_list=validation_prompt_list,
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
                max_grad_norm=max_grad_norm
            )
    else:
        pipeline=loop(
            images=images,
            text_prompt_list=text_prompt_list,
            validation_prompt_list=validation_prompt_list,
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

    end=time.time()
    seconds=end-start
    hours=seconds/3600
    print(f"{training_method} training elapsed {seconds} seconds == {hours} hours")
    result_dict= evaluate_pipeline(ip_adapter_image,description_prompt,entity_name,pipeline,timesteps_per_image,use_ip_adapter,negative_prompt, hot_prompt,clip_processor,clip_model)
    try:
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()
        print("cleared cache after eval!?!?")
    except:
        print("did not clear cache after eval")
    del pipeline, images,unet,vae,text_encoder,optimizer
    if use_ip_adapter:
        del image_encoder
    return result_dict