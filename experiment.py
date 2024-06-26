
import os
import torch
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
from clustering import get_hidden_states,get_best_cluster_kmeans,get_best_cluster_sorted,get_init_dist,get_best_cluster_aesthetic
import time
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from safetensors import safe_open
from aesthetic_reward import get_aesthetic_scorer
import gc
from datasets import load_dataset
import random
import ImageReward as image_reward
import string
def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))
reward_cache="/scratch/jlb638/reward_symbolic/"+generate_random_string(10)
from align_prop_src import align_generate

def prepare_unet(unet,unet_target_modules,adapter_name,lora_alpha):
    config = LoraConfig(
        r=4,
        lora_alpha=lora_alpha,
        target_modules=unet_target_modules,
        lora_dropout=0.0,
        bias="none")
    unet = get_peft_model(unet, config,adapter_name=adapter_name)
    unet.train()
    print("prepare unet trainable parameters")
    unet.print_trainable_parameters()
    return unet

def prepare_unet_from_path(unet,weight_path:str,unet_target_modules=["to_k", "to_q", "to_v", "to_out.0"],lora_alpha=4):
    unet=prepare_unet(unet,unet_target_modules,"default",lora_alpha=lora_alpha)
    state_dict={}
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key]=f.get_tensor(key)
    state_dict={
        "base_model.model."+k.replace("weight","default.weight"):v for k,v in state_dict.items()
    }
    unet.load_state_dict(state_dict,strict=False)
    unet.requires_grad_(False)
    unet.print_trainable_parameters()
    return unet

def get_initializer_token(text_prompt:str)->str:
    initializer_token=""
    for token in [ " man "," woman "," boy "," girl "]:
        if text_prompt.find(token)!=-1:
            initializer_token=token
    if initializer_token=="":
        if text_prompt.find(" female ")!=-1:
            initializer_token="woman"
        elif text_prompt.find(" male ")!=-1:
            initializer_token="man"
        else:
            initializer_token="person"
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
    try:
        ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")
    except FileNotFoundError:
        new_cache=reward_cache+"1"
        os.makedirs(new_cache,exist_ok=True)
        ir_model=image_reward.load("ImageReward-v1.0",download_root=new_cache)
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
    text_list=[negative_prompt, target_prompt] + evaluation_prompt_list
    clip_inputs=clip_processor(text=text_list, images=evaluation_image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    
    negative_text_embeds=outputs.text_embeds.detach().numpy()[0]
    target_text_embeds=outputs.text_embeds.detach().numpy()[1]
    text_embeds_list=outputs.text_embeds.detach().numpy()[2:]
    image_embed_list=outputs.image_embeds.detach().numpy()
    prompt_similarity_list=[]
    negative_prompt_similarity_list=[]
    target_prompt_similarity_list=[]
    aesthetic_score_list=[]
    ir_prompt_list=[evaluation_prompt.format(text_prompt) for evaluation_prompt in evaluation_prompt_list ]
    print(ir_prompt_list)
    ir_score_list=[]
    for image,prompt in zip(evaluation_image_list,ir_prompt_list):
        print(prompt)
        ir_score_list.append(ir_model.score(prompt,image))
    #ir_score_list=[ir_model.score(image, prompt) for image,prompt in zip(evaluation_image_list,ir_prompt_list)]

    for image in evaluation_image_list:
        aesthetic_score_list.append(aesthetic_scorer(image).cpu().numpy()[0])

    identity_consistency_list=[]
    for i in range(len(image_embed_list)):
        vector_i=image_embed_list[i]
        text_embeds=text_embeds_list[i]
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
              "target_prompt_similarity": np.mean(target_prompt_similarity_list),
               "ir_score":np.mean(ir_score_list) }
    
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
                        subfolder:str,
                        text_encoder_target_modules: list,
                        train_embeddings:bool,
                        align_prop:bool,
                        align_from_prompt:bool
                       )->dict:
    """
    init_image_list= the images we are starting with
    of this image and train on the same image a bunch
    text_prompt= the text prompt describing the character or whatever for ex: "a male character with a sword holding a sword and wearing a blue and black outfit"
    prior_text_prompt= for dreambooth this is the prior, (should be Man, woman, boy, girl or person)
    prior_images = images of prior text prompt
    """
    def get_prior_image_mega_list(initializer_token:str):
        prior_dataset="jlbaker361/prior-"
        for flavor in [COLD,HOT,REWARD]:
            if training_method.find(flavor)!=-1:
                prior_dataset+=flavor
                break
        if timesteps_per_image==50:
            prior_dataset+="-50"
        print("prior_dataset",prior_dataset)
        if training_method.find(IP)==-1:
            #prior_image_mega_list=[row[initializer_token] for row in load_dataset(prior_dataset,split="train")]
            prior_image_mega_list=[
                    pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image).images[0] for _ in range(20)
            ]
        else:
            try:
                prior_image_mega_list=load_dataset(prior_dataset+f"_{label}",split="train")[initializer_token]
            except:
                prior_image_mega_list=[
                    pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,ip_adapter_image=ip_adapter_image, num_inference_steps=timesteps_per_image).images[0] for _ in range(20)
                ]
        print("len(prior_image_mega_list)",len(prior_image_mega_list))
        return prior_image_mega_list
    print(f"training method {training_method}")
    start=time.time()
    for flavor in [COLD,HOT,REWARD]:
        if training_method.find(flavor)!=-1:
            prior_dataset+=flavor
            break
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
    pipeline("nothing",num_inference_steps=1,safety_checker=None) #if we dont do this the properties wont instantiate correctly???
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
    initializer_token=get_initializer_token(description_prompt)
    if training_method.find(BASIC)==-1:
        if training_method.find(COLD)!=-1:
            negative_prompt=cold_prompt
        if training_method.find(HOT) !=-1:
            description_prompt+=hot_prompt
        if training_method.find(REWARD)!=-1:
            weight_path=hf_hub_download(repo_id=pretrained_lora_path,subfolder=subfolder,filename="pytorch_lora_weights.safetensors", repo_type="model")
            trainable_modules=["to_k", "to_q", "to_v", "to_out.0"]
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
        #text_encoder_target_modules=args.text_encoder_target_modules #"k_proj","out_proj"]
        if len(text_encoder_target_modules)>0:
            text_encoder_config=LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=text_encoder_target_modules,
                lora_dropout=0.0
            )
            text_encoder=get_peft_model(text_encoder,text_encoder_config,adapter_name="trainable")
            text_encoder.train()
            print("text encoder parameters")
        text_encoder.get_input_embeddings().requires_grad_(train_embeddings)
        if len(text_encoder_target_modules)>0:
            print("text encoder parameters with iput embeddings?")
            text_encoder.print_trainable_parameters()
        prepare_unet(unet, ["to_k", "to_q", "to_v", "to_out.0"],"trainable",16)
        use_chosen_one=True
        entity_name=description_prompt
        validation_prompt_list=[template.format(entity_name) for template in imagenet_template_list]
        if training_method.find(PRIOR)!=-1:
            with_prior_preservation=True
            prior_image_mega_list=get_prior_image_mega_list(initializer_token)
    if training_method.find(CHOSEN)!=-1:
        n_generated_img=int(chosen_one_args["n_generated_img"]/retain_fraction)
    if training_method.find(CHOSEN_TEX_INV)!=-1:
        n_generated_img=chosen_one_args["n_generated_img"]
    if training_method.find(DB_MULTI)!=-1:
        with_prior_preservation=True
        text_prompt_list=[NEW_TOKEN]*n_image
        entity_name=NEW_TOKEN +" "+ initializer_token
        validation_prompt_list=[template.format(entity_name) for template in imagenet_template_list]
        text_encoder_target_modules=["q_proj", "v_proj"]
        text_encoder_config=LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=text_encoder_target_modules,
            lora_dropout=0.0
        )
        text_encoder=get_peft_model(text_encoder,text_encoder_config,adapter_name="trainable")
        text_encoder.train()
        print("text encoder parameters")
        text_encoder.print_trainable_parameters()

        unet_target_modules= ["to_q", "to_v", "query", "value"]
        unet=prepare_unet(unet,unet_target_modules=unet_target_modules,adapter_name="trainable",lora_alpha=16)
        prior_text_prompt_list=[initializer_token]*n_image
        prior_image_mega_list=get_prior_image_mega_list(initializer_token)
    if training_method.find(CHOSEN)==-1 and training_method.find(CHOSEN_TEX_INV)==-1: #TODO everything but chosen should do this
        if align_prop:
            if align_from_prompt:
                images=[
                    align_generate(accelerator,
                    pipeline,
                    description_prompt,
                    PROMPT,
                    True,
                    5,
                    timesteps_per_image-5,
                    0,
                    7.5,
                    optimizer,
                    hot_prompt,
                    NEGATIVE_PROMPT) for _ in range(n_image)
                ]
            else:
                images=[
                    align_generate(
                        accelerator,
                    pipeline,
                    description_prompt,
                    flavor,
                    True,
                    5,
                    timesteps_per_image-5,
                    0,
                    7.5,
                    optimizer,
                    hot_prompt,
                    negative_prompt) for _ in range(n_generated_img)
                ]
        else:
            images=[
                pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image).images[0] for _ in range(n_image)
            ]
    if training_method.find(UNET)!=-1 or training_method.find(ADAPTER)!=-1: #TODO all uNet should do this except for reward
        unet=prepare_unet(unet,unet_target_modules=["to_k", "to_q", "to_v", "to_out.0"],adapter_name="trainable",lora_alpha=16)
        text_prompt_list=[NEW_TOKEN]*n_image
        validation_prompt_list=[template.format(entity_name) for template in imagenet_template_list]
        if training_method.find(ADAPTER)!=-1:
            epochs=0
    if training_method.find(TEX_INV)!=-1: # or training_method.find(CHOSEN)!=-1 or training_method.find(CHOSEN_TEX_INV)!=-1: #TODO all chosen,cte,and tex_inv should do this
        tokenizer,text_encoder=prepare_textual_inversion(description_prompt,tokenizer,text_encoder)
        entity_name=NEW_TOKEN
        text_prompt_list=[imagenet_template.format(entity_name) for imagenet_template in imagenet_template_list]
        random_text_prompt=True
        validation_prompt_list=[template.format(NEW_TOKEN) for template in imagenet_template_list]
    if training_method.find(CHOSEN_TEX_INV)!=-1: #TODD all cte should do this
        cluster_function=get_best_cluster_kmeans
    if training_method.find(CHOSEN)!=-1:
        if training_method.find(COLD)!=-1: #TODO all chosen_cold should do this (incl BASIC)
            cluster_text_prompt=cold_prompt
            cluster_function=get_best_cluster_sorted
        if training_method.find(HOT)!=-1:  #TODO all chosen_hot should do this (incl BASIC)
            cluster_text_prompt=hot_prompt
            cluster_function=get_best_cluster_sorted
            negative=False
        if training_method.find(REWARD)!=-1:
            cluster_function=get_best_cluster_aesthetic
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
        #n_generated_img=chosen_one_args["n_generated_img"] # how many images to generate and then cluster
        convergence_scale=chosen_one_args["convergence_scale"] #when cluster distance < convergence * init_dist then we stop
        min_cluster_size=chosen_one_args["min_cluster_size"] #ignore clusters smaller than this aka d_min_c
        max_iterations=chosen_one_args["max_iterations"] #how many loops before we give up
        #starting_cluster=chosen_one_args["starting_cluster"] #initial images
        target_cluster_size=chosen_one_args["target_cluster_size"] #aka dsize_c
        n_clusters=n_generated_img // target_cluster_size
        if align_prop:
            if align_from_prompt:
                images=[
                    align_generate(
                        accelerator,
                    pipeline,
                    description_prompt,
                    PROMPT,
                    True,
                    5,
                    timesteps_per_image-5,
                    0,
                    7.5,
                    optimizer,
                    hot_prompt,
                    negative_prompt) for _ in range(n_generated_img)
                ]
            else:
                images=[
                    align_generate(
                        accelerator,
                    pipeline,
                    description_prompt,
                    flavor,
                    True,
                    5,
                    timesteps_per_image-5,
                    0,
                    7.5,
                    optimizer,
                    hot_prompt,
                    negative_prompt) for _ in range(n_generated_img)
                ]
        else:
            images=[
                pipeline(description_prompt,negative_prompt=negative_prompt,safety_checker=None,num_inference_steps=timesteps_per_image).images[0] for _ in range(n_generated_img)
            ]
        print("generated initial sets of images")
        last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
        init_dist=get_init_dist(last_hidden_states)
        pairwise_distances=init_dist
        iteration=0
        while pairwise_distances>=convergence_scale*init_dist and iteration<max_iterations:
            print(f"while loop len(image_list)={len(image_list)} n_genertaed_img={n_generated_img}")
            valid_image_list, centroid_distances=cluster_function(image_list,
                                                                  n_clusters, 
                                                                  min_cluster_size,
                                                                  vit_processor,vit_model,
                                                                  cluster_text_prompt,
                                                                  retain_fraction,
                                                                  negative,clip_processor,clip_model)
            print(f"iteration {iteration} centroid distances {centroid_distances}")
            print(f"len(valid_image_list) {len(valid_image_list)}")
            text_prompt_list=[description_prompt]*len(valid_image_list)
            if with_prior_preservation:
                prior_text_prompt_list=[initializer_token]*len(valid_image_list)
                prior_images=random.sample(prior_image_mega_list,len(valid_image_list))
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
            last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
            init_dist=get_init_dist(last_hidden_states)
            pairwise_distances=init_dist
            iteration+=1
            print(f"iteration {iteration} pairwise distances {pairwise_distances} vs {convergence_scale*init_dist}")

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
    del pipeline, images,unet,vae,text_encoder,optimizer, clip_model, clip_processor,vit_processor, vit_model
    if use_ip_adapter:
        del image_encoder
    return result_dict