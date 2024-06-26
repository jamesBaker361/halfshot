from sklearn.cluster import KMeans
import torchvision.transforms as T
import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from transformers import ViTImageProcessor, ViTModel,CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from aesthetic_reward import get_aesthetic_scorer
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"


#remove backgrounds and use faces?

#vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
#vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_init_dist(last_hidden_states)->float:
    n=len(last_hidden_states)
    total=(n*(n+1)/2)-n
    init_dist=0.0
    for i in range(n):
        for j in range(i+1,n):
            init_dist+= np.linalg.norm(last_hidden_states[i] - last_hidden_states[j])/total
    return init_dist

    

def get_hidden_states(image_list:list, vit_processor: ViTImageProcessor, vit_model:ViTModel):
    vit_inputs = vit_processor(images=image_list, return_tensors="pt")
    #print("inputs :)")
    vit_inputs['pixel_values']=vit_inputs['pixel_values'].to(vit_model.device)
    vit_outputs=vit_model(**vit_inputs)
    #print("outputs :))")
    last_hidden_states = vit_outputs.last_hidden_state.detach()
    #print("last hidden :)))")
    last_hidden_states=last_hidden_states.cpu().numpy().reshape(len(image_list),-1)
    return last_hidden_states

def get_best_cluster_kmeans(
        image_list:list,
                            n_clusters:int,
                            min_cluster_size:int,
                            vit_processor: ViTImageProcessor, 
                            vit_model:ViTModel,*args):
    print(f"best cluster kmeans len(image_list) {len(image_list)}")
    last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
    try:
        print('last_hidden_states[0].shape()',last_hidden_states[0].shape())
        print('last_hidden_states[1].shape()',last_hidden_states[1].shape())
    except:
        pass
    try:
        print('last_hidden_states.shape()',last_hidden_states.shape())
    except:
        pass
    try:
        print('len(last_hidden_states)',len(last_hidden_states))
    except:
        pass
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(last_hidden_states)
    
    cluster_dict={}
    for label,embedding in zip(k_means.labels_, last_hidden_states):
        if label not in cluster_dict:
            cluster_dict[label]=[]
        cluster_dict[label].append(embedding)
    cluster_dict={label:v for label,v in cluster_dict.items() if len(v)>=min_cluster_size}

    dist_dict={}
    for label,v in cluster_dict.items():
        center=k_means.cluster_centers_[label]
        dist=np.mean([np.linalg.norm(center-embedding) for embedding in v])
        dist_dict[label]=dist
    print("dist_dict",dist_dict)
    min_label=[label for label in dist_dict.keys()][0]
    for label,dist in dist_dict.items():
        if dist<=dist_dict[min_label]:
            min_label=label
    valid_image_list=[]
    for label,image in zip(k_means.labels_,  image_list):
        if label==min_label:
            valid_image_list.append(image)
    return valid_image_list, dist_dict[min_label]

def get_ranked_images_list(image_list:list, text_prompt:str,clip_processor:CLIPProcessor, clip_model:CLIPModel)->list:
    clip_inputs=clip_processor(text=[text_prompt], images=image_list, return_tensors="pt", padding=True)
    clip_outputs = clip_model(**clip_inputs)
    logits_per_image=clip_outputs.logits_per_image.detach().numpy()
    try:
        print('clip_outputs.logits_per_image.detach().numpy().shape()',clip_outputs.logits_per_image.detach().numpy().shape())
    except:
        pass
    try:
        print('len(clip_outputs.logits_per_image.detach().numpy())',len(clip_outputs.logits_per_image.detach().numpy()))
    except:
        pass
    try:
        print('logits_per_image.shape()',logits_per_image.shape())
    except:
        pass
    try:
        print('len(logits_per_image)',len(logits_per_image))
    except:
        pass
    try:
        print('logits_per_image[0].shape()',logits_per_image[0].shape())
    except:
        pass
    try:
        print('len(logits_per_image[0])',len(logits_per_image[0]))
    except:
        pass
    pair_list=[ (logit,image) for logit,image in zip(logits_per_image, image_list)]
    pair_list.sort(key=lambda x: x[0])
    print(f"len(pair_list), {len(pair_list)}")
    print([logit for (logit,image) in pair_list])
    return [image for (logit,image) in pair_list]


def get_best_cluster_sorted(
        image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        negative:bool,clip_processor:CLIPProcessor, clip_model:CLIPModel,):
    ranked_image_list=get_ranked_images_list(image_list, text_prompt,clip_processor,clip_model)
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    if negative:
        ranked_image_list=ranked_image_list[:limit]
    else:
        ranked_image_list=ranked_image_list[-limit:]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)

def get_best_cluster_aesthetic(
        image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        *args):
    aesthetic_scorer=get_aesthetic_scorer()
    scored_ranked_image_list=[[ aesthetic_scorer(image).cpu().numpy()[0],image   ] for image in image_list]
    scored_ranked_image_list.sort(reverse=True, key=lambda x: x[0])
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    ranked_image_list=[image for [score,image] in scored_ranked_image_list][:limit]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)

def get_best_cluster_ir(
       image_list:list,
        n_clusters:int,
        min_cluster_size:int,
        vit_processor: ViTImageProcessor, vit_model:ViTModel,
        text_prompt:str,
        retain_fraction:float,
        *args):
    ir_model=image_reward.load("ImageReward-v1.0",download_root=reward_cache)
    scored_ranked_image_list=[
        [ir_model.score(text_prompt, image),image] for image in image_list
    ]
    scored_ranked_image_list.sort(reverse=True, key=lambda x: x[0])
    limit=int(len(image_list) * retain_fraction)
    print(f"len(image_list) {len(image_list)} vs limit {limit}")
    ranked_image_list=[image for [score,image] in scored_ranked_image_list][:limit]
    return get_best_cluster_kmeans(ranked_image_list,n_clusters, min_cluster_size,vit_processor,vit_model)