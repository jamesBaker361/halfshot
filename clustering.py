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
from transformers import ViTImageProcessor, ViTModel
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

#remove backgrounds and use faces?

def get_hidden_states(image_list:list):
    vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
    vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
    vit_inputs = vit_processor(images=image_list, return_tensors="pt")
    vit_outputs=vit_model(**vit_inputs)
    last_hidden_states = vit_outputs.last_hidden_state.detach().numpy().reshape(len(image_list),-1)
    return last_hidden_states

def get_best_cluster_kmeans(
        image_list:list,
                            n_clusters:int,
                            min_cluster_size:int):
    last_hidden_states=get_hidden_states(image_list)
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

    min_label=[label for label in dist_dict.keys()][0]
    for label,dist in dist_dict.items():
        if dist<=dist_dict[min_label]:
            min_label=label
    valid_image_list=[]
    for label,image in zip(k_means.labels_,  image_list):
        if label==min_label:
            valid_image_list.append(image)
    return valid_image_list, dist_dict[min_label]