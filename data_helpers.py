from torch.utils.data import DataLoader
from torchvision import transforms
from string_globals import *
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import torch
import random

class CustomDataset(Dataset):
    def __init__(self,mapping,random_text_prompt):
        self.mapping={k:v for k,v in mapping.items() if len(v)!=0}
        self.random_text_prompt=random_text_prompt

    def __len__(self):
        return len(self.mapping[IMAGES])
        
    def __getitem__(self,index):
        example={}
        for k,v in self.mapping.items():
            example[k]=v[index]
        if self.random_text_prompt:
            example[TEXT_INPUT_IDS]=random.choice(self.mapping[TEXT_INPUT_IDS])
        return example

def make_dataloader(images: list, text_prompt_list:list,prior_images:list, prior_text_prompt_list:list,tokenizer:object,size:int,train_batch_size:int, random_text_prompt:bool)->DataLoader:
    '''
    makes a torch dataloader that we can use for training
    '''
    img_transform=transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ])

    mapping={
        TEXT_INPUT_IDS:[], #tokenized texts
        CLIP_IMAGES:[], #images preprocessed by clip processor 
        IMAGES:[], #images used for latents (lora trainign script calls it pixel values)
        PRIOR_IMAGES:[],
        PRIOR_TEXT_INPUT_IDS:[]
    }
    clip_image_processor = CLIPImageProcessor()
    for image in  images:
        clip_image=clip_image_processor(images=image,return_tensors="pt").pixel_values
        mapping[CLIP_IMAGES].append(clip_image)
        mapping[IMAGES].append(img_transform(image.convert("RGB")))
    text_input_ids=tokenizer(
            text_prompt_list,max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
    mapping[TEXT_INPUT_IDS]=text_input_ids
    use_prior_preservation=False
    if len(prior_images)>0 and len(prior_text_prompt_list)>0:
        use_prior_preservation=True
        for prior_image in prior_images:
            mapping[PRIOR_IMAGES].append(img_transform(prior_image.convert("RGB")))
        prior_text_input_ids=tokenizer(
                prior_text_prompt_list,max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        mapping[PRIOR_TEXT_INPUT_IDS]=prior_text_input_ids

    def collate_fn(examples,use_prior_preservation=False):
        if use_prior_preservation:
            return {
                TEXT_INPUT_IDS: torch.stack([example[TEXT_INPUT_IDS] for example in examples]+[example[PRIOR_TEXT_INPUT_IDS] for example in examples]),
                IMAGES: torch.stack([example[IMAGES] for example in examples]+[example[PRIOR_IMAGES] for example in examples])
            }
        else:
            return {
                TEXT_INPUT_IDS: torch.stack([example[TEXT_INPUT_IDS] for example in examples]),
                IMAGES: torch.stack([example[IMAGES] for example in examples])
            }
    train_dataset=CustomDataset(mapping, random_text_prompt)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, use_prior_preservation),
        batch_size=train_batch_size,
    )
    return train_dataloader