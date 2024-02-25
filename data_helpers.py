from torch.utils.data import DataLoader
from torchvision import transforms
from string_globals import *
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self,mapping):
        self.mapping=mapping

    def __len__(self):
        for v in self.mapping.values():
            return len(v)
        
    def __getitem__(self,index):
        example={}
        for k,v in self.mapping.items():
            example[k]=v[index]
        return example

def make_dataloader(images: list, text_prompt_list:list,tokenizer:object,size:int,train_batch_size:int)->DataLoader:
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
        IMAGES:[] #images used for latents (lora trainign script calls it pixel values)
    }
    clip_image_processor = CLIPImageProcessor()
    for image,text_prompt in zip(images,text_prompt_list):
        clip_image=clip_image_processor(images=image,return_tensors="pt").pixel_values
        mapping[CLIP_IMAGES].append(clip_image)
        mapping[IMAGES].append(img_transform(image.convert("RGB")))
    text_input_ids=tokenizer(
            text_prompt_list,max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
    mapping[TEXT_INPUT_IDS]=text_input_ids

    def collate_fn(examples):
        return {
            TEXT_INPUT_IDS: torch.stack([example[TEXT_INPUT_IDS] for example in examples]),
            CLIP_IMAGES: torch.stack([example[CLIP_IMAGES] for example in examples]),
            IMAGES: torch.stack([example[IMAGES] for example in examples])
        }
    train_dataset=CustomDataset(mapping)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
    )
    return train_dataloader