from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from data_helpers import make_dataloader

def train_loop(images: list,
               text_prompt:str,
               src_image: Image,
               pipeline:StableDiffusionPipeline,
               epochs:int,
               optimizer:object,
               lr_scheduler: object,
               accelerator:object,
               size:int,train_batch_size:int
               )->StableDiffusionPipeline:
    '''
    given images generated from text prompt, and the src_image, trains the pipeline for epochs
    using the prompt and the src_image for conditioning and returns the trained pipeline
    
    images: PIL imgs to be used for training
    text_prompt: text prompt describing character
    pipeline: should already have ip adapter loaded
    epochs: how many epochs to do this training
    optimizer: optimizer
    acceelrator: accelerator object
    size: img dim 
    train_batch_size: batch size
    '''
    tokenizer=pipeline.tokenizer
    dataloader=make_dataloader(images,text_prompt,tokenizer,size, train_batch_size)
    unet=tokenizer.unet
    #lora_layers = filter(lambda p: p.requires_grad, unet.parameters()) optimizer should already be listening to lora_layers
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    return pipeline