from diffusers import AutoPipelineForText2Image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

def make_dataloader(images: list, text_prompt:str,tokenizer:object)->DataLoader:
    '''
    makes a torch dataloader that we can use for training
    '''
    src_dict={
        "text_input_ids":[]
        ""
    }
    for image in images:
        text_input_ids=tokenizer(
            [text_prompt for _ in images],max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    return None

def train_loop(images: list,
               text_prompt:str,
               src_image: Image,
               pipeline:AutoPipelineForText2Image,
               epochs:int,
               optimizer:object,
               accelerator:object)->AutoPipelineForText2Image:
    '''
    given images generated from text prompt, and the src_image, trains the pipeline for epochs
    using the prompt and the src_image for conditioning and returns the trained pipeline
    '''
    return None