from diffusers import AutoPipelineForText2Image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from data_helpers import make_dataloader

def train_loop(images: list,
               text_prompt:str,
               src_image: Image,
               pipeline:AutoPipelineForText2Image,
               epochs:int,
               optimizer:object,
               accelerator:object,
               size:int,train_batch_size:int
               )->AutoPipelineForText2Image:
    '''
    given images generated from text prompt, and the src_image, trains the pipeline for epochs
    using the prompt and the src_image for conditioning and returns the trained pipeline
    '''
    dataloader=make_dataloader(images,text_prompt,)
    return None