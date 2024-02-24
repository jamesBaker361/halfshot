from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline

def get_trained_pipeline(
        trainable_unet:bool,
        trainable_vae:bool,
        trainable_text_encoder:bool,
        trainable_tokenizer:bool,
        trainable_ip_adapter: bool,
        chosen_one:bool,
)->StableDiffusionPipeline:
    pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    loop_kwargs={}
    return None

def evaluate_pipeline(image,text_prompt,pipeline)->dict:
    return
    

def train_and_evaluate(image,text_prompt, training_method)->dict:
    if training_method=="dreambooth":
        return
    elif training_method=="ip_adapter":
        return
    elif training_method=="lora":
        return
    elif training_method=="textual_inversion":
        return