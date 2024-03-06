TEXT_INPUT_IDS="text_input_ids"
CLIP_IMAGES='clip_images'
IMAGES="images" #in text_to_image_lora this is aka pixel_values
PRIOR_IMAGES="prior_images"
PRIOR_TEXT_INPUT_IDS="prior_text_input_ids"
NEW_TOKEN="<xyz>" #this is the new token we will use for dreambooth/textual inversion
DB="dreambooth"
DB_MULTI="dreambooth_multi"
IP="ip_adapter"
UNET="unet_lora"
TEX_INV="textual_inversion"
CHOSEN_TEX_INV="chosen_one_textual_inversion"
CHOSEN_TEX_INV_IP="chosen_one_textual_inversion_facial_ip"
CHOSEN_DB="chosen_one_dreambooth"
CHOSEN_NEG="chosen_one_negative_prompt"
CHOSEN_TARGET="chosen_one_target_prompt"
LOL_SUFFIX=" in the style of league of legends"
NEGATIVE_PROMPT="blurry,text,low quality,logo,poorly drawn face,horror,mutation"

CHOSEN_SUITE="chosen_suite"
test_suite_dict={
    CHOSEN_SUITE:[CHOSEN_TARGET, CHOSEN_NEG, CHOSEN_TEX_INV]
}