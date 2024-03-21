TEXT_INPUT_IDS="text_input_ids"
CLIP_IMAGES='clip_images'
IMAGES="images" #in text_to_image_lora this is aka pixel_values
PRIOR_IMAGES="prior_images"
PRIOR_TEXT_INPUT_IDS="prior_text_input_ids"
NEW_TOKEN="<xyz>" #this is the new token we will use for dreambooth/textual inversion
DB="db"
DB_MULTI="db_multi"
DB_MULTI_IP="db_multi_ip"
IP="ip"
CHOSEN="chosen"
UNET="unet"
UNET_IP="unet_ip"
TEX_INV="tex_inv"
TEX_INV_IP="tex_inv_ip"
BASIC="basic"
HOT="hot"
COLD="cold"
REWARD="reward"
CHOSEN_TEX_INV="chosen_tex_inv"
CHOSEN_TEX_INV_IP="chosen_tex_inv_ip"
CHOSEN_DB="chosen_db"
CHOSEN_COLD="chosen_neg"
CHOSEN_HOT="chosen_target"
CHOSEN_COLD_IP="chosen_neg_ip"
CHOSEN_HOT_IP="chosen_target_ip"
REWARD="reward"
CHOSEN_REWARD="chosen_reward"
CHOSEN_REWARD_IP="chosen_reward_ip"
DB_MULTI_REWARD="db_multi_reward"
DB_MULTI_IP_REWARD="db_multi_ip_reward"
TEX_INV_REWARD="tex_inv_reward"
TEX_INV_IP_REWARD="tex_inv_ip_reward"
UNET_REWARD="unet_reward"
UNET_IP_REWARD="unet_ip_reward"


LOL_SUFFIX=" in the style of league of legends"
NEGATIVE_PROMPT="blurry,text,low quality,logo,poorly drawn face,horror,mutation"

CHOSEN_SUITE="chosen_suite"
LIGHT_SUITE="light_suite"
NEGATIVE_SUITE="negative_suite"
BASIC_SUITE="basic_suite"
BASIC_SUITE_IP="basic_suite_ip"
REWARD_SUITE="reward_suite"
REWARD_SUITE_IP="reward_suite_ip"
training_method_suite_dict={
    CHOSEN_SUITE:[CHOSEN_HOT, CHOSEN_COLD, CHOSEN_TEX_INV],
    CHOSEN_HOT: [CHOSEN_HOT],
    CHOSEN_COLD: [CHOSEN_COLD],
    CHOSEN_TEX_INV: [CHOSEN_TEX_INV],
    BASIC_SUITE:[DB_MULTI, UNET, TEX_INV],
    BASIC_SUITE_IP: [DB_MULTI_IP, UNET_IP, TEX_INV_IP],
    REWARD_SUITE:[DB_MULTI_REWARD, UNET_REWARD, TEX_INV_REWARD],
    REWARD_SUITE_IP: [DB_MULTI_IP_REWARD, UNET_IP_REWARD, TEX_INV_IP_REWARD],
    CHOSEN_HOT_IP:[CHOSEN_HOT_IP],
    CHOSEN_COLD_IP: [CHOSEN_COLD_IP],
    CHOSEN_TEX_INV_IP: [CHOSEN_TEX_INV_IP],
    LIGHT_SUITE: [TEX_INV, UNET]
}