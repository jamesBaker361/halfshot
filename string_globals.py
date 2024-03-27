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
CHOSEN_TEX_INV="cte"
CHOSEN_BASIC=f"{CHOSEN}_{BASIC}"

CHOSEN_BASIC_COLD=f"{CHOSEN_BASIC}_{COLD}"
CHOSEN_BASIC_HOT=f"{CHOSEN_BASIC}_{HOT}"
CHOSEN_BASIC_COLD_IP=f"{CHOSEN_BASIC}_{COLD}_{IP}"
CHOSEN_BASIC_HOT_IP=f"{CHOSEN_BASIC}_{HOT}_{IP}"
CHOSEN_BASIC_REWARD=f"{CHOSEN_BASIC}_{REWARD}"
CHOSEN_BASIC_REWARD_IP=f"{CHOSEN_BASIC}_{REWARD}_{IP}"

CHOSEN_COLD=f"{CHOSEN}_{COLD}"
CHOSEN_HOT=f"{CHOSEN}_{HOT}"
CHOSEN_COLD_IP=f"{CHOSEN}_{COLD}_{IP}"
CHOSEN_HOT_IP=f"{CHOSEN}_{HOT}_{IP}"
CHOSEN_REWARD=f"{CHOSEN}_{REWARD}"
CHOSEN_REWARD_IP=f"{CHOSEN}_{REWARD}_{IP}"

DB_MULTI_REWARD=f"{DB_MULTI}_{REWARD}"
DB_MULTI_REWARD_IP=f"{DB_MULTI}_{REWARD}_{IP}"
DB_MULTI_COLD=f"{DB_MULTI}_{COLD}"
DB_MULTI_COLD_IP=f"{DB_MULTI}_{COLD}_{IP}"
DB_MULTI_HOT=f"{DB_MULTI}_{HOT}"
DB_MULTI_HOT_IP=f"{DB_MULTI}_{HOT}_{IP}"


TEX_INV_REWARD=f"{TEX_INV}_{REWARD}"
TEX_INV_REWARD_IP=f"{TEX_INV}_{REWARD}_{IP}"
TEX_INV_COLD=f"{TEX_INV}_{COLD}"
TEX_INV_COLD_IP=f"{TEX_INV}_{COLD}_{IP}"
TEX_INV_HOT=f"{TEX_INV}_{HOT}"
TEX_INV_HOT_IP=f"{TEX_INV}_{HOT}_{IP}"

UNET_REWARD=f"{UNET}_{REWARD}"
UNET_HOT=f"{UNET}_{HOT}"
UNET_COLD=f"{UNET}_{COLD}"
UNET_REWARD_IP=f"{UNET}_{REWARD}_{IP}"
UNET_HOT_IP=f"{UNET}_{HOT}_{IP}"
UNET_COLD_IP=f"{UNET}_{COLD}_{IP}"

CHOSEN_TEX_INV_COLD=f"{CHOSEN_TEX_INV}_{COLD}"
CHOSEN_TEX_INV_HOT=f"{CHOSEN_TEX_INV}_{HOT}"
CHOSEN_TEX_INV_COLD_IP=f"{CHOSEN_TEX_INV}_{COLD}_{IP}"
CHOSEN_TEX_INV_HOT_IP=f"{CHOSEN_TEX_INV}_{HOT}_{IP}"
CHOSEN_TEX_INV_REWARD=f"{CHOSEN_TEX_INV}_{REWARD}"
CHOSEN_TEX_INV_REWARD_IP=f"{CHOSEN_TEX_INV}_{REWARD}_{IP}"


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
    CHOSEN_HOT_IP:[CHOSEN_HOT_IP],
    CHOSEN_COLD_IP: [CHOSEN_COLD_IP],
    LIGHT_SUITE: [TEX_INV, UNET]
}

TOKEN_LIST=[ " man "," woman "," boy "," girl "," male "," female "]
metric_list=["prompt_similarity","identity_consistency","negative_prompt_similarity","target_prompt_similarity","aesthetic_score"]