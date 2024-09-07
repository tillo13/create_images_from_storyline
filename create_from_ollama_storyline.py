import os
import json
import logging
import time
import random
import re
from datetime import datetime
import torch
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download
import faceid_utils
from enhance_image_via_import import enhance_image

# Configuration
logging.basicConfig(level=logging.DEBUG)

# Directories
INCOMING_IMAGES_PATH = "incoming_images"
AI_GENERATED_USERS_PATH = os.path.join(INCOMING_IMAGES_PATH, "ai_generated_users")
GENERATED_IMAGES_PATH = "generated_images"
ENHANCED_IMAGES_PATH = "enhanced_images"
STORYLINES_PATH = os.path.join("..", "ollama_video", "storylines")

# Constants
NUMBER_OF_LOOPS = 1
SEED = 1060
CFG_SCALE = 1.13
NUMBER_OF_STEPS = 20
RANDOMIZE_SEED_VALUE = False

# Models
TOP_MODELS = [
    "DucHaiten/DucHaitenDreamWorld",
    "stablediffusionapi/disney-pixal-cartoon",
    "Hemlok/RainierMix",
    "digiplay/RunDiffusionFXPhotorealistic_v1",
    "luongphamit/NeverEnding-Dream2",
    "digiplay/RealCartoon3D_F16full_v3.1"
]

# Download the IP-Adapter checkpoint if it doesn't exist
v2 = True
ip_ckpt_filename = "ip-adapter-faceid-plusv2_sd15.bin" if v2 else "ip-adapter-faceid-plus_sd15.bin"

# Ensure directories exist
os.makedirs(AI_GENERATED_USERS_PATH, exist_ok=True)
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)
os.makedirs(ENHANCED_IMAGES_PATH, exist_ok=True)

# Ensure required files are downloaded
faceid_utils.download_required_files()

# Find the latest storyline file ending with _10_word_chapter_summaries.json
latest_file = max(
    [f for f in os.listdir(STORYLINES_PATH) if f.endswith('_10_word_chapter_summaries.json')],
    key=lambda f: os.path.getctime(os.path.join(STORYLINES_PATH, f))
)
storyline_path = os.path.join(STORYLINES_PATH, latest_file)

# Load storyline from the latest storyline file
with open(storyline_path, 'r') as f:
    storyline_data = json.load(f)
main_character = storyline_data['main_character']
story_chapters = storyline_data['story_chapters']

# Download the IP-Adapter checkpoint
if not os.path.exists(ip_ckpt_filename):
    logging.info(f"IP-Adapter checkpoint not found. Downloading...")
    ip_ckpt_filename = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename=ip_ckpt_filename)
logging.info(f"IP-Adapter checkpoint available at: {ip_ckpt_filename}")

# Select a model randomly from the top models
selected_model = random.choice(TOP_MODELS)
logging.info(f"Selected model: {selected_model}")

def load_pipeline(model_id):
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        pipeline.safety_checker = None  # Ensure the safety checker is disabled
        logging.info(f"Pipeline VAE model: {pipeline.vae}")
        return pipeline
    except Exception as e:
        logging.error(f"Failed to load model {model_id}: {e}")
        return None

pipeline = load_pipeline(selected_model)
if not pipeline:
    logging.error("Failed to load any model. Exiting.")
    exit(1)

def generate_image(prompt, negative_prompt, seed, width=768, height=1024, pipeline=None):
    generator = torch.manual_seed(seed)
    try:
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        return image
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None

def adjust_aspect_ratio(image, target_width, target_height):
    original_width, original_height = image.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:  # Adjust width
        new_width = int(target_aspect_ratio * original_height)
        left = (original_width - new_width) // 2 
        right = left + new_width
        top = 0
        bottom = original_height
    else:  # Adjust height
        new_height = int(original_width / target_aspect_ratio)
        top = (original_height - new_height) // 2 
        bottom = top + new_height
        left = 0
        right = original_width

    return image.crop((left, top, right, bottom))

def create_initial_character_image(activity, seed, positive_prompt, negative_prompt):
    character_description = main_character.split(": ")[1]  # Extract the character description
    prompt = f"portrait of {character_description}. Highly detailed, solo, cinematic lighting, professional photography, 8K resolution, {positive_prompt}"
    logging.info(f"Generated prompt: {prompt}")

    image = generate_image(prompt, negative_prompt, seed, pipeline=pipeline)
    if image is None:
        raise ValueError("Failed to generate initial image.")

    return image

def save_image(image, path):
    logging.debug(f"Saving generated image to {path}")
    try:
        image.save(path)
    except Exception as e:
        logging.error(f"Error saving image to {path}: {e}")

def extract_embeddings(image_path):
    logging.info(f"Extracting face embedding from {image_path}")
    return faceid_utils.extract_face_embedding(image_path)

def sanitize_filename(filename):
    sanitized = re.sub(r'\W+', '_', filename)
    return sanitized[:15]

def process_images(loop, timestamp, default_embedding, default_aligned_face, model_name, chapter_prompts):
    global total_images_generated, total_generation_time

    for idx, chapter in enumerate(chapter_prompts):
        image_start_time = time.time()

        positive_prompt = "full-body shot, wide-angle, sfw, " + chapter['positive_ai_prompt']
        negative_prompt = "portrait, inactive, closeup, nsfw, " + chapter['negative_ai_prompt']
        
        logging.info(f"Using positive prompt: {positive_prompt}")
        logging.info(f"Using negative prompt: {negative_prompt}")

        if RANDOMIZE_SEED_VALUE:
            seed = random.randint(0, 100000)
            logging.info(f"Generating image with random seed {seed}.")
        else:
            seed = SEED + loop + idx
            logging.info(f"Generating image with fixed seed {seed}.")

        try:
            images = ip_model.generate(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                faceid_embeds=default_embedding,
                face_image=default_aligned_face,
                shortcut=v2,
                s_scale=CFG_SCALE,
                num_samples=1,
                width=768,
                height=1024,
                num_inference_steps=NUMBER_OF_STEPS,
                seed=seed
            )
            if images is None or len(images) == 0:
                raise ValueError("Generated image is None or empty.")
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            continue

        adjusted_image = images[0]
        if adjusted_image:
            try:
                adjusted_image = adjust_aspect_ratio(adjusted_image, 1024, 1024)
            except Exception as e:
                logging.error(f"Error adjusting aspect ratio: {e}")
                continue
        else:
            logging.error("No image generated to adjust aspect ratio.")
            continue

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        sanitized_model_name = model_name.replace("/", "_")[:15]
        sanitized_activity = sanitize_filename(chapter['chapter'])
        result_image_path = f"{GENERATED_IMAGES_PATH}/{timestamp_str}_{sanitized_model_name}_{sanitized_activity}_{seed}.png"

        save_image(adjusted_image, result_image_path)
        logging.info(f"Generated image saved at {result_image_path}")

        try:
            enhanced_image_result = enhance_image(result_image_path, ENHANCED_IMAGES_PATH)
            enhanced_image_result_path = os.path.join(ENHANCED_IMAGES_PATH, f"{sanitized_model_name}_enhanced_{sanitized_activity}_{timestamp_str}.png")
            enhanced_image_result.save(enhanced_image_result_path)
            logging.info(f"Enhanced image saved to {enhanced_image_result_path}")
        except Exception as e:
            logging.error(f"Error enhancing image: {result_image_path}, error: {e}")

        total_images_generated += 1
        elapsed_time = time.time() - image_start_time
        total_generation_time += elapsed_time

        average_time_per_image = total_generation_time / total_images_generated
        estimated_time_remaining = average_time_per_image * (NUMBER_OF_LOOPS * len(chapter_prompts) - total_images_generated)

        logging.info(f"Elapsed time for current image: {elapsed_time:.2f} seconds")
        logging.info(f"Total images generated: {total_images_generated}")
        logging.info(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")

# Extract the first chapter's positive and negative prompts for initial character image
initial_prompt_info = story_chapters[0]

# Generate initial character image using main_character details and first chapter's prompts
logging.info(f"Generating initial character image using the selected model: {selected_model}")

try:
    initial_image = create_initial_character_image(
        initial_prompt_info['chapter'],
        SEED,
        initial_prompt_info['positive_ai_prompt'],
        initial_prompt_info['negative_ai_prompt']
    )
except ValueError as e:
    logging.error(str(e))
    exit(1)

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
sanitized_model_name = selected_model.replace("/", "_")
filename_prefix = f"{sanitized_model_name}"
initial_image_path = os.path.join(AI_GENERATED_USERS_PATH, f"{filename_prefix}_{timestamp_str}.png")
initial_image.save(initial_image_path)

# Enhance Initial Image
enhanced_image = enhance_image(initial_image_path, ENHANCED_IMAGES_PATH)
enhanced_image_path = os.path.join(ENHANCED_IMAGES_PATH, f"{filename_prefix}_enhanced_{timestamp_str}.png")

# Save enhanced image
enhanced_image.save(enhanced_image_path)
logging.info(f"Initial character image enhanced and saved to {ENHANCED_IMAGES_PATH}")

# Load Stable Diffusion pipeline for further processing
logging.info("Setting up Stable Diffusion pipeline for further processing.")
vae_model_path = "stabilityai/sd-vae-ft-mse"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    selected_model,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to("cuda")
logging.info(f"Stable Diffusion pipeline set up using model: {selected_model}")

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

# Load IP-Adapter
logging.info("Loading IP-Adapter.")
device = "cuda"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt_filename, device)
logging.info("IP-Adapter loaded successfully.")

# Extract embeddings for initial character from enhanced image
try:
    initial_embedding, initial_aligned_face = extract_embeddings(enhanced_image_path)
    if initial_embedding is None or initial_aligned_face is None:
        raise ValueError("Failed to extract embedding from enhanced image.")
except ValueError as e:
    logging.error(str(e))
    initial_image.show()
    logging.error("Make sure that the generated image contains a clear, detectable face. Exiting.")
    exit(1)

# Process images using the positive and negative prompts from each chapter
logging.debug("Listing all AI generated image files...")
timestamp = datetime.now().strftime("%H%M%S")
total_start_time = time.time()
total_images_generated = 0
total_generation_time = 0

for loop in range(NUMBER_OF_LOOPS):
    logging.info(f"Starting loop {loop + 1} of {NUMBER_OF_LOOPS}")

    logging.info("Processing AI generated images...")
    process_images(loop, timestamp, initial_embedding, initial_aligned_face, selected_model, story_chapters)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

if total_images_generated > 0:
    average_time_per_image = total_elapsed_time / total_images_generated
else:
    average_time_per_image = 0

logging.info("=== SUMMARY ===")
logging.info(f"Total images generated: {total_images_generated}")
logging.info(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
logging.info(f"Average time per image: {average_time_per_image:.2f} seconds")

logging.info("All images have been generated and enhanced.")