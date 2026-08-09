import argparse
import os 
import numpy as np
import torch

# # Force SDPA to use the safe math backend for visualization
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

from ECE57000.Project.ISP.lib.ISP_prune import prune_SD_3_5_Structured, prune_SD_XL_Structured, check_sparsity, check_size
from lib.prune import prune_OBS_Diff, prune_OBS_Diff_Structured, prune_OBS_Diff_Structured_SDXL

from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from torchvision import transforms
from datasets import load_dataset

import transformers
from torch.nn import Linear

import transformers.modeling_utils
import transformers.pytorch_utils  # <--- Added this

from lib.collect_full_grams import collect_full_model_grams, collect_sdxl_unet_grams #, collect_full_model_grams_sharded
import pandas as pd

# 1. Patch the chunking function (deleted by HF)
def dummy_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    return forward_fn(*input_tensors)
transformers.modeling_utils.apply_chunking_to_forward = dummy_apply_chunking_to_forward

# # 2. Patch the pruning function (moved by HF)
# # Tell Python to grab it from its new home and pretend it lives in the old one
# transformers.modeling_utils.find_pruneable_heads_and_indices = transformers.pytorch_utils.find_pruneable_heads_and_indices
# transformers.modeling_utils.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer  # <--- The final missing piece!

def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
        
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

# Monkey-patch it back into the transformers library for any downstream code that expects it
transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
if hasattr(transformers, "pytorch_utils"):
    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

def prune_linear_layer(layer: Linear, index: torch.LongTensor, dim: int = 0) -> Linear:
    """
    Prune a linear layer to keep only entries in index.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    else:
        b = None
        
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    
    new_layer = Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        
    return new_layer

# Monkey-patch it back into transformers just in case other internal calls expect it
transformers.modeling_utils.prune_linear_layer = prune_linear_layer
if hasattr(transformers, "pytorch_utils"):
    transformers.pytorch_utils.prune_linear_layer = prune_linear_layer
# NOW safely import ImageReward
import ImageReward as RM

import logging
from tqdm import tqdm
from scipy import linalg
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import clip
from cleanfid import fid
import torch_fidelity
from pathlib import Path
import math
import re 
import gc

import warnings
warnings.filterwarnings("ignore") # This kills the "FID noise" warnings

def setup_logging(save_dir, sparsity_ratio):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"pruning_log_{sparsity_ratio}.txt")
    
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file), # Save to file
            logging.StreamHandler()        # Still show in terminal
        ]
    )
    return logging.getLogger(__name__)

# def calculate_clip_score(image_dir, prompts, device="cuda"):
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     total_score = 0
    
#     print("Computing CLIP scores...")
#     for i, prompt in enumerate(tqdm(prompts)):
#         img_path = os.path.join(image_dir, f"gen_{i}.png")
#         image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#         text = clip.tokenize([prompt], truncate=True).to(device)
        
#         with torch.no_grad():
#             image_features = model.encode_image(image)
#             text_features = model.encode_text(text)
            
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
            
#             similarity = (image_features @ text_features.T).item()
#             total_score += similarity
            
#     return total_score / len(prompts)

# def calculate_clip_score(model, preprocess, image_dir, prompts, device="cuda"):
#     total_score = 0
#     valid_samples = 0
    
#     print("\nComputing CLIP scores (Strict Index Matching)...")
#     for i, prompt in enumerate(tqdm(prompts)):
#         img_path = os.path.join(image_dir, f"gen_{i}.png")
        
#         # Robust check to handle paused/incomplete runs or missing files safely
#         if not os.path.exists(img_path):
#             continue
            
#         try:
#             image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt], truncate=True).to(device)
            
#             with torch.no_grad():
#                 image_features = model.encode_image(image)
#                 text_features = model.encode_text(text)
                
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
#                 text_features /= text_features.norm(dim=-1, keepdim=True)
                
#                 similarity = (image_features @ text_features.T).item()
#                 total_score += similarity
#                 valid_samples += 1
#         except Exception as e:
#             print(f"Skipping {img_path} due to error: {e}")
            
#     return total_score / valid_samples if valid_samples > 0 else 0.0

# def load_eval_prompts(num_samples=500, real_images_path="./eval_output/real_images"):
#     # 'common_objects' is a standard path for COCO 2017
#     dataset = load_dataset(
#                 "sayakpaul/coco-30-val-2014", 
#                 split="train", 
#                 cache_dir="/local/a/pmaletti/datasets/"
#             )
#     # dataloader = get_loaders(
#     #     'gcc3m',
#     #     num_samples=num_samples
#     # )

#     # 2. Setup the real images directory
#     os.makedirs(real_images_path, exist_ok=True)
    
#     prompts = []

#     # Check if we need to export real images (only if the folder is empty)
#     export_images = len(os.listdir(real_images_path)) < num_samples

#     print(f"Streaming first {num_samples} prompts from COCO via Hugging Face...")
    
#     for i, entry in enumerate(dataset):
#         if i >= num_samples:
#             break
#         # entry['caption'] is a list of strings for that image; take the first one
#         prompts.append(entry['caption'])  # Assuming each entry has a 'caption' field with a list of strings

#         # Save the real image if it doesn't already exist in the path
#         if export_images:
#             image_path = os.path.join(real_images_path, f"real_{i}.png")
#             if not os.path.exists(image_path):
#                 entry['image'].save(image_path)
#     return prompts


# def calculate_ssim_against_base(gen_dir, base_dir):
#     print(f"\nCalculating SSIM against base model: {base_dir}")
#     if not os.path.exists(base_dir):
#         print("Base directory not found. Skipping SSIM.")
#         return 0.0
        
#     files = [f for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(".png")]
#     scores = []
    
#     for f in tqdm(files, desc="Computing SSIM"):
#         p1, p2 = os.path.join(gen_dir, f), os.path.join(base_dir, f)
#         if not os.path.exists(p2):
#             continue
            
#         img1 = np.array(Image.open(p1).convert('RGB'))
#         img2 = np.array(Image.open(p2).convert('RGB'))
        
#         # Calculate SSIM (channel_axis=2 specifies the RGB color channels)
#         score = ssim(img1, img2, channel_axis=2, data_range=255)
#         scores.append(score)
        
#     return np.mean(scores) if scores else 0.0

def generate_intermediate_visualizations(pipe, args):
    """
    Generates a small set of fixed prompts to quickly visually verify 
    the semantic and structural integrity of the pruned model.
    """
    print("\n--- Starting Intermediate Visualizations ---")
    height = getattr(args, 'height', 1024)
    width = getattr(args, 'width', 1024)
    num_inference_steps = getattr(args, 'num_inference_steps', 25)
    guidance_scale = getattr(args, 'guidance_scale', 7.0)
    batch_size = getattr(args, 'batch_size', 4)

    # prompts = [
    #     "A cat holding a sign that says hello world",
    #     "A highly detailed close-up of a weathered leather boot on a cobblestone street, soft golden hour lighting, 8k resolution, cinematic textures.",
    #     "An astronaut riding a glowing translucent jellyfish through a neon-lit cyberpunk city, vibrant purple and teal color palette.",
    #     "A complex glass sculpture of a DNA double helix made of melting ice, sitting on a mirror, with light refracting through the structure.",
    #     "A vintage chalkboard in a dusty classroom with the words 'RICCI FLOW' written perfectly in white chalk, surrounded by complex mathematical equations.",
    #     "A portrait of a human growing colorful flowers from her hair. Hyperrealistic oil painting. Intricate details.",
    #     "A translucent, glowing jellyfish in the deep dark ocean.",
    #     "A vibrant hummingbird hovering next to a hibiscus flower, macro photo.",
    #     "A lighthouse on a rocky coast during a storm.",
    #     "A steampunk detective on a London street.",
    #     "An astronaut planting a flag on Mars, cinematic.",
    #     "A robot gardening in a futuristic city",
    #     "A hot air balloon floating over a valley",
    #     "A squirrel in a suit reading a book",
    #     "A vintage car parked by a lighthous"
    # ]

    # os.makedirs(args.demo_dir, exist_ok=True)

    # for i in range(0, len(prompts), batch_size):
    #     batch_prompts = prompts[i : i + batch_size]        
    #     current_bs = len(batch_prompts)
    #     if args.prune_method == 'Isometric-Structured':
    #         # generators = [torch.Generator("cuda").manual_seed(args.seed + i + j) for j in range(current_bs)]
    #         generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]
    #     else:
    #         generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]
            
    #     images = pipe(
    #         prompt=batch_prompts,
    #         height=height,
    #         width=width,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         generator=generators
    #         # generator=torch.Generator("cuda").manual_seed(0)
    #     ).images 
        
    #     for j, image in enumerate(images):
    #         absolute_index = i + j
    #         image.save(os.path.join(args.demo_dir, f"eval_{absolute_index}.png"))

    prompts = [
        {
            "prompt": "A cat holding a sign that says hello world",
            "negative_prompt": "blurry, distorted text, misspelled, bad anatomy, mutated fingers, ugly, artifacts"
        },
        {
            "prompt": "A highly detailed close-up of a weathered leather boot on a cobblestone street, soft golden hour lighting, 8k resolution, cinematic textures.",
            "negative_prompt": "blurry, low resolution, flat lighting, overexposed, plastic texture, synthetic, cartoon, CGI"
        },
        {
            "prompt": "An astronaut riding a glowing translucent jellyfish through a neon-lit cyberpunk city, vibrant purple and teal color palette.",
            "negative_prompt": "dull colors, washed out, low contrast, boring, messy composition, murky, lowres"
        },
        {
            "prompt": "A complex glass sculpture of a DNA double helix made of melting ice, sitting on a mirror, with light refracting through the structure.",
            "negative_prompt": "opaque, matte, badly drawn, messy, low quality, noise, grain, flat, 2D"
        },
        {
            "prompt": "A vintage chalkboard in a dusty classroom with the words 'RICCI FLOW' written perfectly in white chalk, surrounded by complex mathematical equations.",
            "negative_prompt": "typos, misspelled, unreadable text, gibberish, modern, clean, digital art, shiny"
        },
        {
            "prompt": "A portrait of a human growing colorful flowers from her hair. Hyperrealistic oil painting. Intricate details.",
            "negative_prompt": "photograph, bad anatomy, deformed face, creepy, ugly, poorly drawn, asymmetrical, modern, low detail"
        },
        {
            "prompt": "A translucent, glowing jellyfish in the deep dark ocean.",
            "negative_prompt": "bright background, daylight, dull, flat, low resolution, murky water, plastic"
        },
        {
            "prompt": "A vibrant hummingbird hovering next to a hibiscus flower, macro photo.",
            "negative_prompt": "out of focus, motion blur, noisy, painting, illustration, drawing, dead, static"
        },
        {
            "prompt": "A lighthouse on a rocky coast during a storm.",
            "negative_prompt": "calm, sunny, daylight, peaceful, low detail, cartoon, interior, dry"
        },
        {
            "prompt": "A steampunk detective on a London street.",
            "negative_prompt": "modern, contemporary, sci-fi, bad anatomy, messy details, futuristic, neon"
        },
        {
            "prompt": "An astronaut planting a flag on Mars, cinematic.",
            "negative_prompt": "earth, green grass, trees, low quality, amateur, grainy, badly framed, poorly drawn"
        },
        {
            "prompt": "A robot gardening in a futuristic city",
            "negative_prompt": "dystopian, rusty, abandoned, human, bad anatomy, lowres, dirty, bleak"
        },
        {
            "prompt": "A hot air balloon floating over a valley",
            "negative_prompt": "grounded, flat terrain, dark, gloomy, low contrast, indoors, claustrophobic"
        },
        {
            "prompt": "A squirrel in a suit reading a book",
            "negative_prompt": "bad anatomy, ugly, cartoon, messy, poorly drawn, abstract, human face"
        },
        {
            "prompt": "A vintage car parked by a lighthouse",
            "negative_prompt": "modern car, ruined, rusty, blurry, out of focus, text, watermark, bad proportions"
        }
    ]
    
    os.makedirs(args.demo_dir, exist_ok=True)

    for i in range(0, len(prompts), batch_size):
        # 1. Slice the batch of dictionaries
        batch_dicts = prompts[i : i + batch_size]        
        
        # 2. Extract the positive and negative strings into separate lists
        batch_prompts = [item["prompt"] for item in batch_dicts]
        batch_negatives = [item["negative_prompt"] for item in batch_dicts]
        
        current_bs = len(batch_prompts)
        if args.prune_method == 'Isometric-Structured':
            # generators = [torch.Generator("cuda").manual_seed(args.seed + i + j) for j in range(current_bs)]
            # generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]
            generators = [torch.Generator("cuda").manual_seed(args.seed) for j in range(current_bs)]
        else:
            # generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]
            generators = [torch.Generator("cuda").manual_seed(args.seed) for j in range(current_bs)]
            
        images = pipe(
            prompt=batch_prompts,
            negative_prompt=batch_negatives,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generators
        ).images 
        
        for j, image in enumerate(images):
            absolute_index = i + j
            image.save(os.path.join(args.demo_dir, f"eval_{absolute_index}.png"))
    
            
    print(f"Intermediate visualizations saved to {args.demo_dir}.")

# def calculate_image_reward(gen_dir, eval_prompts):
#     print("\nLoading ImageReward model...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     reward_model = RM.load("ImageReward-v1.0").to(device)
    
#     # 1. Cleanly discover files and sort them strictly by their numerical ID
#     try:
#         image_files = sorted(
#             [f for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(('.png', '.jpg'))],
#             key=lambda x: int(re.findall(r'\d+', x)[0])
#         )
#     except IndexError:
#         raise ValueError("Some filenames in gen_dir do not match the expected 'gen_[number].png' format.")
    
#     total_score = 0.0
#     valid_samples = 0
    
#     print(f"Scoring {len(image_files)} images for Human Preference (Strict Index Matching)...")
#     with torch.no_grad():
#         for img_file in tqdm(image_files):
#             # 2. Extract the exact global index from the filename
#             file_idx = int(re.findall(r'\d+', img_file)[0])
            
#             # 3. Ensure the extracted index is within the bounds of your prompt list
#             if file_idx >= len(eval_prompts):
#                 print(f"Warning: File {img_file} indicates index {file_idx}, but only {len(eval_prompts)} prompts available. Skipping.")
#                 continue
                
#             prompt = eval_prompts[file_idx]
#             img_path = os.path.join(gen_dir, img_file)
            
#             try:
#                 # with Image.open(img_path).convert("RGB") as img:
#                 #     score = reward_model.score(prompt, img)
#                 score = reward_model.score(prompt, img_path)
#                 total_score += score
#                 valid_samples += 1
#             except Exception as e:
#                 print(f"Skipping scoring for {img_file} due to error: {e}")
                
#     mean_reward = total_score / valid_samples if valid_samples > 0 else 0.0
#     return mean_reward

# def generate_and_evaluate_full_dataset(pipe, args):
#     """
#     Generates the full dataset for statistical metric evaluation and 
#     calculates CLIP, FID, Inception Score, Precision, and Recall.
#     """
#     print("\n--- Starting Full Dataset Generation & Evaluation ---")
    
#     # Setup directories
#     real_dir = "./COCO_Images/real_images"
#     gen_dir = f"{args.demo_dir}/eval_output/generated"
#     base_gen_dir = f"new_output/base_evals/{args.model_path}/eval_output/generated/"
#     os.makedirs(gen_dir, exist_ok=True)

#     height = getattr(args, 'height', 1024)
#     width = getattr(args, 'width', 1024)
#     num_inference_steps = getattr(args, 'num_inference_steps', 25)
#     guidance_scale = getattr(args, 'guidance_scale', 7.0)
#     batch_size = getattr(args, 'batch_size', 4)
    
#     # # Load prompts
#     # if args.model_path in ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-medium']:
#     #     num_samples = 2500
#     # elif args.model_path == 'stabilityai/stable-diffusion-xl-base-1.0': 
#     #     num_samples = 5000
#     num_samples = args.num_gen_samples
#     eval_prompts = load_eval_prompts(num_samples=num_samples, real_images_path=real_dir)
    
#     # --- RESUME LOGIC ---
#     existing_files = [f for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(".png")]
#     if existing_files:
#         indices = [int(re.findall(r'\d+', f)[0]) for f in existing_files]
#         resume_idx = max(indices) + 1
#         print(f"Found {len(existing_files)} existing images. Resuming from prompt #{resume_idx}")
#     else:
#         resume_idx = 0

#     num_batches = (len(eval_prompts) + batch_size - 1) // batch_size
    
#     # clip_model, preprocess = clip.load("ViT-B/32", device=pipe.device)
#     clip_model, preprocess = clip.load("ViT-B/16", device=pipe.device)

#     # --- GENERATION LOOP ---
#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = min(start_idx + batch_size, len(eval_prompts))

#         if end_idx <= resume_idx:
#             continue
            
#         batch_prompts = eval_prompts[start_idx:end_idx]
#         current_bs = len(batch_prompts)

#         if args.prune_method == 'Isometric-Structured':
#             # generators = [torch.Generator("cuda").manual_seed(args.seed + i + j) for j in range(current_bs)]
#             # generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]
#             generators = [torch.Generator("cuda").manual_seed(args.seed) for j in range(current_bs)]
#         else:
#             generators = [torch.Generator("cuda").manual_seed(0) for j in range(current_bs)]

#         output = pipe(
#             prompt=batch_prompts,
#             height=height,
#             width=width,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#             generator=generators,
#         ).images

#         # Save images
#         for j, image in enumerate(output):
#             global_idx = start_idx + j
#             save_path = os.path.join(gen_dir, f"gen_{global_idx}.png")
#             image.save(save_path)

#         # --- PERIODIC SNAPSHOT ---
#         if (i + 1) % 200 == 0 or (i + 1) == num_batches:
#             avg_clip = calculate_clip_score(clip_model, preprocess, gen_dir, eval_prompts[:end_idx])
#             # Fast, noisy FID check during generation to catch catastrophic failure early
#             fid_score = fid.compute_fid(gen_dir, real_dir)

#             # if os.path.exists(base_gen_dir):
#             #     ssim_val = calculate_ssim_against_base(gen_dir, base_gen_dir)
#             # else:
#             #     print(f"Base model's generation not yet run for model {args.model_path}")
#             #     ssim_val = None

#             print(f"Sparsity: {args.sparsity_ratio} | Batch: {i+1:04d}/{num_batches:04d} | Samples: {end_idx:04d} | CLIP: {avg_clip:.4f} | FID: {fid_score:.4f}") #| SSIM: {ssim_val:.4f}
            
#             # The "PhD Panic" Button
#             if fid_score > 350:
#                 print("WARNING: FID is critically high. Manifold might be collapsing!")

#     print("\nGeneration complete. Proceeding to rigorous metrics computation...")

#     # --- FINAL METRICS CALCULATION (torch-fidelity) ---
#     metrics = torch_fidelity.calculate_metrics(
#         input1=gen_dir,
#         input2=real_dir,
#         cuda=True,
#         fid=True,
#         isc=True,             # Inception Score
#         prc=True,             # Precision and Recall
#         batch_size=1,         # <--- ADD THIS: Prevents stacking crash with varying image sizes
#         verbose=False
#     )

#     final_clip = calculate_clip_score(clip_model, preprocess, gen_dir, eval_prompts)

#     # # --- NEW: Calculate ImageReward ---
#     # final_ir = calculate_image_reward(gen_dir, eval_prompts)

#     # if os.path.exists(base_gen_dir):
#     #     final_ssim = calculate_ssim_against_base(gen_dir, base_gen_dir)
#     # else:
#     #     print(f"Base model's generation not yet run for model {args.model_path}")
#     #     final_ssim = None

#     print(metrics)
#     fid_val = metrics['frechet_inception_distance']
#     is_val = metrics['inception_score_mean']
#     precision_val = metrics['precision']
#     recall_val = metrics['recall']

#     print(f"\n" + "="*50)
#     print(f"FINAL EVALUATION METRICS (Sparsity: {args.sparsity_ratio})")
#     print(f"="*50)
#     print(f"CLIP Score      : {final_clip:.4f}  (Text-Image Alignment)")
#     # print(f"ImageReward     : {final_ir:.4f}  (Human Preference Alignment)") # <--- Added here
#     print(f"FID             : {fid_val:.4f}  (Distribution Distance)")
#     print(f"Inception Score : {is_val:.4f}  (Clarity & Diversity)")
#     # print(f"SSIM            : {final_ssim:.4f}  (Structural Similarity to Base Model)") 
#     print(f"Precision       : {precision_val:.4f}  (Image Quality / Realism)")
#     print(f"Recall          : {recall_val:.4f}  (Dataset Coverage / Mode Collapse)")
#     print(f"="*50)
    
#     # Advanced Diagnostics for your logs
#     if fid_val > 100:
#         if recall_val < 0.2:
#             print(">> DIAGNOSTIC: High FID is primarily driven by MODE COLLAPSE (Low Recall).")
#         elif precision_val < 0.4:
#             print(">> DIAGNOSTIC: High FID is primarily driven by MANIFOLD DEGRADATION (Low Precision).")

#     return

def load_eval_prompts(num_samples=500, real_images_path="./eval_output/real_images"):
    dataset = load_dataset(
        "sayakpaul/coco-30-val-2014",
        split="train",
        cache_dir="/local/a/pmaletti/datasets/"
    )
    os.makedirs(real_images_path, exist_ok=True)
    prompts = []
    export_images = len(os.listdir(real_images_path)) < num_samples
    print(f"Streaming first {num_samples} prompts from COCO via Hugging Face...")

    for i, entry in enumerate(dataset):
        if i >= num_samples:
            break
        cap = entry['caption']
        # COCO caption may be str OR list of str — always collapse to a single str
        if isinstance(cap, (list, tuple)):
            cap = cap[0] if len(cap) > 0 else ""
        prompts.append(str(cap))
        if export_images:
            image_path = os.path.join(real_images_path, f"real_{i}.png")
            if not os.path.exists(image_path):
                entry['image'].save(image_path)

    assert all(isinstance(p, str) for p in prompts), "prompts must be a flat list of str"
    return prompts


def calculate_clip_score(model, preprocess, image_dir, prompts, device="cuda"):
    total_score = 0.0
    valid_samples = 0
    print("\nComputing CLIP scores (Strict Index Matching)...")
    for i, prompt in enumerate(tqdm(prompts)):
        img_path = os.path.join(image_dir, f"gen_{i}.png")
        if not os.path.exists(img_path):
            continue
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            text = clip.tokenize([str(prompt)], truncate=True).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                total_score += (image_features @ text_features.T).item()
                valid_samples += 1
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
    return total_score / valid_samples if valid_samples > 0 else 0.0


def calculate_ssim_against_base(gen_dir, base_dir):
    print(f"\nCalculating SSIM against base model: {base_dir}")
    if not os.path.exists(base_dir):
        print("Base directory not found. Skipping SSIM.")
        return None
    files = [f for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(".png")]
    scores = []
    for f in tqdm(files, desc="Computing SSIM"):
        p1, p2 = os.path.join(gen_dir, f), os.path.join(base_dir, f)
        if not os.path.exists(p2):
            continue
        try:
            img1 = np.array(Image.open(p1).convert('RGB'))
            img2 = np.array(Image.open(p2).convert('RGB'))
            if img1.shape != img2.shape:
                continue
            scores.append(ssim(img1, img2, channel_axis=2, data_range=255))
        except Exception as e:
            print(f"Skipping SSIM for {f}: {e}")
    return float(np.mean(scores)) if scores else None


def calculate_image_reward(reward_model, gen_dir, eval_prompts):
    print("\nScoring ImageReward (Human Preference, Strict Index Matching)...")
    try:
        image_files = sorted(
            [f for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(('.png', '.jpg'))],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
    except IndexError:
        raise ValueError("Filenames must match 'gen_[number].png'.")

    total_score = 0.0
    valid_samples = 0
    with torch.no_grad():
        for img_file in tqdm(image_files):
            file_idx = int(re.findall(r'\d+', img_file)[0])
            if file_idx >= len(eval_prompts):
                continue
            prompt = str(eval_prompts[file_idx])
            img_path = os.path.join(gen_dir, img_file)
            try:
                total_score += reward_model.score(prompt, img_path)
                valid_samples += 1
            except Exception as e:
                print(f"Skipping IR for {img_file}: {e}")
    return total_score / valid_samples if valid_samples > 0 else 0.0


def generate_and_evaluate_full_dataset(pipe, args):
    print("\n--- Starting Full Dataset Generation & Evaluation ---")
    real_dir = "./COCO_Images/real_images"
    gen_dir = f"{args.demo_dir}/eval_output/generated"
    base_gen_dir = f"new_output/base_evals/{args.model_path}/eval_output/generated/"
    os.makedirs(gen_dir, exist_ok=True)

    height = getattr(args, 'height', 1024)
    width = getattr(args, 'width', 1024)
    num_inference_steps = getattr(args, 'num_inference_steps', 25)
    guidance_scale = getattr(args, 'guidance_scale', 7.0)
    batch_size = getattr(args, 'batch_size', 4)
    num_samples = args.num_gen_samples
    eval_prompts = load_eval_prompts(num_samples=num_samples, real_images_path=real_dir)

    # --- metrics dataframe: load if resuming, else create ---
    metrics_csv = os.path.join(gen_dir, "metrics_progress.csv")
    cols = ["batch", "samples", "clip", "fid", "ssim", "image_reward",
            "inception_score", "precision", "recall", "sparsity", "final"]
    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        print(f"Loaded existing metrics df with {len(df)} rows from {metrics_csv}")
    else:
        df = pd.DataFrame(columns=cols)

    def _log_metrics(batch, samples, final=False, **vals):
        nonlocal df
        df = df[df["samples"] != samples]
        row = {"batch": batch, "samples": samples,
               "sparsity": args.sparsity_ratio, "final": final}
        for c in cols:
            row.setdefault(c, np.nan)
        row.update(vals)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values("samples").reset_index(drop=True)
        df.to_csv(metrics_csv, index=False)

    # --- resume from first MISSING index (handles half-written batches) ---
    existing = {int(re.findall(r'\d+', f)[0])
                for f in os.listdir(gen_dir) if f.startswith("gen_") and f.endswith(".png")}
    resume_idx = 0
    while resume_idx in existing:
        resume_idx += 1
    print(f"Found {len(existing)} images. Resuming from index {resume_idx}")

    num_batches = (len(eval_prompts) + batch_size - 1) // batch_size
    clip_model, preprocess = clip.load("ViT-B/16", device=pipe.device)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(eval_prompts))
        if end_idx <= resume_idx:
            continue

        batch_prompts = eval_prompts[start_idx:end_idx]
        current_bs = len(batch_prompts)
        if args.prune_method == 'Isometric-Structured':
            generators = [torch.Generator("cuda").manual_seed(args.seed) for _ in range(current_bs)]
        else:
            generators = [torch.Generator("cuda").manual_seed(0) for _ in range(current_bs)]

        output = pipe(prompt=batch_prompts, height=height, width=width,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale, generator=generators).images
        for j, image in enumerate(output):
            image.save(os.path.join(gen_dir, f"gen_{start_idx + j}.png"))

        if (i + 1) % 200 == 0 or (i + 1) == num_batches:
            avg_clip = calculate_clip_score(clip_model, preprocess, gen_dir, eval_prompts[:end_idx])
            fid_score = fid.compute_fid(gen_dir, real_dir)
            print(f"Sparsity: {args.sparsity_ratio} | Batch: {i+1:04d}/{num_batches:04d} | "
                  f"Samples: {end_idx:04d} | CLIP: {avg_clip:.4f} | FID(cleanfid): {fid_score:.4f}")
            _log_metrics(i + 1, end_idx, final=False, clip=avg_clip, fid=fid_score)
            if fid_score > 350:
                print("WARNING: FID critically high. Manifold might be collapsing!")

    print("\nGeneration complete. Computing final rigorous metrics...")

    metrics = torch_fidelity.calculate_metrics(
        input1=gen_dir, input2=real_dir, cuda=True,
        fid=True, isc=True, prc=True, batch_size=1, verbose=False)
    fid_val = metrics['frechet_inception_distance']
    is_val = metrics['inception_score_mean']
    precision_val = metrics['precision']
    recall_val = metrics['recall']

    final_clip = calculate_clip_score(clip_model, preprocess, gen_dir, eval_prompts)
    final_ssim = calculate_ssim_against_base(gen_dir, base_gen_dir)

    # reward_model = RM.load("ImageReward-v1.0").to(pipe.device)
    # final_ir = calculate_image_reward(reward_model, gen_dir, eval_prompts)
    # del reward_model
    # torch.cuda.empty_cache()

    _log_metrics(num_batches, len(eval_prompts), final=True,
                 clip=final_clip, fid=fid_val, ssim=final_ssim, image_reward= None,
                 inception_score=is_val, precision=precision_val, recall=recall_val)

    print(metrics)
    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION METRICS (Sparsity: {args.sparsity_ratio})")
    print("=" * 50)
    print(f"CLIP Score      : {final_clip:.4f}  (Text-Image Alignment)")
    # print(f"ImageReward     : {final_ir:.4f}  (Human Preference Alignment)")
    print(f"FID             : {fid_val:.4f}  (Distribution Distance)")
    print(f"Inception Score : {is_val:.4f}  (Clarity & Diversity)")
    print(f"SSIM            : {final_ssim if final_ssim is not None else 'N/A'}  (Structural Sim. to Base)")
    print(f"Precision       : {precision_val:.4f}  (Image Quality / Realism)")
    print(f"Recall          : {recall_val:.4f}  (Coverage / Mode Collapse)")
    print("=" * 50)

    if fid_val > 100:
        if recall_val < 0.2:
            print(">> DIAGNOSTIC: High FID primarily from MODE COLLAPSE (Low Recall).")
        elif precision_val < 0.4:
            print(">> DIAGNOSTIC: High FID primarily from MANIFOLD DEGRADATION (Low Precision).")

    print(f"\nMetrics history saved to {metrics_csv}")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='text-to-image model, e.g. SD3')
    parser.add_argument('--cache_dir', type=str, help='cache dir for text-to-image model, e.g. SD3')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="structured", choices=["unstructured", "4:8", "2:4", "structured"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "OBS-Diff", "OBS-Diff-Structured", "dsnot", "magnitude_structured", "Isometric", "Isometric-Structured"])
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--dataset', type=str, default="gcc3m", help='Dataset to use for calibration.')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use for calibration.')
    parser.add_argument('--minlayer', type=int, default=None, help='Minimum layer to prune')
    parser.add_argument('--maxlayer', type=int, default=None, help='Maximum layer to prune')
    parser.add_argument('--demo_evaluate', action="store_true", help="A single image evaluation by the pruned model")
    parser.add_argument("--demo_dir", type=str, default="eval_output.png", help="Path to save the demo images.")
    parser.add_argument("--pruned_model", type=str, default="", help="Path to save the demo images.")
    parser.add_argument("--num_pruned_groups", type=int, default=4, help="Number of pruned groups.")
    parser.add_argument("--timestep_weight_strategy", type=str, default="uniform", 
                       choices=["uniform", "linear_increase", "linear_decrease", "log_increase", "log_decrease"], help="Timestep weight strategy for Hessian update")
    parser.add_argument("--timestep_min_weight", type=float, default=0.8, help="Min weight for timestep-aware weighting")
    parser.add_argument("--timestep_max_weight", type=float, default=1.2, help="Max weight for timestep-aware weighting")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=512, help="Height of the image")
    parser.add_argument("--width", type=int, default=512, help="Width of the image")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--no_compensate", action="store_true", help="Skip error compensation in OBS-Diff")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Hessian dampening factor")
    parser.add_argument("--test_generation_only", action="store_true", help="Eval Pruned Model Generation")
    parser.add_argument("--vis_only", action="store_true", help="Eval Pruned Model Generation")
    parser.add_argument("--run_base_model", action="store_true", help="Eval Pruned Model Generation")
    parser.add_argument("--ricci_flow_iters", type=int, default=1, help="Numb er of ricci flow iterations")
    parser.add_argument("--blocksize", type=int, default=128, help="Number of inference steps")
    parser.add_argument("--n_eigen", type=int, default=32, help="Number of inference steps")
    parser.add_argument("--ffn_protect", type=float, default=0.5, help="Number of inference steps")
    parser.add_argument("--attn_protect", type=float, default=0.8, help="Number of inference steps")
    parser.add_argument("--attn_protect_single", type=float, default=0.25, help="Number of inference steps")
    parser.add_argument("--num_gen_samples", type=int, default=2500, help="Number of images to generate for scoring")
    parser.add_argument("--eval_ecodiff", action="store_true", help="Eval EcoDiff Models Generation")
    parser.add_argument("--collect_grams_only", action="store_true",
                        help="One dense pass: accumulate + save full-model Gram matrices, then exit.")
    parser.add_argument("--grams_path", type=str, default=None,
                        help="Path to save/load full-model Gram matrices (.pt).")
    parser.add_argument("--saliency_mode", type=str, default="neumann", choices=["neumann", "mean_norm", "max_norm", "magnitude", "effective_resistance"])
    parser.add_argument("--correction_mode", type=str, default="spectral", choices=["full", "spectral", "none", "robust"])
    parser.add_argument("--n_tbins", type=int, default=1, help="Timestep bins for robust healing")
    parser.add_argument("--robust_iters", type=int, default=6)
    parser.add_argument("--robust_lr", type=float, default=1.0)
    parser.add_argument("--bin_device", type=str, default=None,
                        help="Device to store SNR bins, e.g. 'cuda:1'. Defaults to model device.")
    parser.add_argument("--robust_dro", action="store_true", help="Skip error compensation in OBS-Diff")
    parser.add_argument("--robust_eta", type=float, default=1.0)
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if not args.test_generation_only:
        log_save_path =  f"{args.save_model}/logs_{args.sparsity_type}/nsamples{args.num_samples}_sp{args.sparsity_ratio}_iter{args.ricci_flow_iters}_maxlayer{args.maxlayer}_minlayer{args.minlayer}" if args.save_model else "pruning_log"
        logger = setup_logging(log_save_path, args.sparsity_ratio)

        logger.info("="*30)
        logger.info(f"PRUNING START: {args.sparsity_ratio} sparsity")
        logger.info(f"CONFIG: {args.prune_method} with {args.sparsity_type}")
        logger.info(f"Log path & Model Save Path: {args.save_model}")
        logger.info(f"Demo images path: {args.demo_dir}")
        logger.info("="*30)

        # Example: replacing a print
        logger.info(f"Loading model from {args.model_path}")

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "structured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
  
    device = torch.device("cuda:0")

    if args.test_generation_only:
        
        # 1. Dynamically set the correct precision (Flux REQUIRES bf16, others use fp16)
        is_flux = "flux" in args.model_path.lower()
        weight_dtype = torch.bfloat16 if is_flux else torch.float16
        print(f"Detected model type. Using precision: {weight_dtype}")

        kwargs = {
            "cache_dir": args.cache_dir,
            "torch_dtype": weight_dtype,
        }

        # 2. Check if we are running the base model or injecting pruned weights
        if getattr(args, 'run_base_model', False):
            print("Evaluating BASE model only (bypassing pruned weight injection).")
        elif args.eval_ecodiff:
            from huggingface_hub import hf_hub_download
            import pickle 
            import sys
        
            target_path = "/home/nano01/a/pmaletti/PhD/Projects/GeometricDiffusionPruning/EcoDiff/src"
            
            if target_path not in sys.path:
                sys.path.insert(0, target_path)

            # actual_model_path = hf_hub_download(
            #     repo_id=args.model_path, 
            #     filename=args.pruned_model,
            #     cache_dir=args.cache_dir  # <-- Add this right here!
            # )
            actual_model_path='/local/a/pmaletti/diffusion_weightss/models--LWZ19--ecodiff_flux_prune/snapshots/474298478a898827a89adff4e21821729b46566a/dev/pruned_model_20.pkl'
            # === START OF MONKEY PATCH ===
            import diffusers.models.embeddings
            
            # 1. Save the original class blueprint
            OriginalFluxPosEmbed = diffusers.models.embeddings.FluxPosEmbed
            
            # 2. Create a tolerant subclass that intercepts BOTH __new__ and __init__
            class SafeFluxPosEmbed(OriginalFluxPosEmbed):
                def __new__(cls, *args, **kwargs):
                    if 'theta' not in kwargs:
                        kwargs['theta'] = 10000
                    if 'axes_dim' not in kwargs:
                        kwargs['axes_dim'] = [16, 56, 56]
                    
                    try:
                        return OriginalFluxPosEmbed.__new__(cls, *args, **kwargs)
                    except TypeError:
                        # Fallback if the original __new__ is overly strict
                        return object.__new__(cls)

                def __init__(self, *args, **kwargs):
                    if 'theta' not in kwargs:
                        kwargs['theta'] = 10000
                    if 'axes_dim' not in kwargs:
                        kwargs['axes_dim'] = [16, 56, 56]
                    try:
                        super().__init__(*args, **kwargs)
                    except Exception:
                        pass
            
            # 3. Disguise our safe class as the original so Pickle finds it
            SafeFluxPosEmbed.__name__ = OriginalFluxPosEmbed.__name__
            SafeFluxPosEmbed.__module__ = OriginalFluxPosEmbed.__module__
            
            # 4. Inject into the live library
            diffusers.models.embeddings.FluxPosEmbed = SafeFluxPosEmbed
            # ===============================
            
            pruned_weights = torch.load(actual_model_path, map_location='cpu', weights_only=False)
            diffusers.models.embeddings.FluxPosEmbed = OriginalFluxPosEmbed
            pruned_weights = pruned_weights.to(weight_dtype)

            if hasattr(pruned_weights, "proj_out") or is_flux:
                print("Injecting pruned weights into Transformer (SD3/Flux) during initialization...")
                kwargs["transformer"] = pruned_weights
            else:
                print("Injecting pruned weights into U-Net (SDXL/SD1.5) during initialization...")
                kwargs["unet"] = pruned_weights

        else:
            print(f"Loading pruned weights from {args.pruned_model} into CPU RAM...")
            # Load pruned weights to CPU first to avoid VRAM spikes
            pruned_weights = torch.load(args.pruned_model, map_location='cpu', weights_only=False)
            
            # Cast the pruned weights to the correct precision
            pruned_weights = pruned_weights.to(weight_dtype)

            # Determine the target architecture and inject it into the kwargs
            # This is the magic step: it prevents diffusers from loading the base weights for this component!
            if hasattr(pruned_weights, "proj_out") or is_flux:
                print("Injecting pruned weights into Transformer (SD3/Flux) during initialization...")
                kwargs["transformer"] = pruned_weights
            else:
                print("Injecting pruned weights into U-Net (SDXL/SD1.5) during initialization...")
                kwargs["unet"] = pruned_weights

        # 3. Load the pipeline 
        print(f"Loading pipeline from {args.model_path}...")
        if args.model_path in ['black-forest-labs/FLUX.2-klein-9B']:
            # Ensure torch_dtype is set to float16 or bfloat16 for FLUX
            # FLUX is highly sensitive to precision; bfloat16 is recommended if your GPU supports it
            pipe = Flux2KleinPipeline.from_pretrained(
                args.model_path,
                **kwargs
            ).to(device)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                args.model_path,
                **kwargs
            ).to(device)

        # 4. Clean up dangling references and force garbage collection to free system RAM
        if 'pruned_weights' in locals():
            del pruned_weights
        gc.collect()
        torch.cuda.empty_cache()
        
        # Phase 1: Intermediate Visual Check
        if args.vis_only:
            generate_intermediate_visualizations(pipe, args)
            exit()

        # Phase 2: Full Dataset Evaluation
        generate_and_evaluate_full_dataset(pipe, args)
        return 

    # print(f"loading model {args.model_path}")
    logger.info(f"Loading model from {args.model_path}")
    if args.model_path in ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-medium']:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16
        ).to("cuda")

        pipe.transformer.eval()

        dense_shapes = {name: param.shape for name, param in pipe.transformer.named_parameters()}

    elif args.model_path == 'stabilityai/stable-diffusion-xl-base-1.0': 
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16
        ).to("cuda")
        
        pipe.unet.eval()

        dense_shapes = {name: param.shape for name, param in pipe.unet.named_parameters()}

    elif args.model_path in ['black-forest-labs/FLUX.1-dev', 
                             'black-forest-labs/FLUX.1-schnell'
                        ]: 
        pipe = FluxPipeline.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        pipe.transformer.eval()
        # dense_shapes = {name: param.shape for name, param in pipe.transformer.named_parameters() if 'single' in name}     
        dense_shapes = {name: param.shape for name, param in pipe.transformer.named_parameters()}

    elif args.model_path in ['black-forest-labs/FLUX.2-klein-9B']: 
        pipe = Flux2KleinPipeline.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        pipe.transformer.eval()
        # dense_shapes = {name: param.shape for name, param in pipe.transformer.named_parameters() if 'single' in name}
        dense_shapes = {name: param.shape for name, param in pipe.transformer.named_parameters()} 


    # 1. Dynamically determine the total number of physical blocks
    if hasattr(pipe, "unet"):
        # SDXL / U-Net architectures
        # Count the specific transformer blocks scattered in the U-Net
        total_layers = sum(
            1 for _, module in pipe.unet.named_modules() 
            if module.__class__.__name__ == "BasicTransformerBlock"
        )
    elif hasattr(pipe, "transformer"):
        if hasattr(pipe.transformer, "transformer_blocks") and hasattr(pipe.transformer, "single_transformer_blocks"):
            # Flux architectures (Flow Matching)
            total_layers = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
        elif hasattr(pipe.transformer, "transformer_blocks"):
            # SD3 / SD3.5 architectures (MM-DiT)
            total_layers = len(pipe.transformer.transformer_blocks)
        else:
            # Generic fallback for standard DiTs
            total_layers = pipe.transformer.config.num_layers

    # 2. Safely clamp args.minlayer and args.maxlayer
    if args.minlayer is not None and args.maxlayer is not None:
        args.minlayer = max(args.minlayer, 0)
        args.maxlayer = min(args.maxlayer, total_layers)
    elif args.minlayer is not None:
        args.minlayer = max(args.minlayer, 0)
        args.maxlayer = total_layers
    elif args.maxlayer is not None:
        args.maxlayer = min(args.maxlayer, total_layers)
        args.minlayer = 0
    else:
        args.minlayer = 0
        args.maxlayer = total_layers

    # To ensure the last layer is not pruned (we prune the complete MMDiT layers in structured pruning)
    if args.sparsity_type == "structured":
        if args.maxlayer == total_layers:
            args.maxlayer = total_layers - 1


    # print(f"pruning from layer {args.minlayer} to {args.maxlayer}")
    logger.info(f"Pruning from layer {args.minlayer} to {args.maxlayer}")
    # print(f"use device {device}")
    logger.info(f"Using device: {device}")

   
    # Recommended target list for SD3 / SD3.5
    target_modules = [
        # 1. Image Stream MLPs (Main Manifold)
        "ff.net.0.proj",   # Up-projection
        "ff.net.2",        # Down-projection
        
        # 2. Text/Context Stream MLPs (Guidance Manifold)
        "ff_context.net.0.proj", 
        "ff_context.net.2",
        
        # 3. Attention Projections (Geometric Backbone)
        "attn.to_q", "attn.to_k", "attn.to_v",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        
        # 4. Output Projections
        "attn.to_out.0",
        "attn.to_add_out"
    ]

    if args.sparsity_type == "structured":
        target_modules = {
            "stabilityai/stable-diffusion-3.5-large" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0"
            ],
            "stabilityai/stable-diffusion-3.5-medium" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0"
            ],  
            "stabilityai/stable-diffusion-xl-base-1.0" : [
                "ff.net.2",
                "attn1.to_out.0",
                "attn2.to_out.0"
            ],
            "black-forest-labs/FLUX.1-dev" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0",
                "proj_out"
            ],
            "black-forest-labs/FLUX.1-schnell" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0",
                "proj_out"
            ],
            "black-forest-labs/FLUX.2-dev" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0",
                "proj_mlp"
            ],
            "black-forest-labs/FLUX.2-klein-9B": [
                "ff.linear_out",
                "ff_context.linear_out",
                "attn.to_out.0",
                "attn.to_out"
            ],
            "black-forest-labs/FLUX.2-klein-4B" : [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0",
                "proj_out"
            ]
        }

    

    if args.sparsity_ratio != 0:
        # print("pruning starts")
        logger.info("Pruning starts")
        # if args.prune_method == "OBS-Diff":
        if args.sparsity_type == "unstructured":

            if args.timestep_weight_strategy == "linear_increase":
                timestep_weight = np.linspace(args.timestep_min_weight, args.timestep_max_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "linear_decrease":
                timestep_weight = np.linspace(args.timestep_max_weight, args.timestep_min_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "uniform":
                timestep_weight = np.ones(args.num_inference_steps)
            elif args.timestep_weight_strategy == "log_increase":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)

            elif args.timestep_weight_strategy == "log_decrease":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
                timestep_weight = timestep_weight[::-1]

            # print(f"timestep_weight: {timestep_weight}")

            if args.prune_method == "Isometric":
                prune_Ricci(args, pipe, target_modules[args.model_path], device, prune_n=prune_n, prune_m=prune_m, timestep_weight=timestep_weight, logger=logger)
            elif args.prune_method == "OBS-Diff":
                prune_OBS_Diff(args, pipe, target_modules, device, prune_n=prune_n, prune_m=prune_m, timestep_weight=timestep_weight, logger=logger)
        
       
        elif args.sparsity_type == "structured":
            if args.timestep_weight_strategy == "linear_increase":
                timestep_weight = np.linspace(args.timestep_min_weight, args.timestep_max_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "linear_decrease":
                timestep_weight = np.linspace(args.timestep_max_weight, args.timestep_min_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "uniform":
                timestep_weight = np.ones(args.num_inference_steps)
            elif args.timestep_weight_strategy == "log_increase":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)

            elif args.timestep_weight_strategy == "log_decrease":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
                timestep_weight = timestep_weight[::-1]

            # ─── COLLECT-ONLY: one dense pass, save Grams, exit ───────────────
            if args.collect_grams_only:
                from lib.dataloader import get_loaders 
                from ECE57000.Project.ISP.lib.ISP_prune import find_layers, get_module_by_name, callback_on_step_end

                step_info = {"current": 0}
                assert args.grams_path, "--grams_path required with --collect_grams_only"
                dataloader = get_loaders(args.dataset, num_samples=args.num_samples)
                collect_sdxl_unet_grams(
                    pipe, dataloader,
                    target_modules[args.model_path],
                    args=args, dev=device,
                    save_path=args.grams_path,
                    find_layers=find_layers,
                    # get_module_by_name=get_module_by_name,
                    callback_on_step_end=callback_on_step_end,
                    step_info=step_info,
                    store_device="cpu",
                    # num_shards=6,
                    timestep_weight=timestep_weight,
                    logger=logger,
                )
                logger.info("Gram collection complete; exiting before pruning.")
                return
            # ──────────────────────────────────────────────────────────────────

            if args.prune_method == "Isometric-Structured":
                if args.model_path in ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-medium']:
                    prune_SD_3_5_Structured(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)
                elif args.model_path == 'stabilityai/stable-diffusion-xl-base-1.0': 
                    prune_SD_XL_Structured(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)
                elif args.model_path in ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell']: 
                    prune_Flux_1_Structured(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)
                elif args.model_path in ['black-forest-labs/FLUX.2-klein-9B']: 
                    prune_Flux_2_Structured(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)

            elif args.prune_method == "OBS-Diff-Structured":
                if args.model_path in ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-medium']: 
                    prune_OBS_Diff_Structured(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)
                elif args.model_path == 'stabilityai/stable-diffusion-xl-base-1.0': 
                    prune_OBS_Diff_Structured_SDXL(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)
                elif args.model_path in ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell', 'black-forest-labs/FLUX.2-klein-9B']: 
                    prune_OBS_Diff_Structured_flux(args, pipe, target_modules[args.model_path], device, timestep_weight=timestep_weight, logger=logger)

    # 1. Dynamically identify the core network
    if hasattr(pipe, "unet"):
        core_network = pipe.unet
    elif hasattr(pipe, "transformer"):
        core_network = pipe.transformer
    else:
        raise ValueError("Could not find a 'unet' or 'transformer' in the pipeline.")

    # 2. Run Sanity Checks on the correct network
    if args.sparsity_type != "structured":
        sparsity_ratio = check_sparsity(core_network, target_modules[args.model_path], logger)
        logger.info(f"Sparsity sanity check: {sparsity_ratio:.4f}")
    
    if args.sparsity_type == "structured":
        check_size(core_network, target_modules[args.model_path], logger, dense_shapes, args)
    
    # 3. Save the correct network
    if args.save_model:
        # Note: It's safer to create a new 'save_path' variable rather than 
        # overwriting 'args.save_model', in case you need the directory path later!
        os.makedirs(args.save_model, exist_ok=True)
        save_path = f"{args.save_model}/pruned_model_ricci_prune_{args.sparsity_type}_nsamples{args.num_samples}_{args.sparsity_ratio}_iters{args.ricci_flow_iters}.pth"
        
        torch.save(core_network, save_path)
        logger.info(f"Saved pruned model to {save_path}")


    if args.demo_evaluate:
        generate_intermediate_visualizations(pipe, args)
    
if __name__ == '__main__':
    main()