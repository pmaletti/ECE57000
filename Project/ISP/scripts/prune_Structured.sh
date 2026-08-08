#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ============================================================
# 1. MODEL SELECT  (uncomment exactly one)
# ============================================================
model_path=stabilityai/stable-diffusion-3.5-large
# model_path=stabilityai/stable-diffusion-3.5-medium
# model_path=stabilityai/stable-diffusion-xl-base-1.0
# model_path=black-forest-labs/FLUX.1-dev
# model_path=black-forest-labs/FLUX.1-schnell
# model_path=black-forest-labs/FLUX.2-klein-9B

# ============================================================
# 2. PER-MODEL GEOMETRY
#    NOTE: for SDXL, pruning uses blocksize = d_in//4 (ffn) or d_in (attn)
#          internally — BLOCKSIZE below is ignored for SDXL geometry.
# ============================================================
case "$model_path" in
  *stable-diffusion-3.5-medium*)   # 2.5B | hidden 1536 | 24 heads (64) | FFN 6144
    BLOCKSIZE=1536; FFN_PROTECT=0.80; ATTN_PROTECT=0.50
    N_EIGEN=768;  NUM_GROUPS=4; STEPS=25 ;;
  *stable-diffusion-3.5-large*)    # 8B   | hidden 2432 | 38 heads (64) | FFN 9728
    BLOCKSIZE=2432; FFN_PROTECT=0.80; ATTN_PROTECT=0.60
    N_EIGEN=1216; NUM_GROUPS=4; STEPS=25 ;;
  *stable-diffusion-xl-base-1.0*)  # 2.6B | hidden 1280 | 20 heads (64) | FFN 5120 (GEGLU)
    BLOCKSIZE=128;  FFN_PROTECT=0.60; ATTN_PROTECT=0.40
    N_EIGEN=32;   NUM_GROUPS=4; STEPS=25 ;;
  *FLUX.1-dev*|*FLUX.1-schnell*)   # 12B | hidden 3072 | 24 heads (128)
    BLOCKSIZE=3072; FFN_PROTECT=0.60; ATTN_PROTECT=0.50
    N_EIGEN=1536; NUM_GROUPS=8; STEPS=25 ;;
  *FLUX.2-klein-9B*)
    BLOCKSIZE=4096; FFN_PROTECT=0.60; ATTN_PROTECT=0.50
    N_EIGEN=2048; NUM_GROUPS=8; STEPS=4  ;;
  *)
    echo "Unknown model_path: $model_path"; exit 1 ;;
esac
ATTN_PROTECT_SINGLE=0.25

# ============================================================
# 3. EXPERIMENT CONFIG
# ============================================================
num_samples=20
gs=7.0
SPARSITIES=(0.3)
prune_method=Isometric-Structured        # or OBS-Diff-Structured
saliency_mode=max_norm #neumann                     # or mean_norm  (Isometric only)
correction_mode=robust
n_tbins=4
COLLECT_GRAMS=false #false                       # true = one dense pass, save Grams, exit
seed=24432

# ============================================================
# 4. PATH ROUTING
# ============================================================
is_flux=false
[[ "$model_path" == *black-forest-labs/FLUX.* ]] && is_flux=true

if [[ "$prune_method" == Isometric-Structured ]]; then
    base=${prune_method}/${model_path}/saliency_mode_${saliency_mode}/GS${gs}/new1_correction_mode${correction_mode}_n_tbins${n_tbins}/num_samples_${num_samples}
    cfg=blocksize${BLOCKSIZE}_n_eigen${N_EIGEN}_ffn${FFN_PROTECT}_attn${ATTN_PROTECT}
    $is_flux && cfg=${cfg}_attn_single${ATTN_PROTECT_SINGLE}
    output_dir=new_output/${base}/${cfg}
    log_dir=new_log_output/${base}/${cfg}
    grams_path=${log_dir}/gram_matrices.pt
else  # OBS-Diff-Structured
    output_dir=new_output/OBS_Diff_Structured/${model_path}/new/num_samples_${num_samples}
    log_dir=new_log_output/OBS_Diff_Structured/${model_path}/new/num_samples_${num_samples}
fi

# ============================================================
# 5. RUN
# ============================================================
for sparsity_ratio in "${SPARSITIES[@]}"; do
    echo "=================================================================="
    echo "Model:${model_path}  Sparsity:${sparsity_ratio}  Samples:${num_samples}"
    echo "Method:${prune_method}  Saliency:${saliency_mode}  Correction:${correction_mode}"
    echo "Config:${cfg:-n/a}"
    echo "=================================================================="

    grams_args=()
    if [[ "$COLLECT_GRAMS" == true ]]; then
        grams_args=(--collect_grams_only --grams_path "${grams_path}")
    elif [[ -f "${grams_path:-/nonexistent}" ]]; then
        grams_args=(--grams_path "${grams_path}")   # prune FROM cache
    fi

    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --model_path "${model_path}" \
        --cache_dir /local/a/pmaletti/diffusion_weightss \
        --prune_method "${prune_method}" \
        --seed ${seed} \
        --sparsity_ratio ${sparsity_ratio} \
        --sparsity_type structured \
        --timestep_weight_strategy log_decrease \
        --timestep_min_weight 0.8 \
        --timestep_max_weight 1.2 \
        --dataset gcc3m \
        --num_samples ${num_samples} \
        --num_inference_steps ${STEPS} \
        --batch_size 4 \
        --height 512 --width 512 \
        --guidance_scale ${gs} \
        --num_pruned_groups ${NUM_GROUPS} \
        --demo_evaluate \
        --demo_dir "${output_dir}/sp${sparsity_ratio}/" \
        --save_model "${log_dir}/sp${sparsity_ratio}/" \
        --blocksize ${BLOCKSIZE} \
        --n_eigen ${N_EIGEN} \
        --ffn_protect ${FFN_PROTECT} \
        --attn_protect ${ATTN_PROTECT} \
        --attn_protect_single ${ATTN_PROTECT_SINGLE} \
        --saliency_mode ${saliency_mode} \
        --correction_mode ${correction_mode} \
        --n_tbins ${n_tbins} \
        --bin_device 'cuda:1' \
        "${grams_args[@]}"
done