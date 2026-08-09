#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 1. MODEL SELECT  (uncomment exactly one — MUST match the pruning run)
# ============================================================
model_path="stabilityai/stable-diffusion-3.5-large"
# model_path="stabilityai/stable-diffusion-3.5-medium"
# model_path="stabilityai/stable-diffusion-xl-base-1.0"
# model_path=black-forest-labs/FLUX.1-dev
# model_path=black-forest-labs/FLUX.2-klein-9B

# ============================================================
# 2. PER-MODEL GEOMETRY  (must match the pruning script's values)
# ============================================================
case "$model_path" in
  *stable-diffusion-3.5-medium*)
    BLOCKSIZE=1536; FFN_PROTECT=0.80; ATTN_PROTECT=0.50; N_EIGEN=768
    num_gen_samples=2500; num_inference_steps=25; guidance_scale=7.0; batch_size=4 ;;
  *stable-diffusion-3.5-large*)
    BLOCKSIZE=2432; FFN_PROTECT=0.80; ATTN_PROTECT=0.60; N_EIGEN=1216
    num_gen_samples=5000; num_inference_steps=25; guidance_scale=7.0; batch_size=4 ;;
  *stable-diffusion-xl-base-1.0*)
    BLOCKSIZE=128;  FFN_PROTECT=0.60; ATTN_PROTECT=0.40; N_EIGEN=32
    num_gen_samples=5000; num_inference_steps=25; guidance_scale=7.0; batch_size=4 ;;
  *FLUX.1-dev*)
    BLOCKSIZE=3072; FFN_PROTECT=0.60; ATTN_PROTECT=0.50; N_EIGEN=1536
    num_gen_samples=5000; num_inference_steps=25; guidance_scale=7.0; batch_size=4 ;;
  *FLUX.2-klein-9B*)
    BLOCKSIZE=4096; FFN_PROTECT=0.60; ATTN_PROTECT=0.50; N_EIGEN=2048
    num_gen_samples=5000; num_inference_steps=4;  guidance_scale=1.0; batch_size=1 ;;
  *)
    echo "Unknown model_path: $model_path"; exit 1 ;;
esac

# ============================================================
# 3. EXPERIMENT CONFIG  (must match the pruning run)
# ============================================================
num_samples=100
gs=7.0
ATTN_PROTECT_SINGLE=0.25
SPARSITIES=(0.3 0.2)
prune_method="Isometric-Structured" #"OBS-Diff-Structured" #"Isometric-Structured"     # or OBS-Diff-Structured
saliency_mode=max_norm #max_norm #magnitude #mean_norm #neumann #magnitude                   # or mean_norm — MUST match pruning run
correction_mode=robust #robust #spectral
n_tbins=4
seed=24432 #24432 #0                                   # generation seed (independent of calib seed)

# ============================================================
# 4. PATH ROUTING  (mirrors the pruning script EXACTLY)
# ============================================================
is_flux=false
[[ "$model_path" == *black-forest-labs/FLUX.* ]] && is_flux=true

# trailing-zero fallbacks (0.80 -> 0.8) in case bash printed either form
FFN_SHORT=${FFN_PROTECT%0};  ATTN_SHORT=${ATTN_PROTECT%0}

if [[ "$prune_method" == Isometric-Structured ]]; then
    # base=${prune_method}/${model_path}/saliency_mode_${saliency_mode}/GS${gs}/correction_mode${correction_mode}/num_samples_${num_samples}
    base=${prune_method}_fresh/${model_path}/saliency_mode_${saliency_mode}/timestep_weight_strategyuniform/GS${gs}/correction_mode${correction_mode}_no_bandwidth/robust_dro/num_samples_${num_samples}
    # base=${prune_method}_fresh/${model_path}/saliency_mode_${saliency_mode}/GS${gs}/correction_mode${correction_mode}_no_bandwidth/num_samples_${num_samples}

    # base=${prune_method}/${model_path}/saliency_mode_${saliency_mode}/GS${gs}/num_samples_${num_samples}

    cfg_exact=blocksize${BLOCKSIZE}_n_eigen${N_EIGEN}_ffn${FFN_PROTECT}_attn${ATTN_PROTECT}
    cfg_short=blocksize${BLOCKSIZE}_n_eigen${N_EIGEN}_ffn${FFN_SHORT}_attn${ATTN_SHORT}
    if $is_flux; then
        cfg_exact=${cfg_exact}_attn_single${ATTN_PROTECT_SINGLE}
        cfg_short=${cfg_short}_attn_single${ATTN_PROTECT_SINGLE}
    fi

    out_dir_exact=new_output/${base}/${cfg_exact}
    log_dir_exact=new_log_output/${base}/${cfg_exact}
    out_dir_short=new_output/${base}/${cfg_short}
    log_dir_short=new_log_output/${base}/${cfg_short}
else  # OBS-Diff-Structured
    out_dir_exact=new_output/OBS_Diff_Structured/${model_path}/new/num_samples_${num_samples}
    log_dir_exact=new_log_output/OBS_Diff_Structured/${model_path}/new/num_samples_${num_samples}
    out_dir_short=$out_dir_exact
    log_dir_short=$log_dir_exact
fi

# ============================================================
# 5. EVAL LOOP
# ============================================================
for sparsity in "${SPARSITIES[@]}"; do
    echo "=========================================================="
    echo "Eval: ${prune_method} | ${model_path} | sp${sparsity} | ${saliency_mode}"
    echo "=========================================================="

    ckpt_name="pruned_model_ricci_prune_structured_nsamples${num_samples}_${sparsity}_iters1.pth"

    if [ -f "${log_dir_exact}/sp${sparsity}/${ckpt_name}" ]; then
        pruned_model="${log_dir_exact}/sp${sparsity}/${ckpt_name}"
        demo_dir="${out_dir_exact}/sp${sparsity}/seed${seed}"
        echo "Found checkpoint (exact formatting)."
    elif [ -f "${log_dir_short}/sp${sparsity}/${ckpt_name}" ]; then
        pruned_model="${log_dir_short}/sp${sparsity}/${ckpt_name}"
        demo_dir="${out_dir_short}/sp${sparsity}/seed${seed}"
        echo "Found checkpoint (short formatting)."
    else
        echo "CRITICAL ERROR: checkpoint not found"
        echo "  Path 1: ${log_dir_exact}/sp${sparsity}/${ckpt_name}"
        echo "  Path 2: ${log_dir_short}/sp${sparsity}/${ckpt_name}"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model_path "${model_path}" \
        --cache_dir /local/a/pmaletti/diffusion_weightss \
        --prune_method ${prune_method} \
        --seed ${seed} \
        --sparsity_ratio "${sparsity}" \
        --num_inference_steps ${num_inference_steps} \
        --batch_size ${batch_size} \
        --height 1024 --width 1024 \
        --guidance_scale ${guidance_scale} \
        --pruned_model "${pruned_model}" \
        --demo_dir "${demo_dir}" \
        --test_generation_only \
        --num_gen_samples "${num_gen_samples}" \
        --vis_only \
        # --run_base_model
    echo ""
done