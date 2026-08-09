import time 
import heapq 
import torch 
import torch.nn as nn 
from .IsometricDiffusionPruner_Structured_ import IsometricDiffusionPruner_Structured, IsometricDiffusionJointPruner_Structured
from collections import defaultdict
from .dataloader import get_loaders 
from collections import OrderedDict
import numpy as np
import torch_pruning as tp
import math
import transformers
from .collect_full_grams import (
    load_grams, attach_grams_to_pruner_dict,
)

dependents = { "all" : ["attn.to_q", "attn.to_k", "attn.to_v","attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "ff.net.0.proj", "ff_context.net.0.proj"],
    "ff.net.2" : ["ff.net.0.proj"],
    "ff_context.net.2" : ["ff_context.net.0.proj"],
    "attn.to_out.0" : ["attn.to_q", "attn.to_k", "attn.to_v","attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"]
}

def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def _hinv_diag_neumann(H, eps=1e-8):
    d = torch.diagonal(H).clamp(min=eps)
    row_sq = (H ** 2).sum(dim=1)
    offdiag = (row_sq - d ** 2).clamp(min=0.0)
    ratio = (offdiag / (d ** 2)).clamp(max=1.0)     # ← ADD: keep series valid
    return ((1.0 / d) * (1.0 + ratio)).clamp(min=eps)

def _hinv_diag_effective_resistance(H, blocksize=256, k_ratio=0.99):
    """
    Computes the Effective Resistance (diagonal of the spectral pseudo-inverse) block-by-block.
    Mirrors the spectral embedding approach: embd = evecs * (1 / sqrt(evals)).
    """
    d_in = H.shape[0]
    R_diag = torch.zeros(d_in, device=H.device, dtype=H.dtype)
    
    for i in range(0, d_in, blocksize):
        end = min(i + blocksize, d_in)
        H_block = H[i:end, i:end]
        
        # 1. Spectral decomposition of the block
        evals, evecs = torch.linalg.eigh(H_block)
        evals = evals.flip(0); evecs = evecs.flip(1)
        
        # 2. Spectral truncation (drop noise / near-zero modes)
        total_energy = torch.sum(evals)
        cumulative_energy = torch.cumsum(evals, dim=0)
        k = torch.searchsorted(cumulative_energy, k_ratio * total_energy).item() + 1
        
        # 3. Create the embedding decay (1.0 / sqrt(lambda)) 
        # and square it immediately (1.0 / lambda) for the squared norm
        lambda_inv = 1.0 / evals[:k].clamp(min=1e-8)
        
        # 4. Squared norm of the spectral embedding (Effective Resistance)
        R_block = torch.sum((evecs[:, :k] ** 2) * lambda_inv[None, :], dim=1)
        
        R_diag[i:end] = R_block
        
    return R_diag

def check_sparsity(transformer_model, target_names, logger=None):
    logger.info("\n" + "="*50)
    logger.info("Calculating Sparsity for Target Modules...")
    logger.info("="*50)

    all_linear_layers = find_layers(transformer_model)

    individual_sparsities = OrderedDict()
    
    total_target_params = 0
    total_target_zeros = 0


    for layer_name, layer_module in all_linear_layers.items():
        is_target = False
        for target_suffix in target_names:
            if layer_name.endswith(target_suffix):
                is_target = True
                break
        
        if is_target:
            weight = layer_module.weight
            
            if weight.numel() > 0:
                zeros = torch.sum(weight == 0).item()
                total = weight.numel()
                sparsity_percentage = (zeros / total) * 100 if total > 0 else 0
                
                individual_sparsities[layer_name] = sparsity_percentage
                
                total_target_zeros += zeros
                total_target_params += total

    logger.info("--- Individual Module Sparsity ---")
    if not individual_sparsities:
        logger.info("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            logger.info(f"{name:<50s} | Sparsity: {sparsity:.4f}%")
            
    logger.info("--- Aggregate Sparsity for Targets ---")
    if total_target_params > 0:
        aggregate_percentage = (total_target_zeros / total_target_params) * 100
        logger.info(f"Total Parameters in Target Modules : {total_target_params}")
        logger.info(f"Zero Parameters in Target Modules  : {total_target_zeros}")
        logger.info(f"Overall Sparsity of Target Modules : {aggregate_percentage:.4f}%")
    else:
        logger.info("No parameters found in target modules.")
    
    logger.info("\n" + "="*50)
    logger.info("Calculation finished.")
    logger.info("="*50)

    return aggregate_percentage

def check_size(transformer_model, target_names, logger, dense_shapes, args):
    """
    Checks the true structured sparsity of a physically pruned model by 
    comparing current parameter counts against a dense baseline snapshot.
    """
    logger.info("\n" + "="*50)
    logger.info("Checking Size and Structured Sparsity...")
    logger.info("="*50)

    all_linear_layers = find_layers(transformer_model)

    individual_sparsities = OrderedDict()
    individual_sparsities_structured = OrderedDict()
    individual_sparsities_bias = OrderedDict()
    
    total_target_current = 0
    total_target_original = 0
    
    total_model_current = 0
    total_model_original = 0

    # ==========================================
    # UNIVERSAL DEPENDENCY EXPANSION
    # ==========================================
    expanded_target_names = list(target_names)
    
    # 1. FFN Dependencies
    if "ff.net.2" in target_names and "ff.net.0.proj" not in expanded_target_names:
        expanded_target_names.append("ff.net.0.proj")
    if "ff_context.net.2" in target_names and "ff_context.net.0.proj" not in expanded_target_names:
        expanded_target_names.append("ff_context.net.0.proj")
        
    # 2. Attention Dependencies (Dynamically handles attn, attn1, and attn2)
    attn_targets = [t for t in target_names if "to_out.0" in t]

    if "proj_out" in target_names:
        for suffix in ["proj_mlp", "attn.to_q", "attn.to_k", "attn.to_v"]:
            if suffix not in expanded_target_names:
                expanded_target_names.append(suffix)
                 
    for attn_target in attn_targets:
        # Extract the exact prefix (e.g., 'attn1' from 'attn1.to_out.0')
        prefix = attn_target.split('.to_out')[0]
        
        # Base QKV projections (SDXL, SD3, Flux)
        base_suffixes = [f"{prefix}.to_q", f"{prefix}.to_k", f"{prefix}.to_v"]
        # Joint-Attention residual projections (SD3.5 specific)
        sd3_suffixes = [f"{prefix}.to_add_out", f"{prefix}.add_k_proj", f"{prefix}.add_q_proj", f"{prefix}.add_v_proj"]
        
        for suffix in base_suffixes + sd3_suffixes:
            if suffix not in expanded_target_names:
                expanded_target_names.append(suffix)

    # 1. CALCULATE GLOBAL MODEL SPARSITY
    for name, param in transformer_model.named_parameters():
        if param is not None and param.numel() > 0:
            current_count = param.numel()
            total_model_current += current_count
            
            # Fetch the original size from our snapshot
            if name in dense_shapes:
                original_count = torch.prod(torch.tensor(dense_shapes[name])).item()
                total_model_original += original_count
            else:
                total_model_original += current_count # Fallback

    # 2. CALCULATE TARGET MODULE SPARSITY
    for layer_name, layer_module in all_linear_layers.items():
        is_target = any(layer_name.endswith(suffix) for suffix in expanded_target_names)
        
        if is_target:
            weight = layer_module.weight
            
            if weight is not None and weight.numel() > 0:
                # Map the module name back to the named_parameters key
                weight_name = f"{layer_name}.weight" 
                
                current_total = weight.numel()
                
                if weight_name in dense_shapes:
                    original_total = torch.prod(torch.tensor(dense_shapes[weight_name])).item()
                else:
                    original_total = current_total # Fallback
                
                # Sparsity = (Original - Current) / Original
                pruned_params = original_total - current_total
                sparsity_percentage = (pruned_params / original_total) * 100 if original_total > 0 else 0

                individual_sparsities[layer_name] = sparsity_percentage
                individual_sparsities_structured[layer_name] = layer_module.weight.shape
                individual_sparsities_bias[layer_name] = layer_module.bias.shape if getattr(layer_module, 'bias', None) is not None else None
                
                total_target_current += current_total
                total_target_original += original_total

    # --- LOGGING EXECUTIONS ---
    logger.info("\n--- Individual Target Module Shapes ---")
    if not individual_sparsities:
        logger.info("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            b_shape = individual_sparsities_bias[name]
            b_str = str(b_shape) if b_shape else "No Bias"
            # Printing shape, bias shape, and local structural sparsity percentage
            logger.info(f"{name:<50s} | W: {str(individual_sparsities_structured[name]):<20s} | B: {b_str:<15s} | Sparsity: {sparsity:>5.2f}%")
            
    logger.info("\n--- Overall Structured Sparsity Metrics ---")
    
    if total_target_original > 0:
        target_pruned = total_target_original - total_target_current
        target_sparsity_pct = (target_pruned / total_target_original) * 100
        logger.info(f"Target Modules Original Params : {total_target_original:,}")
        logger.info(f"Target Modules Current Params  : {total_target_current:,}")
        logger.info(f"Target Modules Sparsity        : {target_sparsity_pct:.2f}%")
    else:
        logger.info("Target Modules Sparsity: N/A")

    if total_model_original > 0:
        global_pruned = total_model_original - total_model_current
        global_sparsity_pct = (global_pruned / total_model_original) * 100
        logger.info(f"Global Model Original Params   : {total_model_original:,}")
        logger.info(f"Global Model Current Params    : {total_model_current:,}")
        logger.info(f"GLOBAL MODEL SPARSITY          : {global_sparsity_pct:.2f}%")
    else:
        logger.info("Global Model Sparsity: N/A")

    logger.info("\n" + "="*50)
    logger.info("Calculation finished.")
    logger.info("="*50)

def group_modules_with_parallelism(target_pruned_modules, num_groups):
    # 1. First, group all modules by their block_idx
    modules_by_block = defaultdict(list)
    for block_idx, name in target_pruned_modules:
        modules_by_block[block_idx].append((block_idx, name))
    
    # 2. Sort the blocks to maintain the sequential order of the transformer
    sorted_block_indices = sorted(modules_by_block.keys())
    num_blocks = len(sorted_block_indices)
    
    if num_blocks == 0:
        return []

    # 3. Calculate how many BLOCKS go into each group
    group_size = num_blocks // num_groups
    remainder = num_blocks % num_groups
    
    if group_size == 0:
        group_size = 1
        num_groups = num_blocks
        remainder = 0
    
    final_groups = []
    start_idx = 0
    for i in range(num_groups):
        # Determine which blocks fall into this group
        count = group_size + (1 if i < remainder else 0)
        end_idx = start_idx + count
        
        current_block_indices = sorted_block_indices[start_idx:end_idx]
        
        # Flatten all modules from these specific blocks into the group
        flat_group = []
        for b_idx in current_block_indices:
            flat_group.extend(modules_by_block[b_idx])
            
        final_groups.append(flat_group)
        start_idx = end_idx

    return final_groups

def group_modules_with_parallelism_flux(target_pruned_modules, num_groups):
    # 1. Group all modules by their logical sequential order
    modules_by_block = defaultdict(list)
    
    for block_type, block_idx, name in target_pruned_modules:
        # Flux sequential order: 'double' blocks run before 'single' blocks.
        # We assign priority 0 to double, 1 to single to enforce this order.
        sort_priority = 0 if block_type == 'double' else 1
        sort_key = (sort_priority, block_idx)
        
        # Append the original 3-tuple
        modules_by_block[sort_key].append((block_type, block_idx, name))
        
    # 2. Sort the blocks to maintain the correct forward-pass sequence
    sorted_block_keys = sorted(modules_by_block.keys())
    num_blocks = len(sorted_block_keys)
    
    if num_blocks == 0:
        return []

    # 3. Calculate how many BLOCKS go into each group
    group_size = num_blocks // num_groups
    remainder = num_blocks % num_groups
    
    if group_size == 0:
        group_size = 1
        num_groups = num_blocks
        remainder = 0
        
    final_groups = []
    start_idx = 0
    
    for i in range(num_groups):
        # Determine which blocks fall into this group
        count = group_size + (1 if i < remainder else 0)
        end_idx = start_idx + count
        
        current_block_keys = sorted_block_keys[start_idx:end_idx]
        
        # Flatten all modules from these specific blocks into the group
        flat_group = []
        for b_key in current_block_keys:
            flat_group.extend(modules_by_block[b_key])
            
        final_groups.append(flat_group)
        start_idx = end_idx

    return final_groups

def create_block_input_hook(block_idx, block_inputs):
    # Use a pre_hook to capture the state BEFORE any internal norm/processing
    def hook(module, args, kwargs):
        # MMDiT blocks usually have hidden_states (image) as the first positional arg
        # or as a keyword.
        hidden_states = None
        
        if len(args) > 0:
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            
        if hidden_states is not None:
            # We detach to avoid keeping the graph in memory during collection
            block_inputs[block_idx] = hidden_states.detach()
        else:
            # Fallback for the 'context' stream if you are pruning context layers
            context_states = kwargs.get("encoder_hidden_states", None)
            block_inputs[f"{block_idx}_context"] = context_states.detach() if context_states is not None else None
            
    return hook

def create_hook_fn(block_idx, layer_name, pruner_dict, timestep_weight, block_inputs=None):
    def hook_fn(module, input, output):
        step = step_info["current"]
        pruner = pruner_dict[(block_idx, layer_name)]

        input_data = input[0].data

        safe_step = min(step, len(timestep_weight) - 1)
        current_weight = timestep_weight[safe_step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples

        input_data = input_data * np.sqrt(current_weight)

        pruner.add_batch(input_data, output.data, W_new)

    return hook_fn

def create_joint_hook_fn(block_idx, layer_name, pruner_dict, timestep_weight, block_inputs=None):
    def hook_fn(module, input, output):
        step = step_info["current"]
        
        # --- THE JOINT PRUNER ROUTING FIX ---
        # If this hook is attached to the text stream, it must fetch the master image pruner
        is_text_stream = (layer_name == "attn.to_add_out")
        master_key = "attn.to_out.0" if is_text_stream else layer_name
        
        pruner = pruner_dict[(block_idx, master_key)]

        input_data = input[0].data

        safe_step = min(step, len(timestep_weight) - 1)
        current_weight = timestep_weight[safe_step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples

        input_data = input_data * np.sqrt(current_weight)

        # Pass the routing flag into the Joint Pruner's add_batch method
        pruner.add_batch(input_data, output.data, W_new, is_text_stream=is_text_stream)

    return hook_fn

def create_flux_hook_fn(block_idx, layer_name, pruner_dict, timestep_weight, block_inputs=None):
    def hook_fn(module, input, output):
        # Assumes step_info is accessible globally as in your snippet
        step = step_info["current"]
        pruner = pruner_dict[(block_idx, layer_name)]

        input_data = input[0].data

        safe_step = min(step, len(timestep_weight) - 1)
        current_weight = timestep_weight[safe_step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples

        # --- THE MODALITY-BALANCED HESSIAN FIX FOR SINGLE BLOCKS ---
        # By applying a spatial scaling multiplier before add_batch, we force the 
        # underlying X^T X accumulation to evaluate text and image tokens at exactly 50/50 power.
        if "single" in str(block_idx) and layer_name == "proj_out":
            txt_len = 512 # Default for FLUX.1-dev (Change to 256 if using FLUX-schnell)
            
            # Ensure the sequence is long enough to be the concatenated representation
            if input_data.dim() == 3 and input_data.size(1) > txt_len:
                # Clone here to avoid destructively modifying the live activations during the forward pass
                input_data = input_data.clone()
                
                S_total = input_data.size(1)
                S_txt = txt_len
                S_img = S_total - txt_len
                
                # Mathematically scale the input data so that X^T X natively enforces 
                # a 50/50 balance between text and image variance when accumulated in add_batch.
                scale_txt = np.sqrt((0.5 * S_total) / S_txt)
                scale_img = np.sqrt((0.5 * S_total) / S_img)
                
                input_data[:, :txt_len, :] *= scale_txt
                input_data[:, txt_len:, :] *= scale_img

        # Base timestep weighting
        input_data = input_data * np.sqrt(current_weight)

        pruner.add_batch(input_data, output.data, W_new)

    return hook_fn

step_info = {"current": 0}

def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
    step_info["current"] += 1
    return callback_kwargs

@torch.no_grad()
def calculate_group_energy_threshold_sd3_5(args, pruner_dict, target_sparsity_ratio, dev, headsize=64, percdamp=0.1):
    """
    Calculates the exact global energy threshold by simulating the greedy pruning 
    process across the entire network, perfectly accounting for discrete chunks and armor.
    """
    all_units = []
    total_target_params = 0
    
    # 1. EXTRACT AND MEASURE ALL UNITS GLOBALLY
    for (block_idx, module_name), pruner in pruner_dict.items():
        W = pruner.layer.weight.data.clone().float()
        if isinstance(pruner.layer, nn.Conv2d): W = W.flatten(1)
        if isinstance(pruner.layer, transformers.Conv1D): W = W.t()
        
        d_out, d_in = W.shape
        H_float = torch.nan_to_num(pruner.H.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        is_ffn = any(keyword in module_name.lower() for keyword in ["ff.net", "ff_context.net", "mlp", "w_down", "down_proj", "net.2"])
        
        # --- THE MULTIMODAL FIX: Extract Text Stream parameters if they exist ---
        has_layer2 = hasattr(pruner, 'layer2') and pruner.layer2 is not None
        if has_layer2:
            W2 = pruner.layer2.weight.data.clone().float()
            if isinstance(pruner.layer2, transformers.Conv1D): W2 = W2.t()
            H2_float = torch.nan_to_num(pruner.H2.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        if is_ffn:
            current_headsize = 1
            W_H_project = torch.matmul(W, H_float)
            item_energies = torch.norm(W * W_H_project, dim=0) # [9728]

            block_limit_ratio = args.ffn_protect
            param_multiplier = 1
        else:
            current_headsize = headsize
            num_heads = d_in // current_headsize
            item_energies = torch.zeros(num_heads, device=dev)

            for h in range(num_heads):
                idx_start = h * current_headsize
                idx_end = (h + 1) * current_headsize
                
                # 1. Image Stream Energy
                W_h = W[:, idx_start:idx_end]                      
                H_h = H_float[idx_start:idx_end, idx_start:idx_end] 
                row_importance_img = torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1) 
                energy_img = torch.max(row_importance_img)
                
                # 2. Text Stream Energy
                energy_txt = 0.0
                if has_layer2:
                    W2_h = W2[:, idx_start:idx_end]                      
                    H2_h = H2_float[idx_start:idx_end, idx_start:idx_end] 
                    row_importance_txt = torch.norm(torch.matmul(W2_h, H2_h) * W2_h, dim=-1) 
                    energy_txt = torch.max(row_importance_txt)

                energy_img = energy_img / (energy_img.sum() + 1e-12)
                energy_txt = energy_txt / (energy_txt.sum() + 1e-12)

                item_energies[h] = math.sqrt((energy_img**2) + (energy_txt**2))
                
            block_limit_ratio = args.attn_protect
            param_multiplier = 2 if has_layer2 else 1

        # original
        if is_ffn:
            total_target_params += len(item_energies) * current_headsize
        else:
            total_target_params += len(row_importance_img)

        # --- CHUNK INTO LOCAL BLOCKS ---
        blocksize = args.blocksize
        if current_headsize > 1:
            blocksize = max(blocksize, current_headsize)
            blocksize = (blocksize // current_headsize) * current_headsize
            
        items_per_block = blocksize // current_headsize
        
        for i, energy in enumerate(item_energies):
            local_block_idx = i // items_per_block
            max_allowed = int(items_per_block * block_limit_ratio)
            
            all_units.append({
                'energy': energy.item(),
                'param_weight': 1, #unit_param_weight, #current_headsize, #1
                'block_key': f"{block_idx}_{module_name}_{local_block_idx}",
                'max_allowed': max_allowed
            })

    # 2. SIMULATE THE EXACT PRUNING PROCESS
    all_units.sort(key=lambda x: x['energy'])
    
    target_pruned_params = total_target_params * target_sparsity_ratio
    current_pruned_params = 0
    block_prune_counts = {}
    best_threshold = 0.0
    
    for unit in all_units:
        bk = unit['block_key']
        if bk not in block_prune_counts:
            block_prune_counts[bk] = 0
            
        if block_prune_counts[bk] < unit['max_allowed']:
            block_prune_counts[bk] += 1
            current_pruned_params += unit['param_weight']
            best_threshold = unit['energy'] 
            
            if current_pruned_params >= target_pruned_params:
                break

    print(f"\n" + "="*50)
    print(f"[GREEDY SIMULATOR REPORT]")
    print(f"Target Sparsity   : {target_sparsity_ratio:.4%}")
    print(f"Simulated Sparsity: {(current_pruned_params / total_target_params):.4%}")
    print(f"Target Params     : {int(target_pruned_params):,}")
    print(f"Simulated Pruned  : {int(current_pruned_params):,}")
    print("="*50 + "\n")

    return best_threshold

@torch.no_grad()
def calculate_pool_thresholds_sd3_5(args, pruner_dict, target_sparsity_ratio, dev,
                                    headsize=64, saliency_mode='neumann',
                                    pool_targets=None, logger=None):
    """
    Per-POOL threshold calculator for SD3.5 (MM-DiT) structured pruning.

    Returns a DICT {pool_key -> energy_threshold}, pool_key in {'ffn','attn'}.
    struct_prune must be called per layer with the threshold for THAT layer's pool.

    WHY PER-POOL (not one global, not per-individual-layer)
    -------------------------------------------------------
    * One global threshold pools FFN (W^2*diagH scale) with attention
      (norm/col_score scale); the scale gap skews the FFN/attn split by accident.
    * A per-INDIVIDUAL-LAYER threshold forces every layer to exactly target —
      that is uniform pruning and discards the non-uniform contribution.
    * Per-POOL keeps non-uniformity WITHIN each pool (across its layers, per block)
      while making the FFN-vs-attn allocation an explicit, controlled choice.

    SD3.5 needs only two pools: attention is a single uniform 64-dim family,
    so no per-dimension split (unlike SDXL).

    pool_targets (optional): {'ffn': 0.30, 'attn': 0.10} to prune redundant FFN
    harder and spare attention. Missing pools fall back to target_sparsity_ratio.

    saliency_mode MUST match struct_prune's live selection:
      'neumann'   -> ffn col : sum(W^2,0) / diag(H^-1_neumann)
                     attn head: sqrt(mean(col_img)^2 + mean(col_txt)^2)
      'mean_norm' -> ffn col : sum(W^2,0) * diag(H)
                     attn head: sqrt(mean(norm_img)^2 + mean(norm_txt)^2)
    """
    assert saliency_mode in ('neumann', 'mean_norm')
    from collections import defaultdict

    pools      = defaultdict(list)   # 'ffn'/'attn' -> [unit, ...]
    pool_total = defaultdict(int)

    for (block_idx, module_name), pruner in pruner_dict.items():
        W = pruner.layer.weight.data.clone().float()
        if isinstance(pruner.layer, nn.Conv2d):           W = W.flatten(1)
        if isinstance(pruner.layer, transformers.Conv1D): W = W.t()

        d_out, d_in = W.shape
        H = torch.nan_to_num(pruner.H.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        is_ffn = any(k in module_name.lower() for k in
                     ["ff.net", "ff_context.net", "mlp", "w_down", "down_proj", "net.2"])

        has_layer2 = hasattr(pruner, 'layer2') and pruner.layer2 is not None
        if has_layer2:
            W2 = pruner.layer2.weight.data.clone().float()
            if isinstance(pruner.layer2, transformers.Conv1D): W2 = W2.t()
            H2 = torch.nan_to_num(pruner.H2.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        # ---------------- FFN ----------------
        if is_ffn:
            current_headsize = 1
            if saliency_mode == 'neumann':
                item_energies = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
            else:
                item_energies = torch.sum(W ** 2, dim=0) * torch.diagonal(H)
            block_limit_ratio = args.ffn_protect
            pool_key = 'ffn'

        # ---------------- attention (joint image + text) ----------------
        else:
            current_headsize = headsize
            num_heads = d_in // current_headsize
            item_energies = torch.zeros(num_heads, device=dev)

            if saliency_mode == 'neumann':
                col  = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
                col2 = (torch.sum(W2 ** 2, dim=0) / _hinv_diag_neumann(H2)) if has_layer2 else None
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    ei = col[s:e].mean()
                    et = col2[s:e].mean() if col2 is not None else 0.0
                    item_energies[h] = torch.sqrt(ei ** 2 + (et ** 2 if has_layer2 else 0.0))
            else:
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    W_h = W[:, s:e]; H_h = H[s:e, s:e]
                    ei = torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1).mean()   # ← mean, not max
                    et = torch.tensor(0.0, device=dev)
                    if has_layer2:
                        W2_h = W2[:, s:e]; H2_h = H2[s:e, s:e]
                        et = torch.norm(torch.matmul(W2_h, H2_h) * W2_h, dim=-1).mean()  # ← mean
                    item_energies[h] = torch.sqrt(ei ** 2 + et ** 2)

            block_limit_ratio = args.attn_protect
            pool_key = 'attn'

        blocksize = args.blocksize
        if current_headsize > 1:
            blocksize = max(blocksize, current_headsize)
            blocksize = (blocksize // current_headsize) * current_headsize
        items_per_block = blocksize // current_headsize

        for i, energy in enumerate(item_energies):
            pools[pool_key].append({
                'energy':       energy.item(),
                'param_weight': current_headsize,
                'block_id':     f"{block_idx}_{module_name}_{i // items_per_block}",
                'max_allowed':  int(items_per_block * block_limit_ratio),
            })
            pool_total[pool_key] += current_headsize

    # ---- struct_prune-faithful selection + bisection, PER POOL ----
    def _prune_at(us_sorted, T):
        counts, weight, max_e = {}, 0, 0.0
        for u in us_sorted:
            if u['energy'] > T:
                break
            c = counts.get(u['block_id'], 0)
            if c < u['max_allowed']:
                counts[u['block_id']] = c + 1
                weight += u['param_weight']
                if u['energy'] > max_e:
                    max_e = u['energy']
        return weight, max_e

    def _solve(units, tgt_weight):
        us = sorted(units, key=lambda x: x['energy'])
        if not us or tgt_weight <= 0:
            return 0.0, 0
        lo, hi = 0.0, us[-1]['energy']
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            w, _ = _prune_at(us, mid)
            if w <= tgt_weight:
                lo = mid
            else:
                hi = mid
        w, thr = _prune_at(us, lo)
        return thr, w

    def _ceiling(units):
        seen = {}
        for u in units:
            seen[u['block_id']] = (u['max_allowed'], u['param_weight'])
        return sum(m * w for m, w in seen.values())

    pool_targets = pool_targets or {}
    thresholds   = {}
    grand_total = grand_pruned = 0

    log = logger.info if logger else print
    log("\n" + "=" * 60)
    log(f"[SD3.5 POOL THRESHOLDS]  mode={saliency_mode}  target={target_sparsity_ratio:.4%}")
    log("-" * 60)

    for key in ('attn', 'ffn'):
        units = pools.get(key, [])
        if not units:
            thresholds[key] = 0.0
            continue
        total_w   = pool_total[key]
        tgt_ratio = pool_targets.get(key, target_sparsity_ratio)
        tgt_w     = total_w * tgt_ratio

        thr, w = _solve(units, tgt_w)
        thresholds[key] = thr + 1e-9

        grand_total  += total_w
        grand_pruned += w
        ceil = _ceiling(units)
        flag = f"  <-- CAP-BOUND ceiling {ceil/total_w:.2%}" if tgt_w > ceil else ""
        log(f"{key:<6} | target {tgt_ratio:>6.2%} | cutoff {thr:>13.6f} "
            f"| gives {w/total_w:>6.2%}{flag}")

    if grand_total:
        log("-" * 60)
        log(f"{'GLOBAL':<6} | simulated {grand_pruned/grand_total:.4%} of "
            f"{grand_total/1e9:.3f}B prunable params")
    log("=" * 60 + "\n")

    return thresholds

@torch.no_grad()
def prune_SD_3_5_Structured(args, pipe, target_modules, dev, timestep_weight=None, logger=None):
    logger.info('Starting ...')
    dataloader = get_loaders(
        args.dataset,
        num_samples=args.num_samples  
    )

    blocks = pipe.transformer.transformer_blocks
    target_pruned_modules = []


    for i in range(args.minlayer, args.maxlayer):
        block = blocks[i]
        all_module_dict = find_layers(block)
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    logger.info(f"\n intelligently divided {num_modules} modules into {len(modules_groups)} groups:")

    for g_idx, group_modules in enumerate(modules_groups):
        logger.info(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        pruner_dict = {}
        hooks = []

        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]

            if module_name == "ff.net.2" or module_name == "ff_context.net.2":

                pruner_dict[(block_idx, module_name)] = IsometricDiffusionPruner_Structured(
                                                                module, 
                                                                block_idx, 
                                                                module_name, 
                                                                args, 
                                                                logger=logger
                                                        )

                hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))

            elif module_name == "attn.to_out.0":
                # pruner_dict[(block_idx, module_name)] = IsometricDiffusionPruner_Structured(
                #                                                 module, 
                #                                                 block_idx, 
                #                                                 module_name, 
                #                                                 args, 
                #                                                 logger=logger,
                #                                             )

                # hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
                # hooks.append(module.register_forward_hook(hook_fn))

                target_img_layer = module
                target_txt_layer = get_module_by_name(blocks[block_idx], "attn.to_add_out")

                # 1. Instantiate the Joint Pruner ONCE and store it under the master image key
                pruner_dict[(block_idx, module_name)] = IsometricDiffusionJointPruner_Structured(
                    layer=target_img_layer, 
                    layer_idx=block_idx, 
                    layer_name=module_name, 
                    args=args, 
                    logger=logger,
                    layer2=target_txt_layer,          # Pass the text layer
                    layer2_name="attn.to_add_out"
                )

                # 2. Register the Image Hook
                img_hook_fn = create_joint_hook_fn(block_idx, "attn.to_out.0", pruner_dict, timestep_weight)
                hooks.append(target_img_layer.register_forward_hook(img_hook_fn))

                # 3. Register the Text Hook
                txt_hook_fn = create_joint_hook_fn(block_idx, "attn.to_add_out", pruner_dict, timestep_weight)
                hooks.append(target_txt_layer.register_forward_hook(txt_hook_fn))

        logger.info(f"Running diffusion for group {g_idx + 1} to collect activations...")
        
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]

            current_bs = len(prompts)
            start_idx = i * batch_size

            # generators = [torch.Generator("cuda").manual_seed(args.seed + start_idx + j) for j in range(current_bs)]
                          
            logger.info(f"  Prompts {i}: {prompts}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                # generator=generators
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        for hook in hooks:
            hook.remove()

        if hasattr(pipe, "transformer"):
            native_head_dim = pipe.transformer.config.attention_head_dim

        elif hasattr(pipe, "unet"):
            raw_dim = pipe.unet.config.attention_head_dim
            native_head_dim = raw_dim[0] if isinstance(raw_dim, list) else raw_dim

        else:
            native_head_dim = 64

        logger.info("Calculating non-uniform sparsity threshold for group...")
        # global_threshold = calculate_group_energy_threshold_sd3_5(
        #     args=args,
        #     pruner_dict=pruner_dict, 
        #     target_sparsity_ratio=args.sparsity_ratio, 
        #     dev=dev, 
        #     headsize=native_head_dim
        # )
        pool_thresholds = calculate_pool_thresholds_sd3_5(
            args=args,
            pruner_dict=pruner_dict, 
            target_sparsity_ratio=args.sparsity_ratio, 
            dev=dev, 
            headsize=native_head_dim,
            saliency_mode=args.saliency_mode,
            logger=logger
        )
        # logger.info(f"Group Energy Threshold: {global_threshold:.6f}")
        logger.info(f"Group Energy Threshold: {pool_thresholds}")
        
        # Execute pruning for this group
        # print(f"Pruning group {g_idx + 1}...")
        logger.info(f"Pruning group {g_idx + 1}...")

        master_targets = {"ff_context.net.2", "ff.net.2", "attn.to_out.0"}
        target_modules_prune = [
            (idx, name) for idx, name in group_modules 
            if name in master_targets
        ]

        for block_idx, module_name in target_modules_prune:
            # Skip manual execution of to_add_out since it is handled jointly with to_out.0
            if module_name == "attn.to_add_out":
                continue
            
            is_ffn = any(k in module_name.lower() for k in ["ff.net","ff_context.net","net.2"])
            thr = pool_thresholds['ffn'] if is_ffn else pool_thresholds['attn']
            logger.info(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            if module_name == "attn.to_out.0":
                idx_1 = pruner_dict[(block_idx, module_name)].struct_prune(
                    target_sparsity=sparsity,
                    energy_threshold=thr, #global_threshold,
                    headsize=native_head_dim,
                    iterations = args.ricci_flow_iters,
                )
                
            elif module_name == "ff.net.2" or module_name == "ff_context.net.2":
                idx = pruner_dict[(block_idx, module_name)].struct_prune(
                    target_sparsity=sparsity,
                    energy_threshold=thr, #global_threshold,
                    headsize=1, #128
                    iterations = args.ricci_flow_iters,
                )

            pruner_dict[(block_idx, module_name)].free()
            
            target_layer = get_module_by_name(blocks[block_idx], module_name)

            if module_name == "ff.net.2":
                # target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                # id = idx.tolist()

                # tp.prune_linear_in_channels(target_layer, id)
                # tp.prune_linear_out_channels(target_layer_in, id)

                target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                id = idx.tolist()

                # BUG 2 FIX: The SwiGLU / GEGLU index mirroring
                intermediate_dim = target_layer.weight.shape[1] 
                glu_id = id + [i + intermediate_dim for i in id]

                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, glu_id)
                
            # --- CONTEXT FFN BLOCK ---
            if module_name == "ff_context.net.2":
                # target_layer_in = get_module_by_name(blocks[block_idx], "ff_context.net.0.proj")
                # id = idx.tolist()

                # tp.prune_linear_in_channels(target_layer, id)
                # tp.prune_linear_out_channels(target_layer_in, id)

                target_layer_in = get_module_by_name(blocks[block_idx], "ff_context.net.0.proj")
                id = idx.tolist()

                # BUG 2 FIX: The SwiGLU / GEGLU index mirroring
                intermediate_dim = target_layer.weight.shape[1] 
                glu_id = id + [i + intermediate_dim for i in id]

                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, glu_id)

            # --- ATTENTION BLOCK ---
            if module_name == 'attn.to_out.0':
                target_add_layer = get_module_by_name(blocks[block_idx], "attn.to_add_out")
                target_q_layer = get_module_by_name(blocks[block_idx], "attn.to_q")
                target_k_layer = get_module_by_name(blocks[block_idx], "attn.to_k")
                target_v_layer = get_module_by_name(blocks[block_idx], "attn.to_v")
                target_add_k_layer = get_module_by_name(blocks[block_idx], "attn.add_k_proj")
                target_add_q_layer = get_module_by_name(blocks[block_idx], "attn.add_q_proj")
                target_add_v_layer = get_module_by_name(blocks[block_idx], "attn.add_v_proj")
                
                id_list = idx_1.tolist()
                tp.prune_linear_in_channels(target_layer, id_list)

                tp.prune_linear_in_channels(target_add_layer, id_list)
                tp.prune_linear_out_channels(target_q_layer, id_list)
                tp.prune_linear_out_channels(target_k_layer, id_list)
                tp.prune_linear_out_channels(target_v_layer, id_list)
                tp.prune_linear_out_channels(target_add_k_layer, id_list)
                tp.prune_linear_out_channels(target_add_q_layer, id_list)
                tp.prune_linear_out_channels(target_add_v_layer, id_list)

                logger.info(f"Previous heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")
                
                pipe.transformer.transformer_blocks[block_idx].attn.heads -= len(id_list) // 64
                logger.info(f"Current heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")

        torch.cuda.empty_cache()
        logger.info(f"Group {g_idx + 1} pruning completed.")

@torch.no_grad()
def calculate_group_energy_threshold_sdxl(args, pruner_dict, target_sparsity_ratio, dev,
                                          blocks, headsize=64, saliency_mode='neumann', logger=None):
    """
    Global-budget SINGLE-threshold calculator for SDXL structured pruning.
    Returns one scalar energy threshold, applied by struct_prune to every target
    layer (attn + ffn) with a per-block cap.

    FIX vs previous version
    -----------------------
    The old version walked a GLOBAL param budget and read the last-pruned energy as
    the threshold. struct_prune applies that threshold PER BLOCK with no global
    budget, so blocks the walk left unfinished were fully pruned -> ~5% overshoot.
    This version BISECTS a single T under the EXACT per-block-capped rule
    struct_prune runs, so realized sparsity == target.

    saliency_mode MUST match struct_prune's selection:
      'neumann'   -> ffn col : sum(W^2,0) / diag(H^-1_neumann)
                     attn head: mean(col_score over head)          [shared-class default]
      'mean_norm' -> ffn col : sum(W^2,0) * diag(H)
                     attn head: max(norm(W_h @ H_h * W_h, dim=-1)) [old SDXL path]
    """
    assert saliency_mode in ('neumann', 'mean_norm')

    all_units = []
    total_target_params = 0

    for (block_idx, module_name), pruner in pruner_dict.items():
        W = pruner.layer.weight.data.clone().float()
        if isinstance(pruner.layer, nn.Conv2d):           W = W.flatten(1)
        if isinstance(pruner.layer, transformers.Conv1D): W = W.t()

        d_out, d_in = W.shape
        H = torch.nan_to_num(pruner.H.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        is_ffn = "ff.net" in module_name.lower()

        # ---- saliency + geometry (mode-switched, mirrors struct_prune) ----
        if is_ffn:
            current_headsize = 1
            if saliency_mode == 'neumann':
                item_energies = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
            else:
                item_energies = torch.sum(W ** 2, dim=0) * torch.diagonal(H)
            block_limit_ratio = args.ffn_protect
            blocksize = d_in // 4
        else:
            attn_prefix     = module_name.split('.')[0]
            sample_attn     = getattr(blocks[block_idx], attn_prefix)
            current_headsize = d_in // sample_attn.heads
            num_heads       = sample_attn.heads
            item_energies   = torch.zeros(num_heads, device=dev)

            if saliency_mode == 'neumann':
                col_score = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    item_energies[h] = col_score[s:e].mean()
            else:
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    W_h = W[:, s:e]; H_h = H[s:e, s:e]
                    row_importance = torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1)
                    # item_energies[h] = torch.max(row_importance)
                    item_energies[h] = torch.mean(row_importance)

            block_limit_ratio = args.attn_protect
            blocksize = d_in

        total_target_params += len(item_energies) * current_headsize

        if current_headsize > 1:
            blocksize = max(blocksize, current_headsize)
            blocksize = (blocksize // current_headsize) * current_headsize
        items_per_block = blocksize // current_headsize

        for i, energy in enumerate(item_energies):
            local_block_idx = i // items_per_block
            all_units.append({
                'energy':       energy.item(),
                'param_weight': current_headsize,
                'block_id':     f"{block_idx}_{module_name}_{local_block_idx}",
                'max_allowed':  int(items_per_block * block_limit_ratio),
            })

    # ================= faithful bisection over ONE global threshold =========
    us = sorted(all_units, key=lambda x: x['energy'])
    if not us:
        return 0.0

    def _prune_at(T):
        counts, weight, max_e = {}, 0, 0.0
        for u in us:
            if u['energy'] > T:
                break                               # ascending: nothing else qualifies
            bid = u['block_id']
            c   = counts.get(bid, 0)
            if c < u['max_allowed']:                # per-block cap == struct_prune's cap
                counts[bid] = c + 1
                weight += u['param_weight']
                if u['energy'] > max_e:
                    max_e = u['energy']
        return weight, max_e

    target_pruned = total_target_params * target_sparsity_ratio
    if target_pruned <= 0:
        return 0.0

    lo, hi = 0.0, us[-1]['energy']
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        w, _ = _prune_at(mid)
        if w <= target_pruned:
            lo = mid
        else:
            hi = mid
    pruned_weight, threshold = _prune_at(lo)

    # cap-bound diagnostic
    seen = {}
    for u in all_units:
        seen[u['block_id']] = (u['max_allowed'], u['param_weight'])
    ceiling = sum(m * w for m, w in seen.values())

    logger.info("\n" + "=" * 50)
    logger.info("[GREEDY SIMULATOR REPORT]  (faithful bisection)")
    logger.info(f"Mode              : {saliency_mode}")
    logger.info(f"Target Sparsity   : {target_sparsity_ratio:.4%}")
    logger.info(f"Simulated Sparsity: {pruned_weight / total_target_params:.4%}")
    logger.info(f"Threshold         : {threshold:.6f}")
    logger.info(f"Target Params     : {int(target_pruned):,}")
    logger.info(f"Simulated Pruned  : {int(pruned_weight):,}")
    if target_pruned > ceiling:
        logger.info(f"  <-- CAP-BOUND: ceiling {ceiling / total_target_params:.2%}")
    logger.info("=" * 50 + "\n")

    return threshold + 1e-9

def _sdxl_pool_key(is_ffn, d_in):
    """
    Pool identity for SDXL threshold solving. MUST be reconstructible at
    execution time from (is_ffn, d_in) alone so struct_prune can look up the
    right threshold for each layer.
      FFN            -> ('ffn',)              (all FFN compete together)
      attention d_in -> ('attn', d_in)        (self-attn 1280 vs cross-attn 768
                                               land in SEPARATE pools — this is
                                               what kills the 0/40 bimodality)
    """
    return ('ffn',) if is_ffn else ('attn', d_in)


@torch.no_grad()
def calculate_pool_thresholds_sdxl(args, pruner_dict, target_sparsity_ratio, dev,
                                   blocks, headsize=64, saliency_mode='neumann',
                                   pool_targets=None, logger=None):
    """
    Per-pool threshold calculator for SDXL structured pruning.

    Returns a DICT {pool_key -> energy_threshold}, where pool_key comes from
    _sdxl_pool_key(is_ffn, d_in). struct_prune must be called per layer with the
    threshold for THAT layer's pool.

    WHY PER-POOL (vs the old single global threshold)
    -------------------------------------------------
    SDXL attention comes in two dimension families (self-attn d_in=1280,
    cross-attn d_in=768) whose per-head saliencies form two separated clusters.
    A single global threshold can only land below both (0%), between them (one
    family 0%, the other pinned at attn_protect), or above both (both at cap) —
    the 0/40 bimodality. FFN, on a third scale, gets sliced almost not at all.
    Solving EACH pool to its own target removes the cross-scale comparison, so:
      * each attention family prunes partially (no more 0/40),
      * FFN is solved UP to target instead of drifting at 2-6%,
      * no pool is driven to its cap, so the global overprune disappears.

    Non-uniform-across-pools control
    --------------------------------
    By default every pool targets `target_sparsity_ratio` (uniform across pools;
    non-uniformity still happens WITHIN a pool, per-block). To push the redundant
    FFN harder and spare attention — your method's thesis — pass e.g.
        pool_targets={('ffn',): 0.30, ('attn', 1280): 0.10, ('attn', 768): 0.05}
    Any pool absent from pool_targets falls back to target_sparsity_ratio.

    saliency_mode MUST match struct_prune's live selection:
      'neumann'   -> ffn col : sum(W^2,0) / diag(H^-1_neumann)
                     attn head: mean(col_score over head)
      'mean_norm' -> ffn col : sum(W^2,0) * diag(H)
                     attn head: mean(norm(W_h @ H_h * W_h, dim=-1))
      'max_norm' -> ffn col : sum(W^2,0) * diag(H)
                     attn head: max(norm(W_h @ H_h * W_h, dim=-1))
      'magnitude' -> ffn col : sum(W^2,0)
                     attn head: mean(sum(W^2, dim=0))
    """
    assert saliency_mode in ('neumann', 'mean_norm', 'magnitude', 'max_norm', 'effective_resistance')
    from collections import defaultdict

    pools      = defaultdict(list)   # pool_key -> [unit, ...]
    pool_total = defaultdict(int)    # pool_key -> total prunable params

    # ------------------------------------------------------------------
    # 1. Build units, bucketed by pool
    # ------------------------------------------------------------------
    for (block_idx, module_name), pruner in pruner_dict.items():
        W = pruner.layer.weight.data.clone().float()
        if isinstance(pruner.layer, nn.Conv2d):           W = W.flatten(1)
        if isinstance(pruner.layer, transformers.Conv1D): W = W.t()

        d_out, d_in = W.shape
        H = torch.nan_to_num(pruner.H.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        is_ffn = "ff.net" in module_name.lower()

        if is_ffn:
            current_headsize = 1
            block_limit_ratio = args.ffn_protect
            blocksize = d_in // 4
            if saliency_mode == 'neumann':
                item_energies = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
            elif saliency_mode == 'effective_resistance':
                hinv = _hinv_diag_effective_resistance(H, blocksize=blocksize)
                item_energies = torch.sum(W ** 2, dim=0) / hinv
            elif saliency_mode == 'magnitude':
                item_energies = torch.sum(W ** 2, dim=0)   
            else:
                item_energies = torch.sum(W ** 2, dim=0) * torch.diagonal(H)
        else:
            attn_prefix      = module_name.split('.')[0]
            sample_attn      = getattr(blocks[block_idx], attn_prefix)
            current_headsize = d_in // sample_attn.heads
            num_heads        = sample_attn.heads
            item_energies    = torch.zeros(num_heads, device=dev)

            block_limit_ratio = args.attn_protect
            blocksize = d_in

            if saliency_mode == 'neumann':
                col_score = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H)
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    item_energies[h] = col_score[s:e].mean()
            elif saliency_mode == 'effective_resistance':
                hinv = _hinv_diag_effective_resistance(H, blocksize=blocksize)
                col_score = torch.sum(W ** 2, dim=0) / hinv
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    item_energies[h] = col_score[s:e].mean()
            elif saliency_mode == 'magnitude':
                col_score = torch.sum(W ** 2, dim=0)                        # no H
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    item_energies[h] = col_score[s:e].mean()
            elif saliency_mode == 'mean_norm':
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    W_h = W[:, s:e]; H_h = H[s:e, s:e]
                    item_energies[h] = torch.norm(
                        torch.matmul(W_h, H_h) * W_h, dim=-1).mean()
            elif saliency_mode == 'max_norm':
                for h in range(num_heads):
                    s, e = h * current_headsize, (h + 1) * current_headsize
                    W_h = W[:, s:e]; H_h = H[s:e, s:e]
                    item_energies[h] = torch.norm(
                        torch.matmul(W_h, H_h) * W_h, dim=-1).max()

        if current_headsize > 1:
            blocksize = max(blocksize, current_headsize)
            blocksize = (blocksize // current_headsize) * current_headsize
        items_per_block = blocksize // current_headsize

        key = _sdxl_pool_key(is_ffn, d_in)
        for i, energy in enumerate(item_energies):
            pools[key].append({
                'energy':       energy.item(),
                'param_weight': current_headsize,
                'block_id':     f"{block_idx}_{module_name}_{i // items_per_block}",
                'max_allowed':  int(items_per_block * block_limit_ratio),
            })
            pool_total[key] += current_headsize

    # ------------------------------------------------------------------
    # 2. struct_prune-faithful selection + bisection, PER POOL
    # ------------------------------------------------------------------
    def _prune_at(us_sorted, T):
        counts, weight, max_e = {}, 0, 0.0
        for u in us_sorted:
            if u['energy'] > T:
                break
            c = counts.get(u['block_id'], 0)
            if c < u['max_allowed']:
                counts[u['block_id']] = c + 1
                weight += u['param_weight']
                if u['energy'] > max_e:
                    max_e = u['energy']
        return weight, max_e

    def _solve(units, tgt_weight):
        us = sorted(units, key=lambda x: x['energy'])
        if not us or tgt_weight <= 0:
            return 0.0, 0
        lo, hi = 0.0, us[-1]['energy']
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            w, _ = _prune_at(us, mid)
            if w <= tgt_weight:
                lo = mid
            else:
                hi = mid
        w, thr = _prune_at(us, lo)
        return thr, w

    def _ceiling(units):
        seen = {}
        for u in units:
            seen[u['block_id']] = (u['max_allowed'], u['param_weight'])
        return sum(m * w for m, w in seen.values())

    pool_targets = pool_targets or {}
    thresholds   = {}
    grand_total = grand_pruned = 0

    log = logger.info if logger else print
    log("\n" + "=" * 64)
    log(f"[SDXL POOL THRESHOLDS]  mode={saliency_mode}  target={target_sparsity_ratio:.4%}")
    log("-" * 64)

    for key in sorted(pools.keys(), key=lambda k: (k[0], k[-1])):
        units   = pools[key]
        total_w = pool_total[key]
        tgt_ratio = pool_targets.get(key, target_sparsity_ratio)
        tgt_w   = total_w * tgt_ratio

        thr, w = _solve(units, tgt_w)
        thresholds[key] = thr + 1e-9

        grand_total  += total_w
        grand_pruned += w

        ceil = _ceiling(units)
        flag = f"  <-- CAP-BOUND ceiling {ceil/total_w:.2%}" if tgt_w > ceil else ""
        name = "ffn" if key[0] == 'ffn' else f"attn(d_in={key[1]})"
        log(f"{name:<18} | target {tgt_ratio:>6.2%} | cutoff {thr:>13.6f} "
            f"| gives {w/total_w:>6.2%}{flag}")

    if grand_total:
        log("-" * 64)
        log(f"{'GLOBAL':<18} | simulated {grand_pruned/grand_total:.4%} of "
            f"{grand_total/1e9:.3f}B prunable params")
    log("=" * 64 + "\n")

    return thresholds

@torch.no_grad()
def prune_SD_XL_Structured(args, pipe, target_modules, dev, timestep_weight=None, logger=None):
    logger.info('Starting SDXL U-Net Pruning...')
    dataloader = get_loaders(
        args.dataset,
        num_samples=args.num_samples  
    )

    # 1. FLAT BLOCK EXTRACTION FOR U-NET
    # SDXL nests BasicTransformerBlock inside down_blocks, mid_block, and up_blocks.
    # We recursively find them and flatten them into a standard list for grouping.
    blocks = []
    for name, module in pipe.unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            blocks.append(module)
            
    logger.info(f"Found {len(blocks)} BasicTransformerBlocks in the U-Net.")

    target_pruned_modules = []
    
    # 2. MODULE DISCOVERY
    for i in range(args.minlayer, min(args.maxlayer, len(blocks))):
        block = blocks[i]
        all_module_dict = find_layers(block)
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)
    num_modules = len(target_pruned_modules)
    logger.info(f"\n Intelligently divided {num_modules} modules into {len(modules_groups)} groups.")

    for g_idx, group_modules in enumerate(modules_groups):
        logger.info(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        pruner_dict = {}
        hooks = []

        # 3. HOOK REGISTRATION
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]

            # Register pruners and hooks for all target modules dynamically
            if module_name in ["ff.net.2", "attn1.to_out.0", "attn2.to_out.0"]:
                pruner_dict[(block_idx, module_name)] = IsometricDiffusionPruner_Structured(
                                                                                    module, 
                                                                                    block_idx, 
                                                                                    module_name, 
                                                                                    args, 
                                                                                    logger=logger
                                                                                )
                hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))

        logger.info(f"Running diffusion for group {g_idx + 1} to collect activations...")
        
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
            logger.info(f"  Prompts {i}: {prompts}")
            
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end, # Ensure your callback is defined globally
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        for hook in hooks:
            hook.remove()

        # Extract native head dimension for SDXL U-Net
        if hasattr(pipe, "unet"):
            raw_dim = pipe.unet.config.attention_head_dim
            native_head_dim = raw_dim[0] if isinstance(raw_dim, list) else raw_dim
        else:
            native_head_dim = 64 # Fallback

        logger.info("Calculating non-uniform sparsity threshold for group...")
        # global_threshold = calculate_group_energy_threshold_sdxl(
        #     args=args,
        #     pruner_dict=pruner_dict, 
        #     target_sparsity_ratio=args.sparsity_ratio, 
        #     dev=dev, 
        #     blocks=blocks,
        #     headsize=native_head_dim,
        #     saliency_mode=args.saliency_mode,
        #     logger=logger
        # )
        pool_thresholds = calculate_pool_thresholds_sdxl(
            args=args,
            pruner_dict=pruner_dict, 
            target_sparsity_ratio=args.sparsity_ratio, 
            dev=dev, 
            blocks=blocks,
            headsize=native_head_dim,
            saliency_mode=args.saliency_mode,
            logger=logger
        )
        
        # logger.info(f"Group Energy Threshold: {global_threshold:.6f}")
        logger.info(f"Group Energy Threshold: {pool_thresholds}")
        logger.info(f"Pruning group {g_idx + 1}...")

        master_targets = {"ff.net.2", "attn1.to_out.0", "attn2.to_out.0"}
        target_modules_prune = [
            (idx, name) for idx, name in group_modules 
            if name in master_targets
        ]

        # 4. PHYSICAL PRUNING EXECUTION
        for block_idx, module_name in target_modules_prune:
            logger.info(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            
            target_layer = get_module_by_name(blocks[block_idx], module_name)

            # --- FFN PRUNING ---
            if module_name == "ff.net.2":
                current_headsize = 1
                thr = pool_thresholds[_sdxl_pool_key(True, target_layer.in_features)]
                # Generate TopoPrune Masks
                idx = pruner_dict[(block_idx, module_name)].struct_prune(
                    target_sparsity=sparsity,
                    energy_threshold=thr, #global_threshold,
                    headsize=current_headsize,
                    iterations=args.ricci_flow_iters
                )
                
                pruner_dict[(block_idx, module_name)].free()
                
                target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                id_list = idx.tolist()

                # GEGLU Trap Detection for SDXL
                is_geglu = target_layer_in.out_features == target_layer.in_features * 2
                
                if is_geglu:
                    logger.info("  -> Detected GEGLU activation. Mirroring gate indices.")
                    inner_dim = target_layer.in_features
                    geglu_id_list = id_list + [i + inner_dim for i in id_list]
                    
                    tp.prune_linear_in_channels(target_layer, id_list)
                    tp.prune_linear_out_channels(target_layer_in, geglu_id_list)
                else:
                    tp.prune_linear_in_channels(target_layer, id_list)
                    tp.prune_linear_out_channels(target_layer_in, id_list)

            # --- ATTENTION PRUNING (Self and Cross) ---
            elif module_name in ['attn1.to_out.0', 'attn2.to_out.0']:
                attn_prefix = module_name.split('.')[0] 
                attn_module = getattr(blocks[block_idx], attn_prefix)
                
                # DYNAMIC HEADSIZE CALCULATION
                # This must happen BEFORE struct_prune so the mask slices correctly!
                true_head_dim = target_layer.in_features // attn_module.heads
                current_headsize = true_head_dim
                thr = pool_thresholds[_sdxl_pool_key(False, target_layer.in_features)]
                # Generate TopoPrune Masks using the true dimension
                idx = pruner_dict[(block_idx, module_name)].struct_prune(
                    target_sparsity=sparsity,
                    energy_threshold=thr, #global_threshold,
                    headsize=current_headsize,
                    iterations=args.ricci_flow_iters
                )
                
                pruner_dict[(block_idx, module_name)].free()
                
                target_q_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_q")
                target_k_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_k")
                target_v_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_v")
                
                id_list = idx.tolist()
                
                # Mathematical Sanity Check
                assert len(id_list) % true_head_dim == 0, f"Alignment error: Pruned channels ({len(id_list)}) is not divisible by true head dimension ({true_head_dim})!"
                
                # Prune Output projection In-Channels
                tp.prune_linear_in_channels(target_layer, id_list)
                
                # Prune Q, K, V Out-Channels
                tp.prune_linear_out_channels(target_q_layer, id_list)
                tp.prune_linear_out_channels(target_k_layer, id_list)
                tp.prune_linear_out_channels(target_v_layer, id_list)
                
                logger.info(f"Previous {attn_prefix} heads: {attn_module.heads} (Calculated Head Dim: {true_head_dim})")
                
                # Update the heads using the true mathematical dimension
                attn_module.heads -= len(id_list) // true_head_dim
                
                logger.info(f"Current {attn_prefix} heads: {attn_module.heads}")

        torch.cuda.empty_cache()
        logger.info(f"Group {g_idx + 1} pruning completed.")