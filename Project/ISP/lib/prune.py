import time 
import heapq 
import torch 
import torch.nn as nn 
from .OBS_Diff import OBS_Diff 
from .OBS_Diff_Structured import OBS_Diff_Structured, OBS_Diff_Structured_Joint_Attn
from collections import defaultdict
from .dataloader import get_loaders 
from collections import OrderedDict
import numpy as np
import torch_pruning as tp


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

def check_sparsity(transformer_model, target_names):
    print("\n" + "="*50)
    print("Calculating Sparsity for Target Modules...")
    print("="*50)

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

    print("--- Individual Module Sparsity ---")
    if not individual_sparsities:
        print("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            print(f"{name:<50s} | Sparsity: {sparsity:.4f}%")
            
    print("--- Aggregate Sparsity for Targets ---")
    if total_target_params > 0:
        aggregate_percentage = (total_target_zeros / total_target_params) * 100
        print(f"Total Parameters in Target Modules : {total_target_params}")
        print(f"Zero Parameters in Target Modules  : {total_target_zeros}")
        print(f"Overall Sparsity of Target Modules : {aggregate_percentage:.4f}%")
    else:
        print("No parameters found in target modules.")
    
    print("\n" + "="*50)
    print("Calculation finished.")
    print("="*50)

    return aggregate_percentage


def check_size(transformer_model, target_names):
    print("\n" + "="*50)
    print("Checking Size for Target Modules...")
    print("="*50)

    all_linear_layers = find_layers(transformer_model)

    individual_sparsities = OrderedDict()
    individual_sparsities_structured = OrderedDict()
    individual_sparsities_bias = OrderedDict()
    total_target_params = 0
    total_target_zeros = 0

    if "ff.net.2" in target_names:
        target_names.append("ff.net.0.proj")
    if "ff_context.net.2" in target_names:
        target_names.append("ff_context.net.0.proj")
    if "attn.to_out.0" in target_names:
        target_names.append("attn.to_add_out")
        target_names.append("attn.to_q")
        target_names.append("attn.to_k")
        target_names.append("attn.to_v")
        target_names.append("attn.add_k_proj")
        target_names.append("attn.add_q_proj")
        target_names.append("attn.add_v_proj")
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
                individual_sparsities_structured[layer_name] = layer_module.weight.shape
                individual_sparsities_bias[layer_name] = layer_module.bias.shape
                total_target_zeros += zeros
                total_target_params += total

    print("\n--- Individual Module Sparsity ---")
    if not individual_sparsities:
        print("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            print(f"{name:<50s} | {individual_sparsities_structured[name]} {individual_sparsities_bias[name]}")
            
    
    
    print("\n" + "="*50)
    print("Calculation finished.")
    print("="*50)

def group_modules_with_parallelism(target_pruned_modules, num_groups):

    parallel_sets_rules = [
        {"attn.to_q", "attn.to_k", "attn.to_v", "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj"},
        {"attn.to_out.0", "attn.to_add_out"},
        {"ff_context.net.0.proj", "ff.net.0.proj"},
        {"ff.net.2", "ff_context.net.2"},
    ]
    modules_by_block = defaultdict(list)
    for block_idx, name in target_pruned_modules:
        modules_by_block[block_idx].append(name)

    groupable_items = []
    
    for block_idx in sorted(modules_by_block.keys()):
        block_modules = set(modules_by_block[block_idx])
        processed_modules = set()

        for p_set in parallel_sets_rules:
            intersection = block_modules.intersection(p_set)
            if intersection:
                parallel_unit = [(block_idx, name) for name in sorted(list(intersection))] # 排序以保证确定性
                groupable_items.append(parallel_unit)
                processed_modules.update(intersection)
        
        remaining_modules = block_modules - processed_modules
        for name in sorted(list(remaining_modules)):
            groupable_items.append([(block_idx, name)])

    num_items = len(groupable_items)
    print(f"num_items: {num_items}")
    if num_items == 0:
        return []

    group_size = num_items // num_groups
    remainder = num_items % num_groups
    
    if group_size == 0:
        group_size = 1
        num_groups = num_items
        remainder = 0
    
    final_groups = []
    start_index = 0
    for i in range(num_groups):
        end_index = start_index + group_size + (1 if i < remainder else 0)
        
        # e.g., [ [(0, 'ff.net.2'), (0, 'ff_context.net.2')], [(0, 'attn.to_q')] ]
        current_chunk = groupable_items[start_index:end_index]
        
        flat_group = [module for unit in current_chunk for module in unit]
        final_groups.append(flat_group)
        
        start_index = end_index

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


def create_hook_fn(block_idx, layer_name, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        pruner = pruner_dict[(block_idx, layer_name)]
        
        # get the input data
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples
        
        input_data = input_data * np.sqrt(current_weight)
        # call add_batch, pass the weighted input data
        pruner.add_batch(input_data, output.data, W_new)
        #msg = f"Updated Hessian for Block {block_idx}, {layer_name}, Step {step}, Input Shape: {input[0].shape}, Weight: {timestep_weight[step]:.4f}"
        #logger.info(msg)  
    return hook_fn

def create_hook_fn_Joint_Attn(block_idx, layer_name, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        if layer_name == "attn.to_add_out":
            pruner = pruner_dict[(block_idx, "attn.to_out.0")]
        else:
            pruner = pruner_dict[(block_idx, layer_name)]
        
        # get the input data
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples
        
        input_data = input_data * np.sqrt(current_weight)
        # call add_batch, pass the weighted input data

        #print(f"input_data.shape: {input_data.shape}")
        #print(f"layer_name: {layer_name}")
        pruner.add_batch(input_data, output.data, layer_name, W_new)
        #print(f"input_data.shape: {input_data.shape}")
        #print(f"layer_name: {layer_name}")
        #msg = f"Updated Hessian for Block {block_idx}, {layer_name}, Step {step}, Input Shape: {input[0].shape}, Weight: {timestep_weight[step]:.4f}"
        #print(msg)  
    return hook_fn

step_info = {"current": 0}
# callback function, update the step value in step_info after each denoising step
def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
    step_info["current"] += 1
    return callback_kwargs

@torch.no_grad()
def prune_OBS_Diff(args, pipe, target_modules,  dev, prune_n=0, prune_m=0, timestep_weight=None, logger=None):
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

    # Divide target_pruned_modules into num_pruned_groups groups
    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    logger.info(f"\n intelligently divided {num_modules} modules into {len(modules_groups)} groups:")

    for g_idx, group in enumerate(modules_groups):
        logger.info(f"Group {g_idx + 1}: {[(block_idx, name) for block_idx, name in group]}")
    for g_idx, group_modules in enumerate(modules_groups):
        logger.info(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        # Initialize pruner and hooks for this group
        pruner_dict = {}
        hooks = []
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]
            pruner_dict[(block_idx, module_name)] = OBS_Diff(module, args, logger=logger)
            hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))
     
        # Run prompts in dataloader to collect activations
        logger.info(f"Running diffusion for group {g_idx + 1} to collect activations...")
        # consider batch_size
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
         
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
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        # Remove hooks for this group
        for hook in hooks:
            hook.remove()
        
        # Execute pruning for this group
        logger.info(f"Pruning group {g_idx + 1}...")
        for block_idx, module_name in group_modules:
            logger.info(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            pruner_dict[(block_idx, module_name)].fasterprune(
                sparsity=sparsity,
                percdamp=args.percdamp,
                prunen=prune_n,
                prunem=prune_m
            )
            pruner_dict[(block_idx, module_name)].free()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        logger.info(f"Group {g_idx + 1} pruning completed.")


@torch.no_grad()
def prune_OBS_Diff_Structured(args, pipe, target_modules, dev, timestep_weight=None, logger=None):
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
        logger.info(f"all_module_dict: {all_module_dict}")
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    # Divide target_pruned_modules into num_pruned_groups groups
    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    logger.info(f"\n intelligently divided {num_modules} modules into {len(modules_groups)} groups:")

    for g_idx, group in enumerate(modules_groups):
        logger.info(f"Group {g_idx + 1}: {[(block_idx, name) for block_idx, name in group]}")


    # Process each group
    for g_idx, group_modules in enumerate(modules_groups):
        logger.info(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        # Initialize pruner and hooks for this group
        pruner_dict = {}
        hooks = []
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]
            if module_name == "ff.net.2" or module_name == "ff_context.net.2":
                pruner_dict[(block_idx, module_name)] = OBS_Diff_Structured(module, block_idx, args, logger=logger, layer_name=module_name)

                hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))
            else:
                module_2 = get_module_by_name(blocks[block_idx], "attn.to_add_out")
                pruner_dict[(block_idx, module_name)] = OBS_Diff_Structured_Joint_Attn(module, module_2, block_idx, args, logger=logger, layer_name=module_name, layer2_name="attn.to_add_out")
                hook_fn = create_hook_fn_Joint_Attn(block_idx, module_name, pruner_dict, timestep_weight)
                hook_fn2 = create_hook_fn_Joint_Attn(block_idx, "attn.to_add_out", pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))
                hooks.append(module_2.register_forward_hook(hook_fn2))
        
        # Run prompts in dataloader to collect activations
        logger.info(f"Running diffusion for group {g_idx + 1} to collect activations...")
        # consider batch_size
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
         
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
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        # Remove hooks for this group
        for hook in hooks:
            hook.remove()
        
        # Execute pruning for this group
        logger.info(f"Pruning group {g_idx + 1}...")
        for block_idx, module_name in group_modules:
            logger.info(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            if module_name == "attn.to_out.0":
                idx_1 = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp,
                    headsize=64
                )
            else:
                idx = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp
                )
            pruner_dict[(block_idx, module_name)].free()
            
            target_layer = get_module_by_name(blocks[block_idx], module_name)
            
            if module_name == "ff.net.2":
                target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                id = idx.tolist()
                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, id)
                
            if module_name == "ff_context.net.2":
                target_layer_in = get_module_by_name(blocks[block_idx], "ff_context.net.0.proj")
                id = idx.tolist()
                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, id)

            if module_name == 'attn.to_out.0':
                target_add_layer = get_module_by_name(blocks[block_idx], "attn.to_add_out")
                target_q_layer = get_module_by_name(blocks[block_idx], "attn.to_q")
                target_k_layer = get_module_by_name(blocks[block_idx], "attn.to_k")
                target_v_layer = get_module_by_name(blocks[block_idx], "attn.to_v")
                target_add_k_layer = get_module_by_name(blocks[block_idx], "attn.add_k_proj")
                target_add_q_layer = get_module_by_name(blocks[block_idx], "attn.add_q_proj")
                target_add_v_layer = get_module_by_name(blocks[block_idx], "attn.add_v_proj")
                idx_1 = idx_1.tolist()
                tp.prune_linear_in_channels(target_layer, idx_1)
                tp.prune_linear_in_channels(target_add_layer, idx_1)
                tp.prune_linear_out_channels(target_q_layer, idx_1)
                tp.prune_linear_out_channels(target_k_layer, idx_1)
                tp.prune_linear_out_channels(target_v_layer, idx_1)
                tp.prune_linear_out_channels(target_add_k_layer, idx_1)
                tp.prune_linear_out_channels(target_add_q_layer, idx_1)
                tp.prune_linear_out_channels(target_add_v_layer, idx_1)

                logger.info(f"Previous heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")
                pipe.transformer.transformer_blocks[block_idx].attn.heads -= len(idx_1) // 64
                logger.info(f"Current heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")
        torch.cuda.empty_cache()
        logger.info(f"Group {g_idx + 1} pruning completed.")


@torch.no_grad()
def prune_OBS_Diff_Structured_SDXL(args, pipe, target_modules, dev, timestep_weight=None, logger=None):
    logger.info('Starting SDXL Structural Pruning...')
    dataloader = get_loaders(
        args.dataset,
        num_samples=args.num_samples  
    )

    # 1. FLATTEN THE U-NET HIARARCHY
    # SDXL nests transformers inside down_blocks, mid_block, and up_blocks.
    # We must extract all BasicTransformerBlocks into a flat, indexable list.
    blocks = []
    for name, module in pipe.unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            blocks.append(module)
            
    target_pruned_modules = []

    for i in range(args.minlayer, min(args.maxlayer, len(blocks))):
        block = blocks[i]
        all_module_dict = find_layers(block)
        
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    # Divide target_pruned_modules into num_pruned_groups groups
    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    logger.info(f"\n Intelligently divided {num_modules} modules into {len(modules_groups)} groups.")

    # Process each group
    for g_idx, group_modules in enumerate(modules_groups):
        logger.info(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        # Initialize pruner and hooks for this group
        pruner_dict = {}
        hooks = []
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]
            
            # SDXL only uses standard FF and standard Attention
            pruner_dict[(block_idx, module_name)] = OBS_Diff_Structured(module, block_idx, args, logger=logger, layer_name=module_name)
            hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))
        
        # Run prompts in dataloader to collect activations
        logger.info(f"Running diffusion for group {g_idx + 1} to collect activations...")
        batch_size = getattr(args, 'batch_size', 1)
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
            step_info["current"] = 0
            
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        # Remove hooks for this group
        for hook in hooks:
            hook.remove()
        
        # Execute pruning for this group
        logger.info(f"Pruning group {g_idx + 1}...")
        for block_idx, module_name in group_modules:
            logger.info(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            
            # Identify if it is an attention block (Self or Cross)
            is_attn = "attn" in module_name 
            
            target_layer = get_module_by_name(blocks[block_idx], module_name)
            if is_attn:
                attn_prefix = module_name.split('.')[0] # Extracts 'attn1' or 'attn2'
                attn_module = getattr(blocks[block_idx], attn_prefix)
                true_head_dim = target_layer.in_features // attn_module.heads
                current_headsize = true_head_dim
                # SDXL attention head dimension is strictly 64
                idx_pruned = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp,
                    headsize=current_headsize
                )
            else:
                idx_pruned = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp
                )
                
            pruner_dict[(block_idx, module_name)].free()
            target_layer = get_module_by_name(blocks[block_idx], module_name)
            
            # --- SDXL FEED-FORWARD AMPUTATION ---
            if module_name == "ff.net.2":
                target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                id_list = idx_pruned.tolist()

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
                
            # --- SDXL ATTENTION AMPUTATION ---
            # SDXL has attn1 (Self-Attention) and attn2 (Cross-Attention)
            elif is_attn:
                target_q_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_q")
                target_k_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_k")
                target_v_layer = get_module_by_name(blocks[block_idx], f"{attn_prefix}.to_v")
                
                id_list = idx_pruned.tolist()
                
                # Mathematical Sanity Check
                assert len(id_list) % true_head_dim == 0, f"Alignment error: Pruned channels ({len(id_list)}) is not divisible by true head dimension ({true_head_dim})!"

                # Prune Output Projection Input Channels
                tp.prune_linear_in_channels(target_layer, id_list)
                
                # Prune Q, K, V Output Channels
                tp.prune_linear_out_channels(target_q_layer, id_list)
                tp.prune_linear_out_channels(target_k_layer, id_list)
                tp.prune_linear_out_channels(target_v_layer, id_list)

                # Update the internal head count tracker for xFormers / scaled dot product
                logger.info(f"Previous {attn_prefix} heads: {attn_module.heads}")
                attn_module.heads -= len(id_list) // 64
                logger.info(f"Current {attn_prefix} heads: {attn_module.heads}")
                
        torch.cuda.empty_cache()
        logger.info(f"Group {g_idx + 1} pruning completed.")