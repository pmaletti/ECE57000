import math
import time
import os
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt
import torch_pruning as tp
import torch_pruning.pruner.function as tfun
import logging

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def block_robust_inverse(H, rcond=1e-5):
    """
    Memory-efficient block inversion using pseudo-inverse for rank-deficient 
    matrices common in local isometry mappings.
    """
    n = H.shape[0]
    half = n // 2

    # 1. Extract blocks
    A = H[:half, :half]
    B = H[:half, half:]
    D = H[half:, half:]

    # 2. Invert A robustly
    # hermitian=True ensures it uses eigh, which is highly stable for symmetric matrices
    A_inv = torch.linalg.pinv(A, hermitian=True, rcond=rcond)

    # 3. Compute Schur Complement: S = D - B^T @ A_inv @ B
    B_T = B.transpose(0, 1)
    B_T_A_inv = torch.matmul(B_T, A_inv)
    S = D - torch.matmul(B_T_A_inv, B)

    # 4. Invert S robustly (where the previous Cholesky failure occurred)
    S_inv = torch.linalg.pinv(S, hermitian=True, rcond=rcond)

    # 5. Compute the four blocks of the new inverse matrix
    top_right = -torch.matmul(B_T_A_inv.transpose(0, 1), S_inv)
    bottom_left = top_right.transpose(0, 1)
    top_left = A_inv - torch.matmul(top_right, B_T_A_inv)
    bottom_right = S_inv

    # 6. Reconstruct the full matrix
    top = torch.cat([top_left, top_right], dim=1)
    bottom = torch.cat([bottom_left, bottom_right], dim=1)
    
    return torch.cat([top, bottom], dim=0)

def recursive_robust_inverse(H, rcond=1e-5, depth=2):
    """
    Memory-efficient recursive pseudo-inverse.
    Avoids torch.cat OOMs by pre-allocating the output and aggressively 
    freeing intermediates.
    """
    n = H.shape[0]
    
    if depth <= 0 or n <= 512:
        try:
            # Attempt 1: Fast Hermitian pseudo-inverse (eigh)
            return torch.linalg.pinv(H, hermitian=True, rcond=rcond)
            
        except RuntimeError as e:
            if "linalg.eigh" in str(e) or "converge" in str(e):
                try:
                    # Attempt 2: SVD pseudo-inverse (slower but highly stable)
                    return torch.linalg.pinv(H, hermitian=False, rcond=rcond)
                    
                except RuntimeError:
                    # Attempt 3: Safe Fallback (Diagonal Inverse)
                    # This is the mathematically correct way to "skip" the inversion 
                    # without destroying the subsequent Optimal Brain-style weight updates.
                    diag = torch.diag(H).clamp(min=1e-8)
                    return torch.diag(1.0 / diag)
            else:
                # Re-raise if it's a completely unrelated CUDA/memory error
                raise e

    half = n // 2

    # 1. Views (no extra memory allocation)
    A = H[:half, :half]
    B = H[:half, half:]
    D = H[half:, half:]

    # 2. Invert A
    A_inv = recursive_robust_inverse(A, rcond=rcond, depth=depth-1)

    # 3. Intermediate Math
    B_T_A_inv = torch.matmul(B.transpose(0, 1), A_inv)

    # 4. Schur Complement & its inverse
    S = D - torch.matmul(B_T_A_inv, B)
    S_inv = recursive_robust_inverse(S, rcond=rcond, depth=depth-1)
    
    # Free S immediately before we allocate anything else
    del S 

    # 5. Pre-allocate the final tensor for this level 
    # This prevents the massive double-allocation overhead of torch.cat
    out = torch.empty_like(H)

    # 6. Fill the output tensor directly
    out[half:, half:] = S_inv

    top_right = -torch.matmul(B_T_A_inv.transpose(0, 1), S_inv)
    out[:half, half:] = top_right
    out[half:, :half] = top_right.transpose(0, 1)

    # Compute top_left directly into the 'out' tensor to save memory
    out[:half, :half] = A_inv - torch.matmul(top_right, B_T_A_inv)

    # 7. Aggressive Cleanup
    # Manually drop the references to massive intermediates
    del A_inv, B_T_A_inv, S_inv, top_right
    
    # Force CUDA to sweep the freed blocks back to the memory pool 
    # so the layer above in the recursion tree can use them
    torch.cuda.empty_cache()

    return out

class OBS_Diff_Structured(object):
    def __init__(self, layer, layer_idx, args, logger=None, layer_name=None):
        self.args = args
        self.logger = logger
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # # Flux is massive; offload H to pinned CPU memory to prevent VRAM OOM.
        # # Other models stay on GPU for maximum speed.
        # if getattr(self.args, 'model_path', '') == 'black-forest-labs/FLUX.1-dev':
        #     self.H = torch.zeros((self.columns, self.columns), device='cpu').pin_memory()
        #     self.is_flux = True
        # else:
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.is_flux = False
        self.nsamples = 0
        self.sum_weight = 0

        self.no_compensate = args.no_compensate
        self.is_offloaded = False
    
    def offload_hessian(self):
        """Moves the Hessian to pinned CPU memory to free up VRAM for other layers."""
        if getattr(self.args, 'model_path', '') in ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.2-klein-9B']:
            self.H = self.H.cpu().pin_memory()
            self.is_offloaded = True
            torch.cuda.empty_cache() # Force VRAM release immediately

    def add_batch(self, inp, out, W_new, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()


        W_old = self.sum_weight

        W_total = W_old + W_new
        self.H *= W_old / W_total

        self.sum_weight = W_total
        
        norm_factor = math.sqrt(2 / self.sum_weight)
        inp = norm_factor * inp.float()

        H_update = inp.matmul(inp.t())
        self.H += H_update

    # Structured pruning
    def struct_prune(
        self, sparsity, headsize=1, percdamp=0.0, layer_idx=None, 
    ):
        assert self.columns % headsize == 0

        tick = time.time()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if self.is_offloaded:
            H = self.H.to(self.dev, non_blocking=True)
        else:
            H = self.H

        del self.H

        # Handle the elements on the diagonal of the Hessian matrix that are 0
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Regularize the Hessian matrix
        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev) # 1 for remove
        pruned_columns = column_mask.count_nonzero()
        target_columns = round(self.columns // headsize * sparsity) * headsize

        if headsize > 1:
            pass
        else:
            blocksize = (target_columns - 512) // 2

        while pruned_columns < target_columns:     
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            if headsize > 1:
                Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize] for i in range(0, self.columns, headsize)])
                Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                Hinv_diag = Hinv.diag()

            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            error[column_mask] = torch.inf
            if headsize > 1:
                head_sort_idx = error.view(-1, headsize).sum(1).argsort()
                column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize) for x in head_sort_idx])
                cnt = headsize
            else:
                column_sort_idx = error.argsort()
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            W = W[:, column_sort_idx]
            Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
            Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]
            
            W1 = W[:, :cnt].clone()
            Hinv1 = Hinv[:, :cnt]
            Err1 = torch.zeros_like(W1)

            for i in range(cnt):
                Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                if not self.no_compensate:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # local update

            W[:, :cnt] = 0
            if not self.no_compensate:
                end = self.columns - pruned_columns
                W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # global update

            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W = W[:, column_sort_idx_inv]

            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            if headsize > 1:
                pass
            else:
                blocksize = (blocksize - 512) // 2

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        # print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)))

        zeros = (self.layer.weight.data == 0).all(dim=0).sum().item()
        total = self.layer.weight.size(1)
        self.logger.info(
            f'[{self.layer_name}] Structured Prune Done | pruned columns {zeros / total:.2%}) | Time: {time.time() - tick:.2f}s'
        )
        if DEBUG:
            out_gap = torch.mean((self.layer(self.inp1) - self.out1) ** 2).item()
            out = torch.mean(self.out1 ** 2).item()
            self.logger.info('output_gap: %f, output: %f, output_gap / output: %f' % (out_gap, out, out_gap / out))

        pruned_indices = torch.where(column_mask)[0]

        return pruned_indices
        

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


def reciprocal_rank_fusion(rankings: list, k: int = 60):
    """Use reciprocal rank fusion to merge multiple ranking lists."""
    fused_scores = {}
    for rank_list in rankings:
        for rank, item in enumerate(rank_list):
            if item not in fused_scores:
                fused_scores[item] = 0
            fused_scores[item] += 1 / (k + rank + 1)
    return fused_scores


def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module

class OBS_Diff_Structured_Joint_Attn(object):
    def __init__(self, layer, layer_2, layer_idx, args, logger=None, layer_name=None, layer2_name=None):
        self.args = args
        self.logger = logger
        self.layer_to_out = layer
        self.layer_to_add_out = layer_2
        self.layer_name = layer_name
        self.layer2_name = layer2_name
        self.dev = self.layer_to_out.weight.device
        
        
        self.to_out_columns = layer.weight.data.shape[1]
        self.to_add_out_columns = layer_2.weight.data.shape[1]
        
        # Flux is massive; offload H to pinned CPU memory to prevent VRAM OOM.
        # Other models stay on GPU for maximum speed.
        # if getattr(self.args, 'model_path', '') == 'black-forest-labs/FLUX.1-dev':
        #     self.H_to_out = torch.zeros((self.layer_to_out.in_features, self.layer_to_out.in_features), device='cpu').pin_memory()
        #     self.H_to_add_out = torch.zeros((self.layer_to_add_out.in_features, self.layer_to_add_out.in_features), device='cpu').pin_memory()
        #     self.is_flux = True
        # else:
        self.H_to_out = torch.zeros((self.layer_to_out.in_features, self.layer_to_out.in_features), device=self.dev)
        self.H_to_add_out = torch.zeros((self.layer_to_add_out.in_features, self.layer_to_add_out.in_features), device=self.dev)
        self.is_flux = False

        self.nsamples_to_out = 0
        self.nsamples_to_add_out = 0

        self.sum_weight_to_out = 0
        self.sum_weight_to_add_out = 0

        self.no_compensate = args.no_compensate
        self.is_offloaded = False

    def offload_hessian(self):
        """Moves the Hessian to pinned CPU memory to free up VRAM for other layers."""
        if getattr(self.args, 'model_path', '') in ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.2-klein-9B']:
            self.H_to_out = self.H_to_out.cpu().pin_memory()
            self.H_to_add_out = self.H_to_add_out.cpu().pin_memory()
            self.is_offloaded = True
            torch.cuda.empty_cache()
    
    def add_batch(self, inp, out, layer_name, W_new):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if layer_name == "attn.to_out.0":
            if isinstance(self.layer_to_out, nn.Linear) or isinstance(self.layer_to_out, transformers.Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()  # [hsize, seqlen]
        elif layer_name == "attn.to_add_out":
            if isinstance(self.layer_to_add_out, nn.Linear) or isinstance(self.layer_to_add_out, transformers.Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()  # [hsize, seqlen]

        if layer_name == "attn.to_out.0":
            W_old = self.sum_weight_to_out
            W_total = W_old + W_new
            self.H_to_out *= W_old / W_total
            self.sum_weight_to_out = W_total
            norm_factor = math.sqrt(2 / self.sum_weight_to_out)
            inp = norm_factor * inp.float()
            H_update = inp.matmul(inp.t())
            self.H_to_out += H_update

        elif layer_name == "attn.to_add_out":
            W_old = self.sum_weight_to_add_out
            W_total = W_old + W_new
            self.H_to_add_out *= W_old / W_total
            self.sum_weight_to_add_out = W_total
            norm_factor = math.sqrt(2 / self.sum_weight_to_add_out)
            inp = norm_factor * inp.float()
            H_update = inp.matmul(inp.t())
            self.H_to_add_out += H_update

    def _get_head_ranking(self, layer, H, headsize):
        """Calculate the importance ranking of the heads based on the Hessian matrix."""
        W = layer.weight.data.clone().float()
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        # Regularize the Hessian matrix
        if self.args.percdamp > 0:
            damp = self.args.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp
            
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

        Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize] for i in range(0, layer.weight.data.shape[1], headsize)])
        Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1).reshape(-1)
        Hinv_diag = Hinv_diag ** 2
        error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
        
        head_errors = error.view(-1, headsize).sum(dim=1)
        return head_errors.argsort().tolist()


    # Structured pruning
    def struct_prune(
        self, sparsity, headsize=1, percdamp=0.0, layer_idx=None, 
    ):
        tick = time.time()
        assert self.to_out_columns % headsize == 0
        assert self.to_add_out_columns % headsize == 0

        W1 = self.layer_to_out.weight.data.clone().float()
        W2 = self.layer_to_add_out.weight.data.clone().float()

        # Conditionally pull from pinned memory if it's Flux
        if self.is_offloaded:
            H1 = self.H_to_out.to(self.dev, non_blocking=True)
            H2 = self.H_to_add_out.to(self.dev, non_blocking=True)
        else:
            H1 = self.H_to_out
            H2 = self.H_to_add_out

        del self.H_to_out, self.H_to_add_out

        dead_1 = torch.diag(H1) == 0
        H1[dead_1, dead_1] = 1
        W1[:, dead_1] = 0

        dead_2 = torch.diag(H2) == 0
        H2[dead_2, dead_2] = 1
        W2[:, dead_2] = 0

        if self.args.percdamp > 0:
            damp = self.args.percdamp * torch.mean(torch.diag(H1))
            diag = torch.arange(H1.size(0), device=self.dev)
            H1[diag, diag] += damp

        if self.args.percdamp > 0:
            damp = self.args.percdamp * torch.mean(torch.diag(H2))
            diag = torch.arange(H2.size(0), device=self.dev)
            H2[diag, diag] += damp

        column_mask = torch.zeros(self.to_out_columns, dtype=torch.bool, device=self.dev) # 1 for remove
        pruned_columns = column_mask.count_nonzero()
        target_columns = round(self.to_out_columns // headsize * sparsity) * headsize

        while pruned_columns < target_columns:
            Hinv_1 = torch.cholesky_inverse(torch.linalg.cholesky(H1))
            Hinv_2 = torch.cholesky_inverse(torch.linalg.cholesky(H2))
            if headsize > 1:
                Hinv_diag_1 = torch.stack([Hinv_1[i:i+headsize, i:i+headsize] for i in range(0, self.to_out_columns, headsize)])
                Hinv_diag_1 = torch.diagonal(torch.linalg.cholesky(Hinv_diag_1), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag_1 = Hinv_diag_1 ** 2
                Hinv_diag_2 = torch.stack([Hinv_2[i:i+headsize, i:i+headsize] for i in range(0, self.to_add_out_columns, headsize)])
                Hinv_diag_2 = torch.diagonal(torch.linalg.cholesky(Hinv_diag_2), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag_2 = Hinv_diag_2 ** 2

            error_1 = torch.sum(W1 ** 2 / Hinv_diag_1.unsqueeze(0), dim=0)
            error_2 = torch.sum(W2 ** 2 / Hinv_diag_2.unsqueeze(0), dim=0)

            error_1[column_mask] = torch.inf
            error_2[column_mask] = torch.inf
            if headsize > 1:
                head_sort_idx_1 = error_1.view(-1, headsize).sum(1).argsort()
                head_sort_idx_2 = error_2.view(-1, headsize).sum(1).argsort()
                head_merge_sort_idx = reciprocal_rank_fusion([head_sort_idx_1.tolist(), head_sort_idx_2.tolist()]) 
                # print(f"head_merge_sort_idx: {head_merge_sort_idx}")
                sorted_heads = sorted(head_merge_sort_idx.keys(), key=lambda h: head_merge_sort_idx[h], reverse=True)
                # print(f"sorted_heads: {sorted_heads}")
                column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize) for x in sorted_heads])
                cnt = headsize

            W1 = W1[:, column_sort_idx]
            W2 = W2[:, column_sort_idx]
            Hinv_1 = Hinv_1[column_sort_idx, :][:, column_sort_idx]
            Hinv_2 = Hinv_2[column_sort_idx, :][:, column_sort_idx]
            Hinv_1 = torch.linalg.cholesky(Hinv_1, upper=True)[:cnt]
            Hinv_2 = torch.linalg.cholesky(Hinv_2, upper=True)[:cnt]
            
            W1_prune = W1[:, :cnt].clone()
            W2_prune = W2[:, :cnt].clone()
            Hinv1 = Hinv_1[:, :cnt]
            Hinv2 = Hinv_2[:, :cnt]
            Err1 = torch.zeros_like(W1_prune)
            Err2 = torch.zeros_like(W2_prune)

            for i in range(cnt):
                Err1[:, i:i+1] = W1_prune[:, i:i+1] / Hinv1[i, i]
                Err2[:, i:i+1] = W2_prune[:, i:i+1] / Hinv2[i, i]
                if not self.no_compensate:
                    W1_prune[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # local update
                    W2_prune[:, i:] -= Err2[:, i:i+1].matmul(Hinv2[i:i+1, i:])  # local update

            W1[:, :cnt] = 0
            W2[:, :cnt] = 0
            if not self.no_compensate:
                end = self.to_out_columns - pruned_columns
                
                W1[:, cnt:end] -= Err1.matmul(Hinv_1[:, cnt:end])  # global update
                W2[:, cnt:end] -= Err2.matmul(Hinv_2[:, cnt:end])  # global update

            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W1 = W1[:, column_sort_idx_inv]
            W2 = W2[:, column_sort_idx_inv]

            pruned_idx = column_sort_idx[:cnt]

            H1[pruned_idx, :] = H1[:, pruned_idx] = 0
            H1[pruned_idx, pruned_idx] = 1

            H2[pruned_idx, :] = H2[:, pruned_idx] = 0
            H2[pruned_idx, pruned_idx] = 1
            
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            if headsize > 1:
                pass
            else:
                blocksize = (blocksize - 512) // 2

        if isinstance(self.layer_to_out, transformers.Conv1D):
            W1 = W1.t()
        if isinstance(self.layer_to_add_out, transformers.Conv1D):
            W2 = W2.t()

        self.layer_to_out.weight.data = W1.reshape(self.layer_to_out.weight.shape).to(self.layer_to_out.weight.data.dtype)
        self.layer_to_add_out.weight.data = W2.reshape(self.layer_to_add_out.weight.shape).to(self.layer_to_add_out.weight.data.dtype)
        
        # print('pruned columns %d/%d' % ((self.layer_to_out.weight.sum(0) == 0).sum().item(), self.layer_to_out.weight.size(1)))
        # print('pruned columns %d/%d' % ((self.layer_to_add_out.weight.sum(0) == 0).sum().item(), self.layer_to_add_out.weight.size(1)))
        for name, layer in ((self.layer_name, self.layer_to_out), 
                            (self.layer2_name, self.layer_to_add_out)):
            zeros = (layer.weight.data == 0).all(dim=0).sum().item()
            total = layer.weight.size(1)
            print(f'[attn.{name}] Joint Prune Done | pruned columns {zeros}/{total} '
                f'({zeros / total:.2%}) | Time: {time.time() - tick:.2f}s')

       
        pruned_indices = torch.where(column_mask)[0]

        return pruned_indices
        
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
