import torch
import torch.nn as nn
import math
import time
import transformers
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

import matplotlib.pyplot as plt
import os

# def _hinv_diag_neumann(H, eps=1e-8):
#     d = torch.diagonal(H).clamp(min=eps)                 # (n,)
#     row_sq = (H ** 2).sum(dim=1)                          # sum_k H_jk^2  (n,)
#     offdiag = (row_sq - d ** 2).clamp(min=0.0)            # off-diagonal energy
#     hinv_diag = (1.0 / d) * (1.0 + offdiag / (d ** 2))    # ~ [H^-1]_jj
#     return hinv_diag.clamp(min=eps)  
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
class IsometricDiffusionPruner_Structured(object):
    def __init__(self, 
                layer, 
                layer_idx, 
                layer_name, 
                args, 
                logger=None, 
        ):
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        self.dtype = self.layer.weight.dtype
        self.logger = logger
        self.blocksize = args.blocksize
        self.n_eigen = args.n_eigen
        self.ffn_protect = args.ffn_protect
        self.attn_protect = args.attn_protect
        self.attn_protect_single = args.attn_protect_single

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.model_path = args.model_path
        self.saliency_mode = args.saliency_mode
        self.correction_mode = args.correction_mode

        # Gram Matrix (X^T X)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.sum_weight = 0
        self.original_norm = torch.norm(W, p='fro', dim=0)

    def add_batch(self, inp, out, W_new=1.0):
        if len(inp.shape) == 2: inp = inp.unsqueeze(0)
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        W_old = self.sum_weight
        W_total = W_old + W_new
        self.H *= (W_old / W_total)
        self.sum_weight = W_total
        
        norm_factor = math.sqrt(2 / self.sum_weight)
        inp = (norm_factor * inp).float()
        self.H += inp.matmul(inp.t())

    # @torch.no_grad()
    # def get_topological_skeleton(self, H_block, k_local):
    #     H_block = H_block.float() 
    #     H_block = torch.nan_to_num(H_block, nan=0.0, posinf=65500.0, neginf=-65500.0)

    #     D = torch.diag(H_block)
    #     eps_norm = 1e-4 * torch.mean(D).clamp(min=1e-7)
    #     D_inv_sqrt = 1.0 / torch.sqrt(D + eps_norm)
        
    #     L_local = D_inv_sqrt[:, None] * H_block * D_inv_sqrt[None, :]
    #     L_local = (L_local + L_local.t()) / 2.0

    #     evals, evecs = torch.linalg.eigh(L_local)
    #     return evecs[:, -k_local:], k_local

    @torch.no_grad()
    def get_topological_skeleton(self, H_block, k_local, diffusion_t=1.0, gap_select=True):
        H_block = torch.nan_to_num(H_block.float(), nan=0.0, posinf=65500., neginf=-65500.)
        D = torch.diag(H_block)
        eps_norm = 1e-4 * torch.mean(D).clamp(min=1e-7)
        D_inv_sqrt = 1.0 / torch.sqrt(D + eps_norm)
        A = D_inv_sqrt[:, None] * H_block * D_inv_sqrt[None, :]
        A = (A + A.t()) / 2.0

        evals, evecs = torch.linalg.eigh(A)
        evals = evals.flip(0); evecs = evecs.flip(1)      # descending

        if gap_select:
            # # largest relative spectral gap = natural dimensionality
            # lam = evals[:k_local].clamp(min=1e-10)
            # ratios = lam[:-1] / lam[1:]
            # k = int(torch.argmax(ratios[: max(1, k_local // 2)]).item()) + 1
            # k = max(k, 8)
            total_energy = torch.sum(evals)
            cumulative_energy = torch.cumsum(evals, dim=0)
            k = torch.searchsorted(cumulative_energy, 0.99 * total_energy).item() + 1
        else:
            k = k_local

        # if self.model_path == 'stabilityai/stable-diffusion-3.5-large':
        psi = evecs[:, :k] #* (evals[:k].clamp(min=0) ** diffusion_t)[None, :]
        # else:
        #     psi = evecs[:, :k] * (evals[:k].clamp(min=0) ** diffusion_t)[None, :]
        return psi, k

    @torch.no_grad()
    def struct_prune(self,
                target_sparsity,
                headsize,
                energy_threshold=None,
                epsilon=1e-4,
                iterations=None,
                percdamp=0.01,                    # <-- was 0.1; now a real relative ridge
                force_column_mask=None,           # <-- NEW
                heal_whole=False                  # <-- NEW: one block over all d_in
        ):
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
 
        d_out, d_in = W.shape
        num_heads = d_in // headsize
        tick = time.time()
 
        H_float = torch.nan_to_num(self.H.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        W = W.float()
        is_ffn = any(k in self.layer_name.lower() for k in
                     ["ff.net", "ff_context.net", "ff.linear_out", "ff_context.linear_out"])
 
        # ---- blocksize (needed for both selection and, unless heal_whole, healing) ----
        if self.model_path == 'stabilityai/stable-diffusion-xl-base-1.0':
            blocksize = (d_in // 4) if is_ffn else d_in
        else:
            blocksize = self.blocksize
        if headsize > 1:
            blocksize = max(blocksize, headsize)
            blocksize = (blocksize // headsize) * headsize
 
        # ================================================================
        # PHASE 1 — SELECTION  (skipped entirely if a mask is supplied)
        # ================================================================
        if force_column_mask is not None:
            column_mask = force_column_mask.to(self.dev)
        else:
            head_saliency = torch.zeros(num_heads, device=self.dev)
            if self.layer_name in ["proj_out", "attn.to_out"] and headsize == 1:
                head_saliency = torch.sum(W ** 2, dim=0) * torch.diag(H_float)
            elif is_ffn:
                if self.saliency_mode == 'neumann':
                    hinv = _hinv_diag_neumann(H_float)
                    head_saliency = torch.sum(W ** 2, dim=0) / hinv
                elif self.saliency_mode == 'magnitude':
                    head_saliency = torch.sum(W ** 2, dim=0)
                elif self.saliency_mode == 'effective_resistance':
                    hinv = _hinv_diag_effective_resistance(H_float, blocksize=blocksize)
                    head_saliency = torch.sum(W ** 2, dim=0) / hinv
                else: # mean_norm
                    head_saliency = torch.sum(W ** 2, dim=0) * torch.diag(H_float)
                
            else:
                if self.saliency_mode in ('neumann', 'magnitude', 'effective_resistance'):
                    if self.saliency_mode == 'neumann':
                        hinv = _hinv_diag_neumann(H_float)
                        col_score = torch.sum(W ** 2, dim=0) / hinv
                    elif self.saliency_mode == 'effective_resistance':
                        hinv = _hinv_diag_effective_resistance(H_float, blocksize=blocksize)
                        col_score = torch.sum(W ** 2, dim=0) / hinv
                    else:
                        col_score = torch.sum(W ** 2, dim=0)
                    for h in range(num_heads):
                        s_, e_ = h * headsize, (h + 1) * headsize
                        head_saliency[h] = col_score[s_:e_].mean()
                elif self.saliency_mode == 'mean_norm':
                    for h in range(num_heads):
                        s_, e_ = h * headsize, (h + 1) * headsize
                        W_h = W[:, s_:e_]; H_h = H_float[s_:e_, s_:e_]
                        head_saliency[h] = torch.mean(torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1))
                elif self.saliency_mode == 'max_norm':
                    for h in range(num_heads):
                        s_, e_ = h * headsize, (h + 1) * headsize
                        W_h = W[:, s_:e_]; H_h = H_float[s_:e_, s_:e_]
                        head_saliency[h] = torch.max(torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1))

            sorted_head_indices = torch.argsort(head_saliency)
            column_mask = torch.zeros(d_in, dtype=torch.bool, device=self.dev)
            num_blocks = (d_in + blocksize - 1) // blocksize
            pruned_per_block = torch.zeros(num_blocks, dtype=torch.int32, device=self.dev)
            global_pruned = 0
            num_to_prune = round(num_heads * target_sparsity)
 
            for h_idx in sorted_head_indices:
                if energy_threshold is not None:
                    if head_saliency[h_idx] > energy_threshold: break
                else:
                    if global_pruned >= num_to_prune: break
                col_start = h_idx * headsize
                b = col_start // blocksize
                b_start = b * blocksize
                b_end = min(b_start + blocksize, d_in)
                heads_in_block = (b_end - b_start) // headsize
                if self.layer_name == "proj_out":
                    prot = self.attn_protect_single if col_start < d_out else self.ffn_protect
                elif is_ffn:
                    prot = self.ffn_protect
                else:
                    prot = self.attn_protect
                if pruned_per_block[b] < int(heads_in_block * prot):
                    column_mask[col_start:col_start + headsize] = True
                    pruned_per_block[b] += 1
                    global_pruned += 1
 
        # ================================================================
        # PHASE 2 — ISOMETRIC CORRECTION
        #   heal_whole=True  -> ONE block over all d_in, full uncut H
        #   heal_whole=False -> per-block as before
        # ================================================================
        heal_bs = d_in if heal_whole else blocksize
        global_cols = torch.arange(d_in, device=self.dev)
        k_surv = "N/A"; survivors_local = []
 
        for i1 in range(0, d_in, heal_bs):
            i2 = min(i1 + heal_bs, d_in)
            mask_block = column_mask[i1:i2]
            if not mask_block.any():
                continue
 
            survivors_local = torch.where(~mask_block)[0]
            survivors_global = global_cols[i1:i2][survivors_local]
            dead_global = global_cols[i1:i2][mask_block]
            if len(survivors_local) == 0:
                W[:, dead_global] = 0.0
                continue
 
            H_nb = H_float[i1:i2, i1:i2]
            W_dead = W[:, i1:i2][:, mask_block].clone()
            W[:, dead_global] = 0.0

            # ── MODE: none — prune only, no compensation ──────────────────
            if self.correction_mode == 'none':
                if self.logger:
                    self.logger.info(f"[{self.layer_name}] blk{i1} correction=none "
                                     f"(dead zeroed, no healing)")
                continue

            residual_error = torch.matmul(W_dead, H_nb[mask_block, :]).float()
            H_survivors = H_nb[survivors_local][:, survivors_local]
            residual_survivors = residual_error[:, survivors_local]

            # ── MODE: full — classic OBS/OBC full-space solve ─────────────
            #   correction = residual_survivors @ H_survivors^-1
            #   (no ψ projection; inverts the whole survivor block)
            if self.correction_mode == 'full':
                damp = torch.eye(H_survivors.size(0), device=self.dev) * epsilon
                try:
                    H_inv = torch.linalg.pinv(H_survivors + damp,
                                              rcond=1e-4, hermitian=True)
                except Exception:
                    try:
                        H_inv = torch.inverse(H_survivors + damp)
                    except Exception:
                        if self.logger:
                            self.logger.warning(f"[{self.layer_name}] blk{i1} full-solve "
                                                f"failed. Skipping healing.")
                        continue
                correction = torch.matmul(residual_survivors, H_inv)
                k_surv = f"full({len(survivors_local)})"
            
            # ── MODE: spectral — YOUR truncated-eigenbasis correction ─────
            #   ψ = top-k eigvecs of normalized survivor Gram (99% energy)
            #   correction = (residual @ ψ) (ψᵀ H_S ψ)^-1 ψᵀ
            else:  # 'spectral'
                psi, k_surv = self.get_topological_skeleton(H_survivors, int(len(survivors_local)))
                H_sub = psi.t() @ H_survivors @ psi
    
                try:
                    H_sub_inv = torch.linalg.pinv(
                        H_sub + torch.eye(H_sub.size(0), device=self.dev) * epsilon, 
                        rcond=1e-4, hermitian=True
                    ) #.to(self.dtype)
                except Exception:
                    try:
                        H_sub_inv = torch.inverse(H_sub + torch.eye(H_sub.size(0), device=self.dev) * epsilon) #.to(self.dtype)
                    except Exception:
                        self.logger.warning(f"Inversion failed for {self.layer_name} block {i1}. Skipping healing.")
                        continue
                
                proj = torch.matmul(residual_survivors, psi)
                correction = torch.matmul(torch.matmul(proj, H_sub_inv), psi.t())

            # diagnostic: |dW|/|W| >> 1 means the correction is exploding.
            # This is the metric that distinguishes full (can explode on
            # ill-conditioned H) from spectral (truncation regularizes it).
            if self.logger:
                rel = (torch.norm(correction) / (torch.norm(W[:, survivors_global]) + 1e-8)).item()
                self.logger.info(f"[{self.layer_name}] blk{i1} mode={self.correction_mode} "
                                 f"k={k_surv}/{len(survivors_local)} |dW|/|W|={rel:.4f}")
 
            W[:, survivors_global] += correction.to(W.dtype)
 
        # ================================================================
        # PHASE 3 — FINALIZE
        # ================================================================
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if not torch.isfinite(W).all() and self.logger:
            self.logger.info(f"| WARNING | W for {self.layer_name} non-finite.")
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.dtype)
 
        sp = torch.sum(column_mask).item() / column_mask.numel()
        if self.logger:
            self.logger.info(f'[{self.layer_name}] Prune Done | k:{k_surv}/{len(survivors_local)} '
                             f'| Sparsity: {sp:.4f} | Time: {time.time()-tick:.2f}s')
        return torch.where(column_mask)[0]
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()

class IsometricDiffusionJointPruner_Structured(object):
    def __init__(self, 
                layer, 
                layer_idx, 
                layer_name, 
                args, 
                logger=None, 
                layer2=None, 
                layer2_name=None
        ):
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        self.dtype = self.layer.weight.dtype
        self.logger = logger
        self.blocksize = args.blocksize
        self.n_eigen = args.n_eigen
        self.ffn_protect = args.ffn_protect
        self.attn_protect = args.attn_protect

        self.model_path = args.model_path
        self.saliency_mode = args.saliency_mode
        self.correction_mode = args.correction_mode

        # --- Setup Primary Stream (Image) ---
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.sum_weight = 0
        self.original_norm = torch.norm(W, p='fro', dim=0)

        # --- Setup Secondary Stream (Text) if provided ---
        self.has_layer2 = layer2 is not None
        if self.has_layer2:
            self.layer2 = layer2
            self.layer2_name = layer2_name
            
            W2 = layer2.weight.data.clone()
            if isinstance(self.layer2, nn.Conv2d):
                W2 = W2.flatten(1)
            if isinstance(self.layer2, transformers.Conv1D):
                W2 = W2.t()
            
            self.rows2 = W2.shape[0]
            self.columns2 = W2.shape[1]

            self.H2 = torch.zeros((self.columns2, self.columns2), device=self.dev, dtype=torch.float32)
            self.sum_weight2 = 0
            self.original_norm2 = torch.norm(W2, p='fro', dim=0)

    def add_batch(self, inp, out, W_new=1.0, is_text_stream=False):
        """
        Accumulates activations into the correct Gram Matrix (H or H2).
        Make sure your hook function passes is_text_stream=True for the layer2 hook!
        """
        if len(inp.shape) == 2: inp = inp.unsqueeze(0)
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # ROUTE TO THE CORRECT MATRIX
        if is_text_stream and self.has_layer2:
            W_old = self.sum_weight2
            W_total = W_old + W_new
            self.H2 *= (W_old / W_total)
            self.sum_weight2 = W_total
            
            norm_factor = math.sqrt(2 / self.sum_weight2)
            inp = (norm_factor * inp).float()
            self.H2 += inp.matmul(inp.t())
        else:
            W_old = self.sum_weight
            W_total = W_old + W_new
            self.H *= (W_old / W_total)
            self.sum_weight = W_total
            
            norm_factor = math.sqrt(2 / self.sum_weight)
            inp = (norm_factor * inp).float()
            self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def get_topological_skeleton(self, H_block, k_local, diffusion_t=1.0, gap_select=True):
        H_block = torch.nan_to_num(H_block.float(), nan=0.0, posinf=65500., neginf=-65500.)
        D = torch.diag(H_block)
        eps_norm = 1e-4 * torch.mean(D).clamp(min=1e-7)
        D_inv_sqrt = 1.0 / torch.sqrt(D + eps_norm)
        A = D_inv_sqrt[:, None] * H_block * D_inv_sqrt[None, :]
        A = (A + A.t()) / 2.0

        evals, evecs = torch.linalg.eigh(A)
        evals = evals.flip(0); evecs = evecs.flip(1)      # descending

        if gap_select:
            # # largest relative spectral gap = natural dimensionality
            # lam = evals[:k_local].clamp(min=1e-10)
            # ratios = lam[:-1] / lam[1:]
            # k = int(torch.argmax(ratios[: max(1, k_local // 2)]).item()) + 1
            # k = max(k, 8)
            total_energy = torch.sum(evals)
            cumulative_energy = torch.cumsum(evals, dim=0)
            k = torch.searchsorted(cumulative_energy, 0.99 * total_energy).item() + 1
        else:
            k = k_local
        
        psi = evecs[:, :k] #* (evals[:k].clamp(min=0) ** diffusion_t)[None, :]

        return psi, k
    
    @torch.no_grad()
    def struct_prune(self, 
                target_sparsity, 
                headsize, 
                energy_threshold=None, 
                epsilon=1e-4, 
                iterations=None,
                percdamp=0.1,
                override_saliency=None,
                force_column_mask=None
        ):
        """
        Performs Head-wise Structured Pruning, executing jointly on two layers if configured.
        """
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()

        d_out, d_in = W.shape
        num_heads = d_in // headsize
        tick = time.time()

        H_float = self.H.float()
        W = W.float()

        # Stabilize Text Stream (Layer 2) if it exists
        if self.has_layer2:
            W2 = self.layer2.weight.data.clone().float()
            if isinstance(self.layer2, transformers.Conv1D): W2 = W2.t()
            
            H2_float = self.H2.float()
            W2 = W2.float()

        is_ffn = any(keyword in self.layer_name.lower() for keyword in ["ff.net", "ff_context.net", "ff.linear_out", "ff_context.linear_out"])
        head_saliency = torch.zeros(num_heads, device=self.dev)

        # --- PHASE 1: SALIENCY SELECTION ---
        if force_column_mask is not None:
            column_mask = force_column_mask
        else:
            if override_saliency is not None:
                head_saliency = override_saliency
            elif is_ffn:
                if self.saliency_mode == 'neumann':
                    head_saliency = torch.sum(W ** 2, dim=0) / _hinv_diag_neumann(H_float)
                elif self.saliency_mode == 'magnitude':
                    head_saliency = torch.sum(W ** 2, dim=0)
                else:
                    # UNCONTAMINATED COLUMN MATH: W^2 * diag(H)
                    W_sq_sum = torch.sum(W ** 2, dim=0)
                    H_diag = torch.diag(H_float)
                    head_saliency = W_sq_sum * H_diag
            else:
                if self.saliency_mode in ('neumann', 'magnitude'):
                    if self.saliency_mode == 'neumann':
                        hinv = _hinv_diag_neumann(H_float)
                        col_score = torch.sum(W ** 2, dim=0) / hinv
                        if self.has_layer2:
                            hinv2 = _hinv_diag_neumann(H2_float)
                            col_score2 = torch.sum(W2 ** 2, dim=0) / hinv2
                    else:
                        col_score = torch.sum(W ** 2, dim=0)
                        if self.has_layer2:
                            col_score2 = torch.sum(W2 ** 2, dim=0)

                    for h in range(num_heads):
                        idx_start = h * headsize
                        idx_end = (h + 1) * headsize
                        energy_img = col_score[idx_start:idx_end].mean().item()
                        energy_txt = 0.0
                        if self.has_layer2:
                            energy_txt = col_score2[idx_start:idx_end].mean().item()
                        # Combined Multimodal Saliency
                        # head_saliency[h] = energy_img + energy_txt
                        head_saliency[h] = math.sqrt((energy_img**2) + (energy_txt**2))
                elif self.saliency_mode == 'mean_norm':
                    for h in range(num_heads):
                        idx_start = h * headsize
                        idx_end = (h + 1) * headsize
                        
                        # Image Energy
                        W_h = W[:, idx_start:idx_end]                      
                        H_h = H_float[idx_start:idx_end, idx_start:idx_end] 
                        row_importance = torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1) 
                        energy_img = torch.mean(row_importance)
                        # Text Energy
                        energy_txt = 0.0
                        if self.has_layer2:
                            W2_h = W2[:, idx_start:idx_end]                      
                            H2_h = H2_float[idx_start:idx_end, idx_start:idx_end] 
                            row_importance2 = torch.norm(torch.matmul(W2_h, H2_h) * W2_h, dim=-1)
                            energy_txt = torch.mean(row_importance2)

                        # Combined Multimodal Saliency
                        # head_saliency[h] = energy_img + energy_txt
                        head_saliency[h] = math.sqrt((energy_img**2) + (energy_txt**2))
                elif self.saliency_mode == 'max_norm':
                    for h in range(num_heads):
                        idx_start = h * headsize
                        idx_end = (h + 1) * headsize
                        
                        # Image Energy
                        W_h = W[:, idx_start:idx_end]                      
                        H_h = H_float[idx_start:idx_end, idx_start:idx_end] 
                        row_importance = torch.norm(torch.matmul(W_h, H_h) * W_h, dim=-1) 
                        energy_img = torch.max(row_importance)
                        # Text Energy
                        energy_txt = 0.0
                        if self.has_layer2:
                            W2_h = W2[:, idx_start:idx_end]                      
                            H2_h = H2_float[idx_start:idx_end, idx_start:idx_end] 
                            row_importance2 = torch.norm(torch.matmul(W2_h, H2_h) * W2_h, dim=-1)
                            energy_txt = torch.max(row_importance2)

                        # Combined Multimodal Saliency
                        # head_saliency[h] = energy_img + energy_txt
                        head_saliency[h] = math.sqrt((energy_img**2) + (energy_txt**2))

            blocksize = self.blocksize 
            if headsize > 1:
                blocksize = max(blocksize, headsize)
                blocksize = (blocksize // headsize) * headsize

            sorted_head_indices = torch.argsort(head_saliency)
            column_mask = torch.zeros(d_in, dtype=torch.bool, device=self.dev)
            num_blocks = (d_in + blocksize - 1) // blocksize
            pruned_heads_per_block = torch.zeros(num_blocks, dtype=torch.int32, device=self.dev)
            
            global_pruned_count = 0
            num_to_prune_global = round(num_heads * target_sparsity)
            
            for h_idx in sorted_head_indices:
                if energy_threshold is not None:
                    if head_saliency[h_idx] > energy_threshold:
                        break
                else:
                    if global_pruned_count >= num_to_prune_global:
                        break
                    
                col_start = h_idx * headsize
                block_idx = col_start // blocksize
                b_start = block_idx * blocksize
                b_end = min(b_start + blocksize, d_in)
                heads_in_block = (b_end - b_start) // headsize

                if is_ffn:
                    max_pruned_in_block = int(heads_in_block * self.ffn_protect)
                else:
                    max_pruned_in_block = int(heads_in_block * self.attn_protect)

                if pruned_heads_per_block[block_idx] < max_pruned_in_block:
                    column_mask[col_start : col_start + headsize] = True
                    pruned_heads_per_block[block_idx] += 1
                    global_pruned_count += 1

        # --- PHASE 2: JOINT LOCAL ISOMETRIC CORRECTION ---
        global_cols = torch.arange(d_in, device=self.dev)
        
        # Build list of tensors to heal so we can execute the exact same topological mask on both
        # layers_to_heal = [(W, self.H.to(self.dtype), self.layer, self.layer_name)]
        layers_to_heal = [(W, self.H, self.layer, self.layer_name)]
        if self.has_layer2:
            # layers_to_heal.append((W2, self.H2.to(self.dtype), self.layer2, self.layer2_name))
            layers_to_heal.append((W2, self.H2, self.layer2, self.layer2_name))

        for current_W, current_H, current_layer, current_name in layers_to_heal:
            for i1 in range(0, d_in, blocksize):
                i2 = min(i1 + blocksize, d_in)
                mask_block = column_mask[i1:i2]
                
                if not mask_block.any(): 
                    k_surv = "N/A"
                    survivors_local = []
                    continue
                
                survivors_local = torch.where(~mask_block)[0]
                survivors_global = global_cols[i1:i2][survivors_local]
                dead_global_cols = global_cols[i1:i2][mask_block]
                
                if len(survivors_local) == 0:
                    current_W[:, dead_global_cols] = 0.0
                    continue
                
                H_nb = current_H[i1:i2, i1:i2].float()
                W_block_original = current_W[:, i1:i2].clone()
                target_norm = torch.norm(W_block_original) # Baseline for variance preservation

                Y_target = torch.matmul(W_block_original, H_nb)
                
                # ATOMIC PRUNING
                current_W[:, dead_global_cols] = 0.0

                # ── MODE: none — prune only, no compensation ──────────────
                if self.correction_mode == 'none':
                    k_surv = "none"
                    if self.logger:
                        self.logger.info(f"[{current_name}] blk{i1} correction=none "
                                         f"(dead zeroed, no healing)")
                    continue


                # --- DIRECT RESIDUAL CALCULATION ---
                # Calculate the exact activation error caused strictly by the amputated weights.
                # This avoids the floating-point drift of subtracting two massive matrices.
                W_dead = W_block_original[:, mask_block]
                H_dead_to_all = H_nb[mask_block, :]
                residual_error = torch.matmul(W_dead, H_dead_to_all).float()

                # --- THE CARDINALITY FIX ---
                # Extract topology of the survivors only
                H_survivors = H_nb[survivors_local][:, survivors_local]
                residual_survivors = residual_error[:, survivors_local]

                # ── MODE: full — classic OBS/OBC full-space solve ─────────
                #   correction = residual_survivors @ H_survivors^-1
                if self.correction_mode == 'full':
                    damp = torch.eye(H_survivors.size(0), device=self.dev) * epsilon
                    try:
                        H_inv = torch.linalg.pinv(H_survivors + damp,
                                                  rcond=1e-4, hermitian=True)
                    except Exception:
                        try:
                            H_inv = torch.inverse(H_survivors + damp)
                        except Exception:
                            if self.logger:
                                self.logger.warning(f"[{current_name}] blk{i1} full-solve "
                                                    f"failed. Skipping healing.")
                            continue
                    correction = torch.matmul(residual_survivors, H_inv)
                    k_surv = f"full({len(survivors_local)})"
                
                # ── MODE: spectral — YOUR truncated-eigenbasis correction ─
                else:  # 'spectral'
                    # --- THE LOW-PASS FILTER FIX ---
                    # Use the full rank of the survivors rather than clamping to n_eigen.
                    # This ensures high-frequency spatial details are included in the healing projection.
                    # k_surv = min(self.n_eigen, len(survivors_local), len(survivors_local)//2) 
                    k_surv = int(len(survivors_local))
                    psi_survivors, k_surv = self.get_topological_skeleton(H_survivors, k_surv)

                    # psi_survivors = torch.ones_like(psi_survivors)

                    # # --- ADAPTIVE DAMPING FIX ---
                    # H_sub = psi_survivors.t() @ H_survivors.to(self.dtype) @ psi_survivors
                    H_sub = psi_survivors.t() @ H_survivors @ psi_survivors

                    try:
                        H_sub_inv = torch.linalg.pinv(
                            H_sub + torch.eye(H_sub.size(0), device=self.dev) * epsilon, 
                            rcond=1e-4, hermitian=True
                        ) #.to(self.dtype)
                    except Exception:
                        try:
                            H_sub_inv = torch.inverse(H_sub + torch.eye(H_sub.size(0), device=self.dev) * epsilon) #.to(self.dtype)
                        except Exception:
                            if self.logger:
                                self.logger.warning(f"Inversion failed for {current_name} block {i1}. Skipping healing.")
                            continue

                    # --- CROSS-DIMENSIONAL PROJECTION ---
                    # proj = torch.matmul(residual_survivors.to(self.dtype), psi_survivors.to(self.dtype))
                    proj = torch.matmul(residual_survivors, psi_survivors)
                    correction = torch.matmul(torch.matmul(proj, H_sub_inv), psi_survivors.t())

                # diagnostic: |dW|/|W| >> 1 => correction exploding.
                # full can explode on ill-conditioned H; spectral truncation
                # should stay bounded — this log IS the stability evidence.
                if self.logger:
                    rel = (torch.norm(correction) /
                           (torch.norm(current_W[:, survivors_global]) + 1e-8)).item()
                    self.logger.info(f"[{current_name}] blk{i1} mode={self.correction_mode} "
                                     f"k={k_surv}/{len(survivors_local)} |dW|/|W|={rel:.4f}")
                    
                # Apply correction
                current_W[:, survivors_global] += correction.to(current_W.dtype)

            # --- PHASE 3: FINALIZE INLINE ---
            torch.cuda.synchronize()
            if isinstance(current_layer, transformers.Conv1D):
                current_W = current_W.t()
        
            if not torch.isfinite(current_W).all():
                if self.logger: self.logger.info(f"| WARNING | W for {current_name} contains non-finite values.")
            
            current_layer.weight.data = current_W.reshape(current_layer.weight.shape).to(self.dtype)

        zeros = torch.sum(column_mask).item()
        sp = zeros / column_mask.numel()
        if self.logger:
            prune_msg = f'[{self.layer_name}]'
            if self.has_layer2:
                prune_msg = f'[{self.layer_name} + {self.layer2_name}]'
            self.logger.info(f'{prune_msg} Joint Prune Done | k_surv: {k_surv}/{len(survivors_local)} | Sparsity: {sp:.4f} | Time: {time.time() - tick:.2f}s')
        
        return torch.where(column_mask)[0] #, column_mask

    def free(self):
        self.H = None
        if self.has_layer2:
            self.H2 = None
        torch.cuda.empty_cache()