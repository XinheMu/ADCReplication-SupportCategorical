import torch 
torch.set_grad_enabled(False)
torch.set_num_threads(30)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
import ast
import torch.nn as nn
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
from cuml.cluster import KMeans
import pickle
import os
import glob
import re
from torch import nn
from torch.distributions import Normal
from typing import Tuple, Union

class GridMapper:
    def __init__(self):
        self.a_values = []   # list of tensors, distinct values per dimension
        self.grids = []      # list of tensors, optimal grids per dimension
        self.operate_dims = []

    def build_optimal_grids(self, gmm, data_used, operate_dims=None, blocklist_dims=None, r=0.1,
                            max_iter=3001, lr=0.01, eps=1e-12, verbose=False):
        """
        Train optimal grids for dimensions with few distinct values.
        Automatically adds any dimension not already in `operate_dims` if
        it has fewer than 100 distinct empirical values.

        Args:
            gmm: (2*d+1, K) tensor, means / stds / weights.
            data_used: (M, d) tensor.
            operate_dims: optional initial list of dimension indices (0-based).
            r: float in (0, 0.5) for cut point constraint.
            max_iter, lr, eps: optimisation controls.
            verbose: if True, prints per-step losses.
        """
        device = data_used.device
        gmm = gmm.to(device)
        d = gmm.shape[0] // 2
        K = gmm.shape[1]
        means = gmm[:d, :]
        stds = gmm[d:2*d, :]
        weights = (gmm[-1, :].to(torch.float64)/torch.sum(gmm[-1, :].to(torch.float64))).to(torch.float32)
        M = data_used.shape[0]

        # --- Automatically extend operate_dims ---
        if operate_dims is None:
            operate_dims = []
        operate_set = set(operate_dims)

        for dim in range(d):
            if dim in operate_set or dim in blocklist_dims:
                continue
            # Quick check: number of distinct values in this dimension
            # NOTE: For huge data (M ~ 1e8) you may want to use a random sample
            # instead of the full column. torch.unique on 1e8 elements is heavy.
            col = data_used[:, dim].contiguous()
            unique_vals = torch.unique(col)
            if unique_vals.numel() < 100:
                operate_set.add(dim)

        self.operate_dims = sorted(operate_set)
        self.a_values = []
        self.grids = []

        # --- Core optimisation loop over the final dimension list ---
        for j in self.operate_dims:
            col = data_used[:, j].contiguous()
            unique_vals, counts = torch.unique(col, return_counts=True)
            a = unique_vals.float()
            p = counts.float() / M
            N = a.numel()

            # Set up unconstrained parameters for cut points
            params = []
            raw0 = nn.Parameter(torch.zeros(1, device=device))
            params.append(raw0)

            raw_interior = []
            for i in range(1, N):
                delta = a[i] - a[i-1]
                lower = a[i-1] + r * delta
                upper = a[i-1] + (1.0 - r) * delta
                raw = nn.Parameter(torch.zeros(1, device=device))
                raw_interior.append((raw, lower, upper))
                params.append(raw)

            rawN = nn.Parameter(torch.zeros(1, device=device))
            params.append(rawN)

            # Marginal GMM distribution for dimension j
            normal = Normal(means[j], stds[j])

            optimizer = torch.optim.Adam(params, lr=lr)
            best_loss = float('inf')
            best_grid = None

            for step in range(max_iter):
                def closure():
                    optimizer.zero_grad()
                    g_list = []
                    g0 = a[0] - torch.nn.functional.softplus(raw0)
                    g_list.append(g0)
                    for raw, low, up in raw_interior:
                        g_i = low + (up - low) * torch.sigmoid(raw)
                        g_list.append(g_i)
                    gN = a[-1] + torch.nn.functional.softplus(rawN)
                    g_list.append(gN)

                    g = torch.cat(g_list)               # (N+1,)
                    cdf = normal.cdf(g.unsqueeze(-1))   # (N+1, K)
                    P = (cdf[1:] - cdf[:-1])            # (N, K)
                    P = (P * weights).sum(dim=1).clamp(min=eps)   # (N,)

                    log_ratio = (P / p).log()
                    loss = (p * log_ratio.pow(2)).sum()
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                if verbose and step % 1000 == 0:
                    print(f"dim {j}, step {step}: loss = {loss.item():.6f}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    with torch.no_grad():
                        g0_best = a[0] - torch.nn.functional.softplus(raw0)
                        g_best = [g0_best]
                        for raw, low, up in raw_interior:
                            g_best.append(low + (up - low) * torch.sigmoid(raw))
                        g_best.append(a[-1] + torch.nn.functional.softplus(rawN))
                        best_grid = torch.cat(g_best).detach().clone()

            self.a_values.append(a)
            self.grids.append(best_grid)
            
        # --- Report ---
        print(f"Training completed. Final operate_dims: {self.operate_dims}")

    def map_values(self, Targ):
        """Maps each value in Targ to a grid point."""
        if not self.a_values:
            raise RuntimeError("No grids built. Call build_optimal_grids first.")
        device = Targ.device
        L = len(self.operate_dims)
        assert Targ.shape == (2, L), f"Expected Targ shape (2, {L}), got {Targ.shape}"
        mapped = torch.empty_like(Targ)
        for i, a in enumerate(self.a_values):
            g = self.grids[i].to(device)
            a = a.to(device)
            idx = torch.bucketize(Targ[:, i], a)
            mapped[:, i] = g[idx]
        return mapped

    def save(self, filepath):
        """Persist the mapper to disk."""
        state = {
            'a_values': [t.cpu() for t in self.a_values],
            'grids': [t.cpu() for t in self.grids],
            'operate_dims': self.operate_dims
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath, map_location=None):
        """Load a mapper from disk."""
        state = torch.load(filepath, map_location=map_location)
        mapper = cls()
        mapper.a_values = state['a_values']
        mapper.grids = state['grids']
        mapper.operate_dims = state['operate_dims']
        return mapper

class BayesnetEstimator:
    def __init__(self, bayes_called_attribute, bayes_source_attribute, bayes_assist_attribute, 
                 S1, S2, bin_val, mcv_val):
        """
        Calculates P(Called | Source, Assist).
        Grid is defined by Source (dim 1) and Assist (dim 2).
        Histogram models the Called attribute.
        """
        self.source_attr = bayes_source_attribute # Grid Axis 1
        self.called_attr = bayes_called_attribute # Target Variable (Histogram)
        self.assist_attr = bayes_assist_attribute # Grid Axis 2
        
        # Params (overridden by load)
        self.S1 = S1 
        self.S2 = S2
        self.bin_val = bin_val
        self.mcv_val = mcv_val
        
        self.filename = f"FD_{bayes_called_attribute}_{bayes_source_attribute}_{bayes_assist_attribute}.npy"
        self.filename = dataset_name+'/'+self.filename        

        # Edges for the Conditioning Attributes (Source, Assist)
        self.source_edges = None
        self.assist_edges = None
        
        # Dense Tensors for Vectorized Inference on the Called Attribute
        self.dense_mcv_vals = None   
        self.dense_mcv_probs = None  
        self.dense_hist_edges = None 
        self.dense_hist_probs = None 

    def train(self, data_used):
        global PKFK_Flag, PKFK_info
        # Extract columns
        source_col = data_used[:, self.source_attr]
        called_col = data_used[:, self.called_attr] # The target
        assist_col = data_used[:, self.assist_attr] 
        # --- (3) PK-FK Heuristic ---
        # Logic: Does Source determine Called?
        vals, counts = torch.unique(source_col, return_counts=True)
        if len(vals)<2:
            min_gap=1e-5
        else:
            min_gap=torch.min(vals[1:]-vals[:-1])
        sorted_indices = torch.argsort(counts, descending=True)
        top_k = 50
        is_functional_dep = True
        check_limit = min(top_k, len(vals))
        
        if check_limit > 0:
            for i in range(check_limit):
                val = vals[sorted_indices[i]]
                mask = (source_col == val)
                # Check variance of CALLED column given SOURCE value
                called_subset = called_col[mask]
                if len(torch.unique(called_subset)) > 1:
                    is_functional_dep = False
                    break
        else:
            is_functional_dep = False

        if is_functional_dep:
            n_distinct_source = len(vals)
            self.S1 = min(1000, 1+torch.round((vals[-1]-vals[0])/min_gap))
            self.S2 = 1 
            calc_param = 0.001 * n_distinct_source
            self.mcv_val = min(2, int(np.ceil(calc_param)))
            self.bin_val = min(5, int(np.ceil(calc_param)))
            PKFK_Flag[flag_change_location]=True
            PKFK_info[flag_change_location]=[min(1000, 1+torch.round((vals[-1]-vals[0])/min_gap)),1,5,2]

        # --- (4) Adaptive Slicing (on Conditioning Attributes) ---
        self.source_edges = self._get_adaptive_edges(source_col, self.S1, min_gap)
        self.S1 = len(self.source_edges) - 1
        self.assist_edges = self._get_adaptive_edges(assist_col, self.S2, min_gap)
        self.S2 = len(self.assist_edges) - 1

        # Binning based on Source and Assist
        source_indices = torch.bucketize(source_col, self.source_edges, right=True) - 1
        assist_indices = torch.bucketize(assist_col, self.assist_edges, right=True) - 1
        source_indices = torch.clamp(source_indices, 0, self.S1 - 1)
        assist_indices = torch.clamp(assist_indices, 0, self.S2 - 1)
        
        temp_grid_models = {}
        temp_redirect_map = {}
        non_empty_coords = []

        for j in range(self.S1):
            for k in range(self.S2):
                mask = (source_indices == j) & (assist_indices == k)
                # We build histogram on the CALLED attribute
                subset_called = called_col[mask]
                
                if len(subset_called) > 0:
                    temp_grid_models[(j, k)] = self._build_1d_hist(subset_called)
                    non_empty_coords.append((j, k))
                    temp_redirect_map[(j, k)] = (j, k)
                else:
                    temp_grid_models[(j, k)] = None

        # Nearest Neighbor for Empty Cells
        for j in range(self.S1):
            for k in range(self.S2):
                if temp_grid_models[(j, k)] is None:
                    best_coord = None
                    min_dist = (float('inf'), float('inf'))
                    for (nj, nk) in non_empty_coords:
                        dist_j = abs(j - nj)
                        dist_k = abs(k - nk)
                        if (dist_j, dist_k) < min_dist:
                            min_dist = (dist_j, dist_k)
                            best_coord = (nj, nk)
                    temp_redirect_map[(j, k)] = best_coord

        # [Inside train() method, replacing the final lines starting from save_dict]

        # 1. Densify to create the full tensors first (keeps logic clean)
        self._densify_models(temp_grid_models, temp_redirect_map)

        # 2. OPTIMIZATION: Extract only Min/Max for histograms
        # dense_hist_edges shape: (S1, S2, bin_val + 1)
        # We only need the first (0) and last (-1) element along the last dimension.
        # Shape becomes (S1, S2, 2)
        if self.dense_hist_edges.numel() > 0:
            hist_bounds = torch.stack([
                self.dense_hist_edges[:, :, 0], 
                self.dense_hist_edges[:, :, -1]
            ], dim=-1)
        else:
            hist_bounds = torch.empty((self.S1, self.S2, 2))

        # 3. Save with Compression and Precision Reduction
        save_dict = {
            'source_edges': self.source_edges.cpu().numpy().astype(np.float32), 
            'assist_edges': self.assist_edges.cpu().numpy().astype(np.float32),
            
            # MCV Vals: Float32 (Keep precision for DB values > 65,504)
            'dense_mcv_vals': self.dense_mcv_vals.cpu().numpy().astype(np.float32),
            
            # MCV Probs: Float16 (Safe for probabilities, 50% storage reduction)
            'dense_mcv_probs': self.dense_mcv_probs.cpu().numpy().astype(np.float16),
            
            # Hist Bounds (Min/Max): Float32
            'dense_hist_bounds': hist_bounds.cpu().numpy().astype(np.float32),
            
            # Hist Probs: Float16
            'dense_hist_probs': self.dense_hist_probs.cpu().numpy().astype(np.float16),
            
            'S1': self.S1, 'S2': self.S2,
            'bin_val': self.bin_val, 'mcv_val': self.mcv_val
        }
        print('saving to')
        print(self.filename)
        np.save(self.filename, save_dict)

    def load(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"No model found at {self.filename}")
        
        data = np.load(self.filename, allow_pickle=True).item()
        
        # 1. Metadata
        if 'S1' in data: self.S1 = int(data['S1'])
        if 'S2' in data: self.S2 = int(data['S2'])
        if 'bin_val' in data: self.bin_val = int(data['bin_val'])
        if 'mcv_val' in data: self.mcv_val = int(data['mcv_val'])
        
        # 2. Restore Edges (Float32)
        # We enforce float32 to avoid implicit double precision overhead
        self.source_edges = torch.from_numpy(data['source_edges']).float()
        self.assist_edges = torch.from_numpy(data['assist_edges']).float()
        
        # 3. Restore Dense Tensors
        # MCVs
        self.dense_mcv_vals = torch.from_numpy(data['dense_mcv_vals']).float()
        # Cast Probs back to Float32 for accurate summation/vectorization during inference
        self.dense_mcv_probs = torch.from_numpy(data['dense_mcv_probs']).float()
        
        # Histograms
        self.dense_hist_probs = torch.from_numpy(data['dense_hist_probs']).float()
        
        # 4. Reconstruct Full Histogram Edges from Bounds
        # Input: (S1, S2, 2) -> Output: (S1, S2, bin_val + 1)
        hist_bounds = torch.from_numpy(data['dense_hist_bounds']).float()
        
        # Generate equi-distant steps 0 to 1
        # Shape: (bin_val + 1)
        steps = torch.linspace(0, 1, self.bin_val + 1)
        
        # Vectorized Linear Interpolation:
        # Edge[i] = Start + (End - Start) * Step[i]
        # Broadcasting: (S1, S2, 1) + (S1, S2, 1) * (B+1) -> (S1, S2, B+1)
        
        starts = hist_bounds[:, :, 0].unsqueeze(-1)
        ends = hist_bounds[:, :, 1].unsqueeze(-1)
        
        self.dense_hist_edges = starts + (ends - starts) * steps

    def _densify_models(self, grid_models, redirect_map):
        # Initialize Tensors
        self.dense_mcv_vals = torch.zeros((self.S1, self.S2, self.mcv_val))
        self.dense_mcv_probs = torch.zeros((self.S1, self.S2, self.mcv_val))
        
        self.dense_hist_edges = torch.zeros((self.S1, self.S2, self.bin_val + 1))
        self.dense_hist_probs = torch.zeros((self.S1, self.S2, self.bin_val))
        
        for j in range(self.S1):
            for k in range(self.S2):
                target_j, target_k = redirect_map[(j, k)]
                model = grid_models[(target_j, target_k)]
                
                if model is not None:
                    # Fill MCVs
                    for i, mcv in enumerate(model['mcvs']):
                        if i < self.mcv_val:
                            self.dense_mcv_vals[j, k, i] = mcv['val']
                            self.dense_mcv_probs[j, k, i] = mcv['prob']
                    
                    # Fill Histograms
                    if model['hist'] is not None:
                        edges = model['hist']['edges']
                        probs = model['hist']['probs']
                        curr_bins = len(probs)
                        
                        self.dense_hist_edges[j, k, :curr_bins+1] = edges
                        if curr_bins < self.bin_val:
                            self.dense_hist_edges[j, k, curr_bins+1:] = edges[-1]
                        self.dense_hist_probs[j, k, :curr_bins] = probs

    def _get_adaptive_edges(self, col_data, requested_S, min_gap=1e-5):
        uniques = torch.unique(col_data)
        n_distinct = len(uniques)
        min_val, max_val = col_data.min(), col_data.max()
        epsilon = min_gap/2
        
        if requested_S >= n_distinct:
            sorted_u, _ = torch.sort(uniques)
            if len(sorted_u) == 1:
                return torch.tensor([min_val - epsilon, max_val + epsilon])
            midpoints = (sorted_u[:-1] + sorted_u[1:]) / 2.0
            return torch.cat([torch.tensor([min_val - epsilon]), midpoints, torch.tensor([max_val + epsilon])])
        else:
            return torch.linspace(min_val, max_val + epsilon, requested_S + 1)

    def _build_1d_hist(self, data):
        total = len(data)
        vals, counts = torch.unique(data, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        
        mcvs = []
        remaining_mask = torch.ones(total, dtype=torch.bool)
        mcv_mass = 0
        limit_mcv = self.mcv_val if self.mcv_val is not None else 0
        
        for i in range(min(limit_mcv, len(vals))):
            val = vals[sorted_indices[i]].item()
            count = counts[sorted_indices[i]].item()
            prob = count / total
            mcvs.append({'val': val, 'prob': prob})
            mcv_mass += prob
            remaining_mask = remaining_mask & (data != val)
            
        remaining_data = data[remaining_mask]
        hist_data = None
        
        if len(remaining_data) > 0 and self.bin_val > 0:
            h_min, h_max = remaining_data.min(), remaining_data.max()
            h_edges = torch.linspace(h_min, h_max + 1e-5, self.bin_val + 1)
            h_counts = torch.histc(remaining_data, bins=self.bin_val, min=h_min, max=h_max)
            h_probs = (h_counts / len(remaining_data)) * (1 - mcv_mass)
            hist_data = {'edges': h_edges, 'probs': h_probs}
            
        return {'mcvs': mcvs, 'hist': hist_data}

    def get_all_slice_cond_probs(self, queried_range):
        """
        Calculates P(Called in Range | Source=j, Assist=k)
        """
        p1, q1 = queried_range[0], queried_range[1]
        
        # MCV Contribution
        mcv_mask = (self.dense_mcv_vals >= p1) & (self.dense_mcv_vals <= q1)
        mcv_contrib = torch.sum(self.dense_mcv_probs * mcv_mask.float(), dim=2)
        
        # Histogram Contribution
        b_starts = self.dense_hist_edges[:, :, :-1]
        b_ends = self.dense_hist_edges[:, :, 1:]
        
        o_starts = torch.maximum(b_starts, p1)
        o_ends = torch.minimum(b_ends, q1)
        
        overlap_lens = torch.maximum(o_ends - o_starts, torch.tensor(0.0))
        bucket_lens = b_ends - b_starts
        
        valid_buckets = bucket_lens > 0
        fractions = torch.zeros_like(overlap_lens)
        fractions[valid_buckets] = overlap_lens[valid_buckets] / bucket_lens[valid_buckets]
        
        hist_contrib = torch.sum(fractions * self.dense_hist_probs, dim=2)
        
        return mcv_contrib + hist_contrib

def get_adjust_vals(dim,edges,means,stds,weights):
    # Ensure they are safely evaluated on the CPU with numpy-facing edges and data
    means = means.cpu()
    stds = stds.cpu()
    weights = weights.cpu()

    edges[0]=edges[0]-1
    edges[-1]=edges[-1]+1
    dim_ingmm=dim-np.sum(dim>bayes_called_attributes)
    full_weights=np.zeros((2,len(edges)-1))
    real_weights=np.zeros((len(edges)-1))
    gmm_weights=np.zeros((len(edges)-1))
    for i in range(0,len(edges)-1):
        real_weights[i]=torch.sum((data_used[:,dim_ingmm]>edges[i])*(data_used[:,dim_ingmm]<=edges[i+1]))/size
        gmm_weights[i]=torch.sum((cdf(((edges[i+1]-means[dim_ingmm,:])/stds[dim_ingmm,:]).to(torch.float64))-cdf(((edges[i]-means[dim_ingmm,:])/stds[dim_ingmm,:]).to(torch.float64)))*weights)/torch.sum(weights)
    weight_adjustment=real_weights/gmm_weights
    full_weights[0,:]=real_weights
    full_weights[1,:]=weight_adjustment
    np.save(dataset_name+'/'+dataset_name+'_bias_dim'+str(dim)+'.npy',full_weights)

def cdf(x):
    return 0.5*(1+torch.erf((x)/(1.41421356))).to(torch.float32)

def regu_step1(starting_col=0):
    global reg_const_one, reg_const_two, reg_const_three, data_used_primitive, data_used, size, corrupted_data_size, data_init
    data_init=pd.read_csv(dataset_name+'training/'+'original'+dataset_name+'.csv',index_col=False)
    data_init=data_init.to_numpy()
    data_init=data_init[1:,starting_col:starting_col+dimension_init]
    orisize=(torch.tensor(data_init).size())[0]
    data_init=torch.tensor(data_init).to(torch.float32)
    data_init=data_init[data_init[:,0]>(nan_to+1e-6),:]
    data_init[:,date_like]=date_to_days(data_init[:,date_like]).to(torch.float32)
    reg_const_zero=torch.zeros((6,dimension_init))
    for i in numerical_attributes:
        unique_values=torch.unique(data_init[:,i],sorted=True,return_counts=False)
        gap_size=unique_values[1:]-unique_values[:-1]
        maxgap,maxgap_position=torch.topk(gap_size,2,largest=True,sorted=False)
        reg_const_zero[0,i]=unique_values[min(maxgap_position[0],maxgap_position[1])]
        reg_const_zero[1,i]=unique_values[min(maxgap_position[0],maxgap_position[1])+1]
        reg_const_zero[2,i]=reg_const_zero[1,i]-max((unique_values[min(maxgap_position[0],maxgap_position[1])+1]-unique_values[min(maxgap_position[0],maxgap_position[1])])-(torch.max(data_init[:,i])-torch.min(data_init[:,i]))/15,0)
        reg_const_zero[3,i]=unique_values[max(maxgap_position[0],maxgap_position[1])]
        reg_const_zero[4,i]=unique_values[max(maxgap_position[0],maxgap_position[1])+1]
        reg_const_zero[5,i]=reg_const_zero[4,i]-max((unique_values[max(maxgap_position[0],maxgap_position[1])+1]-unique_values[max(maxgap_position[0],maxgap_position[1])])-(torch.max(data_init[:,i])-torch.min(data_init[:,i]))/15,0)
    data_used=data_init.clone()
    data_used[:,numerical_attributes]=data_init[:,numerical_attributes]-(reg_const_zero[1:2,numerical_attributes]-reg_const_zero[2:3,numerical_attributes])*(data_init[:,numerical_attributes]>(reg_const_zero[2:3,numerical_attributes]))-(reg_const_zero[4:5,numerical_attributes]-reg_const_zero[5:6,numerical_attributes])*(data_init[:,numerical_attributes]>(reg_const_zero[5:6,numerical_attributes]))
    reg_const_one=torch.zeros((3,dimension_init))
    for i in range(0,dimension_init):
        reg_const_one[:,i:i+1]=torch.tensor([[torch.max(data_used[:,i])],[torch.min(data_used[:,i])],[(1/3)*(torch.max(data_used[:,i])-torch.min(data_used[:,i]))]])
    data_used[:,numerical_attributes]=(data_used[:,numerical_attributes]-reg_const_one[1,numerical_attributes])/(reg_const_one[2,numerical_attributes])-1.5
    data_used_primitive=data_used.clone()
    size=(data_used.size())[0]
    corrupted_data_size=orisize-size
    reg_const_two=torch.zeros((4,dimension_init))
    for i in numerical_attributes:
        unique_values,counts=torch.unique(data_used[:,i],return_counts=True)
        grad_left=torch.zeros_like(unique_values)
        for j in range(1,(unique_values.size())[0]):
            if counts[j]<size/200:
                grad_left[j]=0
            else:
                grad_left[j]=counts[j]-torch.sum((data_used[:,i]<unique_values[j])*(data_used[:,i]>(unique_values[j]-0.05)))*(0.05/min(unique_values[j]+1.5,0.05))
        top_counts, top_indices=torch.topk(grad_left,2)
        reg_const_two[0:2,i]=unique_values[top_indices]*(grad_left[top_indices]>(size/200))-100*(grad_left[top_indices]<=(size/200))
        grad_right=torch.zeros_like(unique_values)
        for j in range(0,(unique_values.size())[0]-1):
            if counts[j]<size/200:
                grad_right[j]=0
            else:
                grad_right[j]=counts[j]-torch.sum((data_used[:,i]>unique_values[j])*(data_used[:,i]<(unique_values[j]+0.05)))*(0.05/min(1.5-unique_values[j],0.05))
        top_counts, top_indices=torch.topk(grad_right,2)
        reg_const_two[2:4,i]=unique_values[top_indices]*(grad_right[top_indices]>(size/200))-100*(grad_right[top_indices]<=(size/200))
        if has_categorical:
            data_used[:,i]=data_used[:,i]-0.05*(data_used[:,i]<reg_const_two[0,i])-0.05*(data_used[:,i]<reg_const_two[1,i])+0.05*(data_used[:,i]>reg_const_two[2,i])+0.05*(data_used[:,i]>reg_const_two[3,i])
        else:
            data_used[:,i]=data_used[:,i]-0.05*(data_used[:,i]<reg_const_two[0,i])-0.05*(data_used[:,i]<reg_const_two[1,i])+0.05*(data_used[:,i]>reg_const_two[2,i])+0.05*(data_used[:,i]>reg_const_two[3,i])
    if has_categorical:
        maxgap_categorical=np.load(dataset_name+'training/maxgapstats.npy')
    for i in range(0,dimension_init):
        if i not in numerical_attributes:
            data_used[:,i]=data_used[:,i]/(1*(1+2*(reg_const_one[0,i]>2)+6*(reg_const_one[0,i]>8)+18*(reg_const_one[0,i]>32)+54*(reg_const_one[0,i]>128)))
            for j in range(0,4):
                if (maxgap_categorical[2*j+1,i]>5e-4 and reg_const_one[0,i]>32) or (maxgap_categorical[2*j+1,i]>1e-2 and j<2 and reg_const_one[0,i]>8):
                    reg_const_two[j,i]=(maxgap_categorical[2*j,i]+0.5)/(1*(1+(reg_const_one[0,i]>2)+2*(reg_const_one[0,i]>8)+4*(reg_const_one[0,i]>32)+8*(reg_const_one[0,i]>128)))
                else:
                    reg_const_two[j,i]=-10000
            data_used[:,i]=data_used[:,i]-1/9*torch.sum(data_used[:,i]<reg_const_two[:,i:i+1],0)+1/9*torch.sum(data_used[:,i]>reg_const_two[:,i:i+1],0)
    reg_const_three=torch.zeros((1,dimension_init))
    for i in range(0,dimension_init):
        reg_const_three[0,i]=torch.mean(data_used[:,i])
        data_used[:,i]=data_used[:,i]-reg_const_three[0,i]
    reg_consts=np.zeros((15,dimension_init))
    reg_consts[0:3,:]=reg_const_one[0:3,:]
    reg_consts[3:7,:]=reg_const_two[0:4,:]
    reg_consts[7:8,:]=reg_const_three[0:1,:]
    reg_consts[8,0]=size
    reg_consts[8,1]=orisize-size
    reg_consts[9:15,:]=reg_const_zero[0:6,:]
    np.save(dataset_name+'/'+'reg_consts_'+dataset_name+'.npy',reg_consts)
    data_init_preprocessed=data_init-(reg_const_zero[1:2,:]-reg_const_zero[2:3,:])*(data_init>(reg_const_zero[2:3,:]))-(reg_const_zero[4:5,:]-reg_const_zero[5:6,:])*(data_init>(reg_const_zero[5:6,:]))
    minimum_radius=torch.zeros((dimension_init,64))
    for i in range(0,dimension_init):
        for j in range(0,64):
            lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j-1/2)/64
            upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j+3/2)/64
            checked_slice=data_init_preprocessed[:,i]
            checked_slice=checked_slice[(checked_slice>lower_bound)*(checked_slice<upper_bound)]
            minimum_radius[i,j]=3*reg_const_one[2,i]/(32*max(len(torch.unique(checked_slice)),4))
    np.save(dataset_name+'/'+'minrad_'+dataset_name+'.npy',minimum_radius.numpy())
    indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    minimum_radius_mcvs=torch.zeros((50,dimension_init,2))
    for i in range(dimension_init):
        unique_vals,counts=torch.unique(data_used[:,i],return_counts=True)
        k_actual=min(50,unique_vals.shape[0])
        topk_counts,topk_indices=torch.topk(counts,k_actual)
        topk_vals = unique_vals[topk_indices]
        topk_counts=topk_counts.float()
        for j in range(0,k_actual):
            min_val=topk_vals[j]-0.075
            max_val=topk_vals[j]+0.075
            size_j=torch.sum((data_used[:,i]<max_val)*(data_used[:,i]>min_val))
            topk_counts[j]=topk_counts[j]/size_j
        minimum_radius_mcvs[:k_actual,i,0]=topk_vals
        minimum_radius_mcvs[:k_actual,i,1]=topk_counts.float()
    np.save(dataset_name+'/'+'minrad_mcvs_'+dataset_name+'.npy',minimum_radius_mcvs.numpy())
    used_indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    data_used=data_used[:,used_indexes].to(torch.float32)
    print('end of normalization step')

def sample(draws):
    global size, data_used
    pos=torch.randperm(size)
    sample=data_used[pos[0:draws],:]
    return torch.transpose(sample,0,1)
def set_datapoint(samplesize):
    global size, samples, samples_num
    samplesize=min(samplesize,size)
    z=sample(samplesize)
    samples=torch.transpose(z,0,1).to(device) # Moved dynamically to GPU
    samples_num=samplesize


def likelihood_calc(kernels_matrix,stepsize=1000,return_likelihood_list=False):
    global samples, samples_num, dimension, kernels_num
    samples_start=0
    samples_end=min(samples_start+stepsize,samples_num)
    derivatives=torch.zeros((2*dimension+1,kernels_num), device=device)
    likelihood=0
    likelihood_list=torch.zeros(samples_num, device=device)
    while samples_start<samples_num:
        kernels_matrix=kernels_matrix.reshape((2*dimension+1,kernels_num,1))
        derivatives_matrix=torch.zeros((2*dimension+1,kernels_num,samples_end-samples_start), device=device)
        samples_used=torch.transpose(samples[samples_start:samples_end,:],0,1)
        calc_matrix=torch.zeros((dimension,1,samples_end-samples_start), device=device)
        calc_matrix[:,0,:]=samples_used[:,:]
        calc_matrix=calc_matrix-kernels_matrix[:dimension,:,0:1]
        derivatives_matrix[:dimension,:,:]=calc_matrix/kernels_matrix[dimension:2*dimension,:,:]**2
        derivatives_matrix[dimension:2*dimension,:,:]=calc_matrix**2/kernels_matrix[dimension:2*dimension,:,:]**3-1/kernels_matrix[dimension:2*dimension,:,:]
        derivatives_matrix[2*dimension,:,:]=1.0
        calc_matrix=(1/kernels_matrix[dimension:2*dimension,:])*torch.exp(-0.5*(calc_matrix/kernels_matrix[dimension:2*dimension,:])**2)
        calc_matrix=torch.prod(calc_matrix,0)
        derivatives_matrix=derivatives_matrix*calc_matrix.unsqueeze(0)
        derivatives_matrix[:2*dimension,:,:]=derivatives_matrix[:2*dimension,:,:]*kernels_matrix[2*dimension:,:,:]
        calc_matrix=torch.sum(calc_matrix*kernels_matrix[2*dimension,:],0)+1e-10
        derivatives_matrix=derivatives_matrix/((calc_matrix.unsqueeze(0)).unsqueeze(0))
        calc_matrix=torch.log(calc_matrix)
        derivatives=derivatives+torch.sum(derivatives_matrix,2)
        likelihood=likelihood+torch.sum(calc_matrix)
        likelihood_list[samples_start:samples_end]=calc_matrix
        samples_start=samples_end
        samples_end=min(samples_start+stepsize,samples_num)
    derivatives[2*dimension:,:]=derivatives[2*dimension:,:]-samples_num/torch.sum(kernels_matrix[2*dimension,:])
    total_weight=torch.sum(kernels_matrix[2*dimension,:])
    likelihood=likelihood/samples_num-torch.log(total_weight)
    likelihood_list=likelihood_list-torch.log(total_weight)
    if return_likelihood_list:
        return likelihood,derivatives,likelihood_list
    return likelihood, derivatives

'''def resample(likelihood_list,samples_num):
    global kernels_matrix,dimension,kernels_num,full_kernels_num
    sequence=torch.arange(full_kernels_num, device=device)
    weight_too_small=(kernels_matrix[2*dimension,:]<1e-3)*(sequence<kernels_num)
    z=torch.nonzero(weight_too_small)
    
    likelihood_list_cpu = likelihood_list.to(torch.float64).cpu()
    w_cpu = (1/torch.exp(likelihood_list_cpu)) / torch.sum(1/torch.exp(likelihood_list_cpu))
    
    new_coord=np.random.choice(min(samples_num,size),np.count_nonzero(weight_too_small.cpu().numpy()),False,w_cpu.numpy())
    kernels_matrix[:dimension,weight_too_small]=torch.transpose(samples[new_coord,:],0,1).to(torch.float32)
    kernels_matrix[dimension:2*dimension,weight_too_small]=0.25
    kernels_matrix[2*dimension:,weight_too_small]=0.002
    if torch.sum(weight_too_small*1)>0:
        clusters_mean, clusters_std=calculate_cluster_stats(samples,torch.transpose(kernels_matrix[:dimension,:],0,1),torch.nonzero(weight_too_small).squeeze())
        kernels_matrix[:dimension,weight_too_small]=torch.transpose(clusters_mean,0,1)
        kernels_matrix[dimension:2*dimension,weight_too_small]=torch.clamp(torch.transpose(clusters_std,0,1),min=1/40)
    kernels_matrix[dimension:2*dimension,:]=torch.clamp(kernels_matrix[dimension:2*dimension,:],min=1/400)
    print("Adding new kernels to low density regions and resampling kernels with weights too low. Kernels to be added or resampled are:")
    print(torch.nonzero(weight_too_small))
    return weight_too_small'''

def resample(likelihood_list,samples_num):
    global kernels_matrix,dimension,kernels_num,full_kernels_num
    sequence=torch.arange(full_kernels_num, device=device)
    weight_too_small=(kernels_matrix[2*dimension,:]<1e-3)*(sequence<kernels_num)
    z=torch.nonzero(weight_too_small)

    likelihood_list_cpu = likelihood_list.to(torch.float64).cpu()
    samples_squeezed,index,counts=np.unique(samples.cpu().numpy(),return_index=True,return_counts=True,axis=0)
    counts=torch.tensor(counts)
    index=torch.tensor(index)
    point_prob = (torch.exp(likelihood_list_cpu[index]))/torch.sum(torch.exp(likelihood_list_cpu[index]))
    actual_weight = counts/torch.sum(counts)
    log_loss=torch.clamp(torch.log(actual_weight/point_prob),min=0)
    w_cpu=actual_weight*log_loss**2
    if spike_switch:
        w_cpu = w_cpu*(counts<samples_num/(8*full_kernels_num))+1e-10
        w_cpu = w_cpu/torch.sum(w_cpu)
    else:
        w_cpu = w_cpu*(counts<samples_num/full_kernels_num)+1e-10
        w_cpu = w_cpu/torch.sum(w_cpu)
    new_coord=np.random.choice(len(counts),np.count_nonzero(weight_too_small.cpu().numpy()),False,w_cpu.numpy())
    a, new_coord=torch.topk(w_cpu,k=np.count_nonzero(weight_too_small.cpu().numpy()))
    kernels_matrix[:dimension,weight_too_small]=torch.transpose(samples[index[new_coord],:],0,1).to(torch.float32)
    kernels_matrix[dimension:2*dimension,weight_too_small]=0.25
    kernels_matrix[2*dimension:,weight_too_small]=0.002
    if spike_switch:
        kernels_matrix[dimension:2*dimension,weight_too_small]=0.001
        kernels_matrix[2*dimension:,weight_too_small]=1
    else:
        if torch.sum(weight_too_small*1)>0:
            clusters_mean, clusters_std=calculate_cluster_stats(samples,torch.transpose(kernels_matrix[:dimension,:],0,1),torch.nonzero(weight_too_small).squeeze())
            kernels_matrix[:dimension,weight_too_small]=torch.transpose(clusters_mean,0,1)
            kernels_matrix[dimension:2*dimension,weight_too_small]=torch.clamp(torch.transpose(clusters_std,0,1),min=1/40)
    kernels_matrix[dimension:2*dimension,:]=torch.clamp(kernels_matrix[dimension:2*dimension,:],min=1/400)
    print("Adding new kernels to low density regions and resampling kernels with weights too low. Kernels to be added or resampled are:")
    print(torch.nonzero(weight_too_small))
    return weight_too_small


def calculate_cluster_stats(sample_points,kernel_points,subset_indices):
    samples_sq = torch.sum(sample_points**2, dim=1, keepdim=True)
    kernels_sq = torch.sum(kernel_points**2, dim=1)                
    dot_product = torch.matmul(sample_points, kernel_points.T)
    dist_sq = samples_sq - 2 * dot_product + kernels_sq
    assignments = torch.argmin(dist_sq, dim=1)
    means_list = []
    stds_list = []
    d = sample_points.shape[1]
    for kernel_idx in subset_indices.flatten():
        mask = (assignments == kernel_idx)
        if torch.any(mask):
            points_in_cluster = sample_points[mask]
            mean = torch.mean(points_in_cluster, dim=0)
            std = torch.sqrt(torch.mean((points_in_cluster-mean)**2,dim=0))
            means_list.append(mean)
            stds_list.append(std)
        else:
            means_list.append(kernel_points[kernel_idx,:])
            stds_list.append(torch.ones((dimension), device=device)*0.15)            
    if not means_list: # Handle case where subset_indices is empty
        return torch.empty(0, d, device=device), torch.empty(0, d, device=device)
    return torch.stack(means_list), torch.stack(stds_list)

def call_back(xk):
    iteration_count+=1
    if iteration_count%10==0:
        print(f"Iteration {iteration_count}:Function value={likelihood_calc(xk)}")

def em_step_gmm(kernel_matrix, stepsize=40000, epsilon=1e-9):
    d = dimension
    N = kernels_num
    size = samples_num
    
    means = kernel_matrix[0:d, :N]
    stds = kernel_matrix[d:2*d, :N]
    weights = kernel_matrix[2*d:2*d+1, :N]
    
    samples_used = torch.transpose(samples, 0, 1) # Shape: (d, size)
    
    # ---------------------------------------------------------
    # PASS 1: Calculate exact weights and means
    # ---------------------------------------------------------
    resp_sum_total = torch.zeros(N, device=kernel_matrix.device)
    mean_num_total = torch.zeros((d, N), device=kernel_matrix.device)

    points_start = 0
    while points_start < size:
        points_end = min(points_start + stepsize, size)
        chunk_samples = samples_used[:, points_start:points_end]
        
        # Calculate responsibilities
        resp_calc = chunk_samples.unsqueeze(1) - means.unsqueeze(2)
        resp_calc = resp_calc**2 / (2 * stds.unsqueeze(2)**2)
        resp_calc = torch.exp(-1 * torch.sum(resp_calc, 0))
        resp_calc = resp_calc / (torch.prod(stds, 0) * (2 * 3.14159265)**(d/2)).unsqueeze(1)
        
        chunk_resp = torch.transpose(resp_calc, 0, 1) * weights
        chunk_resp /= (torch.sum(chunk_resp, axis=1, keepdims=True) + epsilon)
        
        # Accumulate sums
        resp_sum_total += torch.sum(chunk_resp, dim=0)
        mean_num_total += chunk_samples @ chunk_resp
        
        points_start = points_end

    # Apply epsilon exactly as original code did
    resp_sum_epsilon = resp_sum_total + epsilon
    
    # Calculate New Weights and Means
    new_weights = resp_sum_epsilon / size
    new_means = mean_num_total / resp_sum_epsilon

    # ---------------------------------------------------------
    # PASS 2: Calculate exact variances using (X - mu)^2
    # ---------------------------------------------------------
    var_num_total = torch.zeros((d, N), device=kernel_matrix.device)
    
    points_start = 0
    while points_start < size:
        points_end = min(points_start + stepsize, size)
        chunk_samples = samples_used[:, points_start:points_end]
        
        # Recompute responsibilities to avoid storing 108GB of data
        resp_calc = chunk_samples.unsqueeze(1) - means.unsqueeze(2)
        resp_calc = resp_calc**2 / (2 * stds.unsqueeze(2)**2)
        resp_calc = torch.exp(-1 * torch.sum(resp_calc, 0))
        resp_calc = resp_calc / (torch.prod(stds, 0) * (2 * 3.14159265)**(d/2)).unsqueeze(1)
        
        chunk_resp = torch.transpose(resp_calc, 0, 1) * weights
        chunk_resp /= (torch.sum(chunk_resp, axis=1, keepdims=True) + epsilon)
        
        # Calculate (X - mu)^2
        # chunk_samples shape: (d, chunk_size, 1)
        # new_means shape:     (d, 1, N)
        # Resulting diff_squared shape: (d, chunk_size, N)
        diff_squared = (chunk_samples.unsqueeze(2) - new_means.unsqueeze(1))**2
        
        # Multiply by responsibilities and sum over the chunk_size dimension (dim=1)
        var_num_total += torch.sum(diff_squared * chunk_resp.unsqueeze(0), dim=1)
        
        points_start = points_end

    # ---------------------------------------------------------
    # FINALIZE MATRIX
    # ---------------------------------------------------------
    new_kernel_matrix = torch.zeros_like(kernel_matrix[:, :N])
    
    # Weights
    new_kernel_matrix[2*d:2*d+1, :] = new_weights / torch.mean(new_weights)
    
    # Means
    new_kernel_matrix[0:d, :] = new_means
    
    # Standard Deviations
    new_stds = torch.sqrt(var_num_total / resp_sum_epsilon)
    new_kernel_matrix[d:2*d, :] = torch.clamp(new_stds, min=1/400)
    
    return new_kernel_matrix

def sgd_step_gmm(kernels_matrix):
    global kernels_num
    likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
    derivatives[:dimension,:kernels_num]=(derivatives[:dimension,:kernels_num]*learnrate[0]*kernels_matrix[dimension:2*dimension,:kernels_num]**3)/kernels_matrix[2*dimension:,:kernels_num]
    derivatives[dimension:2*dimension,:kernels_num]=(derivatives[dimension:2*dimension,:kernels_num]*learnrate[1]*kernels_matrix[dimension:2*dimension,:kernels_num]**2)/kernels_matrix[2*dimension:,:kernels_num]
    derivatives[2*dimension:,:kernels_num]=derivatives[2*dimension:,:kernels_num]*learnrate[2]*torch.sqrt(kernels_matrix[2*dimension:,:kernels_num])
    derivatives=derivatives*(0.2+torch.rand(2*dimension+1,kernels_num, device=device)*0.8)
    derivatives[dimension:2*dimension,:kernels_num]=torch.maximum(torch.minimum(derivatives[dimension:2*dimension,:kernels_num],torch.ones((dimension,kernels_num), device=device)*0.05),kernels_matrix[dimension:2*dimension,:kernels_num]*(-0.2))
    derivatives[2*dimension,:]=torch.minimum(derivatives[2*dimension,:],torch.ones((kernels_num), device=device)*0.1)
    kernels_matrix[:,:kernels_num]=kernels_matrix[:,:kernels_num]+derivatives
    kernels_matrix[dimension:2*dimension,:kernels_num]=torch.clamp(kernels_matrix[dimension:2*dimension,:kernels_num],min=1/400,max=0.2)
    kernels_matrix[2*dimension:,:kernels_num]=torch.maximum(kernels_matrix[2*dimension:,:kernels_num],torch.ones((1,kernels_num), device=device)/10000)
    kernels_matrix[2*dimension:,:kernels_num]=kernels_matrix[2*dimension:,:kernels_num]/torch.mean(kernels_matrix[2*dimension:,:kernels_num])
    print(likelihood)
    return kernels_matrix

def date_to_days(dates: torch.Tensor) -> torch.Tensor:
    dates=dates.to(int)
    device = dates.device
    dtype = dates.dtype

    year = dates // 10000
    month = (dates // 100) % 100
    day = dates % 100

    year_start = 1901
    year_end = 2099
    years_arr = torch.arange(year_start, year_end + 1, device=device, dtype=dtype)
    is_leap_arr = (years_arr % 4 == 0)
    days_in_year_arr = torch.where(is_leap_arr,
                                   torch.tensor(366, dtype=dtype, device=device),
                                   torch.tensor(365, dtype=dtype, device=device))

    cum_days_before = torch.zeros(len(days_in_year_arr) + 1, dtype=dtype, device=device)
    cum_days_before[1:] = torch.cumsum(days_in_year_arr, dim=0)

    offset = (year - year_start).clamp(0, year_end - year_start)
    days_from_19010101 = cum_days_before[offset]

    mask_month_zero = (month == 0)
    month = torch.where(mask_month_zero, torch.tensor(1, dtype=dtype, device=device), month)
    day   = torch.where(mask_month_zero, torch.tensor(1, dtype=dtype, device=device), day)

    mask_month_gt12 = (month > 12)
    month = torch.where(mask_month_gt12, torch.tensor(12, dtype=dtype, device=device), month)
    day   = torch.where(mask_month_gt12, torch.tensor(31, dtype=dtype, device=device), day)

    days_per_month = torch.tensor([31,28,31,30,31,30,31,31,30,31,30,31],
                                  dtype=dtype, device=device)
    max_day = days_per_month[month - 1]
    is_leap = (year % 4 == 0)
    max_day = torch.where((month == 2) & is_leap,
                          torch.tensor(29, dtype=dtype, device=device),
                          max_day)

    day = torch.where(day == 0, torch.tensor(1, dtype=dtype, device=device), day)
    day = torch.where(day > max_day, max_day, day)

    accum = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
                         dtype=dtype, device=device)
    day_of_year = accum[month - 1] + day
    day_of_year = torch.where((month > 2) & is_leap, day_of_year + 1, day_of_year)
    
    return days_from_19010101 + day_of_year

def train_KDE_model(dim,full_kernum,learnrate,resample_and_sgd=True,loadmatrix=False,matrixloaded=0):
    global samples_num, samples, dimension, kernels_num, kernels_matrix, spike_switch
    dimension=dim
    spike_switch=False
    for j in set(bayes_source_attributes):
        if j in categorical_attribute_index:
            j_actual=j-np.sum(bayes_called_attributes<j)
            data_used[:,j_actual]=data_used[:,j_actual]*(3+3*(reg_const_one[0,j]>32))
        if j not in categorical_attribute_index:
            j_actual=j-np.sum(bayes_called_attributes<j)
            data_used[:,j_actual]=data_used[:,j_actual]*3
    kernels_num=full_kernum-128
    samples_num=0
    kernels_matrix=torch.zeros((2*dimension+1,full_kernum), device=device)
    set_datapoint(full_kernum)
    kernels_matrix[:dimension,:]=torch.transpose(samples,0,1)
    kernels_matrix[dimension:2*dimension,:]=torch.ones((dimension,full_kernum), device=device)*(0.1+torch.rand((dimension,full_kernum), device=device)*0.25)
    kernels_matrix[2*dimension,:kernels_num]=1
    if loadmatrix:
        kernels_matrix=torch.tensor(matrixloaded, device=device)
        kernels_num=full_kernum
    else:
        print('Initializing kernel location via KMeans')
        kmeans=KMeans(n_clusters=kernels_num,n_init=5,random_state=0)
        set_datapoint(2000000)
        kmeans.fit(samples.cpu().numpy())
        print('Initialization complete')
        for k in range(0,kernels_num):
            cluster_points_mask=(kmeans.labels_== k)
            cluster_points=(samples)[cluster_points_mask,:]        
            if len(cluster_points) == 0:
                kernels_matrix[2*dimension,k]=1e-6 
                kernels_matrix[dimension:2*dimension,k]=torch.ones((dimension), device=device)
                kernels_matrix[:dimension,k]=torch.tensor(kmeans.cluster_centers_[k], device=device, dtype=torch.float32) 
                continue
            kernels_matrix[2*dimension,k]=len(cluster_points)/size
            kernels_matrix[:dimension,k]=torch.tensor(kmeans.cluster_centers_[k], device=device, dtype=torch.float32)
            cluster_points_dev = cluster_points.to(device)
            kernels_matrix[dimension:2*dimension,k]=torch.sqrt(torch.mean((cluster_points_dev-kernels_matrix[:dimension,k])**2,0))
            kernels_matrix[dimension:2*dimension,:]=torch.clamp(kernels_matrix[dimension:2*dimension,:],min=(1/40))
        kernels_matrix[2*dimension:,:kernels_num]=kernels_matrix[2*dimension:,:kernels_num]/torch.mean(kernels_matrix[2*dimension:,:kernels_num])
    set_datapoint(test_sample_num)
    print("starting random exploration via SDE updates, outputting likelihood after every 10 iterations")
    for i in range(1,151):
        kernels_matrix[:,:kernels_num]=sgd_step_gmm(kernels_matrix[:,:kernels_num])
        if i%10==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
            print(i)
            print(likelihood)
            if i%10==0:
                if i==150:
                    spike_switch=True
                kernels_num=min(full_kernum,kernels_num+16)
                weight_too_small=resample(likelihood_list,test_sample_num)
            if i%20==0:
                set_datapoint(test_sample_num)
            if i==148:
                set_datapoint(15*test_sample_num)
    print("starting focused EM updates, outputting likelihood after every 10 iterations")
    set_datapoint(test_sample_num)
    for i in range(151,241):
        kernels_matrix[:,:kernels_num]=em_step_gmm(kernels_matrix[:,:kernels_num])
        if i%10==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
            print(i)
            print(likelihood)
            if i%20==0:
               set_datapoint(test_sample_num)
    set_datapoint(8*test_sample_num)
    print("finetuning model on larger trainset using EM algorithm")
    for i in range(241,251):
        kernels_matrix[:,:kernels_num]=em_step_gmm(kernels_matrix[:,:kernels_num])
        if i%10==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
    for j in set(bayes_source_attributes):
        if j in categorical_attribute_index:
            j_actual=j-np.sum(bayes_called_attributes<j)
            data_used[:,j_actual]=data_used[:,j_actual]/(3+3*(reg_const_one[0,j]>32))
            kernels_matrix[j_actual,:]=kernels_matrix[j_actual,:]/(3+3*(reg_const_one[0,j]>32))
            kernels_matrix[j_actual+dimension,:]=kernels_matrix[j_actual+dimension,:]/(3+3*(reg_const_one[0,j]>32))
        if j not in categorical_attribute_index:
            j_actual=j-np.sum(bayes_called_attributes<j)
            data_used[:,j_actual]=data_used[:,j_actual]/3
            kernels_matrix[j_actual,:]=kernels_matrix[j_actual,:]/3
            kernels_matrix[j_actual+dimension,:]=kernels_matrix[j_actual+dimension,:]/3
    np.save(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy',kernels_matrix.cpu().numpy().astype('float16'))
    print("fintuning finished, GMM saved to file, calculating gridding plan matching trained GMM")
    mapper=GridMapper()
    mapper.build_optimal_grids(kernels_matrix,data_used,operate_dims,blocklist_dims,0.1,verbose=True)
    mapper.save(dataset_name+'/categorical_grids.pt')


if __name__ == "__main__":
    params=['CardEst','power','7','-100000000',"[]",'False',"[]"]
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    dimension_init=int(params[2])
    nan_to=float(params[3])
    date_like=ast.literal_eval(params[4])
    has_categorical=(params[5]=='True')
    categorical_attribute_index=ast.literal_eval(params[6])
    numerical_attributes=[]
    for i in range(0,dimension_init):
        if i not in categorical_attribute_index:
            numerical_attributes.append(i)
bayes_source_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[0,:])
bayes_called_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[1,:])
bayes_assist_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[2,:])
print(bayes_called_attributes)
torch.manual_seed(123)
np.random.seed(123)
torch.cuda.manual_seed_all(123)
operate_dims=[(i-np.sum(bayes_called_attributes<i)).item() for i in categorical_attribute_index if i not in bayes_called_attributes]
blocklist_dims=[]
load=False
test_sample_num=1000000
calc_step_size=8000
dimension=dimension_init-len(bayes_called_attributes)
full_kernels_num=1280+640*has_categorical
regu_step1()
learnrate=torch.tensor([600,600,300], device=device)/min(test_sample_num,size)
if load:
    train_KDE_model(dimension,full_kernels_num,learnrate,True,True,np.load(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy'))
else:
    train_KDE_model(dimension,full_kernels_num,learnrate,True)
kernels_matrix=torch.tensor(np.load(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy'), device=device)
for i in range(0,len(bayes_source_attributes)):
    bn=BayesnetEstimator(bayes_called_attributes[i],bayes_source_attributes[i],bayes_assist_attributes[i],1,1,1,1)
    bn.load()
    get_adjust_vals(bayes_source_attributes[i],bn.source_edges,kernels_matrix[:dimension,:],kernels_matrix[dimension:2*dimension,:],kernels_matrix[2*dimension,:])

