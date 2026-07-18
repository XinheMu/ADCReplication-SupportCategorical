from __future__ import division
import torch
import sys
import ast
import warnings
from fractions import Fraction
torch.set_num_threads(30) 
torch.set_grad_enabled(False)
import torch.nn as nn
import os
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
import pickle
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from scipy.stats import qmc
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from typing import Tuple, Union
import json

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
        it has fewer than 50 distinct empirical values.

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
        state = torch.load(filepath, map_location=map_location, weights_only=True)
        mapper = cls()
        mapper.a_values = state['a_values']
        mapper.grids = state['grids']
        mapper.operate_dims = state['operate_dims']
        return mapper


class SingleAttributeHistogram:
    def __init__(self, num_bins: int, num_mcvs: int):
        if num_bins <= 0 or num_mcvs < 0:
            raise ValueError("num_bins must be positive and num_mcvs must be non-negative.")
        self.num_bins = num_bins
        self.num_mcvs = num_mcvs
        # Placeholders for loaded data
        self.histograms = None
        self.mcv_info = None
        self.metadata = None

    def train(self, data: np.ndarray, hist_path='histograms.npy', mcv_path='mcv_info.npz', meta_path='metadata.npz'):
        """
        Trains the histograms and MCV lists on the given data and saves them to files.
        """
        print("Starting training...")
        data_torch = torch.from_numpy(data).float()
        num_rows, num_dimensions = data_torch.shape
        # --- 1. Calculate overall metadata ---
        min_vals = torch.min(data_torch, dim=0).values
        max_vals = torch.max(data_torch, dim=0).values
        self.metadata = {
            'min_vals': min_vals.numpy(),
            'max_vals': max_vals.numpy(),
            'total_rows': num_rows
        }
        # --- 2. Initialize storage for histograms and MCVs ---
        hist_counts_tensor = torch.zeros((num_dimensions, self.num_bins), dtype=torch.float32)
        mcv_values_tensor = torch.full((num_dimensions, self.num_mcvs), float('nan'), dtype=torch.float32)
        mcv_counts_tensor = torch.zeros((num_dimensions, self.num_mcvs), dtype=torch.float32)
        # --- 3. Process each dimension (attribute) separately ---
        for d in range(num_dimensions):
            print(f"  Processing dimension {d+1}/{num_dimensions}...")
            column_data = data_torch[:, d]
            # --- 4. Identify and separate MCVs ---
            if self.num_mcvs > 0:
                unique_vals, counts = torch.unique(column_data, return_counts=True)
                sorted_indices = torch.argsort(counts, descending=True)
                num_actual_mcvs = min(self.num_mcvs, len(unique_vals))
                top_indices = sorted_indices[:num_actual_mcvs]
                mcv_values = unique_vals[top_indices]
                mcv_counts = counts[top_indices]
                mcv_values_tensor[d, :num_actual_mcvs] = mcv_values
                mcv_counts_tensor[d, :num_actual_mcvs] = mcv_counts
                # Create a mask to filter out MCVs from the column data for histogramming
                is_mcv = torch.isin(column_data, mcv_values)
                non_mcv_data = column_data[~is_mcv]
            else:
                non_mcv_data = column_data
            # --- 5. Build histogram on the remaining (non-MCV) data ---
            # Use original min/max for consistent binning across all data
            col_min = min_vals[d].item()
            col_max = max_vals[d].item()
            if col_min < col_max and len(non_mcv_data) > 0:
                hist = torch.histc(non_mcv_data, bins=self.num_bins, min=col_min, max=col_max)
                hist_counts_tensor[d, :] = hist
        self.histograms = hist_counts_tensor.numpy()
        self.mcv_info = {'values': mcv_values_tensor.numpy(), 'counts': mcv_counts_tensor.numpy()}
        # --- 6. Save all computed information to files ---
        print(f"Saving histogram data to {hist_path}")
        np.save(hist_path, self.histograms)
        print(f"Saving MCV data to {mcv_path}")
        np.savez(mcv_path, **self.mcv_info)
        print(f"Saving metadata to {meta_path}")
        np.savez(meta_path, **self.metadata)
        print("Training complete.")
    def load(self, hist_path='histograms.npy', mcv_path='mcv_info.npz', meta_path='metadata.npz'):
        """
        Loads pre-trained histogram and MCV data from files.
        """
        print("Loading pre-trained model...")
        if not all(os.path.exists(p) for p in [hist_path, mcv_path, meta_path]):
            raise FileNotFoundError("One or more required model files are missing.")
            
        self.histograms = np.load(hist_path)
        self.mcv_info = np.load(mcv_path)
        self.metadata = np.load(meta_path)
        
        # Verify loaded data matches instance config
        assert self.histograms.shape[1] == self.num_bins, "Loaded histogram has mismatched num_bins."
        assert self.mcv_info['values'].shape[1] == self.num_mcvs, "Loaded MCVs have mismatched num_mcvs."
        print("Model loaded successfully.")

    def estimate(self, dimension: int, lower_bound: float, upper_bound: float) -> float:
        """
        Estimates the cardinality for a single-attribute range query.
        Query: SELECT COUNT(*) FROM table WHERE lower_bound <= attribute[dimension] <= upper_bound.

        Args:
            dimension (int): The index of the attribute (column) to query.
            lower_bound (float): The lower bound of the query range (inclusive).
            upper_bound (float): The upper bound of the query range (inclusive).

        Returns:
            float: The estimated cardinality.
        """
        if self.histograms is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        if not (0 <= dimension < self.histograms.shape[0]):
            raise ValueError(f"Dimension must be between 0 and {self.histograms.shape[0]-1}.")
        estimated_cardinality = 0.0
        # --- 1. Add counts from MCVs that fall within the query range ---
        if self.num_mcvs > 0:
            mcv_vals = self.mcv_info['values'][dimension]
            mcv_counts = self.mcv_info['counts'][dimension]
            # Create a boolean mask for MCVs within the range
            in_range_mask = (mcv_vals >= lower_bound) & (mcv_vals <= upper_bound)
            estimated_cardinality += mcv_counts[in_range_mask].sum()
        # --- 2. Estimate cardinality from the histogram ---
        hist_counts = self.histograms[dimension]
        min_val = self.metadata['min_vals'][dimension]
        max_val = self.metadata['max_vals'][dimension]
        if min_val >= max_val: # All values are the same, already handled by MCVs if frequent
            return estimated_cardinality
        bin_width = (max_val - min_val) / self.num_bins
        # Clamp query bounds to the data's actual min/max
        query_start = max(lower_bound, min_val)
        query_end = min(upper_bound, max_val)
        if query_start > query_end: # Query range is outside the data's range
            return estimated_cardinality
        # Find which bins the query range touches
        start_bin = int((query_start - min_val) / bin_width)
        end_bin = int((query_end - min_val) / bin_width)
        # Clamp bin indices to be safe
        start_bin = max(0, min(start_bin, self.num_bins - 1))
        end_bin = max(0, min(end_bin, self.num_bins - 1))
        if start_bin == end_bin:
            # Query is contained within a single bin
            bin_start_val = min_val + start_bin * bin_width
            overlap = query_end - query_start
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[start_bin]
        else:
            # Query spans multiple bins
            # a) Partial contribution from the start bin
            bin_end_val = min_val + (start_bin + 1) * bin_width
            overlap = bin_end_val - query_start
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[start_bin]
            # b) Full contribution from intermediate bins
            estimated_cardinality += hist_counts[start_bin + 1 : end_bin].sum()
            # c) Partial contribution from the end bin
            bin_start_val = min_val + end_bin * bin_width
            overlap = query_end - bin_start_val
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[end_bin]
        return estimated_cardinality
    

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
        current_source_index=self.source_attr-np.sum(np.array(bcalled_attributes)<self.source_attr)
        if current_source_index in gridmapper.operate_dims:
            if len(self.source_edges)==len(gridmapper.a_values[gridmapper.operate_dims.index(current_source_index)]):
                self.source_edges = gridmapper.grids[gridmapper.operate_dims.index(current_source_index)]
        
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

class GMM_Estimator:
    def __init__(self, dimension_init, kernels_matrix, 
                 bayes_called_attributes, bayes_source_attributes, bayes_assist_attributes, 
                 S1, S2, bin_val, mcv_val):
        """
        kernels_matrix: shape (2n+1, kernels_num). 
        """
        self.dim_init = dimension_init
        self.non_bayes_indexes=[i not in bayes_called_attributes for i in range(0,dimension_init)]
        self.bayes_called = bayes_called_attributes
        self.bayes_source = bayes_source_attributes
        self.bayes_assist = bayes_assist_attributes
        self.is_rightmost = [(bayes_source_attributes[i] not in bayes_source_attributes[i+1:]) for i in range(0,len(bayes_source_attributes))]
        self.mass_shift = [0 for i in range(0,len(bayes_source_attributes))]
        self.bayes_adjust=[0 for i in range(0,len(bayes_source_attributes))]
        for i in range(0,len(bayes_source_attributes)):
            if self.is_rightmost[i]:
                self.bayes_adjust[i]=torch.tensor(np.load(dataset_name+'/'+dataset_name+'_bias_dim'+str(self.bayes_source[i])+'.npy')).to(torch.float32)
        fullrange_query=torch.tensor([[1.5001 for i in range(0,dimension_init)],[-1.5001 for i in range(0,dimension_init)]])
        if has_categorical:
            self.edge_values=fullrange_query-0.05*(fullrange_query<reg_const_two[0:1,:])-0.05*(fullrange_query<reg_const_two[1:2,:])+0.05*(fullrange_query>reg_const_two[2:3,:])+0.05*(fullrange_query>reg_const_two[3:4,:])
        else:
            self.edge_values=fullrange_query-0.05*(fullrange_query<reg_const_two[0:1,:])-0.05*(fullrange_query<reg_const_two[1:2,:])+0.05*(fullrange_query>reg_const_two[2:3,:])+0.05*(fullrange_query>reg_const_two[3:4,:])
 
        # Device handling (assuming CPU for now, but prepared for cuda)
        self.device = kernels_matrix.device
        
        self.kernels_num = kernels_matrix.shape[1]
        n_gmm = (kernels_matrix.shape[0] - 1) // 2
        self.means = kernels_matrix[:n_gmm, :]                         
        kernels_std_unperturbed=torch.clone(kernels_matrix[dimension:2*dimension]).to(torch.float32)
        kernels_std=[kernels_std_unperturbed]
        for i in range(0,5):
            kernels_std.append(torch.sqrt(kernels_matrix[dimension:2*dimension]**2+Var_min[i]).to(torch.float32))
        self.stds = torch.clone(kernels_std[0])
        self.stds_perturbed = [torch.clone(kernels_std[i]) for i in range(1,6)]
        self.weights = (kernels_matrix[-1, :]).to(torch.float64)/torch.sum((kernels_matrix[-1, :]).to(torch.float64))
        self.weights = self.weights.to(torch.float32)        
        self.gaussian_coefficient=[torch.unsqueeze(torch.prod(self.stds_perturbed[i],0),1)*(ma.sqrt(2*3.14159265))**dimension for i in range(0,5)]

        # Initialize Bayesnet Estimators
        self.bayes_estimators = []
        for y, x1, x2 in zip(self.bayes_called, self.bayes_source, self.bayes_assist):
            bn = BayesnetEstimator(y, x1, x2, S1, S2, bin_val, mcv_val)
            bn.load() 
            self.bayes_estimators.append(bn)
            
        # --- 1. Map Attributes and Identify "Pure" vs "Hybrid" ---
        
        # Map: Original Attribute Index -> GMM Row Index (-1 if not in GMM)
        self.orig_to_gmm_map = [-1] * dimension_init
        gmm_counter = 0
        for i in range(dimension_init):
            if i not in self.bayes_called:
                self.orig_to_gmm_map[i] = gmm_counter
                gmm_counter += 1
        
        # Identify GMM indices used by Bayesnets (x1, x2)
        self.hybrid_gmm_indices = set()
        for bn in self.bayes_estimators:
            self.hybrid_gmm_indices.add(self.orig_to_gmm_map[bn.source_attr])
            self.hybrid_gmm_indices.add(self.orig_to_gmm_map[bn.assist_attr])
            
        # Identify "Pure" GMM indices (Attributes in GMM but NOT x1 or x2)
        pure_gmm_list = []
        pure_orig_list = []
        
        for orig_idx, gmm_idx in enumerate(self.orig_to_gmm_map):
            if gmm_idx != -1 and gmm_idx not in self.hybrid_gmm_indices:
                pure_gmm_list.append(gmm_idx)
                pure_orig_list.append(orig_idx)
                
        # Convert to tensors for fast indexing during inference
        self.pure_gmm_indices = torch.tensor(pure_gmm_list, dtype=torch.long, device=self.device)
        self.pure_orig_indices = torch.tensor(pure_orig_list, dtype=torch.long, device=self.device)
        
        # --- 2. Warmup Hybrid Integrals ---
        self.precomputed_integrals = [] 
        self._warmup()
        for i in range(0,len(bayes_source_attributes)):
            if self.is_rightmost[i]:
                bn=self.bayes_estimators[i]
                massi=torch.zeros(bn.S1,bn.S1)
                possi=((bn.source_edges[1:])+(bn.source_edges[0:bn.S1]))/2
                dispi=torch.unsqueeze(possi,0)-torch.unsqueeze(possi,1)
                weighti=[torch.exp(-1*(dispi**2)/(2*Var_min[j])) for j in range(0,len(Var_min))]
                weighti=[weighti[j]*(self.bayes_adjust[i])[0:1,:] for j in range(0,len(Var_min))]
                weighti=[weighti[j]/(torch.sum(weighti[j],1).unsqueeze(1)) for j in range(0,len(Var_min))]
                (self.mass_shift[i])=[weighti[j].clone().detach() for j in range(0,len(Var_min))]
        
    def _cdf(self, x, mean=0, std=1):
        return 0.5 * (1 + torch.erf((x - mean) / (std * 1.41421356)))

    def _integrate_gaussian(self, lower, upper, mean, std):
        # generic integrator supporting broadcasting
        return self._cdf(upper, mean, std) - self._cdf(lower, mean, std)

    def _warmup(self):
        """
        Pre-calculates integrals of Gaussian kernels over the fixed grid slices
        defined by the Bayesnet estimators.
        """
        for i, bn in enumerate(self.bayes_estimators):
            gmm_idx_x1 = self.orig_to_gmm_map[bn.source_attr]
            gmm_idx_x2 = self.orig_to_gmm_map[bn.assist_attr]
            
            m1, s1 = self.means[gmm_idx_x1, :], self.stds[gmm_idx_x1, :]
            m2, s2 = self.means[gmm_idx_x2, :], self.stds[gmm_idx_x2, :]
            
            grid1 = (bn.source_edges)
            grid2 = (bn.assist_edges)
            
            # X1 Integrals (S1, K)
            lows1 = grid1[:-1].unsqueeze(1)
            highs1 = grid1[1:].unsqueeze(1)
            int_x1 = self._integrate_gaussian(lows1, highs1, m1.unsqueeze(0), s1.unsqueeze(0))
            
            # X2 Integrals (S2, K)
            lows2 = grid2[:-1].unsqueeze(1)
            highs2 = grid2[1:].unsqueeze(1)
            int_x2 = self._integrate_gaussian(lows2, highs2, m2.unsqueeze(0), s2.unsqueeze(0))
            
            self.precomputed_integrals.append({
                'x1': int_x1, 
                'x2': int_x2, 
                'gmm_idx_x1': gmm_idx_x1,
                'gmm_idx_x2': gmm_idx_x2
            })

    def predict_analytical(self, queried_rectangle,attributes_not_covered):
        """
        queried_rectangle: Tensor (2, dimension_init)
        """
        q_min = queried_rectangle[0]
        q_max = queried_rectangle[1]
        
        # 1. Optimized Pure GMM Probability Calculation
        if len(self.pure_gmm_indices) > 0:
            # Gather bounds for all pure dimensions: Shape (Num_Pure)
            p_mins = q_min[self.pure_orig_indices]
            p_maxs = q_max[self.pure_orig_indices]
            
            # Gather means/stds for all pure dimensions: Shape (Num_Pure, K)
            means_pure = self.means[self.pure_gmm_indices]
            stds_pure = self.stds[self.pure_gmm_indices]
            
            # Reshape bounds for broadcasting: (Num_Pure, 1)
            p_mins = p_mins.unsqueeze(1)
            p_maxs = p_maxs.unsqueeze(1)
            
            # Calculate Integrals: Shape (Num_Pure, K)
            probs_pure_matrix = self._integrate_gaussian(p_mins, p_maxs, means_pure, stds_pure)
            
            # Collapse dimensions via product to get probability per kernel: Shape (K)
            pure_gmm_prob = probs_pure_matrix.prod(dim=0)
        else:
            # If no pure dimensions exist, start with 1.0
            pure_gmm_prob = torch.ones(self.kernels_num, device=self.device)
                
        # 2. Calculate Hybrid Contributions
        hybrid_probs = torch.ones(self.kernels_num, device=self.device)
        P_matrices=[0 for i in range(0,len(self.bayes_source))]
        P_matrices_combined=[0 for i in range(0,len(self.bayes_source))]
        v1s_combined=[0 for i in range(0,len(self.bayes_source))]
        v2s_combined=[0 for i in range(0,len(self.bayes_source))]
        for i, bn in enumerate(self.bayes_estimators):
            cache = self.precomputed_integrals[i]
            
            a1, b1 = q_min[bn.source_attr], q_max[bn.source_attr]
            a2, b2 = q_min[bn.assist_attr], q_max[bn.assist_attr]
            p1, q1_y = q_min[bn.called_attr], q_max[bn.called_attr]
            
            # Get P matrix (S1, S2)
            if attributes_not_covered[bn.called_attr]==False:
                P_matrix=bn.get_all_slice_cond_probs(torch.tensor([p1, q1_y],dtype=torch.float16))
            else:
                P_matrix=torch.ones((bn.S1,bn.S2),dtype=torch.float16)
            P_matrices[i]=P_matrix.clone().detach()
            
            if self.is_rightmost[i]:
                # --- Construct v1 (x1) ---
                idx_start_1 = (torch.bucketize(a1, bn.source_edges, right=True) - 1).clamp(0, bn.S1 - 1)
                idx_end_1   = (torch.bucketize(b1, bn.source_edges, right=True) - 1).clamp(0, bn.S1 - 1)
            
                m1, s1 = self.means[cache['gmm_idx_x1']], self.stds[cache['gmm_idx_x1']]
            
                # Start with precomputed body
                v1 = cache['x1'].T.clone() # (K, S1)
            
                # Mask body
                seq_1 = torch.arange(bn.S1, device=self.device).unsqueeze(0)
                mask_1 = (seq_1 > idx_start_1) & (seq_1 < idx_end_1)
                v1 = v1 * mask_1.float()
            
                # Handle Head/Tail edges
                if idx_start_1 == idx_end_1:
                    v1[:, idx_start_1] = self._integrate_gaussian(a1, b1, m1, s1)
                else:
                    head_high = bn.source_edges[idx_start_1 + 1]
                    tail_low = bn.source_edges[idx_end_1]
                    v1[:, idx_start_1] = self._integrate_gaussian(a1, torch.min(head_high, b1), m1, s1)
                    v1[:, idx_end_1] = self._integrate_gaussian(torch.max(tail_low, a1), b1, m1, s1)

                # --- Construct v2 (x2) ---
                idx_start_2 = (torch.bucketize(a2, bn.assist_edges, right=True) - 1).clamp(0, bn.S2 - 1)
                idx_end_2   = (torch.bucketize(b2, bn.assist_edges, right=True) - 1).clamp(0, bn.S2 - 1)
            
                m2, s2 = self.means[cache['gmm_idx_x2']], self.stds[cache['gmm_idx_x2']]
            
                v2 = cache['x2'].T.clone() # (K, S2)
            
                seq_2 = torch.arange(bn.S2, device=self.device).unsqueeze(0)
                mask_2 = (seq_2 > idx_start_2) & (seq_2 < idx_end_2)
                v2 = v2 * mask_2.float()
            
                if idx_start_2 == idx_end_2:
                    v2[:, idx_start_2] = self._integrate_gaussian(a2, b2, m2, s2)
                else:
                    head_high = bn.assist_edges[idx_start_2 + 1]
                    tail_low = bn.assist_edges[idx_end_2]
                    v2[:, idx_start_2] = self._integrate_gaussian(a2, torch.min(head_high, b2), m2, s2)
                    v2[:, idx_end_2] = self._integrate_gaussian(torch.max(tail_low, a2), b2, m2, s2)

                # --- Combine ---
                # v1 (K, S1) @ P (S1, S2) -> (K, S2)
                # (Result * v2).sum -> (K)
                mass_orig=torch.sum(torch.sum(v1,dim=1)*torch.sum(v2,dim=1)*self.weights*pure_gmm_prob)+1e-7
                v1=v1*(self.bayes_adjust[i])[1:2,:]
                mass_new=torch.sum(torch.sum(v1,dim=1)*torch.sum(v2,dim=1)*self.weights*pure_gmm_prob)+1e-7
                for j in range(0,i):
                    if self.bayes_source[j]==self.bayes_source[i]:
                        P_matrix=P_matrix*P_matrices[j]
                P_matrices_combined[i]=P_matrix.clone().detach()
                v1s_combined[i]=v1.clone().detach()
                v2s_combined[i]=v2.clone().detach()
                term_1 = torch.matmul(v1, P_matrix.to(torch.float32))
                term_final = (term_1 * v2).sum(dim=1)
                hybrid_probs *= (term_final*mass_orig/mass_new)

            
        # 3. Final Weighted Sum
        final_kernel_probs = pure_gmm_prob * hybrid_probs
        weighted_kernel_probs = (final_kernel_probs * self.weights).to(torch.float64)
        weighted_kernel_probs[0]=weighted_kernel_probs[0]+1e-11
        total_prob=(weighted_kernel_probs).sum()
        kernels_chosen=np.random.choice(kernels_num,draws,True,(weighted_kernel_probs)/total_prob)
        
        return total_prob, P_matrices_combined,v1s_combined,v2s_combined,kernels_chosen

    def predict_and_sample(self,
                        target_rectangle_init,
                        bayes_source_attributes=None,
                        bayes_called_attributes=None,
                        bayes_assist_attributes=None,
                        working_mode='ADC'):
        global decision_features, perturbation_levels
        attribute_exceeded_above=(maxvals[0,:]<=target_rectangle_init[1,:])*torch.tensor(is_numerical)
        attribute_exceeded_below=(minvals[0,:]>=target_rectangle_init[0,:])*torch.tensor(is_numerical)
        attributes_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
        inside_intervalone=(target_rectangle_init[:,is_numerical]>reg_const_zero[0:1,is_numerical])*(target_rectangle_init[:,is_numerical]<reg_const_zero[1:2,is_numerical])
        inside_intervaltwo=(target_rectangle_init[:,is_numerical]>reg_const_zero[3:4,is_numerical])*(target_rectangle_init[:,is_numerical]<reg_const_zero[4:5,is_numerical])
        target_rectangle_init[:,is_numerical]=target_rectangle_init[:,is_numerical]-(target_rectangle_init[:,is_numerical]-goto_value_one[:,is_numerical])*inside_intervalone-(deducted_value_one[:,is_numerical])*(target_rectangle_init[:,is_numerical]>reg_const_zero[1:2,is_numerical])-(target_rectangle_init[:,is_numerical]-goto_value_two[:,is_numerical])*inside_intervaltwo-(deducted_value_two[:,is_numerical])*(target_rectangle_init[:,is_numerical]>reg_const_zero[4:5,is_numerical])
        modified_target_rectangle=target_rectangle_init.clone().detach()
        modified_target_rectangle[:,is_numerical]=(target_rectangle_init[:,is_numerical]-reg_const_one[1:2,is_numerical])/reg_const_one[2:3,is_numerical]-1.5
        if has_categorical:
            modified_target_rectangle[:,is_numerical]=modified_target_rectangle[:,is_numerical]-0.05*(modified_target_rectangle[:,is_numerical]<reg_const_two[0:1,is_numerical])-0.05*(modified_target_rectangle[:,is_numerical]<reg_const_two[1:2,is_numerical])+0.05*(modified_target_rectangle[:,is_numerical]>reg_const_two[2:3,is_numerical])+0.05*(modified_target_rectangle[:,is_numerical]>reg_const_two[3:4,is_numerical])
        else:
            modified_target_rectangle[:,is_numerical]=modified_target_rectangle[:,is_numerical]-0.05*(modified_target_rectangle[:,is_numerical]<reg_const_two[0:1,is_numerical])-0.05*(modified_target_rectangle[:,is_numerical]<reg_const_two[1:2,is_numerical])+0.05*(modified_target_rectangle[:,is_numerical]>reg_const_two[2:3,is_numerical])+0.05*(modified_target_rectangle[:,is_numerical]>reg_const_two[3:4,is_numerical])
        modified_target_rectangle[0:1,attribute_exceeded_below]=2*self.edge_values[1:2,attribute_exceeded_below]-modified_target_rectangle[1:2,attribute_exceeded_below]
        modified_target_rectangle[1:2,attribute_exceeded_above]=2*self.edge_values[0:1,attribute_exceeded_above]-modified_target_rectangle[0:1,attribute_exceeded_above]
        modified_target_rectangle[:,is_cat]=modified_target_rectangle[:,is_cat]/(1*(1+2*(reg_const_one[0,is_cat]>2)+6*(reg_const_one[0,is_cat]>8)+18*(reg_const_one[0,is_cat]>32)+54*(reg_const_one[0,is_cat]>128)))
        modified_target_rectangle[:,is_cat]=modified_target_rectangle[:,is_cat]-1/9*torch.sum(modified_target_rectangle[:,is_cat]<reg_const_two[:,is_cat].unsqueeze(1),0)+1/9*torch.sum(modified_target_rectangle[:,is_cat]>reg_const_two[:,is_cat].unsqueeze(1),0)
        modified_target_rectangle=modified_target_rectangle-reg_const_three
        center=(modified_target_rectangle[0,:]+modified_target_rectangle[1,:])/2
        mcvs_inside=(minimum_radius_mcvs[:,:,0]>modified_target_rectangle[0:1,:])*(minimum_radius_mcvs[:,:,0]<modified_target_rectangle[1:2,:])
        upperbound_mcvs=torch.amax(minimum_radius_mcvs[:,:,0]+minimum_radius_mcvs[:,:,1]*0.05-1000000000.0*(torch.logical_not(mcvs_inside)),0)
        lowerbound_mcvs=torch.amin(minimum_radius_mcvs[:,:,0]-minimum_radius_mcvs[:,:,1]*0.05+1000000000.0*(torch.logical_not(mcvs_inside)),0)
        if len(index_usegridmap)>0:
            gridmapped_rectangle=gridmapper.map_values(modified_target_rectangle[:,index_usegridmap])
            modified_target_rectangle[0,index_usegridmap]=gridmapped_rectangle[0,:]*(torch.logical_not(attribute_exceeded_below[index_usegridmap]))+modified_target_rectangle[0,index_usegridmap]*attribute_exceeded_below[index_usegridmap]
            modified_target_rectangle[1,index_usegridmap]=gridmapped_rectangle[1,:]*(torch.logical_not(attribute_exceeded_above[index_usegridmap]))+modified_target_rectangle[1,index_usegridmap]*attribute_exceeded_above[index_usegridmap]
        modified_target_rectangle[0,index_usemcvrad]=torch.minimum(modified_target_rectangle[0,index_usemcvrad],lowerbound_mcvs[index_usemcvrad])
        modified_target_rectangle[1,index_usemcvrad]=torch.maximum(modified_target_rectangle[1,index_usemcvrad],upperbound_mcvs[index_usemcvrad])
        result, P_matrices_combined, v1s_combined, v2s_combined, kernels_chosen=self.predict_analytical(modified_target_rectangle,attributes_not_covered)
        if result<1/(20*size):
            return result, 0, True, 0
        bayes_used=(result>-1)*1
        query_edge_length=modified_target_rectangle[1,:]-modified_target_rectangle[0,:]
        values=torch.clamp(query_edge_length,min=ma.sqrt(Var_min[0])/2,max=1)
        volume=torch.prod(values[indexes])+1e-20
        decision_features[query_num,0]=volume
        smallest_three_edges,_=torch.topk(values[indexes],k=3,largest=False)
        perturbation_level=torch.sum(torch.prod(smallest_three_edges)>(125*torch.sqrt(torch.tensor(Var_min)[0:4])**3))
        if working_mode=='ADC+':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if classifier.predict(np.array([[ma.log(volume),ma.log(result.item()+1e-20)]]))==0:
                    return result, 0, True, 0
        tempmat=(torch.transpose(modified_target_rectangle[:,self.non_bayes_indexes],0,1).unsqueeze(1)-self.means[:,kernels_chosen].unsqueeze(2))
        tempval=self._cdf(tempmat/(self.stds[:,kernels_chosen]).unsqueeze(2))
        position=torch.rand((dimension,draws))
        position=position*(tempval[:,:,1]-tempval[:,:,0])+(tempval[:,:,0])
        position=normal.icdf(position)*(self.stds)[:,kernels_chosen]+self.means[:,kernels_chosen]
        if bayes_used:
            for i, bn in enumerate(self.bayes_estimators):
                if self.is_rightmost[i]:
                    source_attr_pos=self.orig_to_gmm_map[bn.source_attr]
                    assist_attr_pos=self.orig_to_gmm_map[bn.assist_attr]
                    (v1s_combined[i])=torch.transpose((v1s_combined[i])[kernels_chosen,:],0,1).unsqueeze(1)
                    (v2s_combined[i])=torch.transpose((v2s_combined[i])[kernels_chosen,:],0,1).unsqueeze(0)
                    (P_matrices_combined[i])=(P_matrices_combined[i]).unsqueeze(2)
                    Probs=(v1s_combined[i]*v2s_combined[i]*P_matrices_combined[i])
                    grids_selected=(torch.multinomial(torch.transpose(Probs.view(-1,draws),0,1),1)).squeeze(-1)
                    S1_selected=grids_selected//bn.S2
                    S2_selected=grids_selected%bn.S2
                    S1_bounds=torch.zeros((2,draws))
                    S2_bounds=torch.zeros((2,draws))
                    S1_bounds[0,:]=bn.source_edges[S1_selected]
                    S1_bounds[1,:]=bn.source_edges[S1_selected+1]
                    S1_bounds=torch.clamp(S1_bounds,min=modified_target_rectangle[0,bn.source_attr],max=modified_target_rectangle[1,bn.source_attr])
                    S2_bounds[0,:]=bn.assist_edges[S2_selected]
                    S2_bounds[1,:]=bn.assist_edges[S2_selected+1]
                    S2_bounds=torch.clamp(S2_bounds,min=modified_target_rectangle[0,bn.assist_attr],max=modified_target_rectangle[1,bn.assist_attr])
                    S1_bounds=self._cdf((S1_bounds-self.means[source_attr_pos,kernels_chosen])/self.stds[source_attr_pos,kernels_chosen])
                    S2_bounds=self._cdf((S2_bounds-self.means[assist_attr_pos,kernels_chosen])/self.stds[assist_attr_pos,kernels_chosen])
                    randnums=torch.rand((2,draws))
                    S1_vals=normal.icdf(randnums[0,:]*S1_bounds[0,:]+(1-randnums[0,:])*S1_bounds[1,:])*self.stds[source_attr_pos,kernels_chosen]+self.means[source_attr_pos,kernels_chosen]
                    S2_vals=normal.icdf(randnums[1,:]*S2_bounds[0,:]+(1-randnums[1,:])*S2_bounds[1,:])*self.stds[assist_attr_pos,kernels_chosen]+self.means[assist_attr_pos,kernels_chosen]
                    position[source_attr_pos,:]=S1_vals
                    position[assist_attr_pos,:]=S2_vals
        return result, position, False, perturbation_level
    
    def GMM_OnePointEst(self,
                        position: torch.Tensor,
                        perturbation_level):
        calc_matrix=(position-self.means[:dimension,:].unsqueeze(2))/self.stds_perturbed[perturbation_level].unsqueeze(2)
        calc_matrix=0.5*(calc_matrix**2)
        calc_matrix=torch.exp(-1*torch.sum(calc_matrix,0))/(self.gaussian_coefficient[perturbation_level])
        return torch.sum(calc_matrix*self.weights.unsqueeze(1),0)

def date_to_days(dates: torch.Tensor) -> torch.Tensor:
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

'''神经网络训练时不加时移'''
class net_tail(nn.Module):
    global dimension
    def __init__(self,net_structure):
        super(net_tail,self).__init__()
        layers=[]
        for i in range(0,len(net_structure)-2):
            layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
            layers.append(nn.LeakyReLU(0.3))
        i=len(net_structure)-2
        layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
        self.structure=nn.Sequential(*layers)

    def forward(self,x):
        y=self.structure(torch.transpose(torch.cat((x[0:dimension],torch.sqrt(x[dimension:dimension+1])),0),0,-1))
        return torch.transpose(y,0,-1)*decay_function(x[dimension:dimension+1])
def decay_function(x):
    return 1/(torch.exp(x)-torch.exp(-x))
'''神经网络训练时不考虑时移'''

class net_head(nn.Module):
    global dimension, activated_nodes, activation_array
    def __init__(self,net_structure):
        super(net_head,self).__init__()
        layers=[]
        for i in range(0,len(net_structure)-2):
            layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
            layers.append(nn.Hardtanh(-2,2))
        i=len(net_structure)-2
        layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
        self.structure=nn.Sequential(*layers)
    def forward(self,x):
        if has_categorical:
            y=self.structure(torch.transpose(torch.cat((x[0:dimension]*torch.exp(x[dimension:dimension+1]),0.5*torch.sin(3.14159265*12*(reg_const_three_shift[0,:].view(dimension,1,1)+x[0:dimension]*torch.exp(x[dimension:dimension+1]))),0.5*torch.cos(3.14159265*12*(reg_const_three_shift[0,:].view(dimension,1,1)+x[0:dimension]*torch.exp(x[dimension:dimension+1]))),5*torch.sqrt(x[dimension:dimension+1])),0),0,-1))
        else:
            y=self.structure(torch.transpose(torch.cat((x[0:dimension]*torch.exp(x[dimension:dimension+1]),5*torch.sqrt(x[dimension:dimension+1])),0),0,-1))
        return (torch.transpose(y[:,:,0:dimension],0,-1))*x[dimension+2:dimension+3]*0.04+(torch.transpose(y[:,:,dimension:2*dimension],0,-1))*x[dimension+1:dimension+2]*0.2+(torch.transpose(y[:,:,2*dimension:3*dimension],0,-1))

class net_head_ablated(nn.Module):
    global dimension
    def __init__(self,net_structure,ablation_key):
        super(net_head_ablated,self).__init__()
        layers=[]
        for i in range(0,len(net_structure)-2):
            layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
            layers.append(nn.Hardtanh(-2,2))
        i=len(net_structure)-2
        layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
        self.structure=nn.Sequential(*layers)
        self.ablat_key=ablation_key
    def forward(self,x):
        y=self.structure(torch.transpose(torch.cat((x[0:dimension]*torch.exp(x[dimension:dimension+1]),5*torch.sqrt(x[dimension:dimension+1])),0),0,-1))
        if self.ablat_key==0:
            return torch.transpose(y[:,:,0:dimension],0,-1)*x[dimension+2:dimension+3,:,:]*0.04
        if self.ablat_key==1:
            return (torch.transpose(y[:,:,dimension:2*dimension],0,-1))*x[dimension+1:dimension+2,:,:]*0.2
        if self.ablat_key==2:
            return (torch.transpose(y[:,:,2*dimension:3*dimension],0,-1))
        if self.ablat_key==3:
            return torch.transpose(y[:,:,0:dimension],0,-1)*x[dimension+2:dimension+3]*0.04-x[0:dimension]*x[dimension+2:dimension+3,:,:]

def set_global(Intervals,TimeAllowed_min,batch_size,dim,cutoff,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None):
    global histogram, gmm_estimator, length, BatchSize, TimeMax, TimeChart, TimeChart_InNetwork, SignalDecay, SignalDecay_abbrev, NoiseVar, NoiseVar_inv, NoiseVarSquare, NoiseVarSquare_inv, Weight, Weight_sqrt, Weight_used, Weight_term_additional, PointArray, dimension, PointArrayScore, PointArrayScore_double, cutoff_length, cutoff_time, Var_min, Time_min, residual_tail_adjust_term, normal, log_likelihood_adjust
    histogram=SingleAttributeHistogram(num_bins=200, num_mcvs=20)
    histogram.load(dataset_name+'/'+dataset_name+'_histogram.npy',dataset_name+'/'+dataset_name+'_mcv.npz',dataset_name+'/'+dataset_name+'_meta.npz')
    cutoff_length=cutoff
    cutoff_time=Intervals[cutoff]
    TimeMax=Intervals[len(Intervals)-1]
    dimension=dim
    BatchSize=batch_size
    length=len(Intervals)-1
    TimeChart=np.zeros((1,(length)*batch_size))
    TimeChart[0,:]=np.repeat([(Intervals[0]+Intervals[1])/2]+[(Intervals[i]+Intervals[i+1])/2 for i in range(1,cutoff_length)]+[(Intervals[i]+Intervals[i+1])/2 for i in range(cutoff_length,length)],batch_size)
    SignalDecay=np.exp(-1*TimeChart)
    SignalDecay_abbrev=(torch.exp(-1*torch.tensor([(Intervals[0]+Intervals[1])/2]+[(Intervals[i]+Intervals[i+1])/2 for i in range(1,length)]))).unsqueeze(0)
    NoiseVarSquare=(1-SignalDecay**2)
    NoiseVar=np.sqrt(NoiseVarSquare)
    Weight=np.zeros((1,(length)*batch_size))
    Weight[0,:]=np.repeat([(Intervals[i+1]-Intervals[i]) for i in range(0,length)],batch_size)
    Weight=torch.tensor(Weight)
    Weight_term_additional=torch.zeros(1,(length)*batch_size)
    Weight_term_additional[0,4*batch_size]=TimeAllowed_min/8
    Weight_sqrt=torch.sqrt(Weight)
    SobolSampler=stats.qmc.Sobol(dim,scramble=True,seed=123)
    UniSobol=np.transpose(SobolSampler.random(2**ma.ceil(ma.log2(batch_size*length))))
    UniSobol=(UniSobol-1*(UniSobol>1))
    NormalSobol=stats.norm.ppf(UniSobol*0.999999+0.0000005)
    PointArray=torch.tensor(NormalSobol[:,0:(length)*batch_size]*NoiseVar)
    PointArrayScore=-1*PointArray/NoiseVarSquare
    PointArrayScore_double=2*PointArrayScore
    TimeChart=torch.tensor(TimeChart)
    SignalDecay=torch.tensor(SignalDecay)
    NoiseVar=torch.tensor(NoiseVar)
    NoiseVarSquare=torch.tensor(NoiseVarSquare)
    Time_min=[TimeAllowed_min,TimeAllowed_min*1.5,TimeAllowed_min*2,TimeAllowed_min*3,TimeAllowed_min*4]
    NoiseVarSquare_inv=[(1/(1-torch.exp(-2*(TimeChart+Time_min[i])))).to(torch.float32) for i in range(0,5)]
    NoiseVar_inv=[(torch.sqrt(NoiseVarSquare_inv[i])).to(torch.float32) for i in range(0,5)]
    '''
    Note that NoiseVar_inv is used inside the noise prediction network, trained without implementing the early stopping time-shift; while NoiseVar is used in the density estimator where the early stopping time-shift was already implemented. Therefore, these two are NOT the inverses of one another due to this discrepancy
    '''
    Var_min=[ma.exp(2*Time_min[i])-1 for i in range(0,5)]
    TimeChart_InNetwork=[(TimeChart+Time_min[i]).to(torch.float32) for i in range(0,5)]
    residual_tail_adjust_term=[(1/NoiseVarSquare[:,BatchSize*cutoff_length:]-(NoiseVarSquare_inv[i])[:,BatchSize*cutoff_length:]).unsqueeze(2).to(torch.float32) for i in range(0,5)]
    Weight_used=torch.transpose(Weight/BatchSize,0,1)
    Weight_term_additional=torch.transpose(Weight_term_additional/BatchSize,0,1)
    TimeChart=TimeChart.to(torch.float32)
    SignalDecay=SignalDecay.to(torch.float32)
    SignalDecay_abbrev=SignalDecay_abbrev.to(torch.float32)
    NoiseVar=NoiseVar.to(torch.float32)
    NoiseVarSquare=NoiseVarSquare.to(torch.float32)
    Weight=Weight.to(torch.float32)
    Weight_sqrt=Weight_sqrt.to(torch.float32)
    Weight_used=Weight_used.to(torch.float32)
    PointArray=PointArray.to(torch.float32)
    PointArrayScore=PointArrayScore.to(torch.float32)
    PointArrayScore_double=PointArrayScore_double.to(torch.float32)
    log_likelihood_adjust=np.load(dataset_name+'/ELBOadjust.npy') if os.path.exists(dataset_name+'/ELBOadjust.npy') else np.zeros(2)
    if bayes_source_attributes is not None:
        bcalled_tensor=torch.tensor(bayes_called_attributes)
        bsource_actual_position=[i-torch.sum(i>bcalled_tensor).item() for i in bayes_source_attributes]
    else:
        bsource_actual_position=[]
    gmm_estimator=GMM_Estimator(dimension_init, kernels_matrix, bayes_called_attributes, bayes_source_attributes, bayes_assist_attributes, 150, 8, 10, 1)
    normal=torch.distributions.Normal(loc=0.0, scale=1.0)

def get_reg_consts():
    global reg_const_one, reg_const_two, reg_const_three, reg_const_three_shift, reg_const_zero, maxvals, minvals, goto_value_one, goto_value_two, deducted_value_one, deducted_value_two, size, nan_nums, orisize, minimum_radius, minimum_radius_mcvs
    reg_consts=np.load(dataset_name+'/'+'reg_consts_'+dataset_name+'.npy')
    reg_const_one=torch.tensor(reg_consts[0:3,:]).to(torch.float32)
    reg_const_two=torch.tensor(reg_consts[3:7,:]).to(torch.float32)
    reg_const_three=torch.tensor(reg_consts[7:8,:]).to(torch.float32)
    reg_const_zero=torch.tensor(reg_consts[9:15,:]).to(torch.float32)
    maxvals=reg_const_one[0:1,:]+(reg_const_zero[1:2,:]-reg_const_zero[2:3,:])+(reg_const_zero[4:5,:]-reg_const_zero[5:6,:])
    minvals=reg_const_one[1:2,:]
    goto_value_one=(reg_const_zero[0:1,:]+reg_const_zero[2:3,:])/2
    goto_value_two=(reg_const_zero[3:4,:]+reg_const_zero[5:6,:])/2
    deducted_value_one=reg_const_zero[1:2,:]-reg_const_zero[2:3,:]
    deducted_value_two=reg_const_zero[4:5,:]-reg_const_zero[5:6,:]
    size=reg_consts[8,0].astype(int)
    nan_nums=reg_consts[8,1].astype(int)
    orisize=size+nan_nums
    minimum_radius=torch.tensor(np.load(dataset_name+'/'+'minrad_'+dataset_name+'.npy'))
    minimum_radius_mcvs=torch.tensor(np.load(dataset_name+'/'+'minrad_mcvs_'+dataset_name+'.npy'))
    reg_const_three_shift=reg_const_three[0:1,indexes].to(device)

def CardEst_Implement(target_rectangle,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,attributes_not_covered=None,working_mode='ADC'):
    global estimated_log_density, estimated_gmm_density, perturbation_levels, GMM_estimate_selectivity 
    result,position,predictor_is_more_accurate,perturbation_level=gmm_estimator.predict_and_sample(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,working_mode)
    GMM_estimate_selectivity[query_num]=result
    if predictor_is_more_accurate or working_mode=='ADC-':
        return result
    position=position.unsqueeze(1)
    density_KDE=gmm_estimator.GMM_OnePointEst(position,perturbation_level)
    position=position*(ma.exp(-1*Time_min[perturbation_level]))
    eval_point=torch.cat((PointArray,TimeChart_InNetwork[perturbation_level],NoiseVar_inv[perturbation_level],NoiseVarSquare_inv[perturbation_level]),0)
    eval_point=eval_point.unsqueeze(2).repeat(1,1,draws)
    position_used=(position*(SignalDecay_abbrev.unsqueeze(2))).unsqueeze(2).expand(-1,-1,BatchSize,-1)
    position_used=position_used.reshape(dimension,length*BatchSize,draws)
    eval_point[:dimension,:,:]=eval_point[:dimension,:,:]+position_used
    eval_point_head=eval_point[:,:cutoff_length*BatchSize,:]
    eval_point_tail=eval_point[:,cutoff_length*BatchSize:,:]
    '''score_head无需以此方法修正,是因为时移项只作为输入时的提示项,不直接出现在输出中'''
    score_head=evalnet_head(eval_point_head.to(device)).to('cpu')
    '''神经网络训练时未加时移项训练,故需以此方法人为修正时移项影响,更正residual_tail的取值'''
    residual_tail=evalnet_tail(eval_point_tail.to(device)).to('cpu')+eval_point_tail[:dimension,:]*(residual_tail_adjust_term[perturbation_level])
    eval_value=torch.cat((score_head*(score_head-(2*PointArrayScore[:,0:cutoff_length*BatchSize]).unsqueeze(2))+1,residual_tail*(residual_tail-2*((PointArrayScore[:,cutoff_length*BatchSize:]).unsqueeze(2)+eval_point[0:dimension,cutoff_length*BatchSize:,:]/(NoiseVarSquare[:,cutoff_length*BatchSize:]).unsqueeze(2)))),1)
    eval_value=torch.sum(eval_value,0)*Weight_used
    eval_value=torch.sum(eval_value,0)
    log_density=dimension*(0.5*ma.log(1/(2*3.14159265))-0.5)-eval_value-ma.log(1-ma.exp(-2*cutoff_time))*(dimension/2)-(torch.sum(position*position,0)).unsqueeze(0)*(1/(2*(1-ma.exp(-2*cutoff_time)))-1/2)
    estimated_log_density[query_num,:]=log_density.clone().detach()
    estimated_gmm_density[query_num,:]=density_KDE.clone().detach()    
    perturbation_levels[query_num]=perturbation_level
    log_density=log_density-dimension*Time_min[perturbation_level]
    density_diffusion=(torch.exp(log_density))
    density_diffusion[torch.isnan(density_diffusion)]=0
    adjust_term=(density_diffusion+1e-7)/(density_KDE+1e-7)
    z=min(torch.mean(adjust_term),40)
    return result*z

def CardEst_Implement_Selective(target_rectangle,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,nan_to=-100000000,working_mode='ADC'): 
    global nan_is_queried, GMM_estimate_selectivity
    is_fullrange=(target_rectangle[0,:]==88888888.0)
    target_rectangle[0,is_fullrange]=-88888888.0
    nan_queried=torch.prod(target_rectangle[0:1,:]<=nan_to+1e-8)*torch.prod(target_rectangle[1:2,:]>=nan_to)
    nan_is_queried[query_num]=nan_queried
    torch.manual_seed(123)
    np.random.seed(123)
    attribute_exceeded_above=maxvals<=target_rectangle[1:2,:]
    attribute_exceeded_below=minvals>=target_rectangle[0:1,:]
    attributes_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
    target_rectangle_orig=target_rectangle.clone().detach()
    target_rectangle[0:1,unit_of_variables[0,:]!=0]=(torch.ceil(target_rectangle[0:1,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]-1e-6)-0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[1:2,unit_of_variables[0,:]!=0]=(torch.floor(target_rectangle[1:2,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]+1e-6)+0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[:,date_like]=date_to_days(torch.floor(target_rectangle[:,date_like]).to(int)).to(target_rectangle.dtype)+0.5
    estimate_zero=((torch.sum(minvals[:,is_numerical]>target_rectangle[1:2,is_numerical])+torch.sum(maxvals[:,is_numerical]<target_rectangle[0:1,is_numerical]))>0)
    if estimate_zero:
        return torch.tensor([nan_queried*nan_nums])
    edge_length=target_rectangle[1:2,:]-target_rectangle[0:1,:]
    if torch.min(edge_length)<1e-7:
        return torch.tensor([nan_queried*nan_nums])
    if torch.sum(attributes_not_covered)==dimension_init:
        GMM_estimate_selectivity[query_num]=1
        return torch.tensor([size+nan_queried*nan_nums])
    if torch.sum(attributes_not_covered)==dimension_init-1:
        queriedatt=torch.argmin(attributes_not_covered)
        target_rectangle=target_rectangle.numpy()
        GMM_estimate_selectivity[query_num]=ma.ceil(histogram.estimate(queriedatt,target_rectangle[0,queriedatt],target_rectangle[1,queriedatt]))/size
        return torch.tensor([ma.ceil(histogram.estimate(queriedatt,target_rectangle[0,queriedatt],target_rectangle[1,queriedatt]))])+nan_queried*nan_nums
    return torch.ceil((CardEst_Implement(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,attributes_not_covered,working_mode)*size+nan_queried*nan_nums))

def convert_dataset(dataset_name, orig_val):
    global is_cat
    # Load column metadata (names + is_categorical)
    meta_path = dataset_name+'/'+dataset_name+'iscat.npy'
    meta = np.load(meta_path, allow_pickle=True)
    col_names = meta[0, :]                # 1D array of strings
    is_cat = meta[1, :].astype(bool)      # 1D bool array
    is_cat=[is_cat[i].item() for i in range(0,len(is_cat))]
    # Load categorical mappings
    map_file = dataset_name+'/categoricalmaps.json'
    with open(map_file, 'r') as f:
        saved_maps = json.load(f)

    n_rows, n_cols = orig_val.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    for j in range(n_cols):
        col_str = orig_val[:, j]
        col_name = col_names[j]

        if is_cat[j]:
            mapping = saved_maps[col_name]   # dict: str → int
            numeric_col = np.empty(n_rows, dtype=np.float64)
            for i, val in enumerate(col_str):
                if val.upper() == 'ALLATTRS':
                    numeric_col[i] = 88888888.0
                else:
                    if val not in mapping:
                        raise ValueError(
                            f"Unknown categorical value '{val}' in column '{col_name}'"
                        )
                    numeric_col[i] = float(mapping[val])
            out[:, j] = numeric_col
        else:
            numeric_col = np.empty(n_rows, dtype=np.float64)
            for i, val in enumerate(col_str):
                if val.upper() == 'ALLATTRS':
                    numeric_col[i] = 88888888.0
                else:
                    numeric_col[i] = float(val)
            out[:, j] = numeric_col

    return out

def calibrate_and_classify(count, log_dens_init, pred_dense, raw_pred_selectivity,  actual_cardinality, feat):
    eps = 1e-7
    actual_sel = actual_cardinality.astype(float) / orisize

    # Pre‑slice training data for faster loss evaluation
    log_dens_train = log_dens_init
    pred_dense_train = pred_dense
    raw_sel_train = raw_pred_selectivity
    actual_card_train = actual_cardinality
    # Helper: compute loss for given A, B on training set
    def loss_ab(ab):
        A, B = ab
        A=A/(1.75**perturbation_levels)
        B=B-(0.5*perturbation_levels)
        # Apply correction: log_dens = log_dens_init + A * ReLU(B - log_dens_init)
        relu = np.maximum(0, B - log_dens_train)           # shape (n_train, 25)
        log_dens = log_dens_train + A * relu
        log_dens = log_dens - dimension * (torch.tensor(Time_min)[perturbation_levels]).numpy()
        # Modifier = mean over 25 dims of exp(log_dens) / pred_dense
        modifier = np.mean((np.exp(log_dens) + eps) / (pred_dense_train + eps), axis=1)
        # Predicted cardinality (with rounding)
        pred_card = np.ceil(size * raw_sel_train * modifier).astype(int)+nan_nums*nan_is_queried
        # Clamp zero cardinalities to 1 (as per "cardinality 0 -> 1")
        pred_card = np.where(pred_card == 0, 1, pred_card)
        actual_card = np.where(actual_card_train == 0, 1, actual_card_train)
        # Q-error = max(pred, actual) / min(pred, actual)
        ratio = np.maximum(pred_card, actual_card) / np.minimum(pred_card, actual_card)
        log_q = np.log(ratio)
        loss = np.mean(log_q**2)
        return loss

    # Optimise A and B with bounds
    bounds = [(0.001, 0.1), (-10.0, 20.0)]   # A, B
    x0 = [0.02, 0.0]                         # initial guess
    res = minimize(loss_ab, x0, method='Powell', bounds=bounds,
                   options={'xtol': 1e-4, 'ftol': 1e-4})
    A_opt, B_opt = res.x
    print(f"Optimal A: {A_opt:.6f}, B: {B_opt:.6f}")

    # --- Compute Q‑errors on full dataset using optimal A, B ---
    # Corrected predictions
    A_opt_individual=A_opt/(1.75**perturbation_levels)
    B_opt_individual=B_opt-(0.5*perturbation_levels)
    relu_full = np.maximum(0, B_opt_individual - log_dens_init)
    log_dens_opt = log_dens_init + A_opt_individual * relu_full
    log_dens_opt = log_dens_opt - dimension * (torch.tensor(Time_min)[perturbation_levels]).numpy()
    modifier_opt = np.mean((np.exp(log_dens_opt) + eps) / (pred_dense + eps), axis=1)
    modifier_opt = np.clip(modifier_opt,a_min=0,a_max=40)
    pred_sel_mod = raw_pred_selectivity * modifier_opt
    pred_card_mod = np.ceil(size * pred_sel_mod).astype(int)+nan_nums*nan_is_queried
    pred_card_mod = np.where(pred_card_mod == 0, 1, pred_card_mod)

    # Raw predictions
    pred_card_raw = np.ceil(size * raw_pred_selectivity).astype(int)+nan_nums*nan_is_queried
    pred_card_raw = np.where(pred_card_raw == 0, 1, pred_card_raw)

    # Actual cardinalities with zero clamping
    actual_card_clamped = np.where(actual_cardinality == 0, 1, actual_cardinality)

    # Q‑error function
    def compute_q(pred_card, actual_card):
        ratio = np.maximum(pred_card, actual_card) / np.minimum(pred_card, actual_card)
        return ratio
    q_mod = compute_q(pred_card_mod, actual_card_clamped)
    q_init = compute_q(pred_card_raw, actual_card_clamped)
    q_max=np.argmax(q_mod)
    sorted_q_mod=np.sort(q_mod)
    print('Under Calculated Likelihood Adjustment Scheme')
    print('Max Q-error is '+str(sorted_q_mod[-1]))
    print('Location with Max Q-error is '+str(q_max))
    print('99th Q-error is '+str(sorted_q_mod[ma.ceil(count*99/100)]))
    print('95th Q-error is '+str(sorted_q_mod[ma.ceil(count*95/100)]))
    print('50th Q-error is '+str(sorted_q_mod[ma.ceil(count/2)]))
    # --- Train decision tree classifier to choose raw (0) or corrected (1) ---
    L0 = (np.log(q_init)) ** 2
    L1 = (np.log(q_mod)) ** 2
    y = (L1 < L0).astype(int)                     # 1 if corrected is better
    sample_weight = np.abs(L0 - L1)               # give more weight to large differences

    X = feat                                    # shape (count, 2)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1000, random_state=123, criterion='gini')
    clf.fit(X, y, sample_weight=sample_weight)
    with open(dataset_name+'/'+'classifier_'+dataset_name+'.pkl','wb') as classifierfile:
        pickle.dump(clf,classifierfile)
    np.save(dataset_name+'/ELBOadjust.npy',np.array((A_opt,B_opt)))
    # Store results (or return them)
    return A_opt, B_opt, clf

def run(dname,uvar,dim_init,kernum,tm,workload_size,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,nan_to=-1000000000,working_mode='ADC',output_mode='qerror',threshold=100):
    global dataset_name, unit_of_variables, dimension, dimension_init, indexes, indexestwo, gridmapper, index_usegridmap, index_usemcvrad, kernels_num, tmin, evalnet_head, evalnet_tail, kernels_matrix, classifier, timeused, j, actualcard, minimum_rad, bayesnet, is_cat, is_numerical, is_binary, GMM_estimate_selectivity, ADC_estimate_selectivity, estimated_log_density, estimated_gmm_density, nan_is_queried, decision_features, query_num, perturbation_levels
    actualcard=np.load(dataset_name+'training/'+dataset_name+'_real_train.npy',allow_pickle=True)[:workload_size]
    dataset_name=dname
    unit_of_variables=(torch.tensor(uvar)).float()
    dimension_init=dim_init
    if bayes_called_attributes is None:
        indexes=[i for i in range(0,dimension_init)]
        indexestwo=[i for i in range(0,dimension_init)]
    else:
        indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
        indexestwo=[(i not in (bayes_called_attributes+bayes_source_attributes)) for i in range(0,dimension_init)]
    if bayes_called_attributes is None:
        dimension=dimension_init
    else:
        dimension=dimension_init-len(bayes_called_attributes)
    bayesnet=None
    if bayes_source_attributes is None:
        bayes_source_attributes=[]
    if bayes_called_attributes is None:
        bayes_called_attributes=[]
    if bayes_assist_attributes is None:
        bayes_assist_attributes=[]
    kernels_num=kernum
    tmin=tm
    get_reg_consts()
    kernels_matrix=torch.tensor(np.load(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy')).float()
    evalnet_head=torch.load(dataset_name+'/'+dataset_name+'_head.pkl',weights_only=False).to(device)
    evalnet_tail=torch.load(dataset_name+'/'+dataset_name+'_tail.pkl',weights_only=False).to(device)
    j=0
    min_sel=1/orisize
    gridmapper=GridMapper.load(dataset_name+'/categorical_grids.pt')
    set_global(timestep,tmin,32*(1+ma.floor(dimension/4)),dimension,cutoff,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes)
    if not has_categorical:
        workloads=pd.read_csv(dataset_name+'training/'+dataset_name+'_trainset.csv', delimiter=',', dtype=np.float32, header=None)
    else:
        workloads=pd.read_csv(dataset_name+'training/'+dataset_name+'_trainset.csv', delimiter=',', dtype=str,  header=None)
    workloads=workloads.to_numpy()
    if has_categorical:
        workloads=convert_dataset(dataset_name,workloads)
    else:
        is_cat=[False for i in range(0,dimension_init)]
    is_numerical=[(not is_cat[i]) for i in range(0,dimension_init)]
    is_binary=[(is_cat[i] and reg_const_one[0,i].item()==2) for i in range(0,len(is_cat))]
    indexes_numeric=[i for i in range(0,dimension_init) if indexes[i]]
    index_usegridmap=[indexes_numeric[i] for i in gridmapper.operate_dims]
    index_usemcvrad=[i for i in range(0,dimension_init) if i not in bayes_called_attributes and i not in index_usegridmap]
    workloads=torch.tensor(workloads)
    actual_selectivity=np.zeros((workload_size))
    nan_is_queried=(np.zeros((workload_size))>1)
    GMM_estimate_selectivity=np.zeros((workload_size))
    ADC_estimate_selectivity=np.zeros((workload_size))
    decision_features=np.zeros((workload_size,2))
    error=np.zeros((workload_size))
    estimated_log_density=np.ones((workload_size,draws))*-1000.0
    estimated_gmm_density=np.zeros((workload_size,draws))
    perturbation_levels=np.zeros((workload_size,draws))
    for query_num in range(0,workload_size):
        if query_num%500==499:
            print("Processed "+str(query_num+1)+" queries")
        target_rectangle=workloads[2*query_num:2*query_num+2,:]
        if output_mode=='sel':
            e=1
        else:
            e=actualcard[query_num]/orisize
        CardEst_Implement_Selective(target_rectangle.clone().detach(),bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,nan_to,working_mode)/orisize
    estimated_log_density=np.nan_to_num(estimated_log_density,nan=-1000.0)
    estimated_gmm_density=np.nan_to_num(estimated_gmm_density,nan=0)
    GMM_estimate_selectivity=np.nan_to_num(GMM_estimate_selectivity,nan=0)
    decision_features[:,1]=GMM_estimate_selectivity
    decision_features=np.nan_to_num(decision_features,0)
    decision_features=np.log(np.abs(decision_features)+1e-20)
    calibrate_and_classify(workload_size,estimated_log_density,estimated_gmm_density,GMM_estimate_selectivity,actualcard,decision_features)

if __name__ == "__main__":
    timestep=(np.load('timestep.npy')).tolist()
    cutoff=43
    device=torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    params=['Estimator','higgs',"[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]",'7','1/1280','10000','-10000000000',"[]",'False','25']
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    uvar=[ast.literal_eval(params[2])]
    dimension_init=int(params[3])
    Time_min=float(Fraction(params[4]))
    workload_size=int(params[5])
    nanto=int(params[6])
    date_like=ast.literal_eval(params[7])
    has_categorical=(params[8]=='True')
    gmmker=1280+640*has_categorical
    draws=int(params[9])
    ablation_key=-1
    bsource_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[0,:]).tolist()
    bcalled_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[1,:]).tolist()
    bassist_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[2,:]).tolist()
    run(dataset_name,uvar,dimension_init,gmmker,Time_min,workload_size,bayes_source_attributes=bsource_attributes,bayes_called_attributes=bcalled_attributes,bayes_assist_attributes=bassist_attributes,nan_to=nanto)

