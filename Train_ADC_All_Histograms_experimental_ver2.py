import torch 
import sys
import ast
torch.set_num_threads(60)
torch.set_grad_enabled(False)
import torch.nn as nn
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
import pickle
import os
import glob
import re
from typing import Tuple, Union
from adc_categorical_params import (
    cat_data_denominator,
    cat_gap_mask,
    cat_shift,
    cat_threshold_denominator,
)

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
            self.S1 = min(1000, 3+torch.round((vals[-1]-vals[0])/min_gap))
            self.S2 = 1 
            calc_param = 0.001 * n_distinct_source
            self.mcv_val = min(2, int(np.ceil(calc_param)))
            self.bin_val = min(5, int(np.ceil(calc_param)))
            PKFK_Flag[flag_change_location]=True
            PKFK_info[flag_change_location]=[min(1000, 1+torch.round((vals[-1]-vals[0])/min_gap)),1,5+35*has_categorical,2]

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
                return torch.tensor([min_val - 1e-6, max_val + 1e-6])
            midpoints = (sorted_u[:-1] + sorted_u[1:]) / 2.0
            return torch.cat([torch.tensor([min_val - epsilon]), midpoints, torch.tensor([max_val + epsilon])])
        else:
            return torch.linspace(min_val - 1e-6, max_val + 1e-6, requested_S + 1)

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


def regu_step1(starting_col=0):
    global reg_const_one, reg_const_two, reg_const_three, data_used_primitive, data_used, size, corrupted_data_size, data_init, data_init_orig
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
    data_init=data_used.clone()
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
            data_used[:,i]=data_used[:,i]/cat_data_denominator(reg_const_one[0,i])
            for j in range(0,4):
                if cat_gap_mask(maxgap_categorical[2*j+1,i], j, reg_const_one[0,i]):
                    reg_const_two[j,i]=(maxgap_categorical[2*j,i]+0.5)/cat_threshold_denominator(reg_const_one[0,i])
                else:
                    reg_const_two[j,i]=-10000
            data_used[:,i]=data_used[:,i]-cat_shift()*torch.sum(data_used[:,i]<reg_const_two[:,i:i+1],0)+cat_shift()*torch.sum(data_used[:,i]>reg_const_two[:,i:i+1],0)
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
    used_indexes=[i for i in range(0,dimension_init)]
    data_used=data_used[:,used_indexes].to(torch.float32)
    data_init_orig=data_init.clone().detach()
    randsample=np.random.choice(size,min(size,2000000),replace=False)
    data_init=data_init[randsample]

def test_functional_dependency(source_target=-1,called_target=-1):
    if source_target<-0.5:
        shared_volume=torch.zeros((dimension,dimension))    
        for i in range(0,dimension):
            print(i)
            for j in range(0,dimension):
                print(j)
                if i==j:
                    shared_volume[i,j]=10
                else:
                    correlation_data=torch.zeros(2,1024)
                    counts=torch.zeros(1,1024)
                    for k in range(0,1024):
                        lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*k/1024
                        upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(k+1)/1024+0.01*(k==1023)
                        mask=(data_init[:,i]<upper_bound)*(data_init[:,i]>=lower_bound)
                        counts[0,k]=torch.sum(mask)
                        if torch.sum(mask)==0:
                            correlation_data[0,k]=1000000
                            correlation_data[1,k]=1000000
                        else:
                            correlation_data[0,k]=torch.min(data_init[mask,j])
                            correlation_data[1,k]=torch.max(data_init[mask,j])
                    shared_volume[i,j]=0.8*torch.sum(counts*(correlation_data[1,:]-correlation_data[0,:])/(reg_const_one[2,j]*min(size,2000000)))+0.2*torch.mean((correlation_data[1,:]-correlation_data[0,:])/reg_const_one[2,j])
            print('fimished evaluating '+str(i+1)+' out of '+str(dimension)+' attributes')
    else:
        i=source_target
        j=called_target
        print('testing specific pair '+str(source_target)+' and '+str(called_target)+'using full dataset')
        correlation_data=torch.zeros(2,1024)
        counts=torch.zeros(1,1024)
        for k in range(0,1024):
            lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*k/1024
            upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(k+1)/1024+0.01*(k==1023)
            mask=(data_init[:,i]<upper_bound)*(data_init[:,i]>=lower_bound)
            counts[0,k]=torch.sum(mask)
            if torch.sum(mask)==0:
                correlation_data[0,k]=1000000
                correlation_data[1,k]=1000000
            else:
                correlation_data[0,k]=torch.min(data_init[mask,j])
                correlation_data[1,k]=torch.max(data_init[mask,j])
        shared_volume=0.8*torch.sum(counts*(correlation_data[1,:]-correlation_data[0,:])/(reg_const_one[2,j]*size))+0.2*torch.mean((correlation_data[1,:]-correlation_data[0,:])/reg_const_one[2,j])
    print(shared_volume)
    return shared_volume

def test_finegrain_functional_dependency(source_attribute,called_attribute,cuts,cutstwo):
    print('testing more finegrained functional dependency to choose assisting attribute')
    shared_volume=torch.zeros((dimension))
    for i in range(0,dimension):
        if i in bayes_source_attribute or i in bayes_called_attribute or i in bayes_assist_attribute:
            shared_volume[i]=100
        else:
            correlation_data=torch.zeros((2,cuts,cutstwo))
            counts=torch.zeros((cuts,cutstwo))
            for j in range(0,cuts):
                lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*j/cuts
                upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j+1)/cuts+0.01*(j==cuts-1)
                mask_j=(data_init[:,i]<upper_bound)*(data_init[:,i]>=lower_bound)
                for k in range(0,cutstwo):
                    lower_bound=reg_const_one[1,source_attribute]+3*reg_const_one[2,source_attribute]*k/cutstwo
                    upper_bound=reg_const_one[1,source_attribute]+3*reg_const_one[2,source_attribute]*(k+1)/cutstwo+0.01*(k==cutstwo-1)
                    mask_k=(data_init[:,source_attribute]<upper_bound)*(data_init[:,source_attribute]>=lower_bound)
                    mask=mask_k*mask_j
                    counts[j,k]=torch.sum(mask)
                    if torch.sum(mask)==0:
                        correlation_data[0,j,k]=1000000
                        correlation_data[1,j,k]=1000000
                    else:
                        correlation_data[0,j,k]=torch.min(data_init[mask,called_attribute])
                        correlation_data[1,j,k]=torch.max(data_init[mask,called_attribute])
            shared_volume[i]=0.8*torch.sum(counts*(correlation_data[1,:,:]-correlation_data[0,:,:])/(reg_const_one[2,called_attribute]*min(size,2000000)))+0.2*torch.mean((correlation_data[1,:,:]-correlation_data[0,:,:])/reg_const_one[2,called_attribute])
    print('assisting atribute chosen to be the one who can best lower the bayes_called_attribute range')
    print(shared_volume)
    return torch.argmin(shared_volume)

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

class SingleAttributeHistogram:
    """
    A class to create, save, load, and query single-attribute histograms
    for a multi-dimensional database. It handles Most Common Values (MCVs)
    separately for improved accuracy.
    """

    def __init__(self, num_bins: int, num_mcvs: int):
        """
        Initializes the histogram model with hyperparameters.

        Args:
            num_bins (int): The number of bins to use for the histogram part.
            num_mcvs (int): The number of most common values to store separately.
        """
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

        Args:
            data (np.ndarray): The database table, with shape (num_rows, num_dimensions).
            hist_path (str): Path to save the histogram counts array.
            mcv_path (str): Path to save the MCV information (values and counts).
            meta_path (str): Path to save metadata (min/max values, total rows).
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
                
                # Sort by counts in descending order
                sorted_indices = torch.argsort(counts, descending=True)
                
                # Get the top N MCVs
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

        Args:
            hist_path (str): Path to the histogram counts array file.
            mcv_path (str): Path to the MCV information file.
            meta_path (str): Path to the metadata file.
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

if __name__ == "__main__":
    params=['CardEst','power','7','-100000000',"[]",'True',"False","[]"]
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    dimension_init=int(params[2])
    dimension=dimension_init
    nan_to=float(params[3])
    date_like=ast.literal_eval(params[4])
    use_bayes=(params[5]=='True')
    has_categorical=(params[6]=='True')
    categorical_attribute_index=ast.literal_eval(params[7])
    numerical_attributes=[]
    for i in range(0,dimension_init):
        if i not in categorical_attribute_index:
            numerical_attributes.append(i)
    data_init=pd.read_csv(dataset_name+'training/'+'original'+dataset_name+'.csv',index_col=False)
    data_init=data_init.to_numpy()
    data_init=data_init[data_init[:,0]>(nan_to+1e-6),:]
    data_init[:,date_like]=date_to_days(torch.tensor(data_init[:,date_like]))
    print("Training Single Attribute Histograms")
    hist_estimator = SingleAttributeHistogram(num_bins=200, num_mcvs=20)
    hist_estimator.train(data_init,dataset_name+'/'+dataset_name+'_histogram.npy',dataset_name+'/'+dataset_name+'_mcv.npz',dataset_name+'/'+dataset_name+'_meta.npz')
    print('Preparing data for Bayesnet test')
    regu_step1()
    bayes_source_attribute=[]
    bayes_called_attribute=[]
    bayes_assist_attribute=[]
    if use_bayes:
        print("Testing Functional Dependency: small number indicate larger dependency with 0.15 being the Bayesnet threshold")
        shared_volume=test_functional_dependency()
        data_init=data_init_orig
        while True:
            max_correlation=torch.argmin(shared_volume)
            max_correlation=[max_correlation//dimension,max_correlation%dimension]
            if shared_volume[max_correlation]>=(0.15+0.1*has_categorical):
                break
            if shared_volume[max_correlation]<(0.15+0.1*has_categorical):
                if test_functional_dependency(max_correlation[0],max_correlation[1])<(0.15+0.1*has_categorical):
                    bayes_source_attribute.append(max_correlation[0])
                    bayes_called_attribute.append(max_correlation[1])
                    shared_volume[:,bayes_called_attribute]=1000
                    shared_volume[:,bayes_source_attribute]=1000
                    shared_volume[bayes_called_attribute,:]=1000
                    bayes_assist_attribute.append(-1)
                    print("Near functional dependency detected")
                    print("Bayes_source_attribute is:")
                    print(max_correlation[0])
                    print("Bayes_called_attribute is:")
                    print(max_correlation[1])
                else:
                    shared_volume[max_correlation[0],max_correlation[1]]=1000
        for i in range(len(bayes_source_attribute)):
            if bayes_source_attribute[len(bayes_source_attribute)-1-i] not in bayes_source_attribute[len(bayes_source_attribute)-i:]:
                bayes_assist_attribute[len(bayes_source_attribute)-1-i]=test_finegrain_functional_dependency(bayes_source_attribute[len(bayes_source_attribute)-1-i],bayes_called_attribute[len(bayes_source_attribute)-1-i],10,100)     
            else:
                for j in range(len(bayes_source_attribute)-i,len(bayes_source_attribute)):
                    if bayes_source_attribute[len(bayes_source_attribute)-1-i]==bayes_source_attribute[j]:
                        bayes_assist_attribute[len(bayes_source_attribute)-1-i]=bayes_assist_attribute[j]
        PKFK_Flag=np.array([False for i in range(len(bayes_source_attribute))])
        PKFK_info=[[] for i in range(len(bayes_source_attribute))]
        print('Training Round 1')
        for i in range(len(bayes_source_attribute)):
            flag_change_location=i
            BayesNet=BayesnetEstimator(bayes_called_attribute[i],bayes_source_attribute[i],bayes_assist_attribute[i],150,8,10,1)
            BayesNet.train(data_used)
            print(i)
        print('Adjustment Round 1')
        print(PKFK_Flag)
        print(PKFK_info)
        for i in range(len(bayes_source_attribute)):
            flag_change_location=i
            for j in range(len(bayes_source_attribute)):
                if bayes_source_attribute[i]==bayes_source_attribute[j] and i!=j and PKFK_Flag[j]==True and PKFK_Flag[i]==False:
                    BayesNet=BayesnetEstimator(bayes_called_attribute[i],bayes_source_attribute[i],bayes_assist_attribute[i],(PKFK_info[j])[0],(PKFK_info[j])[1],(PKFK_info[j])[2],(PKFK_info[j])[3])
                    BayesNet.train(data_used)
                    break
    else:
        bayes_source_attribute=[]
        bayes_called_attribute=[]
        bayes_assist_attribute=[]
    bayesarray=np.array([bayes_source_attribute,bayes_called_attribute,bayes_assist_attribute])
    np.save(dataset_name+'/'+dataset_name+'_bayesarray.npy',bayesarray)
