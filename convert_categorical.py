import pandas as pd
import numpy as np
import json
import sys
import ast
from itertools import permutations

def quantize_dataset(dataset_name, num_bins, is_categorical):
    """
    Convert a mixed dataset into purely numeric form by replacing
    each categorical column with an integer encoding that respects
    inter-attribute dissimilarities.

    Algorithm:
      - Dissimilarity based on absolute row counts (L1 distance).
      - Placeholder 0 represents "no rows" at both ends of the order.
      - Greedy construction starting from the most common value,
        inserting each remaining value at its best position.
      - Local 2-opt improvement on the final permutation.
    """
    filepath = dataset_name+'training/original'+dataset_name+'_withcat.csv'
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()

    # Precompute bin indices for numerical columns
    num_col_bins = {}
    for idx, col in enumerate(cols):
        if not is_categorical[idx]:
            ser = df[col].astype(float)
            min_val, max_val = ser.min(), ser.max()
            if max_val == min_val:
                num_col_bins[col] = np.zeros(len(df), dtype=int)
            else:
                width = (max_val - min_val) / num_bins
                bins = np.floor((ser - min_val) / width).astype(int)
                bins = np.clip(bins, 0, num_bins - 1)
                num_col_bins[col] = bins.values
        else:
            num_col_bins[col] = None

    cat_attrs = [col for idx, col in enumerate(cols) if is_categorical[idx]]

    # -------- Absolute counts for each categorical value ----------
    cond_counts = {}
    value_counts = {}   # total rows per categorical value, for MCV and d(0, v)
    for attr in cat_attrs:
        cond_counts[attr] = {}
        unique_vals = df[attr].unique()
        val_cnt = {}
        cat_universe = {}
        for j, other_col in enumerate(cols):
            if other_col == attr:
                continue
            if is_categorical[j]:
                cat_universe[other_col] = df[other_col].unique()

        for val in unique_vals:
            mask = df[attr] == val
            sub = df.loc[mask]
            n_sub = len(sub)
            val_cnt[val] = n_sub

            counts = {}
            for j, other_col in enumerate(cols):
                if other_col == attr:
                    continue
                if is_categorical[j]:
                    cnt = sub[other_col].value_counts()
                    cnt_dict = {cat: cnt.get(cat, 0) for cat in cat_universe[other_col]}
                    counts[other_col] = cnt_dict
                else:
                    bins = num_col_bins[other_col][mask.values]
                    bin_counts = np.bincount(bins, minlength=num_bins)
                    counts[other_col] = bin_counts

            cond_counts[attr][val] = counts
        value_counts[attr] = val_cnt
    # ----------------------------------------------------------------
    print('end of binning step')
    # L1 distance between two count vectors
    def l1_distance(count_p, count_q, is_cat):
        if is_cat:
            return sum(abs(count_p[k] - count_q[k]) for k in count_p)
        else:
            return np.sum(np.abs(count_p - count_q))

    # Compute pairwise dissimilarities (including placeholder 0)
    dissimilarity = {}   # attr -> dict with key (u, v) for u,v in values ∪ {0}
    for attr in cat_attrs:
        vals = list(cond_counts[attr].keys())
        diss = {}
        num_other_cols = len(cols) - 1

        # Placeholder 0 distances
        for p in vals:
            n_p = value_counts[attr][p]
            # d(0, p) = (d-1) * n_p^2
            diss[(0, p)] = num_other_cols * (n_p ** 2)
            diss[(p, 0)] = diss[(0, p)]   # symmetric

        # Pairwise distances between real values
        for i, p in enumerate(vals):
            for q in vals[i:]:
                if p == q:
                    diss[(p, q)] = 0.0
                    continue
                total_sq = 0.0
                for j, other_col in enumerate(cols):
                    if other_col == attr:
                        continue
                    d_ij = l1_distance(
                        cond_counts[attr][p][other_col],
                        cond_counts[attr][q][other_col],
                        is_categorical[j]
                    )
                    total_sq += d_ij * d_ij
                diss[(p, q)] = total_sq
                diss[(q, p)] = total_sq

        dissimilarity[attr] = diss
    print('end of dissimilarity calculation')
    # ============== NEW GREEDY INSERTION + SWAPS ==============
    maps = {}
    for attr, diss in dissimilarity.items():
        values = list(value_counts[attr].keys())
        m = len(values)

        # 1. Most common value
        mcv = max(values, key=lambda v: value_counts[attr][v])
        L = [mcv]   # ordered list of real values
        remaining = set(values) - {mcv}

        # Helper: cost of an ordered list (including placeholder 0 at both ends)
        def path_cost(order):
            if not order:
                return 0.0
            cost = diss[(0, order[0])] + diss[(order[-1], 0)]
            for i in range(len(order)-1):
                cost += diss[(order[i], order[i+1])]
            return cost

        # 2. Greedy insertion
        while remaining:
            best_cost = float('inf')
            best_x = None
            best_pos = None
            for x in remaining:
                # Try inserting x at every position 0..len(L)
                for pos in range(len(L)+1):
                    new_L = L[:pos] + [x] + L[pos:]
                    c = path_cost(new_L)
                    if c < best_cost:
                        best_cost = c
                        best_x = x
                        best_pos = pos
            # Insert the best found
            L = L[:best_pos] + [best_x] + L[best_pos:]
            remaining.remove(best_x)

        # 3. Local swaps (2-opt) to refine
        improved = True
        while improved:
            improved = False
            current_cost = path_cost(L)
            for i in range(len(L)-1):
                # swap L[i] and L[i+1]
                new_L = L.copy()
                new_L[i], new_L[i+1] = new_L[i+1], new_L[i]
                new_cost = path_cost(new_L)
                if new_cost < current_cost:
                    L = new_L
                    current_cost = new_cost
                    improved = True
                    break   # restart scanning from beginning after each successful swap
        # (When improved is False, we stop.)

        # 4. Mapping: L[0] -> 1, L[1] -> 2, ..., L[-1] -> m
        mapping = {val: idx + 1 for idx, val in enumerate(L)}
        maps[attr] = mapping

    # =================================================================

    # Save converted dataset
    df_converted = df.copy()
    for attr, mapping in maps.items():
        df_converted[attr] = df[attr].map(mapping)

    out_csv = dataset_name+'training/original'+dataset_name+'.csv'
    df_converted.to_csv(out_csv, index=False)

    map_file = dataset_name+'/categoricalmaps.json'
    maps_serialisable = {
        col: {str(k): v for k, v in mapping.items()} for col, mapping in maps.items()
    }
    with open(map_file, 'w') as f:
        json.dump(maps_serialisable, f, indent=2)

    print(f"Converted dataset saved to {out_csv}")

    return maps

def save_column_metadata(dataset_name, is_categorical):
    """
    Read the header of the original CSV and save a 2×d numpy array
    containing column names (row 0) and is_categorical flags (row 1).

    Parameters
    ----------
    dataset_name : str
        The base name of the dataset (same as used for quantisation).
    is_categorical : list[bool]
        Boolean list, one entry per column.
    """
    filepath = dataset_name+'training/original'+dataset_name+'.csv'
    df_header = pd.read_csv(filepath, nrows=0)
    cols = df_header.columns.tolist()

    # Create a 2×d object array (strings + bools)
    meta = np.empty((2, len(cols)), dtype=object)
    meta[0, :] = cols
    meta[1, :] = is_categorical

    outpath = dataset_name+'/'+dataset_name+'iscat.npy'
    np.save(outpath, meta, allow_pickle=True)
    print(f"Column metadata saved to {outpath}")

def analyze_encoding(dataset_name, num_bins):
    """
    For each attribute, find the four adjacently encoded pairs with the
    highest dissimilarity and return a (8, d) numpy array.

    Numerical columns: all 8 entries = 0.
    Categorical columns: entries arranged as
        [ code_largest,  diss_largest,
          code_2nd,     diss_2nd,
          code_3rd,     diss_3rd,
          code_4th,     diss_4th ]
    where code is the smaller (leftmost) integer of the adjacent pair,
    and diss is the squared L1 dissimilarity.

    Parameters
    ----------
    dataset_name : str
        Base name of the dataset (same as during quantisation).
    num_bins : int
        Number of bins used for numerical columns during encoding.

    Returns
    -------
    result : np.ndarray, shape (8, d)
        The jump analysis for all columns.
    """
    # 1. Load metadata
    meta_path = dataset_name + '/' + dataset_name + 'iscat.npy'
    meta = np.load(meta_path, allow_pickle=True)
    cols = list(meta[0, :])
    is_categorical = list(meta[1, :].astype(bool))
    d = len(cols)

    # 2. Load original dataset (with categorical values)
    filepath = dataset_name + 'training/original' + dataset_name + '_withcat.csv'
    df = pd.read_csv(filepath)
    size=len(df)

    # 3. Load saved mapping
    map_file = dataset_name + '/categoricalmaps.json'
    with open(map_file, 'r') as f:
        maps_serial = json.load(f)

    # 4. Precompute bins for numerical columns (identical to quantize_dataset)
    num_col_bins = {}
    for idx, col in enumerate(cols):
        if not is_categorical[idx]:
            ser = df[col].astype(float)
            min_val, max_val = ser.min(), ser.max()
            if max_val == min_val:
                num_col_bins[col] = np.zeros(len(df), dtype=int)
            else:
                width = (max_val - min_val) / num_bins
                bins = np.floor((ser - min_val) / width).astype(int)
                bins = np.clip(bins, 0, num_bins - 1)
                num_col_bins[col] = bins.values
        else:
            num_col_bins[col] = None

    # 5. Build conditional absolute counts for each categorical value
    cond_counts = {}
    cat_attrs = [col for idx, col in enumerate(cols) if is_categorical[idx]]
    cat_universe = {}
    for attr in cat_attrs:
        cat_universe[attr] = {}
        for j, other_col in enumerate(cols):
            if other_col == attr:
                continue
            if is_categorical[j]:
                cat_universe[attr][other_col] = df[other_col].unique()

    for attr in cat_attrs:
        cond_counts[attr] = {}
        unique_vals = df[attr].unique()
        for val in unique_vals:
            mask = df[attr] == val
            sub = df.loc[mask]
            counts = {}
            for j, other_col in enumerate(cols):
                if other_col == attr:
                    continue
                if is_categorical[j]:
                    cnt = sub[other_col].value_counts()
                    cnt_dict = {cat: cnt.get(cat, 0) for cat in cat_universe[attr][other_col]}
                    counts[other_col] = cnt_dict
                else:
                    bins = num_col_bins[other_col][mask.values]
                    bin_counts = np.bincount(bins, minlength=num_bins)
                    counts[other_col] = bin_counts
            cond_counts[attr][val] = counts

    # Helper: L1 distance between two count vectors
    def l1_distance(count_p, count_q, is_cat):
        if is_cat:
            return sum(abs(count_p[k] - count_q[k]) for k in count_p)
        else:
            return np.sum(np.abs(count_p - count_q))

    # 6. Prepare result array
    result = np.zeros((8, d))

    # 7. For each categorical column, compute jumps
    for attr in cat_attrs:
        j = cols.index(attr)   # column index

        # Recover ordered list of actual values (original types)
        mapping_serial = maps_serial[attr]   # dict: str(val) -> int
        code_to_val = [(code, val_str) for val_str, code in mapping_serial.items()]
        code_to_val.sort(key=lambda x: x[0])
        ordered_vals_str = [val_str for _, val_str in code_to_val]

        # Convert string keys back to original dtype using cond_counts keys
        actual_val_lookup = {str(v): v for v in cond_counts[attr].keys()}
        ordered_actual = []
        for s in ordered_vals_str:
            if s in actual_val_lookup:
                ordered_actual.append(actual_val_lookup[s])
            else:
                # fallback (should not normally happen)
                ordered_actual.append(s)

        m = len(ordered_actual)
        if m <= 1:
            continue   # no jumps, column stays zero

        # Compute dissimilarities for adjacent pairs
        jumps = []
        for i in range(m - 1):
            a = ordered_actual[i]
            b = ordered_actual[i+1]
            code = i + 1   # lower integer code (1‑based)
            total_sq = 0.0
            for other_idx, other_col in enumerate(cols):
                if other_col == attr:
                    continue
                d_ij = l1_distance(
                    cond_counts[attr][a][other_col],
                    cond_counts[attr][b][other_col],
                    is_categorical[other_idx]
                )
                total_sq += ((d_ij/size)**2)/(len(cols)-1)
            jumps.append((code, total_sq))

        # Sort by dissimilarity descending, keep top 4
        jumps.sort(key=lambda x: x[1], reverse=True)
        top4 = jumps[:4]

        # Fill the column: pairs of (code, diss)
        for k, (code, diss) in enumerate(top4):
            result[2*k,   j] = code
            result[2*k+1, j] = diss
    print(result)
    return result

if __name__ == "__main__":
    params=['Converter','power','20',"[0,0,0,0,0,0,0]"]
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    num_bins=int(params[2])
    categorical_marker=ast.literal_eval(params[3])
    is_categorical=[categorical_marker[i]==1 for i in range(0,len(categorical_marker))]
    '''quantize_dataset(dataset_name, num_bins,is_categorical)
    save_column_metadata(dataset_name,is_categorical)'''
    np.save(dataset_name+'training/maxgapstats.npy',analyze_encoding(dataset_name,num_bins))
