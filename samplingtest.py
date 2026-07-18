import sys
import time
import numpy as np
import pandas as pd

def run_estimator(dataset_name, is_cat):
    # ----- Training Phase -----
    t_train_start = time.perf_counter()

    # Construct training file path as specified:
    # dataset_name + "training/original" + dataset_name + ".csv"
    # (or with "_withcat" if is_cat is not empty)
    if len(is_cat) == 0:
        train_path = f"{dataset_name}training/original{dataset_name}.csv"
    else:
        train_path = f"{dataset_name}training/original{dataset_name}_withcat.csv"

    # Read the full dataset (contains headers)
    df = pd.read_csv(train_path)
    N = len(df)   # total number of rows in the original table

    # Draw a sample of up to 10000 rows (or all rows if dataset is smaller)
    sample_size = min(10000, N)
    sample = df.sample(n=sample_size, random_state=None)
    if sample_size == 0:
        sample = pd.DataFrame(columns=df.columns)

    t_train_end = time.perf_counter()
    sample_time_min = (t_train_end - t_train_start) / 60.0   # minutes

    # ----- Model size (KB) -----
    # The "model" here is simply the sampled DataFrame kept in memory.
    # We measure its deep memory usage (including the actual data of objects/strings)
    # and convert bytes to kilobytes.
    model_size_bytes = sample.memory_usage(deep=True).sum()
    model_size_kb = model_size_bytes / 1024.0

    # Scaling factor to convert sample counts to full‑table estimates
    scale = N / sample_size if sample_size > 0 else 1.0

    # ----- Testing Phase -----
    # Load workload: dataset_name + "/" + dataset_name + "_testset.csv" (no headers)
    test_path = f"{dataset_name}/{dataset_name}_testset.csv"
    workload = pd.read_csv(test_path, header=None)

    # Load true cardinalities
    true_path = f"{dataset_name}/{dataset_name}_real_test.npy"
    true_card = np.load(true_path)

    num_queries = len(workload) // 2
    assert num_queries == len(true_card), \
        "Mismatch between workload queries and true cardinalities"

    cols = sample.columns.tolist()
    cat_set = set(is_cat)

    t_test_start = time.perf_counter()

    q_errors = []
    for q_idx in range(num_queries):
        row_lower = workload.iloc[2 * q_idx]
        row_upper = workload.iloc[2 * q_idx + 1]

        mask = pd.Series(True, index=sample.index)

        for col_idx, col_name in enumerate(cols):
            val_lower = row_lower[col_idx]
            if val_lower == "ALLATTRS":
                continue  # no predicate on this attribute

            if col_idx in cat_set:
                # categorical: exact match (both rows give the same value)
                mask &= (sample[col_name] == val_lower)
            else:
                # continuous: lower bound from first row, upper bound from second row
                lower = float(val_lower)
                upper = float(row_upper[col_idx])
                col_vals = pd.to_numeric(sample[col_name], errors='coerce')
                mask &= (col_vals >= lower) & (col_vals <= upper)

        # Count matching rows in the sample and scale to full table size
        match_cnt = mask.sum()
        est_card = match_cnt * scale

        real_card = true_card[q_idx]

        # Q‑error: treat zero cardinalities as 1
        r = real_card if real_card != 0 else 1.0
        e = est_card if est_card != 0 else 1.0
        q_err = max(r, e) / min(r, e)
        q_errors.append(q_err)

    t_test_end = time.perf_counter()
    latency_per_query_ms = ((t_test_end - t_test_start) / num_queries) * 1000.0

    # Compute statistics
    qe = np.array(q_errors)
    median_q = np.median(qe)
    q95 = np.percentile(qe, 95)
    q99 = np.percentile(qe, 99)
    max_q = np.max(qe)

    # ----- Verbose output (seven lines) -----
    print(f"Latency per query (ms): {latency_per_query_ms:.6f}")
    print(f"Training time (min): {sample_time_min:.6f}")
    print(f"Model size (KB): {model_size_kb:.2f}")
    print(f"Median Q-error: {median_q:.6f}")
    print(f"95th percentile Q-error: {q95:.6f}")
    print(f"99th percentile Q-error: {q99:.6f}")
    print(f"Max Q-error: {max_q:.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python estimator.py dataset_name [cat_idx1 cat_idx2 ...]")
        sys.exit(1)

    ds_name = sys.argv[1]
    cat_indices = [int(arg) for arg in sys.argv[2:]]
    run_estimator(ds_name, cat_indices)
