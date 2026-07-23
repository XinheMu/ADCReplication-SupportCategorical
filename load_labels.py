import sys
import os
from dotenv import load_dotenv
import pickle
import pandas as pd # Still useful for intermediate data handling if needed, though final output is custom
from collections import OrderedDict
import csv # For writing the custom CSV format
import numpy as np
for dataset_name in [sys.argv[1]]:
    for query_class in [sys.argv[2]]:
        project_root = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(project_root, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded .env from {dotenv_path}")
        else:
            print(f"Warning: .env file not found at {dotenv_path}")
        data_root_env = os.getenv('DATA_ROOT')
        with open("data/"+dataset_name+"/workload/base-original"+dataset_name+"-label.pkl", "rb") as f:
            loaded_labels=pickle.load(f)
        all_cardinalities=[]
        all_labels=loaded_labels[query_class]
        for i, label in enumerate(all_labels):
            all_cardinalities.append(label.cardinality)
        np.save(dataset_name+'_real_'+query_class+'.npy',np.array(all_cardinalities))
        print('task completed for '+dataset_name+' of query type '+query_class)
