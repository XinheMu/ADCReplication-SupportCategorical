Welcome to the replication repository of "ADC: An Accelerated Diffusion Model for Cardinality Estimation"

Our train, test, and valid sets, as well as the results for all compare group models, are generated and derived using the open source provided by 

Xiaoying Wang, Changbo Qu, Weiyuan Wu, Jiannan Wang, and Qingqing Zhou. 2021. Are we ready for learned cardinality estimation? Proceedings of the VLDB Endowment (2021)

At website

https://github.com/sfu-db/AreCELearnedYet

Please consult the original paper and the above link for how to set up their model and run their experiments. 

The instructions on how to train and test the ADC models are provided below.

The packages needed to run the ADC code are provided in the environment.yml file.

To train a model on the dataset dataset_name, eg. "forest", please go to the anonymous Harvard Dataverse Repository 

https://dataverse.harvard.edu/previewurl.xhtml?token=9f86c322-8644-40e0-8256-024eb5ed9709

Download the files ${dataset_name}_real_train.npy; ${dataset_name}_trainset.csv; original${dataset_name}.csv (for continuous-only datasets) and original${dataset_name}_withcat.csv (for categorical datasets census and dmv), and then create a directory named ${dataset_name}training (eg. "foresttraining") under the ADCReplication-SupportCategorical home directory, and put the required files inside.

Then, for datasets without categorical attributes, run the following programs in order:

Train_ADC_All_Histograms_experimental_ver3.py 

Train_ADC_GMM_GPU_ver4.py 

Train_ADC_Network_GPU_ver3.py 

Train_ADC_Classifier_ver6.py

For datasets with categorical attributes, run convert_categorical.py, then run the aforementioned programs in order.

To test the cardinality estimator on any dataset, run ADC_Cardest_experimental_ver7.py

The parameter settings for the tested datasets are (note that the dataset "Modulo" is codenamed "advantage" in our programs):

For convert_categorical.py (eg. run " conda run -n my_conda_env python convert_categorical.py census 20 "[0,1,1,0,1,1,1,1,1,0,0,0,1]" "):

census 20 "[0,1,1,0,1,1,1,1,1,0,0,0,1]"

dmv 20 "[1,1,1,1,1,1,0,1,1,1,1]"

For Train_All_Histograms_experimental_ver2.py:

forest 10 -10000000

power 7 -1

higgs 7 -10000000

advantage 5 -10000000

taxi 23 -10000000

census 13 -10000000 "[]" True True "[1,2,4,5,6,7,8,12]"

dmv 11 -10000000 "[6]" True True "[0,1,2,3,4,5,7,8,9,10]"

For Train_ADC_GMM_GPU_ver4.py:

forest 10 -10000000 

power 7 -1 

higgs 7 -10000000 

advantage 5 -10000000 

taxi 23 -10000000 

census 13 -10000000 "[]" True "[1,2,4,5,6,7,8,12]" 

dmv 11 -10000000 "[6]" True "[0,1,2,3,4,5,7,8,9,10]"

For Train_ADC_Network_GPU_ver2.py:

False 1 forest 10 1/160 32768 True True False 2 -10000000 

False 1 power 7 3/2560 32768 True True False 1 -1 

False 1 higgs 7 1/1280 32768 True True False 1 -10000000 

False 1 advantage 5 1/640 32768 True True False 1 -10000000 

False 1 taxi 23 3/1280 32768 True True False 2 -10000000 

False 1 census 13 1/80 32768 True True False 1 -10000000 "[]" True "[1,2,4,5,6,7,8,12]" 

False 1 dmv 11 1/80 32768 True True False 1 -10000000 "[6]" True "[0,1,2,3,4,5,7,8,9,10]"

For Train_ADC_Classifier_ver6.py:

forest "[1,1,1,1,1,1,1,1,1,1]" 10 1/160 20000 -10000000 

power "[1e-3,1e-3,1e-2,2e-1,1,1,1]" 7 3/2560 20000 -1 

higgs "[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]" 7 1/1280 20000 -10000000000 

advantage "[1,1,1,1,1]" 5 1/640 20000 -10000000000 

taxi "[1,1,1,1e-2,1e-4,1e-4,1e-4,1e-4,5e-2,5e-1,1e-2,1e-2,1e-2,1e-2,1,1,1e-2,1,1e-2,1e-2,0.25,1,1]" 23 3/1280 20000 -1000000000000 

census "[1,1,1,1,1,1,1,1,1,1,1,1,1]" 13 1/80 20000 -100000000 "[]" True 

dmv "[1,1,1,1,1,1,1,1,1,1,1]" 11 1/80 20000 -100000000 "[6]" True

For ADC_Cardest_experimental_ver7.py (Note that “ADC-” can be replaced by “ADC” or “ADC+”):

forest ADC- qerror "[1,1,1,1,1,1,1,1,1,1]" 10 1/160 10000 -10000000 

power ADC- qerror "[1e-3,1e-3,1e-2,2e-1,1,1,1]" 7 3/2560 10000 -1 

higgs ADC- qerror "[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]" 7 1/1280 10000 -10000000000 

advantage ADC- qerror "[1,1,1,1,1]" 5 1/640 10000 -10000000000 

taxi ADC- qerror "[1,1,1,1e-2,1e-4,1e-4,1e-4,1e-4,5e-2,5e-1,1e-2,1e-2,1e-2,1e-2,1,1,1e-2,1,1e-2,1e-2,0.25,1,1]" 23 3/1280 10000 -1000000000000 

census ADC- qerror "[1,1,1,1,1,1,1,1,1,1,1,1,1]" 13 1/80 10000 -100000000 "[]" True 

dmv ADC- qerror "[1,1,1,1,1,1,1,1,1,1,1]" 11 1/80 10000 -100000000 "[6]" True

The meaning for each parameter (please open my python files to see which name matches to each corresponding parameter, other training files follow roughly the same naming convention) of ADC_Cardest_experimental_ver7.py is:

dataset_name: The dataset on which to run our experiment. Currently chosen among the values 'forest', 'power', 'higgs', 'advantage'. Note that the dataset 'modulo' is codenamed 'advantage' in our numerical experiments.

ADCversion: Choose among the values 'ADC-','ADC','ADC+'

output_type: Set to 'qerror' for the program to output and display the Q-error; set to 'sel' for the program to calculate the selectivity without calculating th Q-error. The answer sheet found at location dataset_name+'/'+dataset_name+'_real_test.npy', eg. 'power/power_real_test.npy', is needed for output type 'qerror' but not for output type 'sel'.

unit_of_variables: a list enclosed by "" indicating the numerical precision of each attribute, used for preprocessing the query. Eg. unit_of_variables equal 1 for integral attributes, 1e-1 for attributes rounded to 1 digit decimals, 1e-2 for those rounded to 2 digit decimals, etc.

dimension: dimensionality of the dataset

Time_min: early stopping time of the diffusion model

workload_size: Total number of queries to test

nan_to: Which number did missing values get converted to, used for the dataset 'power' which contain missing values, and whose missing values we converted to -1 in accrodance with the paper "Are We Ready for Learned Cardinality Estimtion"

date_like: A parameter listing all attributes corresponding to dates but are stored in integer format, defaults to "[]".

has_categorical: whether the tested dataset contains categorical attributes, defaults to False

threshold (optional): If output_type is set to 'qerror', all queries with an error bigger than threshold will be outputted and their index will be stored to location dataset_name+'/'+dataset_name+'_high_error_list.npy', eg. 'power/power_high_error_list.npy'

draws (optional): Number of draws for predictor-corrector Monte Carlo scheme, default number 25 balances speed with precision according to my tests, but feel free to adjust if you like.

The overall results for each testing run will be directly printed after each test run The detailed results for each testing run will be saved to the latest sheet in the file "Statistics_"+dataset_name+".xlsx" (eg. Statistics_census.xlsx).

Meaning of the five columns are: 

relseldis: The distribution of the actual selectivity (i.e. actual selectivity sorted in ascending order) 

relsel: The actual selectivity of query number 0 to 9999 

estsel: The estimated selectivity of query number 0 to 9999 

Q: The Q-error of queries 0 to 9999 

SortQ: The Q-error of all queries sorted in ascending order

To run the ablation studies with Bayesnet disabled, change the 4th parameter for "Train_ADC_All_Histograms_experimental_ver2.py" to "False", then run the aforementioned programs in exactly the same sequence, using exactly the same parameter settings. Doing so will not produce different results on the datasets forest, higgs, and advantage, due to no functional dependency being detected in the first place.

To conver queries and labels in the format of "Are We Ready for Learned Cardinality Estimation" into our preferred format (training and testing queries in ADC's preferred format are also provided in the dataverse repository), please run convert_queries.py and convert_labels.py, with parameter setting

${dataset_name}(choose between "forest, power, higgs, advantage, taxi, census, dmv") ${query_type} (choose between "train, test, valid")

Also to run the dataframe-based 10k sampling program samplingtest.py, the parameters are: 

forest 

power 

higgs 

advantage 

taxi 

census 1 2 4 5 6 7 8 12 

dmv 0 1 2 3 4 5 7 8 9 10 

i.e. the name of the dataset followed by all categorical columns.
