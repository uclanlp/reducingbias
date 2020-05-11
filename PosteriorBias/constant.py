import torch
import os

# TODO: find gender.json on server
is_filter = 0   # Filter the instances that are not related to gender when set to be 1.
margin = 0.01
batch_size = 35 # Must be same when training and doing calibrating.

cnn_type = 'resnet_101'
splits = [50, 100, 283]
splits_offset = [0, 1125, 1643]
num_verb = 504
num_gender = 50
num_cons_verb = 212
num_agent_verb = 364
number_obj = 100

# used in baseline model
training_epochs = 20
eval_frequency = 1
print_freq = 5
top_k = 5
learning_rate = 1e-5
weight_decay = 5e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

initial_path = "/home/jiasy16/jiasy16/ReduceBias_ratio_PR"
data_dir = '/local/jiasy16/PRdata/'
image_dir = data_dir + 'resized_256/'

# Here eval files is what used to be evaled calibrating in the future
if is_filter == 1:
  file_type = 'test_gender'  # 'test_gender' or 'dev_gender'
  eval_files = data_dir + 'data/test_gender.json'
  # following are used for model training and getting training gender ratio
  training_file = data_dir + 'data/train_gender.json'
  test_file = data_dir + 'data/test_gender.json'
  dev_file = data_dir + 'data/dev_gender.json'
else:
  file_type = 'test'  # 'test' or 'dev'
  eval_files = data_dir + 'data/test.json'
  training_file = data_dir + 'data/train.json'
  test_file = data_dir + 'data/test.json'
  dev_file = data_dir + 'data/dev.json'
  training_file_gender = data_dir + 'data/train_gender.json'
  test_file_gender = data_dir + 'data/test_gender.json'
  dev_file_gender = data_dir + 'data/dev_gender.json'

# Encoder and model files.
encoding_file = data_dir + 'baseline_models/baseline_encoder'
weights_file = data_dir + 'baseline_models/0.3878453207671934.model'

# Some necessary files for processing data.
words_file = data_dir + 'words.txt'
cons_verbs_file = data_dir + 'verbs'
agent_verbs_file = data_dir + 'agent_verbs'


save_dir = data_dir + file_type + '/'

vrn_potential_table_file = save_dir + 'vrn_potential_table'

# initial potential score
vrn_potential_dir = save_dir + 'vrn_potential/' 
v_potential_dir = save_dir + 'v_potential/'

# potential score after logsoftmax (to make it have probability meaning, which is required in PR)
v_logProb_dir = save_dir + 'v_logProb/'
vrn_logProb_dir = save_dir + 'vrn_logProb/'

# potential score after logsoftmax and EXP(!!!), saved in another formation
vrn_grouped_dir = save_dir + 'vrn_grouped/'   

# For saving model.
output_dir = data_dir + 'model/'

# For saving parameters in LR method.
save_lambda = 'results/imSitu_gender_ratio_lambdas'
save_iteration = 'results/imSitu_gender_ratio_iterations'

if not os.path.exists(save_dir):
  os.mkdir(save_dir)
if not os.path.exists(vrn_potential_dir):
  os.mkdir(vrn_potential_dir)
if not os.path.exists(v_potential_dir):
  os.mkdir(v_potential_dir)
if not os.path.exists(v_logProb_dir):
  os.mkdir(v_logProb_dir)
if not os.path.exists(vrn_logProb_dir):
  os.mkdir(vrn_logProb_dir)
if not os.path.exists(vrn_grouped_dir):
  os.mkdir(vrn_grouped_dir)
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

