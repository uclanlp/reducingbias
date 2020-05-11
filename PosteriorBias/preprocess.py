#MODIFY THE PATH HERE TO POINT TO CAFFE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import pickle
from matplotlib.legend_handler import HandlerLine2D
import myplot as myplt
import mystat as mystat
import fairCRF_utils as myutils
import baseline_crf_prob as mycrf
import baseline_crf as mycrf_LR
from constant import *
import shutil

def preprocess(margin, isPR = 1):
    print ("preprocessing imSitu dataset")
    # load data, 1-- dev file
    print ("start loading potential files")
    eval_file, encoder, model, dataset, loader = myutils.load_potential_files()
    num_instance = len(dataset)

    # generate new potentials for test and dev dataset
    if not os.path.exists(vrn_potential_dir + "1") and isPR == 0:
        mycrf_LR.save_potential(loader, encoder, model)
    if not os.path.exists(vrn_logProb_dir + "1") and isPR == 1:
        mycrf.save_potential(loader, encoder, model)

    if not os.path.exists(vrn_potential_table_file):
        mycrf.generate_potential_table(encoder, model)

    print ("calcuating the accuracy before calibrating")
    acc = myutils.get_acc(encoder, model, loader)
    print ("arg-acc cross before calibrating: ", acc)

    cons_verbs = myutils.read_cons_verbs(cons_verbs_file, encoder.v_id)
    agent_verbs = myutils.read_cons_verbs(agent_verbs_file, encoder.v_id)
    word_map, wordmap2 = myutils.get_word_gender_map()
    output_index, arg_to_v = myutils.process_mapfile(vrn_potential_table_file)
    if not os.path.exists(vrn_grouped_dir + "1"):
        mycrf.generate_probability(loader, model, encoder, cons_verbs)

    arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman = myutils.get_arg_idx_map(words_file, vrn_potential_table_file) #{verb_id-m:[arg_idx], verb_id-f:[arg_idx]}
    all_gender_idx = [item for key in arg_idx_map for item in arg_idx_map[key] ] #all the arg_idx related to "man" or "woman"
    all_man_idx = [item for key in arg_idx_map for item in arg_idx_map[key] if '-m' in key]
    all_woman_idx = [item for key in arg_idx_map for item in arg_idx_map[key] if '-f' in key]
    # all_man_idx: 367, all_woman_idx: 314, all_gender_idx: 681, arg_idx_map: 376
    # arg_idx_map: {'verb_id-m/f/mf': [arg_idx1, arg_idx2, ...]}

    print ("generating the constraints")
    constraints = myutils.generate_gender_constraints(training_file, words_file, num_gender, cons_verbs, margin, encoder.v_id, val = 0.0) #constraint = {verb_id:((m_c1,m_c2), (f_c1,f_c2), val)}

    print ("total number of constraints: ", len(constraints) * 2)
    print ("-------------------")

    if is_filter == 0:
        myutils.get_dataset_comparison(encoder, cons_verbs)
    reargs = (eval_file, encoder, model, dataset, loader, arg_idx_map, word_map, wordmap2, output_index, 
                arg_to_v, all_man_idx, all_woman_idx, constraints, cons_verbs, agent_verbs)
    return reargs
