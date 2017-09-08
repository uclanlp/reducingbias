#MODIFY THE PATH HERE TO POINT TO CAFFE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib.legend_handler import HandlerLine2D
import myplot as myplt
import mystat as mystat
import inference_debias_tobeModified as cocoutils
import fairCRF_utils_tobeModified as myutils
myutils.set_GPU(0)

def preprocess(margin, vSRL, is_dev = 1):
    if vSRL == 1:
        print "preprocessing imSitu dataset"
        #load data, 1-- dev file
        print "start loading potential files"
        file_num = int(myutils.configs['file_num'])
        arg_inner_all, value_frame_all, label_all,len_verb_file = myutils.load_potential_files(file_num, is_dev)

        print "calcuating the accuracy before calibrating"
        acc = myutils.get_acc(arg_inner_all, value_frame_all, label_all, len_verb_file)
        print "arg-acc cross before calibrating: ", acc

        role_potential_file = myutils.configs['role_potential_file']
        words_file = myutils.configs['words_file']
        num_cons_verb = int(myutils.configs['num_cons_verb'])
        output_index,id_verb,verb_id,verb_roles = myutils.process_mapfile(role_potential_file)
        # cons_verbs  = myutils.get_cons_verbs_for_gender(words_file, role_potential_file,
        #                                                 arg_inner_all, value_frame_all, label_all,
        #                                                 output_index, id_verb, verb_id,verb_roles, num_cons_verb)
        cons_verbs = myutils.read_cons_verbs(myutils.configs['cons_verbs_file'], verb_id)

        arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman = myutils.get_arg_idx_map(words_file, role_potential_file) #{verb_id-m:[arg_idx], verb_id-f:[arg_idx]}
        all_gender_idx = [item for key in arg_idx_map for item in arg_idx_map[key] ] #all the arg_idx related to "man" or "woman"
        all_man_idx = [item for key in arg_idx_map for item in arg_idx_map[key] if '-m' in key]
        all_woman_idx = [item for key in arg_idx_map for item in arg_idx_map[key] if '-f' in key]

        #generate the constraints
        print "generating the constraints"
        training_file = myutils.configs['training_file']
        num_gender = int(myutils.configs['num_gender'])
        constraints = myutils.generate_gender_constraints(training_file, words_file, num_gender, cons_verbs, margin, verb_id, val = 0.0) #constraint = {verb_id:((m_c1,m_c2), (f_c1,f_c2), val)}
        print "total number of constraints: ", len(constraints) * 2
        print "-------------------"
        reargs = (arg_inner_all, value_frame_all, label_all, len_verb_file, all_man_idx, all_woman_idx, constraints, output_index, id_verb, verb_roles, cons_verbs, num_gender,words_file, training_file, role_potential_file, verb_id)
        return reargs
    
    else:
        print "preprocessing COCO dataset"
        train_samples = pickle.load(open(myutils.configs['coco_train_file']))
        dev_samples = pickle.load(open(myutils.configs['coco_dev_file']))
        test_samples = pickle.load(open(myutils.configs['coco_test_file']))
        count_train = cocoutils.compute_man_female_per_object_322(train_samples)
        if is_dev == 1:
            arg_inner_raw = pickle.load(open(myutils.configs['coco_dev_potential']))
            target = np.array([sample['annotation'] for sample in dev_samples])
            print "finish loading coco dev potential files"
        else:
            arg_inner_raw = pickle.load(open(myutils.configs['coco_test_potential']))
            target = np.array([sample['annotation'] for sample in test_samples])
            print "finish loading coco test potential files"
        
        arg_inner_list = []
        for i in range(len(arg_inner_raw)):
            arg_inner_list.append(arg_inner_raw[i]['output'])
        arg_inner_all = np.concatenate(arg_inner_list, axis = 0)
        
        all_man_idx = [item for item in range(2,162)]
        all_woman_idx = [item for item in range(162, 322)]
        acc1 = cocoutils.accuracy(arg_inner_all, target)
        print "accuracy before calibrating: ", acc1
        
        top1_bef, pred_objs_bef = cocoutils.inference(arg_inner_all)
        id2obj = cocoutils.id2object
        obj2id = {id2obj[key]:key for key in id2obj}
        cons_verbs_raw = pd.read_csv(myutils.configs['cons_objs_file'], header=None)
        cons_verbs = [obj2id[verb] for verb in list(cons_verbs_raw[0].values)]
        number_obj  = int(myutils.configs["number_obj"])
        all_constraints, cons_objs = cocoutils.get_constraints(train_samples, number_obj, margin) # in format {verb_id:((m_c1,m_c2), (f_c1,f_c2), val)}
        constraints = cocoutils.get_partial_cons(all_constraints, cons_verbs)
        print "total number of constraints: ", len(constraints) * 2
        print "--------------------------"
        reargs = (constraints, all_man_idx, all_woman_idx, arg_inner_all, target, pred_objs_bef, cons_verbs, train_samples)
        return reargs
        

def show_results(margin, imSitu, *args):
    if imSitu == 1:
        res = myutils.get_gender_ratio_res(*args)
        myplt.plot_tr_ax_gender(res, margin, 1)
        myplt.plot_mean_ratio_dis_trax_gender(res, margin, 1)
        mystat.get_violated_verbs(res, margin)
        mystat.get_dis_gound_all(res,margin)
        mystat.get_bias_score(res, margin)
    else:
        res = myutils.coco_get_res(*args)
        myplt.coco_plot_train_x(res, margin)
        myplt.coco_plot_mean_ratio_dis_trax(res, margin, 1)
        mystat.coco_get_violated_objs(res, margin)
        mystat.coco_get_dis_ground(res, margin)
        mystat.coco_get_bias_score(res, margin)