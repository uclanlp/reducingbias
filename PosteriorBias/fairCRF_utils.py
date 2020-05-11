import numpy as np
import pandas as pd
import re
import json
from operator import add
import copy
from collections import Counter
import operator
import os
import sys
import configparser
import io
import ast
import myplot as myplt
import pickle
from baseline_crf_prob import *
from constant import *
import time

os.chdir(initial_path)

def load_potential_files(): # Load files and pretrained models
    eval_file = json.load(open(eval_files))
    encoder = torch.load(encoding_file)
    model = baseline_crf(encoder, cnn_type = cnn_type)
    model.load_state_dict(torch.load(weights_file, map_location='cpu'))
    model.to(device)
    dataset = imSituSituation(image_dir, eval_file, encoder, model.dev_preprocess())
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)
    return eval_file, encoder, model, dataset, loader

# word_map: a map of words and their id, {n00002452: thing}
# wordmap2: {noun_id: 'f'/'m'}
def get_word_gender_map(wordsfile = "words.txt"):
    word_map = {}
    wordmap2 = {}
    M = ['man']
    F = ['woman']
    with open(wordsfile) as words_f:
        for line in words_f:
            words = line.strip().split('\t')
            word_map[words[0]] = words[1]
            all_words = re.split(' |, ', words[1])
            gender = set()
            for m in M:
                if m in all_words:
                    gender.add('m')
                    break
            for f in F:
                if f in all_words:
                    gender.add('f')
                    break
            tmp = ''.join(list(gender))
            wordmap2[words[0]] = tmp
        wordmap2['null'] = 'null'
    return word_map, wordmap2

# to get all the agents for each image in training data.
# training_agents: {image_name: ['m']}
def get_training_agents(training_file, words_file):
    ref = json.load(open(training_file))
    word_map, wordmap2 = get_word_gender_map(words_file)
    training_agents = {}
    for (image, s) in ref.items():
        image_name = str(image.split(".")[0])
        agents = set()
        for r in s["frames"]:
            if 'agent' in r and r['agent'] != '':
                for item in wordmap2[r['agent']]:
                    agents.add(item)
        if image_name not in training_agents:
            training_agents[image_name] = list(agents)
        else:
            training_agents[image_name].extend(list(agents))
    return training_agents

def filter_nongender_instances(instance_file, words_file, type):
    ref = json.load(open(instance_file))
    word_map, wordmap2 = get_word_gender_map(words_file)
    ref_gender = copy.deepcopy(ref)
    training_agents = {}
    for (image, s) in ref.items():
        image_name = str(image.split(".")[0])
        agents = set()
        for r in s["frames"]:
            if 'agent' in r and r['agent'] != '':
                for item in wordmap2[r['agent']]:
                    agents.add(item)
        if len(agents) == 0:
            del ref_gender[image]
            continue
        if image_name not in training_agents:
            training_agents[image_name] = list(agents)
        else:
            training_agents[image_name].extend(list(agents))
    if type == "train":
        filename = training_file_gender
    elif type == "test":
        filename = test_file_gender
    else:
        filename = dev_file_gender
    json.dump(ref_gender, open(filename, "w"))
    print ("Number of all instances:", len(ref))
    print ("Number of instances related to gender:", len(ref_gender))
    return training_agents

# to calculate the m/m+f ratio in training dataset with m+f> 50 for gender ratio version
# train_collect_agents: {img_verb: [m_count, f_count]}
# training_ratio: {img_verb: ratio}
def get_training_gender_ratio(training_file, words_file, num_gender = 50):
    training_agents = get_training_agents(training_file, words_file)
    all_images = training_agents.keys()
    training_ratio = {}
    train_collect_gender = {}
    train_collect_agents = {}
    for image in all_images:
        img_verb = image.split("_")[0]
        tmp_gender = []
        tmp_gender = copy.deepcopy(training_agents[image])
        if img_verb not in train_collect_gender.keys():
            train_collect_gender[img_verb] = tmp_gender
        else:
            train_collect_gender[img_verb].extend(tmp_gender)
    col_keys = train_collect_gender.keys()
    for colkey in col_keys:
        m_count = train_collect_gender[colkey].count('m')
        f_count = train_collect_gender[colkey].count('f')
        if m_count + f_count > num_gender:
            ratio = float(m_count)/(m_count + f_count)
            training_ratio[colkey] = ratio
            train_collect_agents[colkey] = [m_count, f_count]
    agent_verbs = [verb for verb in training_ratio]
    return training_ratio, train_collect_agents

# cons_verbs: [verb_id1, verb_id2, ...]
# verb_id: {verb: verb_id}
def read_cons_verbs(cons_verb_file, verb_id):
    cons_verbs = []
    with open(cons_verb_file) as f:
        for line in f:
            cons_verbs.append(verb_id[line.strip()]) 
    return cons_verbs

def get_ori_gender_ratio(loader, encoding, model, wordmap2, output_index, cons_verbs):
    pred_agents_ori = {}
    predictions = []
    target = []
    idx_sorted = []
    mx = len(loader)
    for batch_idx_, (index_, input, _target_) in enumerate(loader):
        if batch_idx_ % 100 == 0:
            print ("Test batch: {}/{}".format(batch_idx_, mx))
        vrn_potential_tmp = torch.load(vrn_logProb_dir + "%d"%(batch_idx_+1))
        v_potential_tmp = torch.load(v_logProb_dir + "%d"%(batch_idx_+1)).to(device)
        for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
            vrn_potential_tmp[vrn_idx] = _vrn.to(device)

        _top1, _pred_agents_ori, _idx_sorted, _predictions = inference_step(encoding, model, wordmap2, vrn_potential_tmp, v_potential_tmp, output_index, cons_verbs)
        for v_id in _pred_agents_ori:
            if v_id in pred_agents_ori:
                pred_agents_ori[v_id][0] += _pred_agents_ori[v_id][0]
                pred_agents_ori[v_id][1] += _pred_agents_ori[v_id][1]
            else:
                pred_agents_ori[v_id] = _pred_agents_ori[v_id]

        if batch_idx_ == 0:
            idx_sorted = _idx_sorted
            predictions = _predictions
            target = _target_
        else:
            idx_sorted = torch.cat((idx_sorted, _idx_sorted))
            predictions = torch.cat((predictions, _predictions))
            target = torch.cat((target, _target_))

    top1_eval = imSituTensorEvaluation(1, 3, encoding)
    top1_eval.add_point(target, predictions.data, idx_sorted.data)
    acc1 = top1_eval.get_average_results()["value"]

    return acc1, pred_agents_ori

# not used
def get_ini_posterior_ratio(dataset_loader, model, encoding, agent_verbs, wordmap2):
    mx = len(dataset_loader)
    gender_ratio = {v_id:0 for v_id in agent_verbs}   #{v_id: ratio}
    male_prob_sum = {v_id:0 for v_id in agent_verbs}
    female_prob_sum = {v_id:0 for v_id in agent_verbs}
    others_prob_sum = {v_id:0 for v_id in agent_verbs}

    for batch_idx, (index, input, target) in enumerate(dataset_loader):
        input_var, target = input.to(device), target.to(device)
        v_prob = torch.exp(torch.load(v_logProb_dir + '%d'%(batch_idx+1))).to(device)
        vrn_prob = torch.load(vrn_logProb_dir + '%d'%(batch_idx+1))

        for vrn_idx, _vrn in enumerate(vrn_prob):
            _vrn = torch.exp(_vrn).to(device)
            (n_ins, n_vr, n_noun) = _vrn.size()
            n_add = 283 - n_noun
            vrn_add = torch.zeros(n_ins, n_vr, n_add).to(device)
            _vrn = torch.cat((_vrn, vrn_add), 2)
            vrn_prob[vrn_idx] = _vrn.to(device)
        vrn_prob = torch.cat(vrn_prob, 1)
        zeros_add = torch.zeros(n_ins, 1, 283).to(device)
        vrn_prob = torch.cat((zeros_add, vrn_prob), 1)
        # vrn_prob: (n_ins, 1789, 283)

        v_vr = torch.tensor(model.v_vr).to(device)
        vrn_grouped = vrn_prob.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles(), 283)
        # vrn_grouped.size() = (n_ins, n_verb = 504, max_roles = 6, n_roles = 283)
        vrn_grouped = vrn_grouped.transpose(0,1).transpose(1,2).transpose(2,3)

        for v_id in agent_verbs:
            r_id = encoding.r_id["agent"]
            vr_id = encoding.vr_id[(v_id, r_id)]
            vr_localid = model.vr_id_to_local[vr_id]
            r_localid = model.vr_v[vr_localid][1]

            for n_localid in range(len(encoding.vr_n_id[vr_id])):
                n_id = encoding.vr_id_n[vr_id][n_localid]
                n = encoding.id_n[n_id]
                if n != '':
                    gender = wordmap2[n]
                    if gender == "m" or gender == "mf":
                    # if gender == "m":
                        male_prob_sum[v_id] += torch.sum(vrn_grouped[v_id][r_localid][n_localid]).item()
                    if gender == "f" or gender == "mf":
                    # elif gender == "f":
                        female_prob_sum[v_id] += torch.sum(vrn_grouped[v_id][r_localid][n_localid]).item()
                    # else:
                    #     others_prob_sum[v_id] += torch.sum(vrn_grouped[v_id][r_localid][n_localid]).item()

    for v_id in agent_verbs:
        gender_ratio[v_id] = male_prob_sum[v_id] / (female_prob_sum[v_id] + male_prob_sum[v_id])

    return gender_ratio

# constraints: {verb_id: all_constraints[verb_id]}
def generate_gender_constraints(training_file, words_file, number_gender, cons_verbs, margin, verb_id, val = 0.0):
    """{verb_id: ((m_c1, f_c1), (m_c2, f_c2), val)};
    Here (m_c1, f_c1) stands for ratio >= training_ratio, and (m_c2, f_c2) stands for ratio <= training_ratio.
    """
    training_ratio, train_collect_agents = get_training_gender_ratio(training_file, words_file, number_gender)
    all_constraints = {}
    for verb in training_ratio.keys():
        all_constraints[verb_id[verb]] = (((training_ratio[verb]- 1 - margin), (training_ratio[verb] - margin)),
                                        ((1 - margin - training_ratio[verb]), -(margin + training_ratio[verb])),
                                        val)   # Refer to the notebook for explanation
    constraints = {verb: all_constraints[verb] for verb in cons_verbs if verb in all_constraints}
    return constraints

def get_pred_agents(words_file, role_potential_file, id_verb, top1): #{pred_verb_id-m:[pred_arg_idx], pred_verb_id-f:[pred_arg_idx]}
    pred_agents = {}
    arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman = get_arg_idx_map(words_file, role_potential_file)
    for i in range(len(top1)):
        agent_m = '-'.join([str(top1[i][0]), 'm'])  # verb_id-gender
        agent_f = '-'.join([str(top1[i][0]), 'f'])
        agent_m2 = '-'.join([id_verb[top1[i][0]], 'm']) # verb-gender
        agent_f2 = '-'.join([id_verb[top1[i][0]], 'f'])
        for arg_id in top1[i][1] :
            if agent_m in arg_idx_map:
                if arg_id in arg_idx_map[agent_m]:
                    if agent_m2 not in pred_agents:
                        pred_agents[agent_m2] = [arg_id]
                    else:
                        pred_agents[agent_m2].append(arg_id)
            if agent_f in arg_idx_map:
                if arg_id in arg_idx_map[agent_f]:
                    if agent_f2 not in pred_agents:
                        pred_agents[agent_f2] = [arg_id]
                    else:
                        pred_agents[agent_f2].append(arg_id)
    return pred_agents

def get_pred_gender_ratio(pred_agents, num_verbs):
    pred_ratio = {}
    for verb in pred_agents:
        if pred_agents[verb][0] + pred_agents[verb][1] >= num_verbs:
            pred_ratio[verb] = float(pred_agents[verb][0])  / (pred_agents[verb][0] + pred_agents[verb][1])
    return pred_ratio

def inference_step(encoding, model, wordmap2, vrn_potential, v_potential, vrn_map, cons_verb):
    # generate dict vr_r: {vr_localid, (v_id, r_localid)}
    # len(vr_v) == 1788
    t0 = time.time()

    # generate top1 and pred_agents
    top1 = []
    pred_agents = {}

    n_ins = vrn_potential[0].size()[0]
    vrn_marginal = []
    vr_max = []
    vr_maxi = []
    for i, vrn_group in enumerate(vrn_potential):
        _vrn = vrn_group.view(n_ins * vrn_potential[i].size()[1], vrn_potential[i].size()[2])
        _vr_maxi, _vr_max, _vrn_marginal = model.log_sum_exp(_vrn)
        _vr_maxi = _vr_maxi.view(-1, len(model.split_vr[i]))
        _vr_max = _vr_max.view(-1, len(model.split_vr[i]))
        _vrn_marginal = _vrn_marginal.view(-1, len(model.split_vr[i]))

        vr_maxi.append(_vr_maxi)
        vr_max.append(_vr_max)
        vrn_marginal.append(_vrn_marginal)

    # concat role groups with the padding symbol
    zeros = torch.zeros(n_ins, 1).to(device)  # this is the padding
    zerosi = torch.LongTensor(n_ins, 1).zero_().to(device)
    vrn_marginal.insert(0, zeros)
    vr_max.insert(0, zeros)
    vr_maxi.insert(0, zerosi)

    vrn_marginal = torch.cat(vrn_marginal, 1)
    vr_max = torch.cat(vr_max, 1)
    vr_maxi = torch.cat(vr_maxi, 1)

    v_vr = torch.tensor(model.v_vr).to(device)   # v_vr: [vr_id, ..., vr_id], max_vr in a role for one v. 
    vr_max_grouped = vr_max.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())
    vr_maxi_grouped = vr_maxi.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())

    v_max = torch.sum(vr_max_grouped, 2).view(n_ins, model.n_verbs) + v_potential
    (s_sorted, idx_sorted) = torch.sort(v_max, 1, True)

    scores = v_max
    predictions = vr_maxi_grouped

    for ins_localid in range(len(idx_sorted)):
        v_id = idx_sorted[ins_localid][0].item()
        if v_id not in cons_verb:
            continue
        r_id = encoding.r_id["agent"]
        if (v_id, r_id) not in encoding.vr_id:
            continue
        vr_id = encoding.vr_id[(v_id, r_id)]
        vr_localid = model.vr_id_to_local[vr_id]
        r_localid = model.vr_v[vr_localid][1]
        if r_localid > encoding.verb_nroles(v_id):
            print ("Wrong! r_localid = ", r_localid)
            sys.exit()

        n_localid = vr_maxi_grouped[ins_localid][v_id][r_localid].item()
        n_id = encoding.vr_id_n[vr_id][n_localid]
        n = encoding.id_n[n_id]
        arg_id = vrn_map[(v_id, r_localid, n_localid)][0]
        # print (n_localid, n_id, n)
        
        if n != '':
            if wordmap2[n] != '':   # Thus the verbs in pred_agents are associated with gender, because wordmap2 contains nouns that are associated with gender.
                gender = wordmap2[n]
                n_male = 0
                n_female = 0
                if gender == 'm' or gender == 'mf':
                    n_male = 1
                if gender == 'f' or gender == 'mf':
                    n_female = 1
                if v_id not in pred_agents:
                    pred_agents[v_id] = [n_male, n_female]
                else:
                    pred_agents[v_id][0] += n_male
                    pred_agents[v_id][1] += n_female
                top1.append((ins_localid, v_id, r_id, n_localid, vr_localid, arg_id, n_id, r_id))

    # print ("TIME:", time.time() - t0)
    # print (len(top1))
    # print (pred_agents, len(pred_agents))
    # sys.exit()

    return (top1, pred_agents, idx_sorted, predictions)

def inference_grouped(encoding, model, wordmap2, vrn_q, v_potential, vrn_map, cons_verb, isPR):
    # generate dict vr_r: {vr_localid, (v_id, r_localid)}
    # len(vr_v) == 1788
    # vrn_potential: [(3, , 50), (3, , 100), (3, , 283)]
    # vrn_q: (3, 504, 6, 283)

    # generate top1 and pred_agents
    top1 = []
    pred_agents = {}

    n_ins = vrn_q.size()[0]
    vr_max_grouped, vr_maxi_grouped = torch.max(vrn_q, dim = 3)
    vr_max_grouped = torch.where(vr_max_grouped == float("-infinity"), torch.full_like(vr_max_grouped, 0), vr_max_grouped)
    vr_maxi_grouped = torch.where(vr_maxi_grouped == 282, torch.full_like(vr_maxi_grouped, 0), vr_maxi_grouped)

    # if isPR == 1:
    #     v_max = torch.prod(vr_max_grouped, 2).view(n_ins, model.n_verbs) + v_potential
    # else:
    
    v_max = torch.sum(vr_max_grouped, 2).view(n_ins, model.n_verbs) + v_potential
    (s_sorted, idx_sorted) = torch.sort(v_max, 1, True)

    scores = v_max
    predictions = vr_maxi_grouped

    for ins_localid in range (len(idx_sorted)):
        v_id = idx_sorted[ins_localid][0].item()
        if v_id not in cons_verb:
            continue
        r_id = encoding.r_id["agent"]
        if (v_id, r_id) not in encoding.vr_id:
            continue
        vr_id = encoding.vr_id[(v_id, r_id)]
        vr_localid = model.vr_id_to_local[vr_id]
        r_localid = model.vr_v[vr_localid][1]
        if r_localid > encoding.verb_nroles(v_id):
            print ("Wrong! r_localid = ", r_localid)
            sys.exit()

        n_localid = vr_maxi_grouped[ins_localid][v_id][r_localid].item()
        n_id = encoding.vr_id_n[vr_id][n_localid]
        n = encoding.id_n[n_id]
        arg_id = vrn_map[(v_id, r_localid, n_localid)][0]
        # print (n_localid, n_id, n)
        
        if n != '':
            if wordmap2[n] != '':   # Thus the verbs in pred_agents are associated with gender, because wordmap2 contains nouns that are associated with gender.
                gender = wordmap2[n]
                n_male = 0
                n_female = 0
                if gender == 'm' or gender == 'mf':
                    n_male = 1
                if gender == 'f' or gender == 'mf':
                    n_female = 1
                if v_id not in pred_agents:
                    pred_agents[v_id] = [n_male, n_female]
                else:
                    pred_agents[v_id][0] += n_male
                    pred_agents[v_id][1] += n_female
                top1.append((ins_localid, v_id, r_id, n_localid, vr_localid, arg_id, n_id, r_id))

    # print ("TIME:", time.time() - t0)
    # print (len(top1))
    # print (pred_agents, len(pred_agents))
    # sys.exit()
    return (top1, pred_agents, idx_sorted, predictions)

# use vrn to get the acc; associated with inference function
def get_acc(encoding, model, dataset_loader):
    model.eval()
    top1 = imSituTensorEvaluation(1, 3, encoding)
    # top5 = imSituTensorEvaluation(5, 3, encoding)

    mx = len(dataset_loader)
    for batch_idx, (index, input, target) in enumerate(dataset_loader):
        input_var, target = input.to(device), target.to(device)
        v_potential = torch.load(v_potential_dir + '%d'%(batch_idx+1)).to(device)
        vrn_potential = torch.load(vrn_potential_dir + '%d'%(batch_idx+1))
        for vrn_idx, _vrn in enumerate(vrn_potential):
            vrn_potential[vrn_idx] = _vrn.to(device)
        n_ins = len(input_var)

        vrn_marginal = []
        vr_max = []
        vr_maxi = []
        for i, vrn_group in enumerate(vrn_potential):
            _vrn = vrn_group.view(n_ins * vrn_potential[i].size()[1], vrn_potential[i].size()[2])
            _vr_maxi, _vr_max, _vrn_marginal = model.log_sum_exp(_vrn)
            _vr_maxi = _vr_maxi.view(-1, len(model.split_vr[i]))
            _vr_max = _vr_max.view(-1, len(model.split_vr[i]))
            _vrn_marginal = _vrn_marginal.view(-1, len(model.split_vr[i]))

            vr_maxi.append(_vr_maxi)
            vr_max.append(_vr_max)
            vrn_marginal.append(_vrn_marginal)

        # concat role groups with the padding symbol
        zeros = torch.zeros(n_ins, 1).to(device)  # this is the padding
        zerosi = torch.LongTensor(n_ins, 1).zero_().to(device)
        vrn_marginal.insert(0, zeros)
        vr_max.insert(0, zeros)
        vr_maxi.insert(0, zerosi)

        vrn_marginal = torch.cat(vrn_marginal, 1)
        vr_max = torch.cat(vr_max, 1)
        vr_maxi = torch.cat(vr_maxi, 1)

        v_vr = torch.tensor(model.v_vr).to(device)    # v_vr: [vr_id, ..., vr_id], max_vr in a role for one v. 
        vr_max_grouped = vr_max.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())
        vr_maxi_grouped = vr_maxi.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())

        v_max = torch.sum(vr_max_grouped, 2).view(n_ins, model.n_verbs) + v_potential

        scores = v_max
        predictions = vr_maxi_grouped
        (s_sorted, idx_sorted) = torch.sort(scores, 1, True)

        top1.add_point(target, predictions.data, idx_sorted.data)
        # top5.add_point(target, predictions.data, idx.data)

    acc = top1.get_average_results()["value"]
    return (acc)

def get_acc_prob(encoding, model, dataset_loader):
    model.eval()
    top1 = imSituTensorEvaluation(1, 3, encoding)
    # top5 = imSituTensorEvaluation(5, 3, encoding)

    mx = len(dataset_loader)
    for batch_idx, (index, input, target) in enumerate(dataset_loader):
        input_var, target = input.to(device), target.to(device)
        v_potential = torch.load(v_potential_dir + '%d'%(batch_idx+1)).to(device)
        vrn_potential = torch.load(vrn_potential_dir + '%d'%(batch_idx+1))

        n_ins = len(input_var)
        log_softmax_v = torch.nn.LogSoftmax(dim = 1)
        log_softmax_vrn = torch.nn.LogSoftmax(dim = 2)
        v_potential = log_softmax_v(v_potential)

        for vrn_idx, _vrn in enumerate(vrn_potential):
            _vrn = log_softmax_vrn(_vrn.to(device))
            vrn_potential[vrn_idx] = _vrn

        vrn_marginal = []
        vr_max = []
        vr_maxi = []
        for i, vrn_group in enumerate(vrn_potential):
            _vrn = vrn_group.view(n_ins * vrn_potential[i].size()[1], vrn_potential[i].size()[2])
            _vr_maxi, _vr_max, _vrn_marginal = model.log_sum_exp(_vrn)
            _vr_maxi = _vr_maxi.view(-1, len(model.split_vr[i]))
            _vr_max = _vr_max.view(-1, len(model.split_vr[i]))
            _vrn_marginal = _vrn_marginal.view(-1, len(model.split_vr[i]))

            vr_maxi.append(_vr_maxi)
            vr_max.append(_vr_max)
            vrn_marginal.append(_vrn_marginal)

        # concat role groups with the padding symbol
        zeros = torch.zeros(n_ins, 1).to(device)  # this is the padding
        zerosi = torch.LongTensor(n_ins, 1).zero_().to(device)
        vrn_marginal.insert(0, zeros)
        vr_max.insert(0, zeros)
        vr_maxi.insert(0, zerosi)

        vrn_marginal = torch.cat(vrn_marginal, 1)
        vr_max = torch.cat(vr_max, 1)
        vr_maxi = torch.cat(vr_maxi, 1)

        v_vr = torch.tensor(model.v_vr).to(device)    # v_vr: [vr_id, ..., vr_id], max_vr in a role for one v. 
        vr_max_grouped = vr_max.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())
        vr_maxi_grouped = vr_maxi.index_select(1, v_vr).view(n_ins, model.n_verbs, model.encoding.max_roles())

        v_max = torch.sum(vr_max_grouped, 2).view(n_ins, model.n_verbs) + v_potential

        scores = v_max
        predictions = vr_maxi_grouped
        (s_sorted, idx_sorted) = torch.sort(scores, 1, True)

        top1.add_point(target, predictions.data, idx_sorted.data)
        # top5.add_point(target, predictions.data, idx.data)

    acc = top1.get_average_results()["value"]
    return (acc)

# used to get the arg_idx in predictions
def process_mapfile(role_potential_file):
    mapfile = open(role_potential_file)
    output_index = {}
    arg_to_v = {}
    for line in mapfile.readlines():
        tabs = line.split("\t")
        output_index[(int(tabs[4]), int(tabs[5]), int(tabs[6]))] = (int(tabs[0]), (tabs[1], tabs[2], tabs[3]))
        arg_to_v[int(tabs[0])] = (int(tabs[4]), int(tabs[6]), tabs[3])
    return  output_index, arg_to_v

# rv: [{"verb_potential": {"i":verb_id, "n": verb}, {"role_potential": {"i":potential[0], "n":potential[1]}}}]
def map_from_output(map_table, id_verb, verb_roles, vector) :
    if len(vector) % 7 > 0:
        print ("Error: mapping a vector whose length is not divisible by 7")
        exit()
    rv = []
    for i in range(0, len(vector), 7):
        active = {}
        verb_id = vector[i]
        verb = id_verb[verb_id]
        active["verb_potential"] = {"i":verb_id, "n": verb}
        roles = []
        for j in range(0, len(verb_roles[verb_id])):
            potential = map_table[(verb_id, j, vector[i+j+1])]
            roles.append({"i":potential[0], "n":potential[1]})
        active["role_potential"] = roles
        rv.append(active)
    return rv

#return {verb_id-gender:[arg_idx]} from 121381 combinations
def get_arg_idx_map(words_file, role_potential_file):
    word_map, wordmap2 = get_word_gender_map(words_file)
    role_file = open(role_potential_file).readlines()
    # item: arg_id, verb, role, noun, verb_id, r_localid, n_localid
    role_map2 = [(item.split()[0], item.split()[1], item.split()[2], wordmap2[item.strip().split()[3]], item.split()[4])
                    for item in role_file
                    if (item.split()[2] == 'agent' or item.split()[2] == 'agents') 
                        and (wordmap2[item.strip().split()[3]]=='m' 
                            or wordmap2[item.strip().split()[3]]=='f'
                                or wordmap2[item.strip().split()[3]] =='mf')]
    arg_idx_map = {'-'.join([item[4], gender]):[] for item in role_map2 for gender in ['m', 'f'] }
    arg_idx_agent_not_man = [int(item.split()[0]) for item in role_file 
                                if (item.split()[2] == 'agent' or item.split()[2] == 'agents')
                                    and wordmap2[item.strip().split()[3]] != 'm'
                                        and wordmap2[item.strip().split()[3]] != 'mf' ]
    arg_idx_agent_not_woman = [int(item.split()[0]) for item in role_file
                                if (item.split()[2] == 'agent' or item.split()[2] == 'agents') 
                                    and wordmap2[item.strip().split()[3]] != 'f'
                                        and wordmap2[item.strip().split()[3]] != 'mf' ]
    for item in role_map2:
        if item[3] == 'm' or item[3] == 'mf':
            arg_idx_map['-'.join([item[4], 'm'])].append(int(item[0]))
        if item[3] == 'f' or item[3] == 'mf':
            arg_idx_map['-'.join([item[4], 'f'])].append(int(item[0]))
    # print (len(role_map2), len(arg_idx_map))
    return arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman

def save_lambdas(path, lambdas):
    with open(path, 'w') as f:
        for k in lambdas:
            f.write(str(k) + '\t' + str(lambdas[k][0]) + '\t' + str(lambdas[k][1]) + '\n')

def save_iterations(path, results):
    with open(path, 'w') as f:
        for item in results:
            f.write(str(item[0]) + '\t' + str(item[1]) + "\t" + str(item[2]) + '\n')

def load_lambdas(lambdas_file):
    lambdas = {}
    with open(lambdas_file) as f:
        lines = f.readlines()
        for line in lines:
            words = line.split('\t')
            lambdas[int(words[0])] = [float(words[1]), float(words[2])]
    return lambdas

def get_dataset_comparison(encoder, cons_verbs):
    training_agents = filter_nongender_instances(training_file, words_file, type = "train")
    ori_agents = filter_nongender_instances(test_file, words_file, type = "test")
    after_agents = filter_nongender_instances(dev_file, words_file, type = "dev")
    training_ratio, _ = get_training_gender_ratio(training_file_gender, words_file, num_gender)
    ori_ratio, _ = get_training_gender_ratio(test_file_gender, words_file, num_gender)
    after_ratio, _ = get_training_gender_ratio(dev_file_gender, words_file, num_gender)

    golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])
    train_df = pd.DataFrame(training_ratio.items(), columns = ["verb", "training_ratio"])
    ori_df = pd.DataFrame(ori_ratio.items(), columns=['verb', 'bef_ratio'])
    after_df = pd.DataFrame(after_ratio.items(), columns=['verb', 'after_ratio'])
    train_df["verb_id"] = train_df["verb"].apply(lambda x: encoder.v_id[x])
    ori_df["verb_id"] = ori_df["verb"].apply(lambda x: encoder.v_id[x])
    after_df["verb_id"] = after_df["verb"].apply(lambda x: encoder.v_id[x])
    tmp = ori_df.merge(after_df)
    com_df = train_df.merge(tmp)
    res0 = com_df.sort_values(by = ['training_ratio'], ascending=1)
    res0['bef_diff'] = res0['training_ratio'] - res0['bef_ratio']   # training_ratio - bef_ratio
    res0['after_diff'] = res0['training_ratio'] - res0['after_ratio']   # training_ratio - after_ratio
    resx = res0.merge(golden_verbs_df)

    resx.sort_values(by = ['training_ratio'], ascending = 1, inplace = True)
    del resx['verb_id']
    resx.reset_index(inplace = True, drop = True)

    # myplt.plot_tr_ax_gender(resx, margin = 0.05, filename = "Comparison of training & test or dev set")

def arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model):
    v_id, n_localid, noun = arg_to_v[arg_id]
    r_id = encoding.r_id["agent"]
    vr_id = encoding.vr_id[(v_id, r_id)]
    vr_localid = model.vr_id_to_local[vr_id]
    if vr_localid == 0:
        print ("Wrong! v_id is", v_id)
        sys.exit()
    if vr_localid > splits_offset[2]:
        split_id = 2
        vr_split_localid = vr_localid - splits_offset[2] - 1
    elif vr_localid > splits_offset[1]:
        split_id = 1
        vr_split_localid = vr_localid - splits_offset[1] - 1
    else:
        split_id = 0
        vr_split_localid = vr_localid - 1
    return (split_id, n_localid, vr_split_localid)

def get_gender_ratio_res_PR(cons_verbs, num_gender, words_file, training_file, encoding, model, loader, 
                            all_man_idx, all_woman_idx, output_index, wordmap2, arg_to_v, lambdas, constraints):
    # get the original and after_PR inference and their ratios with top1
    # 1) Get training ratio 
    # 2) Load: vrn_logProb(before calibrating), vrnq_dir(after_calibrating)
    # 3) Calculate gender ratio: for each verb, sum up its male and female probability, then calculate [male:female]

    training_ratio, train_collect_agents = get_training_gender_ratio(training_file, words_file, num_gender)

    pred_agents = {}
    predictions = []
    target = []
    idx_sorted = []
    mx = len(loader)

    before_ratio = {item:[0.0, 0.0, 0.0] for item in constraints}
    after_ratio = {item:[0.0, 0.0, 0.0] for item in constraints}
    for batch_idx_, (index_, input, _target_) in enumerate(loader):
        if batch_idx_ % 100 == 0:
            print ("Calculate batch: {}/{}".format(batch_idx_, mx))
        v_potential_tmp = torch.load(v_logProb_dir + "%d"%(batch_idx_+1)).cpu().detach().numpy()
        vrn_before = torch.load(vrn_logProb_dir + "%d"%(batch_idx_+1))
        vrn_after = torch.load(vrn_logProb_dir + "%d"%(batch_idx_+1))
        
        for vrn_idx, _vrn in enumerate(vrn_before):
            vrn_before[vrn_idx] = vrn_before[vrn_idx].transpose(0,2).cpu().detach().numpy()
            vrn_after[vrn_idx] = vrn_after[vrn_idx].transpose(0,2).cpu().detach().numpy()
        
        n_ins = np.shape(vrn_before[0])[2]

        # vrn_potential: [(50, 1125, n_ins), (100, 518, n_ins), (283, 145, n_ins)]
        # update vrn_after using lambdas
        for arg_id in all_man_idx:
            v_id, n_localid, noun = arg_to_v[arg_id]
            if v_id not in constraints:
                continue
            if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                continue
            split_id, n_localid, vr_split_localid = arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
            temp = (-lambdas[v_id][0] * constraints[v_id][0][0] - lambdas[v_id][1] * constraints[v_id][1][0])
            vrn_after[split_id][n_localid][vr_split_localid] += temp

        for arg_id in all_woman_idx:
            v_id, n_localid, noun = arg_to_v[arg_id]
            if v_id not in constraints:
                continue
            if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                continue
            split_id, n_localid, vr_split_localid = arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
            temp = (-lambdas[v_id][0] * constraints[v_id][0][1] - lambdas[v_id][1] * constraints[v_id][1][1])
            vrn_after[split_id][n_localid][vr_split_localid] += temp
        
        # size = [(50, 1125, n_ins), (100, 518, n_ins), (283, 145, n_ins)]  (n_localid, vr_localid, ins_id)
        # calculate gender ratio
        for arg_id in all_man_idx:
            v_id, n_localid, noun = arg_to_v[arg_id]
            if v_id not in constraints:
                continue
            split_id, n_localid, vr_split_localid = arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
            for ins_id in range(n_ins):
                before_ratio[v_id][0] += (math.exp(vrn_before[split_id][n_localid][vr_split_localid][ins_id]) * math.exp(v_potential_tmp[ins_id][v_id]))
                after_ratio[v_id][0] += (math.exp(vrn_after[split_id][n_localid][vr_split_localid][ins_id]) * math.exp(v_potential_tmp[ins_id][v_id]))

        for arg_id in all_woman_idx:
            v_id, n_localid, noun = arg_to_v[arg_id]
            if v_id not in constraints:
                continue
            split_id, n_localid, vr_split_localid = arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
            for ins_id in range(n_ins):
                before_ratio[v_id][1] += (math.exp(vrn_before[split_id][n_localid][vr_split_localid][ins_id]) * math.exp(v_potential_tmp[ins_id][v_id]))
                after_ratio[v_id][1] += (math.exp(vrn_after[split_id][n_localid][vr_split_localid][ins_id]) * math.exp(v_potential_tmp[ins_id][v_id]))

    for v_id in before_ratio:
        before_ratio[v_id][2] = before_ratio[v_id][0] / (before_ratio[v_id][0] + before_ratio[v_id][1])
        after_ratio[v_id][2] = after_ratio[v_id][0] / (after_ratio[v_id][0] + after_ratio[v_id][1])

    before_ratio_readable = {}
    after_ratio_readable = {}
    for v_id in before_ratio:
        before_ratio_readable[v_id] = before_ratio[v_id][2]
        after_ratio_readable[v_id] = after_ratio[v_id][2]

    golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])
    train_df = pd.DataFrame(training_ratio.items(), columns = ["verb", "training_ratio"])
    ori_df = pd.DataFrame(before_ratio_readable.items(), columns=['verb_id', 'bef_ratio'])
    after_df = pd.DataFrame(after_ratio_readable.items(), columns=['verb_id', 'after_ratio'])
    tmp = ori_df.merge(after_df)
    train_df["verb_id"] = train_df["verb"].apply(lambda x: encoding.v_id[x])
    com_df = train_df.merge(tmp)
    res0 = com_df.sort_values(by = ['training_ratio'], ascending=1)
    res0['bef_diff'] = res0['training_ratio'] - res0['bef_ratio']   # training_ratio - bef_ratio
    res0['after_diff'] = res0['training_ratio'] - res0['after_ratio']   # training_ratio - after_ratio
    resx = res0.merge(golden_verbs_df)
    resx.sort_values(by = ['training_ratio'], ascending=1, inplace = True)
    del resx['verb_id']
    resx.reset_index(inplace=True, drop = True)
    return resx, after_ratio_readable

def get_update_index(top1, i, vSRL):
    # top1: [(0 ins_localid, 1 v_id, 2 r_id, 3 n_localid, 4 vr_localid, 5 arg_id, 6 n_id, 7 r_id)]
    if vSRL == 1:
        ins_id = top1[i][0]
        v_id = top1[i][1]
        n_localid = top1[i][3]
        vr_localid = top1[i][4]
        n_id = top1[i][6]
        r_id = top1[i][7]
        if vr_localid > splits_offset[2]:
            split_id = 2
            vr_split_localid = vr_localid - splits_offset[2] - 1
        elif vr_localid > splits_offset[1]:
            split_id = 1
            vr_split_localid = vr_localid - splits_offset[1] - 1
        else:
            split_id = 0
            vr_split_localid = vr_localid - 1
    # else:
    #     if is_man == 1:
    #         k = (arg_idx - 2) / 2
    #     else:
    #         k = (arg_idx - 162) / 2
    return (ins_id, v_id, vr_localid, n_localid, split_id, n_id, r_id, vr_split_localid)