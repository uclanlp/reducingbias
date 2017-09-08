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
import ConfigParser
import io
import ast
import inference_debias_tobeModified as cocoutils
import myplot as myplt
import pickle


def parse_config(config_file = "config.ini"):
    res = {}
    with open(config_file) as f:
        sample_config = f.read()
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.readfp(io.BytesIO(sample_config))
    for section in config.sections():
        if section == "list_values":
            cur_res = {}
            for item in config.items(section):
                cur_res[item[0]] =  ast.literal_eval(config.get(section, item[0]))
        else:
            cur_res = dict(config.items(section))
        z = res.update(cur_res)
    return res

configs = parse_config()
crf_path = configs['crf_path']
os.chdir(crf_path)
caffe_root = configs['caffe_root']
sys.path.insert(0, caffe_root+"python")
import caffe

def set_GPU(GPU_ID = 0):
    caffe.set_mode_gpu()
    if GPU_ID != 0:
        caffe.set_device(GPU_ID)

def load_potential_files(file_num, is_dev = 1):
    arg_inner_raw = []
    value_frame_raw = []
    label_raw=[]
    len_verb_file = []
    #fix the following
    for i in range(file_num):
        if is_dev == 1:
            data = np.load(open(configs['dev_potential_path'] + "%s.npz"%i))
        else:
            data = np.load(open(configs['test_potential_path'] + "%s.npz"%i))
        arg_inner_raw.append(data["arg"]) #role potential
        value_frame_raw.append(data["verb"]) #verb potential
        label_raw.append(data["label"]) #label
        len_verb_file.append(len(arg_inner_raw[i])) #number of instances for this verb i
        if i %50 == 0:
            print ".",
    if is_dev == 1:
        print "\nFinish loading dev potential files"
    else:
        print "\nFinish loading test potential files"
    arg_inner_all = np.concatenate(arg_inner_raw,axis=0)
    value_frame_all = np.concatenate(value_frame_raw,axis=0)
    label_all = np.concatenate(label_raw,axis=0)
    return arg_inner_all, value_frame_all, label_all, len_verb_file



def get_word_gender_map(wordsfile): #wordmap2:{noun:'f'/'m'}
    word_map = {}
    wordmap2 = {}
    M = configs['m']
    F = configs['f']
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
    return wordmap2

# to get all the agents for each image in training data.
def get_training_agents(training_file, words_file):
    ref = json.load(open(training_file))
    wordmap2 = get_word_gender_map(words_file)
    training_agents = {}
    for (image,s) in ref.items():
        image_name = str(image.split(".")[0])
        agents = set()
        for r in s["frames"]:
            if r.has_key('agent') and r['agent'] != '':
                for item in wordmap2[r['agent']]:
                    agents.add(item)
        if image_name not in training_agents:
            training_agents[image_name] = list(agents)
        else:
            training_agents[image_name].extend(list(agents))
    return training_agents

#to calculate the m/m+f ratio in training dataset with m+f> 50 for gender ratio version
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
    return training_ratio, train_collect_agents

def read_cons_verbs(cons_verb_file, verb_id):
    cons_verbs = []
    with open(cons_verb_file) as f:
        for line in f:
            cons_verbs.append(verb_id[line.strip()]) 
    return cons_verbs

def generate_gender_constraints(training_file, words_file, number_gender, cons_verbs, margin, verb_id, val = 0.0):
    """{verb_id: ((m_c1, f_c1), (m_c2, f_c2), val)}; this is the constraints for all verbs; later will filter them with interested verbs"""
    training_ratio, train_collect_agents = get_training_gender_ratio(training_file, words_file, number_gender)
    all_constraints = {}
    for verb in training_ratio.keys():
        all_constraints[verb_id[verb]] = (((training_ratio[verb]- 1 - margin), (training_ratio[verb] - margin)),
                                      ((1 - margin - training_ratio[verb]), -(margin + training_ratio[verb])),
                                     val)
    constraints = {verb:all_constraints[verb] for verb in cons_verbs if verb in all_constraints}
    return constraints



def inference(arg_inner, value_frame, label, output_index, id_verb, verb_roles):
    batchsize = 3000
    words_file = configs['words_file']
    role_potential_file = configs['role_potential_file']
    for i in range(arg_inner.shape[0]/batchsize + 1):
        net2 = caffe.Net("crf_only_score.prototxt", caffe.TEST)
        arg_inner_tmp = arg_inner[i*batchsize: (i+1)*batchsize]
        value_frame_tmp = value_frame[i*batchsize: (i+1)*batchsize]
        label_tmp = label[i*batchsize: (i+1)*batchsize]
        net2.params['arg-inner'][0].reshape(arg_inner_tmp.shape[0], arg_inner_tmp.shape[1])
        net2.params['value-frame'][0].reshape(value_frame_tmp.shape[0], value_frame_tmp.shape[1])
        net2.params['label'][0].reshape(label_tmp.shape[0], label_tmp.shape[1])
        net2.params['arg-inner'][0].data[...] = arg_inner_tmp
        net2.params['value-frame'][0].data[...] = value_frame_tmp
        net2.params['label'][0].data[...] = label_tmp
        result_tmp = net2.forward()
        if i == 0:
            result = result_tmp
        else:
            result['frame-structure'] = np.concatenate((result['frame-structure'], result_tmp['frame-structure']),
                                                      axis = 0)

            result['frame-score'] = np.concatenate((result['frame-score'], result_tmp['frame-score']),
                                                  axis = 0)

    top1 = []
    arg_idx = []
    for k in range(len(arg_inner)):
        output=result['frame-structure'][k][0][0]
        score= result['frame-score'][k][0][0]
        verb_idx = score.argmax()
        vector = output[7 * verb_idx: 7 * (verb_idx + 1)]
        rv = map_from_output(output_index, id_verb, verb_roles, vector)
        arg_idx = [t['i'] for t in rv[0]['role_potential']]
        top1.append((verb_idx, arg_idx))

    pred_agents = {item[0]:[0,0] for item in top1}
    arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman = get_arg_idx_map(words_file, role_potential_file)
    for i in range(len(top1)):
        agent_m = '-'.join([str(top1[i][0]), 'm'])
        agent_f = '-'.join([str(top1[i][0]), 'f'])
        for arg_id in top1[i][1] :
            if agent_m in arg_idx_map:
                if arg_id in arg_idx_map[agent_m]:
                    pred_agents[top1[i][0]][0] += 1
            if agent_f in arg_idx_map:
                if arg_id in arg_idx_map[agent_f]:
                    pred_agents[top1[i][0]][1] += 1
    return top1, pred_agents

def get_pred_agents(words_file, role_potential_file, id_verb, top1): #{pred_verb_id-m:[pred_arg_idx], pred_verb_id-f:[pred_arg_idx]}
    pred_agents = {}
    arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman = get_arg_idx_map(words_file, role_potential_file)
    for i in range(len(top1)):
        agent_m = '-'.join([str(top1[i][0]), 'm'])
        agent_f = '-'.join([str(top1[i][0]), 'f'])
        agent_m2 = '-'.join([id_verb[top1[i][0]], 'm'])
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

def get_pred_gender_ratio(words_file, role_potential_file, id_verb, pred_agents, num_verbs):
    pred_ratio = {}

    for verb in pred_agents:
        if pred_agents[verb][0] + pred_agents[verb][1] >= num_verbs:
            pred_ratio[verb] = float(pred_agents[verb][0])  / (pred_agents[verb][0] + pred_agents[verb][1])
    return pred_ratio


def get_cons_verbs_for_gender(words_file, role_potential_file, arg_inner,
    value_frame, label, output_index, id_verb, verb_id, verb_roles, num_verbs = 20):
    ori_inference, pred_agents = inference(arg_inner, value_frame, label, output_index, id_verb, verb_roles)
    ori_ratio = get_pred_gender_ratio(words_file, role_potential_file, id_verb, pred_agents, num_verbs)
    cons_verbs = [verb for verb in ori_ratio]
    return cons_verbs



def get_acc(arg_inner, value_frame, label, len_verb_file):
    cur_len = 0
    acc = 0.0
    for i in len_verb_file:
        net = caffe.Net(crf_path + "crf_only.prototxt", caffe.TEST)
        arg_inner_tmp = arg_inner[cur_len: cur_len + i]
        value_frame_tmp = value_frame[cur_len: cur_len + i]
        label_tmp = label[cur_len: cur_len + i]
        cur_len += i
        net.params['arg-inner'][0].reshape(len(arg_inner_tmp), arg_inner.shape[1])
        net.params['value-frame'][0].reshape(len(value_frame_tmp), value_frame.shape[1])
        net.params['label'][0].reshape(len(label_tmp), label.shape[1])
        net.params['arg-inner'][0].data[...] = arg_inner_tmp
        net.params['value-frame'][0].data[...] = value_frame_tmp
        net.params['label'][0].data[...] = label_tmp
        acc += net.forward()['top1-arg-acc']
    acc /= len(len_verb_file)
    return acc

#used to get the arg_idx in predictions
def process_mapfile(role_potential_file):
    mapfile = open(role_potential_file)
    output_index = {}
    id_verb = {}
    verb_id = {}
    verb_roles = {}
    argid_gender = {}
    for line in mapfile.readlines():
        tabs = line.split("\t")
        output_index[(int(tabs[4]), int(tabs[5]), int(tabs[6]))] = (int(tabs[0]), (tabs[1], tabs[2], tabs[3]))
        id_verb[int(tabs[4])] = tabs[1]
        if int(tabs[4]) not in verb_roles: verb_roles[int(tabs[4])] = set()
        verb_roles[int(tabs[4])].add(int(tabs[5]))
        verb_id[tabs[1]] = int(tabs[4])
    return  output_index,id_verb,verb_id,verb_roles

def map_from_output( map_table, id_verb, verb_roles, vector ) :
  if len(vector) % 7 > 0:
    print "Error: mapping a vector whose length is not divisible by 7"
    exit()
  rv = []
  for i in range(0, len(vector), 7):
    active = {}
    verb_id = vector[i]
    verb = id_verb[verb_id]
    active["verb_potential"] = {"i":verb_id, "n": verb}
    roles = []
    for j in range( 0, len(verb_roles[verb_id])):
      potential = map_table[(verb_id, j, vector[i+j+1])]
      roles.append({"i":potential[0], "n":potential[1]})
    active["role_potential"] = roles
    rv.append(active)
  return rv




#return {verb_id-gender:[arg_idx]} from 121381 combinations
def get_arg_idx_map(words_file, role_potential_file):
    wordmap2 = get_word_gender_map(words_file)
    role_file = open(role_potential_file).readlines()
    #item: arg_id, verb, role, noun, verb_id, local_idx, number_for_this_role
    role_map2 = [(item.split()[0], item.split()[1],item.split()[2], wordmap2[item.strip().split()[3]], item.split()[4])
                 for item in role_file
                 if (item.split()[2] == 'agent' or item.split()[2] == 'agents') and
                 (wordmap2[item.strip().split()[3]]=='m' or wordmap2[item.strip().split()[3]]=='f'
                  or wordmap2[item.strip().split()[3]] =='mf')]
    arg_idx_map = {'-'.join([item[4], gender]):[] for item in role_map2 for gender in ['m', 'f'] }
    arg_idx_agent_not_man = [int(item.split()[0]) for item in role_file if (item.split()[2] == 'agent' or item.split()[2] == 'agents')
                             and   wordmap2[item.strip().split()[3]] != 'm'
                                     and wordmap2[item.strip().split()[3]] != 'mf' ]
    arg_idx_agent_not_woman = [int(item.split()[0]) for item in role_file
                               if (item.split()[2] == 'agent' or item.split()[2] == 'agents') and
                                    wordmap2[item.strip().split()[3]] != 'f'
                                     and wordmap2[item.strip().split()[3]] != 'mf' ]
    for item in role_map2:
        if item[3] == 'm' or item[3] == 'mf':
            arg_idx_map['-'.join([item[4], 'm'])].append(int(item[0]))
        if item[3] == 'f' or item[3] == 'mf':
            arg_idx_map['-'.join([item[4], 'f'])].append(int(item[0]))
    return arg_idx_map, arg_idx_agent_not_man, arg_idx_agent_not_woman


#[verb_ids] for verbs which occur more than 5 times in the prediction
def get_cons_verbs(arg_inners, value_frames, labels, iamge_agent, output_index, id_verb, verb_roles,verb_threshold = 0):
    ori_top1 = inference(arg_inners, value_frames, labels, output_index, id_verb, verb_roles)
    pred_verb, pred_verb_ratio_m, pred_verb_ratio_f = get_pred_verb_ratio(ori_top1, image_agent)

    cons_verbs = [item for item in pred_verb if pred_verb[item] > verb_threshold]
    return cons_verbs

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

def get_gender_ratio_res(cons_verbs,num_gender, words_file, training_file, role_potential_file, arg_inner_gold, value_frame_gold, label, arg_inner_tmp, value_frame_tmp, output_index, id_verb, verb_id, verb_roles):
    #get the original and after_LR inference and their ratios with top1
    ori_inference, ori_pred_agents = inference(arg_inner_gold, value_frame_gold, label, output_index, id_verb, verb_roles)
    after_inference, after_pred_agents = inference(arg_inner_tmp, value_frame_tmp, label, output_index, id_verb, verb_roles)
    ori_ratio = get_pred_gender_ratio(words_file, role_potential_file, id_verb, ori_pred_agents, 1)
    after_ratio = get_pred_gender_ratio(words_file, role_potential_file, id_verb, after_pred_agents, 1)
    training_ratio,train_collect_agents = get_training_gender_ratio(training_file, words_file, num_gender)
    golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])

    train_df = pd.DataFrame(training_ratio.items(), columns = ["verb", "training_ratio"])
    ori_df = pd.DataFrame(ori_ratio.items(), columns=['verb_id', 'bef_ratio'])
    after_df = pd.DataFrame(after_ratio.items(), columns=['verb_id', 'after_ratio'])
    tmp = ori_df.merge(after_df)
    train_df["verb_id"] = train_df["verb"].apply(lambda x: verb_id[x])
    com_df = train_df.merge(tmp)
    res0 = com_df.sort_values(by = ['training_ratio'], ascending=1)
    res0['bef_diff'] = res0['training_ratio'] - res0['bef_ratio']
    res0['after_diff'] = res0['training_ratio'] - res0['after_ratio']
    resx = res0.merge(golden_verbs_df)
    resx.sort_values(by = ['training_ratio'], ascending=1, inplace = True)
    del resx['verb_id']
    resx.reset_index(inplace=True, drop = True)
    return resx

def adopt_lambdas( margin, lambdas, constraints, test_arg_inner_all, test_value_frame_all, test_label_all, test_len_verb_file, output_index,
id_verb, verb_roles, all_man_idx, all_woman_idx, get_acc, inference ):
    arg_inner_tmp_test = test_arg_inner_all.copy()
    value_frame_tmp_test = test_value_frame_all.copy()
    label_tmp_test = test_label_all.copy()
    acc1 = get_acc(arg_inner_tmp_test, value_frame_tmp_test, label_tmp_test,test_len_verb_file)
    print "ori arg acc on test without adopting lr: ", acc1
#     lambdas = load_lambdas(margin)
    top1_test = inference(arg_inner_tmp_test, value_frame_tmp_test, label_tmp_test,
                                  output_index, id_verb, verb_roles)

    for i in range(len(arg_inner_tmp_test)):
        for arg_idx in top1_test[i][1]:
            k = int(top1_test[i][0])
            if arg_idx in all_man_idx:
                if k in lambdas: #the first constraint for this verb
                    arg_inner_tmp_test[i][arg_idx] -= lambdas[k][0] * constraints[k][0][0]

                    arg_inner_tmp_test[i][arg_idx] -= lambdas[k][1] * constraints[k][1][0]
            if arg_idx in all_woman_idx:
                if k in lambdas: #the first constraint
                    arg_inner_tmp_test[i][arg_idx] -= lambdas[k][0] * constraints[k][0][1]

                    arg_inner_tmp_test[i][arg_idx] -= lambdas[k][1] * constraints[k][1][1]

    acc1 = get_acc(arg_inner_tmp_test, value_frame_tmp_test, label_tmp_test, test_len_verb_file)
    print "ori arg acc on test with adopting lr: ", acc1
    return arg_inner_tmp_test, value_frame_tmp_test, label_tmp_test

def get_update_index(pred_results, i, arg_idx, is_man, vSRL):
    if vSRL == 1:
        k = int(pred_results[i][0])
    else:
        if is_man == 1:
            k = (arg_idx - 2) / 2
        else:
            k = (arg_idx - 162) / 2
    return k
    
###################used for COCO analysis ################
def coco_get_res(arg_inner_tmp, cons_verbs, train_samples, pred_objs_bef):
    top1_aft, pred_objs_aft = cocoutils.inference(arg_inner_tmp)
    id_objs = cocoutils.id2object
    id_objs_df = pd.DataFrame.from_dict(id_objs, orient="index")
    id_objs_df.reset_index(inplace = True)
    id_objs_df.columns=["obj_id", "object"]
    cons_obj_df0 = pd.DataFrame(cons_verbs, columns = ['obj_id'])
    cons_obj_df= cons_obj_df0.merge(id_objs_df)
    training_ratio = cocoutils.get_training_ratio(train_samples)
    train_df = pd.DataFrame(training_ratio.items())
    train_df.columns = ['obj_id', 'train_ratio']
    aft_df = pd.DataFrame.from_dict(pred_objs_aft, orient='index')
    bef_df = pd.DataFrame.from_dict(pred_objs_bef, orient='index')
    bef_df.reset_index(inplace = True)
    aft_df.reset_index(inplace = True)
    bef_df.columns = ['obj_id', 'bef_man', 'bef_woman']
    bef_df['bef_ratio'] = bef_df['bef_man'] / ( bef_df['bef_man']+  bef_df['bef_woman'])
    aft_df.columns = ['obj_id', 'aft_man', 'aft_woman']
    aft_df['aft_ratio'] = aft_df['aft_man'] / ( aft_df['aft_man']+  aft_df['aft_woman'])
    lr_df = bef_df.merge(aft_df, how = 'outer')
    df = lr_df.merge(train_df)
    df.fillna(value = 0)
    df = df.merge(cons_obj_df)
    df.sort_values(by = ['train_ratio'], inplace = True)
    df.reset_index(inplace = True, drop = True)
    df['bef_diff'] = df['bef_ratio'] - df['train_ratio']
    df['aft_diff'] = df['aft_ratio'] - df['train_ratio']
    return df


####################for vSRL example ###################################
def show_amplified_bias(margin, vSRL, is_dev = 1):
    if vSRL == 1:
        print "start loading potential files"
        file_num = int(configs['file_num'])
        arg_inner_all, value_frame_all, label_all, len_verb_file = load_potential_files(file_num, is_dev)
        arg_inner_gold = arg_inner_all.copy()
        value_frame_gold = value_frame_all.copy()
        label = label_all.copy()
        role_potential_file = configs['role_potential_file']
        words_file = configs['words_file']
        training_file = configs['training_file']
        num_cons_verb = int(configs['num_cons_verb'])
        output_index, id_verb, verb_id, verb_roles = process_mapfile(role_potential_file)
        cons_verbs = read_cons_verbs(configs['cons_verbs_file'], verb_id)
        num_gender = int(configs['num_gender'])
        ori_inference, ori_pred_agents = inference(arg_inner_gold, value_frame_gold, label, output_index, id_verb, verb_roles)
        ori_ratio = get_pred_gender_ratio(words_file, role_potential_file, id_verb, ori_pred_agents, 1)
        training_ratio,train_collect_agents = get_training_gender_ratio(training_file, words_file, num_gender)
        golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])
        train_df = pd.DataFrame(training_ratio.items(), columns = ["verb", "training_ratio"])
        ori_df = pd.DataFrame(ori_ratio.items(), columns=['verb_id', 'bef_ratio'])
        train_df["verb_id"] = train_df["verb"].apply(lambda x: verb_id[x])
        res = train_df.merge(ori_df)
        res = res.merge(golden_verbs_df)
        res.sort_values(by = ['training_ratio'], ascending=1, inplace = True)
        myplt.plot_bias(res, margin, is_dev)
    else:
        print "preprocessing COCO dataset"
        train_samples = pickle.load(open(configs['coco_train_file']))
        dev_samples = pickle.load(open(configs['coco_dev_file']))
        count_train = cocoutils.compute_man_female_per_object_322(train_samples)
        id2obj = cocoutils.id2object
        obj2id = {id2obj[key]:key for key in id2obj}
        if is_dev == 1:
            arg_inner_raw = pickle.load(open(configs['coco_dev_potential']))
            target = np.array([sample['annotation'] for sample in dev_samples])
            print "finish loading coco dev potential files"
        else:
            arg_inner_raw = pickle.load(open(configs['coco_test_potential']))
            target = np.array([sample['annotation'] for sample in test_samples])
            print "finish loading coco test potential files"
        arg_inner_list = []
        for i in range(len(arg_inner_raw)):
            arg_inner_list.append(arg_inner_raw[i]['output'])
        arg_inner_all = np.concatenate(arg_inner_list, axis = 0)
        top1_bef, pred_objs_bef = cocoutils.inference(arg_inner_all)
        cons_verbs_raw = pd.read_csv(configs['cons_objs_file'], header=None)
        cons_verbs = [obj2id[verb] for verb in list(cons_verbs_raw[0].values)]

        id_objs = cocoutils.id2object
        id_objs_df = pd.DataFrame.from_dict(id_objs, orient="index")
        id_objs_df.reset_index(inplace = True)
        id_objs_df.columns=["obj_id", "object"]
        cons_obj_df0 = pd.DataFrame(cons_verbs, columns = ['obj_id'])
        cons_obj_df= cons_obj_df0.merge(id_objs_df)
        training_ratio = cocoutils.get_training_ratio(train_samples)
        train_df = pd.DataFrame(training_ratio.items())
        train_df.columns = ['obj_id', 'training_ratio']

        bef_df = pd.DataFrame.from_dict(pred_objs_bef, orient='index')
        bef_df.reset_index(inplace = True)
        bef_df.columns = ['obj_id', 'bef_man', 'bef_woman']
        bef_df['bef_ratio'] = bef_df['bef_man'] / ( bef_df['bef_man']+  bef_df['bef_woman'])
        res = train_df.merge(bef_df)
        res = res.merge(cons_obj_df)
        res.sort_values(by = ['training_ratio'], inplace = True)
        res.reset_index(inplace = True, drop = True)
        myplt.plot_bias(res, margin, is_dev)
