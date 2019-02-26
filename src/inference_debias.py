import itertools
import numpy as np
import pdb
import pickle
import os
import copy

id2object = {0: 'toilet', 1: 'teddy_bear', 2: 'sports_ball', 3: 'bicycle', 4: 'kite', 5: 'skis', 6: 'tennis_racket', 7: 'donut', 8: 'snowboard', 9: 'sandwich', 10: 'motorcycle', 11: 'oven', 12: 'keyboard', 13: 'scissors', 14: 'chair', 15: 'couch', 16: 'mouse', 17: 'clock', 18: 'boat', 19: 'apple', 20: 'sheep', 21: 'horse', 22: 'giraffe', 23: 'person', 24: 'tv', 25: 'stop_sign', 26: 'toaster', 27: 'bowl', 28: 'microwave', 29: 'bench', 30: 'fire_hydrant', 31: 'book', 32: 'elephant', 33: 'orange', 34: 'tie', 35: 'banana', 36: 'knife', 37: 'pizza', 38: 'fork', 39: 'hair_drier', 40: 'frisbee', 41: 'umbrella', 42: 'bottle', 43: 'bus', 44: 'zebra', 45: 'bear', 46: 'vase', 47: 'toothbrush', 48: 'spoon', 49: 'train', 50: 'airplane', 51: 'potted_plant', 52: 'handbag', 53: 'cell_phone', 54: 'traffic_light', 55: 'bird', 56: 'broccoli', 57: 'refrigerator', 58: 'laptop', 59: 'remote', 60: 'surfboard', 61: 'cow', 62: 'dining_table', 63: 'hot_dog', 64: 'car', 65: 'cup', 66: 'skateboard', 67: 'dog', 68: 'bed', 69: 'cat', 70: 'baseball_glove', 71: 'carrot', 72: 'truck', 73: 'parking_meter', 74: 'suitcase', 75: 'cake', 76: 'wine_glass', 77: 'baseball_bat', 78: 'backpack', 79: 'sink'}

def get_training_ratio(train_samples):
    count_train = compute_man_female_per_object_322(train_samples)
    train_ratio = {}
    for k in count_train:
        train_ratio[k] = float(count_train[k][0]) / (count_train[k][0] + count_train[k][1])
    return train_ratio

def get_constraints(train_samples, number_objs, margin, val = 0.0):
    margin2 = margin
    count_train = compute_man_female_per_object_322(train_samples)
    train_ratio = {}
    constraints = {}
    obj_cons_train = []
    for k in count_train:
        if (count_train[k][0] + count_train[k][1]) > number_objs:
            obj_cons_train.append(k)
            train_ratio[k] = float(count_train[k][0]) / (count_train[k][0] + count_train[k][1])
    for k in train_ratio:
        constraints[k] = (((train_ratio[k]- 1 - margin), (train_ratio[k] - margin)), # tr-margin<= m/(m+f)
                            ((1 - margin2 - train_ratio[k]), -(margin2 + train_ratio[k])),
                                     val)
    return constraints, obj_cons_train

def get_partial_cons(all_cons, cons_verbs):
    
    partial_cons = {verb:all_cons[verb] for verb in cons_verbs if verb in all_cons}
    return partial_cons

def compute_man_female_per_object_322(samples):
    count = dict()
    for i in range(80):
        count[i] = [0,0]
    for sample in samples:
        sample = sample['annotation']
        if sample[0] == 1: #man
            objs = sample[2:162]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][0] += 1
        else:#woman
            objs = sample[162:]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][1] += 1
    return count

def compute_man_female_per_object_81(samples): #for the predicted results
    count = dict()
    for i in range(80):
        count[i] = [0,0]
    for sample in samples:
        if sample[0] == 0: #man
            for j in sample[1]:
                count[(j-2)/2][0] += 1
        else:#woman
            for j in sample[1]:
                count[(j - 162)/2][1] += 1
    return count

def inference(output):
    """outputshould be list, num_sample*322"""
    results = list()
    top1 = list()
    for i in range(len(output)):
        output_one = output[i]
        man_score = output_one[0]
        woman_score = output_one[1]
        man_objects = output_one[2:162]
        woman_objects = output_one[162:]
        man_index = list()
        woman_index = list()

        for j in range(80):
            if man_objects[j*2] > man_objects[j*2+1]:
                man_index.append(j)
                man_score += man_objects[j*2]
            else:
                man_score += man_objects[j*2+1]

        for j in range(80):
            if woman_objects[j*2] > woman_objects[j*2+1]:
                woman_index.append(j)
                woman_score += woman_objects[j*2]
            else:
                woman_score += woman_objects[j*2+1]

        result = list()
        result_num = [0]*81
        if man_score > woman_score:
            result.append("man")
            tmp = []
            for elem in man_index:
                result.append(id2object[elem])
                tmp.append(2 + elem*2)
            top1.append((0, tmp))
        else:
            result.append("woman")
            tmp = []
            for elem in woman_index:
                result.append(id2object[elem])
                tmp.append(162 + elem*2)
            top1.append((1, tmp))
        results.append(result)
    pred_agents = compute_man_female_per_object_81(top1)
    return top1, pred_agents

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def accuracy(output, target):
    """output and target should be numpy array, num_sample*322"""
    output_gender = output[:, :2].copy()
    target_gender = target[:, :2].copy()
    output_obj_tmp = output[:, 2:].copy()
    target_obj_tmp = target[:, 2:].copy()
    output_objects = output_obj_tmp.reshape(len(output), 2,80,2)
    target_objects = target_obj_tmp.reshape(len(output), 2,80,2)

    target_scores = np.zeros((len(output),1, 80, 2))
    for i in range(len(output)):
        if target_gender[i, 1] == 0:
            target_scores[i, 0, :, :] = target_objects[i, 0, :, :]
        else:
            target_scores[i, 0, :, :] = target_objects[i, 1, :, :]

    target_idx = np.argmin(target_scores, 3)
    target_idx = target_idx.squeeze()
    
    pred_scores_unknown = np.zeros((len(output),1, 80, 2))
    for i in range(len(output)):
        if output_gender[i,0] > output_gender[i,1]:
            output_gender[i,0] = 1
            output_gender[i,1] = 0
            pred_scores_unknown[i, 0, :, :] = output_objects[i, 0, :, :]
        else:
            output_gender[i,0]= 0
            output_gender[i,1] = 1
            pred_scores_unknown[i, 0, :, :] = output_objects[i, 1, :, :]
    pred_idx_unknown = np.argmin(pred_scores_unknown, 3)
    pred_idx_unknown = pred_idx_unknown.squeeze() #batch_size*80
    
    pred_gender = []
    for i in range(len(output)):
        if output_gender[i,0] == 0:
            gender = 1
        else: 
            gender = 0
        pred_gender.append([gender])
    
    targ_gender = []
    for i in range(len(output)):
        if target_gender[i,0] == 0:
            gender = 1
        else: 
            gender = 0
        targ_gender.append([gender])
    
    pred_with_gender = np.concatenate((pred_idx_unknown, pred_gender), axis=1)
    targ_with_gender = np.concatenate((target_idx, targ_gender), axis=1)
    f1_with_gender = f1_score(targ_with_gender, pred_with_gender, average='macro' )
    f1_with_gender = f1_score(target_idx, pred_idx_unknown, average='macro' )
    return f1_with_gender

if __name__=='__main__':
    main()
