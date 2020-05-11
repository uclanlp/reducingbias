import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib.legend_handler import HandlerLine2D
import myplot as myplt
import mystat as mystat
import fairCRF_utils as myutils
import preprocess as mypreprocess
from constant import *
import time
import copy
import math
from imsitu import imSituTensorEvaluation
import random
import torch

# PR process:
#   for epoch in range(max_epoch):
#       1) compute gradientZ
#       2) update lambdas
#       3) inference once, using lambdas
#           a. get new q(y) from p(y|x) and lambdas
#           b. Do inference
#           c. Update best results and judge whether meet the condition of stopping the whole process.

def posterior_regularization(constraints, dataset, loader, encoding, cons_verbs, agent_verbs, model, word_map, wordmap2, output_index, 
                                all_man_idx, all_woman_idx, arg_to_v, is_dev):
    training_ratio, train_collect_agents = myutils.get_training_gender_ratio(training_file, words_file, num_gender)
    acc, pred_agents_ori = myutils.get_ori_gender_ratio(loader, encoding, model, wordmap2, output_index, cons_verbs)
    # inference results before using PR
    print ("TEST acc is:{}.".format(acc))
    
    # apply gradient descent Adam to find the lambda

    max_epoch = 1
    learning_rate = 0.01
    lr_decay = 0.998
    alpha = 0.9
    beta = 0.999
    epsilon = 1e-6
    inference_freq = 1    # frequency to inference
    print_freq = 1        # frequency to print ratio and lambdas
    update_freq = 1       # frequency to update lambdas

    # other initialization
    results = []
    mx = len(loader)
    batch_num = 0   # number of batches that have been used in total
    min_error = 300
    batch_bst = 0
    lambdas_bst = {item:[0.0, 0.0] for item in constraints}
    lambdas = {item:[0.0, 0.0] for item in constraints}     # lambdas: {v_id: [lambda_left, lambda_right]}
    first_order_sum = {item:[0.0, 0.0] for item in constraints}
    second_order_sum = {item:[0.0, 0.0] for item in constraints}
    M = len(constraints)
    N = len(dataset)

    # Get results of bias amplification before PR.
    res, ori_ratio_PR = myutils.get_gender_ratio_res_PR(cons_verbs, num_gender, words_file, training_file, encoding, model, loader, 
                            all_man_idx, all_woman_idx, output_index, wordmap2, arg_to_v, lambdas_bst, constraints)

    print ("=== Before calibrating ===")
    mystat.get_violated_verbs(res, margin)
    mystat.get_bias_score(res, margin)

    # Start PR process
    print ("Start PR process.")
    gradientZ = {item:[0.0, 0.0] for item in constraints}

    for epoch in range(max_epoch):  # max_epoch can't be too large
        for batch_idx, (index, input, _target) in enumerate(loader):
            batch_num += 1
            print ("Epoch:{}, batch_num:{}".format(epoch, batch_num))
            vrn_grouped = torch.load(vrn_grouped_dir + "%d"%(batch_idx+1)).cpu().detach().numpy()
            v_prob = torch.load(v_logProb_dir + "%d"%(batch_idx+1)).cpu().detach().numpy()

            # compute gradientZ
            sum0_tot = 0.0
            sum1_tot = {item:[0.0, 0.0] for item in constraints}

            n_batch = len(v_prob)
            for ins_id in range(n_batch):
                # print ("------")
                # print ("ins_id:", ins_id)
                # if ins_id % 10 == 0:
                #     print ("batch:{}, ins_id:{}".format(batch_num, ins_id))
                sum0_tot = 0.0
                sum1_tot = {item:[0.0, 0.0] for item in constraints}

                for v_id in cons_verbs:
                    sum0 = 0.0
                    sum1 = {item:[0.0, 0.0] for item in constraints}

                    r_id = encoding.r_id["agent"]
                    vr_id = encoding.vr_id[(v_id, r_id)]
                    vr_localid = model.vr_id_to_local[vr_id]
                    r_localid = model.vr_v[vr_localid][1]

                    for n_localid in range(len(encoding.vr_id_n[vr_id])):
                        # an enumerate of v_g
                        n_id = encoding.vr_id_n[vr_id][n_localid]
                        n = encoding.id_n[n_id]
                        temp = 0   # lambda dot feature function
                        # constraints: {v_id: ((m_c1, f_c1), (m_c2, f_c2), val)}
                        if n != "":
                            if wordmap2[n] == "m":
                                temp -= constraints[v_id][0][0] * lambdas[v_id][0]
                                temp -= constraints[v_id][1][0] * lambdas[v_id][1]
                            elif wordmap2[n] == "f":
                                temp -= constraints[v_id][0][1] * lambdas[v_id][0]
                                temp -= constraints[v_id][1][1] * lambdas[v_id][1]

                        sum0 += (math.exp(temp) * (vrn_grouped[ins_id][v_id][r_localid][n_localid]))
                        if n != "":
                            if wordmap2[n] == "m":
                                sum1[v_id][0] -= constraints[v_id][0][0] * math.exp(temp) * (vrn_grouped[ins_id][v_id][r_localid][n_localid])
                                sum1[v_id][1] -= constraints[v_id][1][0] * math.exp(temp) * (vrn_grouped[ins_id][v_id][r_localid][n_localid])
                            elif wordmap2[n] == "f":
                                sum1[v_id][0] -= constraints[v_id][0][1] * math.exp(temp) * (vrn_grouped[ins_id][v_id][r_localid][n_localid])
                                sum1[v_id][1] -= constraints[v_id][1][1] * math.exp(temp) * (vrn_grouped[ins_id][v_id][r_localid][n_localid])
                    
                    sum0 *= math.exp(v_prob[ins_id][v_id])
                    sum1[v_id][0] *= math.exp(v_prob[ins_id][v_id])
                    sum1[v_id][1] *= math.exp(v_prob[ins_id][v_id])
                    
                    sum0_tot += sum0
                    sum1_tot[v_id][0] += sum1[v_id][0]
                    sum1_tot[v_id][1] += sum1[v_id][1]

                for v_id_ordi in range(num_verb):
                    if v_id_ordi not in cons_verbs:
                        sum0_tot += math.exp(v_prob[ins_id][v_id_ordi])

                for v_id in cons_verbs:
                    gradientZ[v_id][0] += (sum1_tot[v_id][0] / sum0_tot)
                    gradientZ[v_id][1] += (sum1_tot[v_id][1] / sum0_tot)
            
            # update lambdas
            if batch_num % update_freq != 0:
                continue
            count = 0
            # print ("update lambdas!")
            for cons_id in constraints:
                for direc in range(0,2):
                    first_order_sum[cons_id][direc] = first_order_sum[cons_id][direc] * alpha + (1.0 - alpha) * gradientZ[cons_id][direc]
                    second_order_sum[cons_id][direc] = second_order_sum[cons_id][direc] * beta + (1.0 - beta) * gradientZ[cons_id][direc] * gradientZ[cons_id][direc]
                    first_order_mean = first_order_sum[cons_id][direc] / (1.0 - pow(alpha, batch_num))
                    second_order_mean = second_order_sum[cons_id][direc] / (1.0 - pow(beta, batch_num))
                    lambdas[cons_id][direc] -= first_order_mean / (math.sqrt(second_order_mean) + epsilon) * learning_rate
                    
                    if lambdas[cons_id][direc] < 0:
                        lambdas[cons_id][direc] = 0
                    if lambdas[cons_id][direc] > 0:
                        count += 1
                
            gradientZ = {item:[0.0, 0.0] for item in constraints}
            learning_rate *= lr_decay

            # Inference once, using lambdas.
            if batch_num % inference_freq == 0:
                print ("=== inference begin ===")
                pred_agents = {}
                predictions = []
                target = []
                idx_sorted = []

                for batch_idx_, (index_, input, _target_) in enumerate(loader):
                    if batch_idx_ % 100 == 0:
                        print ("Test batch: {}/{}".format(batch_idx_, mx))
                    vrn_potential_tmp = torch.load(vrn_logProb_dir + "%d"%(batch_idx_+1))
                    v_potential_tmp = torch.load(v_logProb_dir + "%d"%(batch_idx_+1)).to(device)
                    for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                        vrn_potential_tmp[vrn_idx] = _vrn.to(device)

                    n_ins = vrn_potential_tmp[0].size()[0]
                    for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                        vrn_potential_tmp[vrn_idx] = _vrn.transpose(0,2)
                    # vrn_potential: [(50, 1125, n_ins), (100, 518, n_ins), (283, 145, n_ins)]

                    # update vrn_after using lambdas
                    for arg_id in all_man_idx:
                        v_id, n_localid, noun = arg_to_v[arg_id]
                        if v_id not in constraints:
                            continue
                        if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                            continue
                        split_id, n_localid, vr_split_localid = myutils.arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
                        temp = -lambdas[v_id][0] * constraints[v_id][0][0] - lambdas[v_id][1] * constraints[v_id][1][0]
                        vrn_potential_tmp[split_id][n_localid][vr_split_localid] += temp

                    for arg_id in all_woman_idx:
                        v_id, n_localid, noun = arg_to_v[arg_id]
                        if v_id not in constraints:
                            continue
                        if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                            continue
                        split_id, n_localid, vr_split_localid = myutils.arg_id_to_more(arg_id, arg_to_v, constraints, lambdas, encoding, model)
                        temp = -lambdas[v_id][0] * constraints[v_id][0][1] - lambdas[v_id][1] * constraints[v_id][1][1]
                        vrn_potential_tmp[split_id][n_localid][vr_split_localid] += temp

                    for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                        vrn_potential_tmp[vrn_idx] = _vrn.transpose(0,2)
                    
                    _top1, _pred_agents, _idx_sorted, _predictions = myutils.inference_step(encoding, model, wordmap2, vrn_potential_tmp, v_potential_tmp, output_index, cons_verbs)
                    if len(_pred_agents) != 0:
                        for v_id in _pred_agents:
                            if v_id in pred_agents:
                                pred_agents[v_id][0] += _pred_agents[v_id][0]
                                pred_agents[v_id][1] += _pred_agents[v_id][1]
                            else:
                                pred_agents[v_id] = [0, 0]
                                pred_agents[v_id][0] = _pred_agents[v_id][0]
                                pred_agents[v_id][1] = _pred_agents[v_id][1]

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
                print ("{}-batch, acc is:{}, {} constraints are not satisfied".format(batch_num, acc1, count))
                results.append([epoch, count, acc1])

                # Ratio here is got from inference results, not probability. In fact, we want both for analysis.
                after_ratio = myutils.get_pred_gender_ratio(pred_agents, 1)
                ori_ratio = myutils.get_pred_gender_ratio(pred_agents_ori, 1)
                _, after_ratio_PR = myutils.get_gender_ratio_res_PR(cons_verbs, num_gender, words_file, training_file, encoding, model, loader, 
                            all_man_idx, all_woman_idx, output_index, wordmap2, arg_to_v, lambdas, constraints)

                if batch_num % print_freq == 0:
                    for v_id in after_ratio_PR:
                        if not ori_ratio.__contains__(v_id):
                            ori_ratio[v_id] = 0
                        if not after_ratio.__contains__(v_id):
                            after_ratio[v_id] = 0
                        print ("{} {}:\ttrain:{:f}\tbefore_prob:{:f}\tafter_prob:{:f}\tbefore_inf:{:f}\tafter_inf:{:f}\tlambdas:{}".format(v_id, encoding.id_v[v_id], 
                            training_ratio[encoding.id_v[v_id]], ori_ratio_PR[v_id], after_ratio_PR[v_id], ori_ratio[v_id], after_ratio[v_id], lambdas[v_id]))

                golden_verbs_df = pd.DataFrame([verb for verb in cons_verbs], columns = ['verb_id'])
                train_df = pd.DataFrame(training_ratio.items(), columns = ["verb", "training_ratio"])
                ori_df = pd.DataFrame(ori_ratio.items(), columns=['verb_id', 'bef_ratio'])
                after_df = pd.DataFrame(after_ratio.items(), columns=['verb_id', 'after_ratio'])
                ori_df_PR = pd.DataFrame(ori_ratio_PR.items(), columns=['verb_id', 'bef_ratio_PR'])
                after_df_PR = pd.DataFrame(after_ratio_PR.items(), columns=['verb_id', 'after_ratio_PR'])
                tmp = ori_df.merge(after_df).merge(ori_df_PR).merge(after_df_PR)
                train_df["verb_id"] = train_df["verb"].apply(lambda x: encoding.v_id[x])
                com_df = train_df.merge(tmp)
                res0 = com_df.sort_values(by = ['training_ratio'], ascending=1)
                res0['bef_diff'] = res0['training_ratio'] - res0['bef_ratio']   # training_ratio - bef_ratio
                res0['after_diff'] = res0['training_ratio'] - res0['after_ratio']   # training_ratio - after_ratio
                res0['bef_diff_PR'] = res0['training_ratio'] - res0['bef_ratio_PR']   # training_ratio - bef_ratio
                res0['after_diff_PR'] = res0['training_ratio'] - res0['after_ratio_PR']   # training_ratio - after_ratio
                resx = res0.merge(golden_verbs_df)
                resx.sort_values(by = ['training_ratio'], ascending=1, inplace = True)
                del resx['verb_id']
                resx.reset_index(inplace=True, drop = True)

                before_error, after_error, before_error_PR, after_error_PR = mystat.get_violated_verbs(resx, margin)

                # To find in which batch we get the best performance on reducing bias
                if after_error_PR < min_error:
                    min_error = after_error_PR
                    res_bst = resx
                    lambdas_bst = lambdas
                    batch_bst = batch_num
                print ("**Epoch:{}, Batch_num:{}, Best batch:{}, min_error:{}**".format(epoch, batch_num, batch_bst, min_error))
                
                print ("|____inference end____|\n")

    print ("**Stop batch:{}. Best batch:{}, min_error:{}**".format(batch_num, batch_bst, min_error))

    # show results and calculate bias score.
    print ("=== After calibrating ===")
    myplt.plot_tr_ax_gender(res_bst, margin, filename = "Gender ratio predicted")
    mystat.get_violated_verbs(res_bst, margin)
    mystat.get_bias_score(res_bst, margin)


def lagrange_with_margins(margin, constraints, encoder, model, dataset, loader, arg_to_v, all_man_idx, 
                                all_woman_idx, wordmap2, arg_idx_map, vrn_map, cons_verbs):

    eta = 0.1
    max_epoch = 2

    lambdas = {item:[0,0] for item in constraints}
    training_ratio, train_collect_agents = myutils.get_training_gender_ratio(training_file, words_file, num_gender)
    training_ratio_id = {}
    for verb in training_ratio:
        v_id = encoder.v_id[verb]
        training_ratio_id[v_id] = training_ratio[verb]

    mx = len(loader)
    top1 = []
    pred_agents = {}
    print ("Initial inferencing.")
    for batch_idx, (index, input, target) in enumerate(loader):
        if batch_idx % 100 == 0:
            print ("Batch: {}/{}".format(batch_idx, mx))
        vrn_potential_tmp = torch.load(vrn_potential_dir + "%d"%(batch_idx+1))
        v_potential_tmp = torch.load(v_potential_dir + "%d"%(batch_idx+1)).to(device)
        for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
            vrn_potential_tmp[vrn_idx] = _vrn.to(device)

        _top1, _pred_agents, _, _ = myutils.inference_step(encoder, model, wordmap2, vrn_potential_tmp, v_potential_tmp, vrn_map, cons_verbs, isPR = 0)
        top1.append(_top1)
        for v_id in _pred_agents:
            if v_id in pred_agents:
                pred_agents[v_id][0] += _pred_agents[v_id][0]
                pred_agents[v_id][1] += _pred_agents[v_id][1]
            else:
                pred_agents[v_id] = _pred_agents[v_id]

    print ("Starting Lagrangian part")
    for epoch in range(max_epoch):
        print ("Epoch: {}/{}".format(epoch, max_epoch))
        count = 0
        error = {item:[0,0] for item in constraints}
        results = []
        t0 = time.time()

        # update lambdas and error
        for k in constraints:
            if k in pred_agents:
                lambdas[k][0] += eta * constraints[k][0][0] * pred_agents[k][0]
                lambdas[k][0] += eta * constraints[k][0][1] * pred_agents[k][1]
                error[k][0] += constraints[k][0][0] * pred_agents[k][0]
                error[k][0] += constraints[k][0][1] *  pred_agents[k][1]
                lambdas[k][1] += eta * constraints[k][1][0] * pred_agents[k][0]
                lambdas[k][1] += eta * constraints[k][1][1] * pred_agents[k][1]
                error[k][1] +=  constraints[k][1][0] * pred_agents[k][0]
                error[k][1] +=  constraints[k][1][1] *  pred_agents[k][1]
        for k in lambdas:
            for i in range(2):
                if lambdas[k][i] <= 0:
                    lambdas[k][i] = 0
        for k in error:
            for i in range(2):
                if error[k][i] > 0:
                    count += 1

        # update potential scores
        mx = len(loader)
        top1_before = copy.deepcopy(top1)
        top1 = []
        pred_agents_before = copy.deepcopy(pred_agents)
        pred_agents = {}
        idx_sorted = 0
        predictions = 0
        target = 0
        print ("Start new inference.")
        for batch_idx, (index, input, _target) in enumerate(loader):
            if batch_idx % 100 == 0:
                print ("Batch: {}/{}".format(batch_idx, mx))
            vrn_potential_tmp = torch.load(vrn_potential_dir + "%d"%(batch_idx+1))
            v_potential_tmp = torch.load(v_potential_dir + "%d"%(batch_idx+1)).to(device)
            for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                vrn_potential_tmp[vrn_idx] = _vrn.to(device)

            n_ins = vrn_potential_tmp[0].size()[0]
            for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                vrn_potential_tmp[vrn_idx] = _vrn.transpose(0,2)
            # vrn_potential: [(50, 1125, n_ins), (100, 518, n_ins), (283, 145, n_ins)]

            for arg_id in all_man_idx:
                v_id, n_localid, noun = arg_to_v[arg_id]
                if v_id not in constraints:
                    continue
                if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                    continue
                r_id = encoder.r_id["agent"]
                vr_id = encoder.vr_id[(v_id, r_id)]
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
                vrn_potential_tmp[split_id][n_localid][vr_split_localid] -= lambdas[v_id][0] * constraints[v_id][0][0] 
                vrn_potential_tmp[split_id][n_localid][vr_split_localid] -= lambdas[v_id][1] * constraints[v_id][1][0]

            for arg_id in all_woman_idx:
                v_id, n_localid, noun = arg_to_v[arg_id]
                if v_id not in constraints:
                    continue
                if lambdas[v_id][0] == 0 and lambdas[v_id][1] == 0:
                    continue
                r_id = encoder.r_id["agent"]
                vr_id = encoder.vr_id[(v_id, r_id)]
                vr_localid = model.vr_id_to_local[vr_id]
                if vr_localid > splits_offset[2]:
                    split_id = 2
                    vr_split_localid = vr_localid - splits_offset[2] - 1
                elif vr_localid > splits_offset[1]:
                    split_id = 1
                    vr_split_localid = vr_localid - splits_offset[1] - 1
                else:
                    split_id = 0
                    vr_split_localid = vr_localid - 1
                vrn_potential_tmp[split_id][n_localid][vr_split_localid] -= lambdas[v_id][0] * constraints[v_id][0][1] 
                vrn_potential_tmp[split_id][n_localid][vr_split_localid] -= lambdas[v_id][1] * constraints[v_id][1][1]

            for vrn_idx, _vrn in enumerate(vrn_potential_tmp):
                vrn_potential_tmp[vrn_idx] = _vrn.transpose(0,2)
                        
            _top1, _pred_agents, _idx_sorted, _predictions = myutils.inference_step(encoder, model, wordmap2, vrn_potential_tmp, v_potential_tmp, vrn_map, cons_verbs, isPR = 0)
            top1.append(_top1)
            for v_id in _pred_agents:
                if v_id in pred_agents:
                    pred_agents[v_id][0] += _pred_agents[v_id][0]
                    pred_agents[v_id][1] += _pred_agents[v_id][1]
                else:
                    pred_agents[v_id] = _pred_agents[v_id]
            if batch_idx == 0:
                idx_sorted = _idx_sorted
                predictions = _predictions
                target = _target
            else:
                idx_sorted = torch.cat((idx_sorted, _idx_sorted))
                predictions = torch.cat((predictions, _predictions))
                target = torch.cat((target, _target))

        for k in pred_agents:
            if k in pred_agents_before:
                if pred_agents[k] != pred_agents_before[k] and k == 243:
                    print ("v_id:{}, before:{}, after:{}, tr:{}, constraint:{}".format(k, pred_agents_before[k], pred_agents[k], training_ratio_id[k], constraints[k]))

        if epoch % 10 == 0 or epoch == max_epoch-1:
            top1_eval = imSituTensorEvaluation(1, 3, encoder)
            top1_eval.add_point(target, predictions.data, idx_sorted.data)
            acc1 = top1_eval.get_average_results()["value"]
            print ("%s-epoch, acc is: "%(epoch), acc1)
            results.append([epoch, count, acc1])
        
        print ("Time for one epoch:", time.time()-t0)
        print ("%s-th epoch, number of times that constrints are not satisfied:"%(epoch), count)

        if count == 0:
            break

    myutils.save_iterations(save_iteration + "_margin_" + str(margin), results)
    myutils.save_lambdas(save_lambda + "_margin_" + str(margin), lambdas)
    return lambdas

def run(margin, is_dev, isPR):
    reargs = mypreprocess.preprocess(margin, is_dev, isPR)
    print ("Processing imSitu dataset.")
    (eval_file, encoder, model, dataset, loader, arg_idx_map, word_map, wordmap2, output_index, 
                arg_to_v, all_man_idx, all_woman_idx, constraints, cons_verbs, agent_verbs) = reargs

    # lambdas = lagrange_with_margins(margin, constraints, encoder, model, dataset, loader, arg_to_v, all_man_idx, 
    #                                             all_woman_idx, wordmap2, arg_idx_map, output_index, cons_verbs)
    posterior_regularization(constraints, dataset, loader, encoder, cons_verbs, agent_verbs, model, word_map, wordmap2, output_index, 
                            all_man_idx, all_woman_idx, arg_to_v, is_dev)

if __name__=='__main__':
    run(margin, is_dev = 1, isPR = 1)
