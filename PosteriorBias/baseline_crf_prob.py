import os
import time
import argparse
from torch import optim
import random as rand
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tvt
import numpy as np
import math
from imsitu import imSituVerbRoleLocalNounEncoder
from imsitu import imSituTensorEvaluation
from imsitu import imSituSituation
from imsitu import imSituSimpleImageFolder
import json
import sys
import re
from constant import *

def initLinear(linear, val = None):
    if val is None:
        fan = linear.in_features +  linear.out_features 
        spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
    else:
        spread = val
        linear.weight.data.uniform_(-spread,spread)
        linear.bias.data.uniform_(-spread,spread)

class vgg_modified(nn.Module):
    def __init__(self):
        super(vgg_modified, self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        # self.classifier = nn.Sequential(
        # nn.Dropout(),
        self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        initLinear(self.lin1)
        initLinear(self.lin2)

    def rep_size(self): return 1024

    def forward(self, x):
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))


class resnet_modified_large(nn.Module):
    def __init__(self):
        super(resnet_modified_large, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class resnet_modified_medium(nn.Module):
    def __init__(self):
        super(resnet_modified_medium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))


def get_word_gender_map(wordsfile = words_file):
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
    return wordmap2


class baseline_crf(nn.Module):
    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    # these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
    def __init__(self, encoding, splits=[50, 100, 283], prediction_type="max_max", ngpus=1, cnn_type="resnet_101"):
        super(baseline_crf, self).__init__()

        self.normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.broadcast = []
        self.nsplits = len(splits)
        self.splits = splits
        self.encoding = encoding
        self.prediction_type = prediction_type
        self.n_verbs = encoding.n_verbs()
        self.split_vr = {}
        self.v_roles = {}
        self.vr_id_to_local = {}
        self.vr_local_to_id = {}
        self.vr_v = {}
        # cnn
        print(cnn_type)
        if cnn_type == "resnet_101":
            self.cnn = resnet_modified_large()
        elif cnn_type == "resnet_50":
            self.cnn = resnet_modified_medium()
        elif cnn_type == "resnet_34":
            self.cnn = resnet_modified_small()
        else:
            print("unknown base network")
            exit()
        self.rep_size = self.cnn.rep_size()
        for s in range(0, len(splits)):
            self.split_vr[s] = []

        # sort by length
        remapping = []
        for (vr, ns) in encoding.vr_id_n.items():
            remapping.append((vr, len(ns)))

        # find the right split
        for (vr, l) in remapping:
            i = 0
            for s in splits:
                if l <= s:
                    break
                i += 1
            _id = (i, vr)
            self.split_vr[i].append(_id)
        total = 0
        for (k, v) in self.split_vr.items():
            # print "{} {} {}".format(k, len(v), splits[k]*len(v))
            total += splits[k]*len(v)
        # print "total compute : {}".format(total)
        # print ('self.split_vr:', self.split_vr)

        # keep the splits sorted by vr id, to keep the model const w.r.t the encoding
        for i in range(0, len(splits)):
            s = sorted(self.split_vr[i], key=lambda x: x[1])
            self.split_vr[i] = []
            for (x, vr) in s:
                _id = (x, len(self.split_vr[i]), vr)
                self.split_vr[i].append(_id)
                (v, r) = encoding.id_vr[vr]
                if v not in self.v_roles:
                    self.v_roles[v] = []
                self.v_roles[v].append(_id)

        # create the mapping for grouping the roles back to the verbs later
        max_roles = encoding.max_roles()

        # need a list that is nverbs by 6
        self.v_vr = [0 for i in range(0, self.encoding.n_verbs()*max_roles)]
        splits_offset = []
        for i in range(0, len(splits)):
            if i == 0:
                splits_offset.append(0)
            else:
                splits_offset.append(splits_offset[-1] + len(self.split_vr[i-1]))

        # and we need to compute the position of the corresponding roles, and pad with the 0 symbol
        for i in range(0, self.encoding.n_verbs()):
            offset = max_roles * i
            # stored in role order
            roles = sorted(self.v_roles[i], key=lambda x: x[2])
            self.v_roles[i] = roles
            k = 0
            for (s, pos, r) in roles:
                # add one to account of the 0th element being the padding
                self.v_vr[offset + k] = splits_offset[s] + pos + 1
                k += 1
            # pad
            while k < max_roles:
                self.v_vr[offset + k] = 0
                k += 1
        
        v_id = 0
        for i in range(len(self.v_vr)):
            if (i%6 == 0 and i != 0):
                v_id += 1
            if self.v_vr[i] != 0:
                self.vr_v.update({self.v_vr[i]: (v_id, i%6)})

        for v_id in self.v_roles:
            for (s, pos, vr_id) in self.v_roles[v_id]:
                vr_localid = splits_offset[s] + pos + 1
                self.vr_id_to_local[vr_id] = vr_localid
                self.vr_local_to_id[vr_localid] = vr_id

        # verb potential
        self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
        # verb-role-noun potentials
        self.linear_vrn = nn.ModuleList([nn.Linear(
            self.rep_size, splits[i]*len(self.split_vr[i])) for i in range(0, len(splits))])
        self.total_vrn = 0
        for i in range(0, len(splits)):
            self.total_vrn += splits[i]*len(self.split_vr[i])
        print ("--------")
        print("total encoding vrn : {0}, with padding in {1} groups : {2}".format(
            encoding.n_verbrolenoun(), self.total_vrn, len(splits)))

        # initilize everything
        initLinear(self.linear_v)
        for _l in self.linear_vrn:
            initLinear(_l)
        self.mask_args()

    def mask_args(self):
        # go through the and set the weights to negative infinity for out of domain items
        neg_inf = float("-infinity")
        for v in range(0, self.encoding.n_verbs()):
            for (s, pos, r) in self.v_roles[v]:
                linear = self.linear_vrn[s]
                # get the offset
                # print self.splits
                start = self.splits[s]*pos+len(self.encoding.vr_n_id[r])
                end = self.splits[s]*(pos+1)
                for k in range(start, end):
                    linear.bias.data[k] = neg_inf  # neg_inf

    def log_sum_exp(self, vec):
        max_score, max_i = torch.max(vec, 1)
        max_score_broadcast = max_score.view(-1, 1).expand(vec.size())
        return (max_i, max_score,  max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1)))

    def forward_max(self, images):
        (_, _, _, _, scores, values) = self.forward(images)
        return (scores, values)     # (scores, predictions)
    
    def forward_potential(self, images):
        (_, v, vrn, _, _, _) = self.forward(images)
        return (v, vrn)

    def forward_features(self, images):
        return self.cnn(images)
    
    def forward(self, image):
        self.mask_args()
        n_ins = image.size()[0]

        rep = self.cnn(image)
        log_softmax = torch.nn.LogSoftmax(dim = 1)

        v_potential = log_softmax((self.linear_v(rep)))

        vrn_potential = []
        vrn_marginal = []
        vr_max = []
        vr_maxi = []
        # first compute the norm
        # step 1 compute the verb-role marginals
        # this loop allows a memory/parrelism tradeoff.
        # To use less memory but achieve less parrelism, increase the number of groups
        for i, vrn_group in enumerate(self.linear_vrn):
            # linear for the group
            _vrn = log_softmax(vrn_group(rep).view(-1, self.splits[i]))

            _vr_maxi, _vr_max, _vrn_marginal = self.log_sum_exp(_vrn)
            _vr_maxi = _vr_maxi.view(-1, len(self.split_vr[i]))
            _vr_max = _vr_max.view(-1, len(self.split_vr[i]))
            _vrn_marginal = _vrn_marginal.view(-1, len(self.split_vr[i]))

            vr_maxi.append(_vr_maxi)
            vr_max.append(_vr_max)
            vrn_potential.append(_vrn.view(n_ins, -1, self.splits[i]))
            vrn_marginal.append(_vrn_marginal)

        # concat role groups with the padding symbol
        zeros = Variable(torch.zeros(n_ins, 1).to(device))  # this is the padding
        zerosi = Variable(torch.LongTensor(n_ins, 1).zero_().to(device))
        vrn_marginal.insert(0, zeros)
        vr_max.insert(0, zeros)
        vr_maxi.insert(0, zerosi)

        vrn_marginal = torch.cat(vrn_marginal, 1)
        vr_max = torch.cat(vr_max, 1)
        vr_maxi = torch.cat(vr_maxi, 1)

        # print vrn_marginal
        # step 2 compute verb marginals
        # we need to reorganize the role potentials so it is BxVxR
        # gather the marginals in the right way
        # v_vr = self.broadcast[torch.cuda.current_device()]
        v_vr = torch.tensor(self.v_vr).to(device)   # v_vr: [vr_id, ..., vr_id], max_vr in a role for one v. 
        # E.g., when we have 2 verbs, max_rv == 5, the length will be 10, values can be [1,2,3,0,0,4,5,6,7,8], every value stands for a vr_id
        vrn_marginal_grouped = vrn_marginal.index_select(1, v_vr).view(n_ins, self.n_verbs, self.encoding.max_roles())
        vr_max_grouped = vr_max.index_select(1, v_vr).view(n_ins, self.n_verbs, self.encoding.max_roles())
        vr_maxi_grouped = vr_maxi.index_select(1, v_vr).view(n_ins, self.n_verbs, self.encoding.max_roles())

        # product ( sum since we are in log space )
        v_marginal = torch.sum(vrn_marginal_grouped, 2).view(n_ins, self.n_verbs) + v_potential

        # step 3 compute the final sum over verbs
        _, _, norm = self.log_sum_exp(v_marginal)
        # compute the maxes

        # max_max probs
        v_max = torch.sum(vr_max_grouped, 2).view(n_ins, self.n_verbs) + v_potential  # these are the scores
        # we don't actually care, we want a max prediction per verb
        # max_max_v_score, max_max_vi = torch.max(v_max, 1)
        # max_max_prob = torch.exp(max_max_v_score - norm)
        # max_max_vrn_i = vr_maxi_grouped.gather(1,max_max_vi.view(n_ins,1,1).expand(n_ins,1,self.encoding.max_roles()))

        # offset so we can use index select... is there a better way to do this?
        # max_marginal probs
        max_marginal_verb_score, max_marg_vi = torch.max(v_marginal, 1)
        max_marginal_prob = torch.exp(max_marginal_verb_score - norm)
        max_marg_vrn_i = vr_maxi_grouped.gather(1,max_marg_vi.view(n_ins,1,1).expand(n_ins,1,self.encoding.max_roles()))

        # this potentially does not work with parrelism, in which case we should figure something out
        if self.prediction_type == "max_max":
            rv = (rep, v_potential, vrn_potential, norm, v_max, vr_maxi_grouped)
        elif self.prediction_type == "max_marginal":
            rv = (rep, v_potential, vrn_potential, norm, v_marginal, vr_maxi_grouped)
        else:
            print("unkown inference type")
            rv = ()
        return rv

    # computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
    # returns the the log of V

    def logsumexp_nx_ny_xy(self, x, y):
        # _,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
        if x > y:
            return torch.log(torch.exp(y-x) + 1 - torch.exp(y) + 1e-8) + x
        else:
            return torch.log(torch.exp(x-y) + 1 - torch.exp(x) + 1e-8) + y

    def sum_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
        # compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        n_ins = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, n_ins):
            _norm = norm[i]
            _v = v_potential[i]
            _vrn = []
            _ref = situations[i]
            for pot in vrn_potential:
                _vrn.append(pot[i])
            for r in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, (s, idx, rid)) in enumerate(self.v_roles[v]):
                    pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
                if pots.data[0] > _norm.data[0]:
                    print("inference error")
                    print(pots)
                    print(_norm)
                if i == 0 and r == 0:
                    loss = pots-_norm
                else:
                    loss = loss + pots - _norm
        return -loss/(n_ins*n_refs)

    def mil_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
        # compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        n_ins = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, n_ins):
            _norm = norm[i]
            _v = v_potential[i]
            _vrn = []
            _ref = situations[i]
            for pot in vrn_potential:
                _vrn.append(pot[i])
            for r in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, (s, idx, rid)) in enumerate(self.v_roles[v.item()]):
                    # print (_vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]])
                    pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
                if pots.item() > _norm.item():
                    print("inference error")
                    print(pots)
                    print(_norm)
                if r == 0:
                    _tot = pots-_norm
                else:
                    _tot = self.logsumexp_nx_ny_xy(_tot, pots-_norm)
            if i == 0:
                loss = _tot
            else:
                loss = loss + _tot
        return -loss/n_ins


def format_dict(d, s, p):
    rv = ""
    for (k, v) in d.items():
        if len(rv) > 0:
            rv += " , "
        rv += p+str(k) + ":" + s.format(v*100)
    return rv


def predict_human_readable(dataset_loader, simple_dataset, model, outdir, top_k):
    model.eval()
    print("predicting...")
    mx = len(dataset_loader)
    for i, (input, index) in enumerate(dataset_loader):
        print("{}/{} batches".format(i+1, mx))
        with torch.no_grad():
            input = input.to(device)
            (scores, predictions) = model.forward_max(input)
        # (s_sorted, idx) = torch.sort(scores, 1, True)
        human = encoder.to_situation(predictions)
        (b, p, d) = predictions.size()
        for _b in range(0, b):
            items = []
            offset = _b * p
            for _p in range(0, p):
                items.append(human[offset + _p])
                items[-1]["score"] = scores.data[_b][_p].item()
            items = sorted(items, key=lambda x: -x["score"])[:top_k]
            name = simple_dataset.images[index[_b][0]].split(".")[:-1]
            name.append("predictions")
            outfile = outdir + ".".join(name)
            json.dump(items, open(outfile, "w"))

# save_potential in v_potential and vrn_potential
def save_potential(dataset_loader, encoding, model):
    model.eval()
    print("saving potential...")
    mx = len(dataset_loader)
    
    for idx, (index, input, target) in enumerate(dataset_loader):
        vrn_potential = [0 for item in range(3)]
        print("{}/{} batches\r".format(idx+1, mx))
        input_var, target_var = input.to(device), target.to(device)
        (v, vrn) = model.forward_potential(input_var)
        for _vrn_id, _vrn in enumerate(vrn):
            vrn_potential[_vrn_id] = _vrn.cpu()
        v_potential = v.cpu()
        torch.save(v_potential, v_logProb_dir + '%d'%(idx+1))
        torch.save(vrn_potential, vrn_logProb_dir + '%d'%(idx+1))
    print ("Finish potential saving.")


def generate_potential_table(encoding, model):
    # generate dict vr_v: {vr_localid: (v_id, r_localid)}; len(vr_v) == 1788
    print ("Generate potential table...")
    v_id = 0

    arg_id = 0
    f = open(vrn_potential_table_file, 'w')
    for split_id in range(0, len(model.splits)):    # range(0, 3)
        for i in range(len(model.split_vr[split_id])):  # range(0, 1125/518/145)
            vr_localid = i + splits_offset[split_id] + 1
            v_id = model.vr_v[vr_localid][0]
            r_localid = model.vr_v[vr_localid][1]
            vr_id = model.vr_local_to_id[vr_localid]
            _, r_id = encoding.id_vr[vr_id]
            if r_localid >= encoding.verb_nroles(v_id):
                break
            for n_localid in range(len(encoding.vr_id_n[vr_id])):    # range(0, <50/100/283)
                n_id = encoding.vr_id_n[vr_id][n_localid]
                vrn_list = [str(arg_id), encoding.id_v[v_id], encoding.id_r[r_id], encoding.id_n[n_id], str(v_id), str(r_localid), str(n_localid)]
                if vrn_list[3] == "":
                    vrn_list[3] = "null"
                vrn_str = "\t".join(vrn_list)
                f.write(str(vrn_str)+"\t \n")
                arg_id += 1
    f.close()
    # print (encoding.n_verbs(), encoding.n_nouns(), encoding.n_roles())

    print("Finish generating potential table.")

# generate_probability, saved in vrn_grouped
def generate_probability(dataset_loader, model, encoding, cons_verbs):
    print ("Generating probability...")

    t0 = time.time()
    mx = len(dataset_loader)

    for batch_idx, (index, input, target) in enumerate(dataset_loader):
        if batch_idx % 10 == 0:
            print ("Batch: {}/{}".format(batch_idx, mx))
            if batch_idx != 0:
                print ("Time for 10 batches:", time.time()-t0)
                t0 = time.time()
        vrn_prob = torch.load(vrn_logProb_dir + "%d"%(batch_idx+1))
        v_prob = torch.exp(torch.load(v_logProb_dir + "%d"%(batch_idx+1)).to(device))
        for vrn_idx, _vrn in enumerate(vrn_prob):
            # ATTENTION: Here is an exp for _vrn, thus vrn_grouped is something after exp!!!
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
        # p_vr_cons: p_vr_cons[ins_id][v_id]
        # p_vr_ordi: p_vr_ordi[ins_id]

        cons_verbs_tensor = torch.tensor(cons_verbs)
        vrn_grouped_ = vrn_grouped.transpose(0,1).transpose(1,2).transpose(2,3)
        # vrn_grouped_.size() = (n_verb = 504, max_roles = 6, n_roles = 283, n_ins)

        torch.save(vrn_grouped, vrn_grouped_dir + '%d'%(batch_idx+1))


def eval_model(dataset_loader, encoding, model):
    model.eval()
    print("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)

    mx = len(dataset_loader)
    for i, (index, input, target) in enumerate(dataset_loader):
        if i % 50 == 0:
            print("{}/{} batches\r".format(i+1, mx))
        with torch.no_grad():
            input = input.to(device)
            (scores, predictions) = model.forward_max(input)
        (s_sorted, idx) = torch.sort(scores, 1, True)

        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)

    print("\ndone.")
    return (top1, top5)


def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, output_dir, timing=False):
    model.train()
    print ("Training model...")

    time_all = time.time()

    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
    loss_total = 0
    total_steps = 0
    avg_scores = []

    for k in range(0, max_epoch):
        for i, (index, input, target) in enumerate(train_loader):
            total_steps += 1

            t0 = time.time()
            t1 = time.time()

            input = input.to(device)
            # target_var = torch.autograd.Variable(target.to(device))
            (_, v, vrn, norm, scores, predictions) = model(input)

            (s_sorted, idx) = torch.sort(scores, 1, True)
            # print norm
            if timing:
                print("forward time = {}".format(time.time() - t1))
            optimizer.zero_grad()
            t1 = time.time()
            loss = model.mil_loss(v, vrn, norm, target, 3)

            if timing:
                print("loss time = {}".format(time.time() - t1))
            t1 = time.time()
            loss.backward()
            # print loss
            if timing:
                print("backward time = {}".format(time.time() - t1))
            optimizer.step()
            loss_total += loss.item()
            # score situation
            t2 = time.time()
            top1.add_point(target, predictions.data, idx.data)
            top5.add_point(target, predictions.data, idx.data)

            if timing:
                print("eval time = {}".format(time.time() - t2))
            if timing:
                print("batch time = {}".format(time.time() - t0))
            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                verb_1 = top1_a["verb"]
                value_1 = top1_a["value"]
                print ("Epoch {}/{}, batch {}/{} | verb_1: {:.2f}, value_1: {:.2f}, loss={:.5f}, avg loss={:.5f}, batch time = {:.2f}".format(k+1, max_epoch, i+1, len(train_loader), 
                            verb_1, value_1, loss.item(), loss_total / ((total_steps) % (eval_frequency*len(train_loader) + 1)), (time.time() - time_all)))
                time_all = time.time()
            
        if (k+1) % eval_frequency == 0:
            print("eval...")
            etime = time.time()
            (top1, top5) = eval_model(dev_loader, encoding, model)
            model.train()
            print("... done after {:.2f} s".format(time.time() - etime))
            top1_a = top1.get_average_results()
            top5_a = top5.get_average_results()

            avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + \
                top5_a["value"] + top5_a["value-all"] + \
                top5_a["value*"] + top5_a["value-all*"]
            avg_score /= 8

            print("Epoch {} average :{:.2f} {} {}".format(k+1, avg_score*100,
                                                        format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))

            avg_scores.append(avg_score)
            maxv = max(avg_scores)

            if maxv == avg_scores[-1]:
                torch.save(model.state_dict(), output_dir + "/{0}.model".format(maxv))
                print("new best model saved! {0}".format(maxv))

            top1 = imSituTensorEvaluation(1, 3, encoding)
            top5 = imSituTensorEvaluation(5, 3, encoding)
            loss_total = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="imsitu Situation CRF. Training, evaluation, prediction and features.")
    parser.add_argument("--command", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--encoding", choices=["true", "false"])
    parser.add_argument("--weights", choices=["true", "false"])
    args = parser.parse_args()

    if args.command == "train":
        print("command = training")
        train_set = json.load(open(training_file))
        dev_set = json.load(open(dev_file))

        if args.encoding is None or args.encoding == "false":
            encoder = imSituVerbRoleLocalNounEncoder(train_set)
            torch.save(encoder, output_dir)
        else:
            encoder = torch.load(encoding_file)

        model = baseline_crf(encoder, cnn_type=cnn_type)
        if args.weights == "true":
            model.load_state_dict(torch.load(weights_file))

        dataset_train = imSituSituation(image_dir, train_set, encoder, model.train_preprocess())
        dataset_dev = imSituSituation(image_dir, dev_set, encoder, model.dev_preprocess())
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dev_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_model(training_epochs, eval_frequency, train_loader, 
                    dev_loader, model, encoder, optimizer, output_dir)


    elif args.command == "eval":
        print("command = evaluating")
        eval_file = json.load(open(eval_files))

        encoder = torch.load(encoding_file)
        print("creating model...")
        model = baseline_crf(encoder, cnn_type=cnn_type)

        print("loading model weights...")
        model.load_state_dict(torch.load(weights_file))
        model.to(device)

        dataset = imSituSituation(image_dir, eval_file, encoder, model.dev_preprocess())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        (top1, top5) = eval_model(loader, encoder, model)
        top1_a = top1.get_average_results()
        top5_a = top5.get_average_results()

        avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + \
            top5_a["value"] + top5_a["value-all"] + \
            top5_a["value*"] + top5_a["value-all*"]
        avg_score /= 8

        print("Average :{:.2f} {} {}".format(
            avg_score*100, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))


    elif args.command == "predict":
        print("command = predict")
        encoder = torch.load(encoding_file)

        print("creating model...")
        model = baseline_crf(encoder, cnn_type=cnn_type)

        print("loading model weights...")
        model.load_state_dict(torch.load(weights_file))
        model.to(device)

        folder_dataset = imSituSimpleImageFolder(image_dir, model.dev_preprocess())
        image_loader = torch.utils.data.DataLoader(folder_dataset, batch_size=batch_size, shuffle=False)

        predict_human_readable(image_loader, folder_dataset, model, output_dir, top_k)


