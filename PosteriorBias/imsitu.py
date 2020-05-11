import torch as torch
import torch.utils.data as data
import json
from PIL import Image
import os
import os.path
import sys
from constant import *

class imSituTensorEvaluation():
    def __init__(self, topk, nref, image_group={}):
        self.score_cards = {}
        self.topk = topk
        self.nref = nref
        self.image_group = image_group

    def clear(self):
        self.score_cards = {}

    def add_point(self, encoded_reference, encoded_predictions, sorted_idx, image_names=None):
        # encoded predictions should be batch x verbs x values, assumes they are the same order as the references
        # encoded reference should be batch x 1+ references*roles,values (sorted)
        (b, tv, l) = encoded_predictions.size()
        for i in range(0, b):
            _pred = encoded_predictions[i]
            _ref = encoded_reference[i].cpu()
            _sorted_idx = sorted_idx[i]
            if image_names is not None:
                _image = image_names[i]

            lr = _ref.size()[0]
            max_r = int((lr - 1)/2/self.nref)

            gt_v = _ref[0]
            if image_names is not None and _image in self.image_group:
                sc_key = (gt_v, self.image_group[_image])
            else:
                sc_key = (gt_v, "")

            if sc_key not in self.score_cards:
                new_card = {"verb": 0.0, "n_image": 0.0, "value": 0.0,
                            "value*": 0.0, "n_value": 0.0, "value-all": 0.0, "value-all*": 0.0}
                self.score_cards[sc_key] = new_card

            v_roles = []
            for k in range(0, int(max_r)):
                _id = _ref[2*k + 1]
                if _id == -1:
                    break
                v_roles.append(_id)

            _score_card = self.score_cards[sc_key]
            _score_card["n_image"] += 1
            _score_card["n_value"] += len(v_roles)

            k = 0
            p_frame = None
            verb_found = (torch.sum(_sorted_idx.cpu()
                                    [0:self.topk] == gt_v) == 1)
            if verb_found:
                _score_card["verb"] += 1
            p_frame = _pred[gt_v]
            all_found = True
            # print p_frame
            # print _ref
            # print len(v_roles)
            for k in range(0, len(v_roles)):
                nv = p_frame[k].cpu()
                found = False
                for r in range(0, self.nref):
                    #          print nv
                    #          print _ref[1 + 2*max_r*r + 2*k+1]
                    if (nv == _ref[1 + 2*max_r*r + 2*k+1]):
                        found = True
                        break
                if not found:
                    all_found = False
                if found and verb_found:
                    _score_card["value"] += 1
                if found:
                    _score_card["value*"] += 1
                # print found
            # print all_found
            if all_found and verb_found:
                _score_card["value-all"] += 1
            if all_found:
                _score_card["value-all*"] += 1

    def combine(self, rv, card):
        for (k, v) in card.items():
            rv[k] += v

    def get_average_results(self, groups=[]):
        # average across score cards.
        rv = {"verb": 0, "value": 0, "value*": 0,
                "value-all": 0, "value-all*": 0}
        agg_cards = {}
        for (key, card) in self.score_cards.items():
            (v, g) = key
            if len(groups) == 0 or g in groups:
                if v not in agg_cards:
                    new_card = {"verb": 0.0, "n_image": 0.0, "value": 0.0,
                                "value*": 0.0, "n_value": 0.0, "value-all": 0.0, "value-all*": 0.0}
                    agg_cards[v] = new_card
                self.combine(agg_cards[v], card)
        nverbs = len(agg_cards)
        for (v, card) in agg_cards.items():
            img = card["n_image"]
            nvalue = card["n_value"]
            rv["verb"] += card["verb"]/img
            rv["value-all"] += card["value-all"]/img
            rv["value-all*"] += card["value-all*"]/img
            rv["value"] += card["value"]/nvalue
            rv["value*"] += card["value*"]/nvalue

        rv["verb"] /= nverbs
        rv["value-all"] /= nverbs
        rv["value-all*"] /= nverbs
        rv["value"] /= nverbs
        rv["value*"] /= nverbs

        return rv


class imSituVerbRoleNounEncoder:

    def n_verbs(self): return len(self.v_id)
    def n_nouns(self): return len(self.n_id)
    def n_roles(self): return len(self.r_id)
    def verbposition_role(self, v, i): return self.v_r[v][i]
    def verb_nroles(self, v): return len(self.v_r[v])
    def max_roles(self): return self.mr
    def pad_symbol(self): return -1
    def unk_symbol(self): return -2

    def __init__(self, dataset):
        self.v_id = {}
        self.id_v = {}

        self.r_id = {}
        self.id_r = {}

        self.id_n = {}
        self.n_id = {}

        self.mr = 0  # max of role number of a verb

        self.v_r = {}

        for (image, annotation) in dataset.items():
            v = annotation["verb"]
            if v not in self.v_id:
                _id = len(self.v_id)
                self.v_id[v] = _id
                self.id_v[_id] = v
                self.v_r[_id] = []
            vid = self.v_id[v]
            for frame in annotation["frames"]:
                for (r, n) in frame.items():
                    if r not in self.r_id:
                        _id = len(self.r_id)
                        self.r_id[r] = _id
                        self.id_r[_id] = r

                    if n not in self.n_id:
                        _id = len(self.n_id)
                        self.n_id[n] = _id
                        self.id_n[_id] = n

                    rid = self.r_id[r]
                    if rid not in self.v_r[vid]:
                        self.v_r[vid].append(rid)

        for (v, rs) in self.v_r.items():
            if len(rs) > self.mr:
                self.mr = len(rs)

        for (v, vid) in self.v_id.items():
            self.v_r[vid] = sorted(self.v_r[vid])

    def encode(self, situation):
        rv = {}
        verb = self.v_id[situation["verb"]]
        rv["verb"] = verb
        rv["frames"] = []
        for frame in situation["frames"]:
            _e = []
            for (r, n) in frame.items():
                if r in self.r_id:
                    _rid = self.r_id[r]
                else:
                    _rid = self.unk_symbol()
                if n in self.n_id:
                    _nid = self.n_id[n]
                else:
                    _nid = self.unk_symbol()
                _e.append((_rid, _nid))
            rv["frames"].append(_e)
        return rv

    def decode(self, situation):
        verb = self.id_v[situation["verb"]]
        rv = {"verb": verb, "frames": []}
        for frame in situation["frames"]:
            _fr = {}
            for (r, n) in frame.items():
                _fr[self.id_r[r]] = self.id_n[n]
            rv["frames"].append(_fr)
        return rv

    # takes a list of situations
    def to_tensor(self, situations, use_role=True, use_verb=True):
        rv = []
        for situation in situations:
            _rv = self.encode(situation)
            verb = _rv["verb"]
            items = []
            if use_verb:
                items.append(verb)
            for frame in _rv["frames"]:
                # sort roles
                _f = sorted(frame, key=lambda x: x[0])
                k = 0
                for (r, n) in _f:
                    if use_role:
                        items.append(r)
                    items.append(n)
                    k += 1
                while k < self.mr:
                    if use_role:
                        items.append(self.pad_symbol())
                    items.append(self.pad_symbol())
                    k += 1
            rv.append(torch.LongTensor(items))
        return torch.cat(rv)

    # the tensor is BATCH x VERB X FRAME
    def to_situation(self, tensor):
        (batch, verbd, _) = tensor.size()
        rv = []
        for b in range(0, batch):
            _tensor = tensor[b]
            #_rv = []
            for verb in range(0, verbd):
                args = []
                __tensor = _tensor[verb]
                for j in range(0, self.verb_nroles(verb)):
                    n = __tensor.data[j]
                    args.append((self.verbposition_role(verb, j), n))
                situation = {"verb": verb, "frames": [args]}
                rv.append(self.decode(situation))
            # rv.append(_rv)
        return rv


class imSituVerbRoleLocalNounEncoder(imSituVerbRoleNounEncoder):

    def n_verbrole(self): return len(self.vr_id)
    def n_verbrolenoun(self): return self.total_vrn
    def verbposition_role(self, v, i): return self.v_vr[v][i]
    def verb_nroles(self, v): return len(self.v_vr[v])

    def __init__(self, dataset):
        imSituVerbRoleNounEncoder.__init__(self, dataset)
        self.vr_id = {}
        self.id_vr = {}

        self.vr_n_id = {}
        self.vr_id_n = {}

        self.vr_v = {}
        self.v_vr = {}

        self.total_vrn = 0

        for (image, annotation) in dataset.items():
            v = self.v_id[annotation["verb"]]

            for frame in annotation["frames"]:
                for(r, n) in frame.items():
                    r = self.r_id[r]
                    n = self.n_id[n]

                    if (v, r) not in self.vr_id:
                        _id = len(self.vr_id)
                        self.vr_id[(v, r)] = _id
                        self.id_vr[_id] = (v, r)
                        self.vr_n_id[_id] = {}
                        self.vr_id_n[_id] = {}

                    vr = self.vr_id[(v, r)]
                    if v not in self.v_vr:
                        self.v_vr[v] = []
                    self.vr_v[vr] = v
                    if vr not in self.v_vr[v]:
                        self.v_vr[v].append(vr)

                    if n not in self.vr_n_id[vr]:
                        _id = len(self.vr_n_id[vr])
                        self.vr_n_id[vr][n] = _id
                        self.vr_id_n[vr][_id] = n
                        self.total_vrn += 1

    # change a situation into id
    # rv: {'verb': 1, 'frames': [[(2, 0), (3, 0), (4, 0)], [(2, 1), (3, 1), (4, 0)], [(2, 2), (3, 2), (4, 0)]]}
    def encode(self, situation):
        v = self.v_id[situation["verb"]]
        rv = {"verb": v, "frames": []}
        for frame in situation["frames"]:
            _e = []
            for (r, n) in frame.items():
                if r not in self.r_id:
                    r = self.unk_symbol()
                else:
                    r = self.r_id[r]
                if n not in self.n_id:
                    n = self.unk_symbol()
                else:
                    n = self.n_id[n]
                if (v, r) not in self.vr_id:
                    vr = self.unk_symbol()
                else:
                    vr = self.vr_id[(v, r)]
                if vr not in self.vr_n_id:
                    vrn = self.unk_symbol()
                elif n not in self.vr_n_id[vr]:
                    vrn = self.unk_symbol()
                else:
                    vrn = self.vr_n_id[vr][n]
                _e.append((vr, vrn))
            rv["frames"].append(_e)
        return rv

    # change the situation_id into situation
    # rv: {'verb': 'flapping', 'frames': [{}, {}, {}]}
    def decode(self, situation):
        # print situation
        verb = self.id_v[situation["verb"]]
        rv = {"verb": verb, "frames": []}
        for frame in situation["frames"]:
            _fr = {}
            for (vr, vrn) in frame:
                vrn = vrn.item()
                # print(self.vr_id_n)
                if vrn not in self.vr_id_n[vr]:
                    print("index error, verb = {}".format(verb))
                    n = -1
                else:
                    n = self.id_n[self.vr_id_n[vr][vrn]]
                r = self.id_r[self.id_vr[vr][1]]
                _fr[r] = n
            rv["frames"].append(_fr)
        return rv


class imSituSimpleImageFolder(data.Dataset):
# partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.ext)

    def get_images(self, dir):
        images = []
        for target in os.listdir(dir):
            f = os.path.join(dir, target)
            if os.path.isdir(f):
                continue
            if self.is_image_file(f):
                images.append(target)
        return images

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # list all images
        self.ext = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]
        self.images = self.get_images(root)

    def __getitem__(self, index):
        _id = os.path.join(self.root, self.images[index])
        img = Image.open(_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.LongTensor([index])

    def __len__(self):
        return len(self.images)


class imSituSituation(data.Dataset):
    def __init__(self, root, annotation_file, encoder, transform=None):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform

    def index_image(self, index):
        rv = []
        index = index.view(-1)
        for i in range(index.size()[0]):
            rv.append(self.ids[index[i]])
        return rv

    def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]

        img = Image.open(os.path.join(self.root, _id)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        target = self.encoder.to_tensor([ann])

        return (torch.LongTensor([index]), img, target)

    def __len__(self):
        return len(self.ids)
