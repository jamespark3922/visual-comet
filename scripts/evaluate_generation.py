import os
import time
import sys

sys.path.append(os.getcwd())


sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from tqdm import tqdm
import json
import numpy as np

import argparse

def flatten(outer):
    return [el for key in outer for el in key]

def use_same_id(sent):
    r_sent = sent.replace("'", " '")
    r_sent = ' '.join([g if not g.isdigit() else '1' for g in r_sent.split()]).strip()
    r_sent = r_sent.replace(" '","'")
    return r_sent


def compute_metric_inference_old(gens_list, refs_list, calculate_diversity=False, train_file=None):

    scorers = [
        (Bleu(4), ["Bleu_1","Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    tokenizer = PTBTokenizer()

    refs = {}
    preds = {}
    output = {}
    cnt = 0

    for i, gens in tqdm(enumerate(gens_list)):
        for pred in gens['generated_relations']:
            ref = gens['relation_text_orig']
            pred = pred['relation']
            pred = pred.replace('<|det', '').replace('|>', '')
            preds[cnt] = [{'caption': pred}]
            refs[cnt] = [{'caption': r} for r in ref]
            cnt += 1

    refs = tokenizer.tokenize(refs)
    preds = tokenizer.tokenize(preds)

    if calculate_diversity:
        unique_sents = []
        novel_sents = []

        # store train sentence to calculate novelty
        train_sents = json.load(open(train_file))
        ts = set()
        for d in train_sents:
            for r in ['intent', 'before', 'after']:
                if r in d:
                    for sent in d[r]:
                        r_sent = use_same_id(sent)
                        ts.add(r_sent)

        for pred in preds.values():
            pred_same_id = use_same_id(pred[0])
            unique_sents.append(pred_same_id)
            novel_sents.append(pred_same_id not in ts)

        print(len(unique_sents))
        unique = len(set(unique_sents)) / len(unique_sents)
        output['Unique'] = unique
        print('Unique Inferences:', unique)

        novel = np.mean(novel_sents)
        output['Novel'] = novel
        print('Novel Inferences:', novel)

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for m in range(len(method)):
                output[method[m]] = score[m]
                print(method[m], score[m])
        else:
            output[method] = score
            print(method, score)

    return output

def compute_metric_inference(gens_list, refs_list, calculate_diversity=False, train_file=None):

    scorers = [
        (Bleu(4), ["Bleu_1","Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    tokenizer = PTBTokenizer()

    refs = {}
    preds = {}
    output = {}
    cnt = 0

    for i, gens in tqdm(enumerate(gens_list)):
        event_idx = gens['event_idx']
        relation = gens['inference_relation']
        ref = refs_list[event_idx][relation]
        if len(ref) > 0:
            for pred in gens['generations']:
                pred = pred.replace('<|det', '').replace('|>', '')
                preds[cnt] = [{'caption': pred}]
                refs[cnt] = [{'caption': r} for r in ref]
                cnt += 1

    refs = tokenizer.tokenize(refs)
    preds = tokenizer.tokenize(preds)

    if calculate_diversity:
        unique_sents = []
        novel_sents = []

        # store train sentence to calculate novelty
        train_sents = json.load(open(train_file))
        ts = set()
        for d in train_sents:
            for r in ['intent', 'before', 'after']:
                if r in d:
                    for sent in d[r]:
                        r_sent = use_same_id(sent)
                        ts.add(r_sent)

        for pred in preds.values():
            pred_same_id = use_same_id(pred[0])
            unique_sents.append(pred_same_id)
            novel_sents.append(pred_same_id not in ts)

        print(len(unique_sents))
        unique = len(set(unique_sents)) / len(unique_sents)
        output['Unique'] = unique
        print('Unique Inferences:', unique)

        novel = np.mean(novel_sents)
        output['Novel'] = novel
        print('Novel Inferences:', novel)

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for m in range(len(method)):
                output[method[m]] = score[m]
                print(method[m], score[m])
        else:
            output[method] = score
            print(method, score)

    return output

# def process_reference(refs):
#     inference_relations = ['intent', 'before', 'after']
#     refs_dict = {}
#     for ref in refs:
#         for relation in inference_relations:
#             if relation in ref:
#                 k = ref['img_fn'] + ' '.join(ref['event'].replace('s','').lower().strip().split()[:4]) + relation
#                 refs_dict[k] = ref[relation]
#     return refs_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs_file", type=str, required=True)
    parser.add_argument("--gens_file", type=str, required=True)
    parser.add_argument("--mode", type=str, default="inference")

    parser.add_argument("--diversity", action='store_true')
    parser.add_argument("--train_file", type=str)

    args = parser.parse_args()

    if args.diversity:
        assert args.train_file is not None, \
            "Running diversity option requires train_file to calculate novel inferences."

    gens_list = json.load(open(args.gens_file))
    refs_list = json.load(open(args.refs_file))

    output = compute_metric_inference(gens_list, refs_list,
                                      calculate_diversity=args.diversity, train_file=args.train_file)

    write_name = args.gens_file.replace(".json", ".evaluate.json")
    print("Saving to: {}".format(write_name))
    with open(write_name, "w") as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()
