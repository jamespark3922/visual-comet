#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging

import numpy as np
import torch
import tqdm
from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader

from dataloaders.tokenizers import VisualCometTokenizer
from dataloaders.vcg_rank import VCGRankDataset
from models.model import GPT2VisionAttentiveLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2_vc': (GPT2Config, GPT2VisionAttentiveLMHead, VisualCometTokenizer),
}

def load_and_cache_vcg_examples(args, tokenizer):
    dataset = VCGRankDataset(
        file_path=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        include_image=args.include_image,
        include_text=args.include_text,
        rank_mode=args.rank_mode,
        only_use_relevant_dets=not args.use_all_dets,
        overwrite_cache=args.overwrite_cache,
        cache_postfix=args.cache_postfix,
    )
    return dataset

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--model_type", default="gpt2_vc", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Directory containing train, val, test files")
    parser.add_argument("--split", default=None, type=str, required=True,
                        help="split to use for generation (val/test)")
    parser.add_argument("--task", type=str, default=None,
                        help="Which task for file input. If None, prompt is read as raw text 1 prompt per line in input-file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to load instance prompts from")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--rank_mode", default='random_50', type=str,
                        help="Type and number of samples to rank. {[random,movie,inference]}_{num_samples}")
    parser.add_argument("--no_image", dest='include_image', action='store_false',
                        help="Do not use image context to generate the inference sentences.")
    parser.add_argument("--no_text", dest='include_text', action='store_false',
                        help="Do not use text event and place to generate the inference sentences.")

    parser.add_argument("--use_all_dets", action='store_true',
                        help="Use all detections.")
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="Max seq length")

    # misc
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--cache_postfix', type=str, default=None,
                       help="postfix for cache")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    config, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    assert len(args.rank_mode.split('_')) == 2 , 'rank_mode should be in following format: {mode}_{num_samples}'
    assert args.rank_mode.split('_')[1].isdigit() , 'rank_mode should be in following format: {mode}_{num_samples}'
    assert args.rank_mode.split('_')[0] in ['random', 'movie', 'inference'], 'rank_mode should be in following format: {mode}_{num_samples}'

    print(args)

    output_name = args.output_file if args.output_file is not None else args.model_name_or_path + '/'
    output_file = '{}{}_rank_{}.json'.format(
        output_name,
        args.split,
        args.rank_mode)
    print(output_file)

    dataset = load_and_cache_vcg_examples(args, tokenizer)
    test_dataset = dataset.get_dataset(args.split)
    all_records = test_dataset.records  # read_and_parse_val_json(args.input_file)
    eval_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 batch_size=1)

    results = []
    idx = 0
    recalls = []

    with torch.no_grad():
        unique_inferences = set()
        for input_record, (
                data_input) in tqdm.tqdm(
            zip(all_records, eval_dataloader), total=len(all_records)):

            set_seed(args)

            ur = input_record['img_fn'] + input_record['event'] + input_record['inference_relation']
            if ur not in unique_inferences:
                unique_inferences.add(ur)
            else:
                continue
            # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, torch.clone(batch))
            if args.include_image:
                inputs, labels, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [d.to(args.device).squeeze(0) for d in data_input[:-1]]
                num_samples = inputs.size(0)

                img_feats = img_feats.repeat(num_samples, 1, 1)
                boxes = boxes.repeat(num_samples, 1, 1)
                boxes_mask = boxes_mask.repeat(num_samples, 1)
                objects = objects.repeat(num_samples, 1)
                segments = segments.repeat(num_samples, 1, 1, 1)
                person_ids = person_ids.repeat(num_samples, 1)
                subject_ids = subject_ids.repeat(num_samples, 1)
            else:
                inputs, labels = [d.to(args.device).squeeze(0) for d in data_input[:-1]]
                img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [None] * 7
            choices = data_input[-1]

            outputs = model(inputs, labels=labels, img_feats=img_feats, boxes=boxes, boxes_mask=boxes_mask, objects=objects,
                            segments=segments, person_ids=person_ids, subject_ids=subject_ids, use_rank=True)
            lm_loss = outputs[0]
            perplexities = torch.exp(torch.tensor(lm_loss))

            _, sorted_indices = torch.sort(perplexities)
            sorted_indices = sorted_indices.data.cpu().numpy()

            num_gt = sum(choices).item()

            gt_correct = 0
            for i in range(num_gt):
                if sorted_indices[i] < num_gt:
                    gt_correct+=1
            recall = gt_correct / num_gt

            input_record['rank'] = sorted_indices.tolist()
            input_record['num_gt'] = num_gt
            input_record['recall'] = recall
            recalls.append(recall)
            results.append(input_record)

            idx += 1

            if idx < 5:
                print('inputs', inputs[0], inputs[-1])
                print('labels', labels[0], labels[-1])

    mean_recall = np.mean(recalls)
    print('mean recall:', mean_recall)
    final_results = {'mean_recall' : mean_recall, 'results' : results}
    json.dump(final_results, open(output_file,'w'))


if __name__ == '__main__':
    main()
