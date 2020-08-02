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

import os
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from dataloaders.tokenizers import VisualCometTokenizer
from dataloaders.vcg_generation import VCGGenDataset
from models.model import GPT2VisionAttentiveLMHead
from utils.file_utils import replace_names

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

def load_and_cache_vcg_examples(args, config, tokenizer):
    dataset = VCGGenDataset(
        file_path=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_event=config.max_event,
        max_place=config.max_place,
        max_inference=config.max_inference,
        include_image=args.include_image,
        include_text=args.include_text,
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

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty

def beam_search_sequence(model, length, context, img_feats, boxes, boxes_mask, objects,
                    segments, person_ids, subject_ids,
                    end_token_id, pad_token_id, num_beams=1, temperature=1, device=None):
    """ Generate sequences for each example with beam search.
    """
    if device is None:
        device = torch.device("cpu")

    if img_feats is not None:
        img_feats = img_feats.repeat(num_beams, 1, 1, 1)
        boxes = boxes.repeat(num_beams, 1, 1)
        boxes_mask = boxes_mask.repeat(num_beams, 1)
        objects = objects.repeat(num_beams, 1)
        segments = segments.repeat(num_beams, 1, 1, 1)
        person_ids = person_ids.repeat(num_beams, 1)
        subject_ids = subject_ids.repeat(num_beams, 1)

    generated = context

    # generated hypotheses
    generated_hyps = BeamHypotheses(num_beams, length, 1, early_stopping=False)

    # scores for each sentence in the beam
    beam_scores = torch.zeros((1,num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    with torch.no_grad():
        for cur_len in range(length):

            inputs = {'input_ids': generated}

            if img_feats is not None:
                inputs = {
                    'input_ids': generated,
                    'img_feats': img_feats,
                    'boxes': boxes,
                    'boxes_mask': boxes_mask,
                    'objects': objects,
                    'segments': segments,
                    'person_ids': person_ids,
                    'subject_ids': subject_ids
                }

            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature

            vocab_size = next_token_logits.size(-1)
            scores = F.log_softmax(next_token_logits, dim=-1)  # (num_beams, vocab_size)

            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            _scores = _scores.view(-1)
            next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=0, largest=True, sorted=True)

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, score in zip(next_words, next_scores):

                # get beam and word IDs
                beam_id = idx // vocab_size
                word_id = idx % vocab_size

                # end of sentence, or next word
                if word_id.item() == end_token_id or cur_len + 1 == length:
                    generated_hyps.add(
                        generated[beam_id].clone(), score.item()
                    )
                else:
                    next_sent_beam.append((score, word_id, beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == length else num_beams
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch

            # sanity check / prepare next batch
            beam_scores = beam_scores.new([x[0] for x in next_sent_beam])
            beam_words = generated.new([x[1] for x in next_sent_beam])
            beam_idx = generated.new([x[2] for x in next_sent_beam])

            # re-order batch
            generated = generated[beam_idx, :]
            generated = torch.cat([generated, beam_words.unsqueeze(1)], dim=-1)

    # select the best hypotheses
    tgt_len = generated.new(num_beams)
    best = []

    for i, hypotheses in enumerate(generated_hyps.hyp):
        best_hyp = hypotheses[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = generated.new(num_beams, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, : tgt_len[i] - 1] = hypo
        decoded[i, tgt_len[i] - 1] = end_token_id

    return decoded

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        probs = F.softmax(logits, dim=1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)

        _cumsum = sorted_probs.cumsum(1)
        mask = _cumsum < top_p
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
        sorted_probs = sorted_probs * mask.float()
        sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)

        logits.scatter_(1, sorted_indices, sorted_probs.log())

    return logits

def sample_sequence(model, length, context, end_token, pad_token, img_feats, boxes, boxes_mask, objects,
                    segments, person_ids, subject_ids,
                    do_sample=True, num_samples=1, temperature=1, top_k=0, top_p=0.0, device=None):
    if not do_sample:
        return beam_search_sequence(model, length, context, img_feats, boxes, boxes_mask, objects, segments,
                                    person_ids, subject_ids, end_token, pad_token,
                                    num_beams=num_samples, temperature=1, device=device)
    if device is None:
        device = torch.device("cpu")

    if img_feats is not None:
        img_feats = img_feats.repeat(num_samples, 1, 1)
        boxes = boxes.repeat(num_samples, 1, 1)
        boxes_mask = boxes_mask.repeat(num_samples, 1)
        objects = objects.repeat(num_samples, 1)
        segments = segments.repeat(num_samples, 1, 1, 1)
        person_ids = person_ids.repeat(num_samples, 1)
        subject_ids = subject_ids.repeat(num_samples, 1)

    generated = context
    with torch.no_grad():
        for tok_idx in range(length):

            inputs = {'input_ids': generated}

            if img_feats is not None:
                inputs = {
                    'input_ids': generated,
                    'img_feats': img_feats,
                    'boxes': boxes,
                    'boxes_mask': boxes_mask,
                    'objects': objects,
                    'segments': segments,
                    'person_ids': person_ids,
                    'subject_ids': subject_ids
                }

            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            if do_sample:
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                           num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated = torch.cat((generated, next_token), dim=-1)
    return generated


def main():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--model_type", default='gpt2_vc', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Directory containing train, val, test files")
    parser.add_argument("--split", default=None, type=str, required=True, choices=['val', 'test'],
                        help="split to use for generation (val/test)")

    parser.add_argument("--task", type=str, default=None,
                        help="Which task for file input. If None, prompt is read as raw text 1 prompt per line in input-file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to generate inferences; otherwise, created in the same directpry in the model")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument('--max_seq_len', type=int, default=128, help="input sequence length after tokenization.")
    parser.add_argument("--no_image", dest='include_image', action='store_false',
                        help="Do not use image context to generate the inference sentences.")
    parser.add_argument("--no_text", dest='include_text', action='store_false',
                        help="Do not use text event and place to generate the inference sentences.")
    parser.add_argument("--use_all_dets", action='store_true',
                        help="Use all detections.")

    # sampling based parameters
    parser.add_argument("--length", type=int, default=20, help='max length of sequence to generate')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--inference_type", default='all', type=str, choices=['all', 'intent', 'need', 'react'],
                        help="inference type to generate")
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--num_samples", default=5, type=int, help="No. of samples to obtain.")
    parser.add_argument("--gen_batch_size", default=1, type=int, help="No. of instances per batch (for now, it only supports batch size 1).")

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
    # train_args = torch.load(os.path.join(args.model_name_or_path, 'training_args.bin'))
    config, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)

    def _prompt_to_gen(context_orig, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids):

        set_seed(args)

        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raise Exception("Not supported")

        context = context_orig.repeat(args.num_samples, 1)
        text_gen = [{} for _ in range(args.num_samples)]

        # set the start token to signal when to start generation
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]
        possible_inferences = [tokenizer.convert_tokens_to_ids([br])[0] for br in tokenizer.begin_inferences.values()]
        begin_inference = [r for r in possible_inferences if r in context_orig]
        assert len(begin_inference) == 1
        prompt_token_idx = begin_inference[0]
        end_token = tokenizer.convert_tokens_to_ids([tokenizer.end_inference])[0]

        # cut off
        idx_of_prompt_token = (context_orig == prompt_token_idx).nonzero()[0][1].item()
        context[:,idx_of_prompt_token] = prompt_token_idx
        context_input = context[:, :idx_of_prompt_token + 1]

        # begin sampling sequence starting from context_input
        out = sample_sequence(
            model=model,
            context=context_input,
            pad_token=pad_token,
            end_token=end_token,
            img_feats=img_feats,
            boxes=boxes,
            boxes_mask=boxes_mask,
            objects=objects,
            segments=segments,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device=args.device,
            num_samples=args.num_samples,
            person_ids=person_ids,
            subject_ids=subject_ids
        )

        # ensure to end the sequence with end token, and pad the rest of the sequence.
        out = out[:, idx_of_prompt_token + 1:]
        out[:,-1] = end_token
        ending_idx = (out == end_token).nonzero()
        processed = []
        for i in range(ending_idx.size(0)):
            sample_idx = ending_idx[i][0].item()
            if sample_idx not in processed:
                processed.append(sample_idx)
                end_idx = ending_idx[i][1].item()
                if end_idx < out.size(1)-1:
                    out[sample_idx,end_idx+1:] = pad_token
                context_end_idx = idx_of_prompt_token+1+end_idx
                context[sample_idx,idx_of_prompt_token+1:context_end_idx+1] = out[sample_idx, :end_idx+1]
                if context_end_idx < context.size(1)-1:
                    context[sample_idx:,context_end_idx+1:] = pad_token

        # decode the sequence to text
        text_gen = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in out.tolist()]
        return text_gen

    # output file to store the generations
    output_name = args.output_file if args.output_file is not None else args.model_name_or_path + '/'
    output_file = '{}{}_sample_{}_num_{}_top_k_{}_top_p_{}.json'.format(
        output_name,
        args.split,
        args.do_sample,
        args.num_samples,
        args.top_k,
        args.top_p,
        )
    print(output_file)

    # Get Dataset Loader
    dataset = load_and_cache_vcg_examples(args, config, tokenizer)
    eval_dataset = dataset.get_dataset(args.split)
    all_records = eval_dataset.records
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.gen_batch_size)

    output_keys = ['img_fn', 'movie', 'metadata_fn', 'split', 'place', 'event', 'inference_relation', 'event_idx']
    results = []
    idx = 0
    context_inputs = set()
    for input_record, (
            data_input) in tqdm.tqdm(
        zip(all_records, eval_dataloader), total=len(all_records)):

        if args.include_image:
            inputs, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = \
                [d.to(args.device) for d in data_input]
        else:
            inputs = data_input.to(args.device) # inputs, labels = [d.to(args.device) for d in data_input]
            img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [None] * 7

        # Skip if we have processed this image, event, and inference type.
        # context = input_record['img_fn'] + input_record['event'] + input_record['inference_relation']
        # if context not in context_inputs:
        #     context_inputs.add(context)
        # else:
        #     continue

        # Now, generate the inferences and decode using original ids.
        generations = _prompt_to_gen(inputs, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids)
        for i in range(len(generations)):
            generations[i] = replace_names(generations[i], input_record['name2person'])
        output_record = {k: input_record[k] for k in output_keys}
        output_record['generations'] = generations
        results.append(output_record)

        if idx < 30:
            print("Image: {}".format(output_record['img_fn']))
            print("Event Text: {}".format(output_record['event']))
            print("Place: {}".format(output_record['place']))
            print("Inference Type: {}".format(output_record['inference_relation']))
            print("Inference Generations: {}".format(generations))
        idx += 1

    json.dump(results, open(output_file,'w'))
    print('Saved to', output_file)

if __name__ == '__main__':
    main()
