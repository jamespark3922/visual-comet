# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""
Fine-tuning the library models for language modeling on WikiText-2 (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import shutil
import random

import numpy as np
import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, GPT2Config)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from dataloaders.tokenizers import VisualCometTokenizer
from dataloaders.vcg import VCGDataset

from models.model import GPT2VisionAttentiveLMHead

import time
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2_vc': (GPT2Config, GPT2VisionAttentiveLMHead, VisualCometTokenizer),
}

def load_and_cache_vcg_examples(args, tokenizer):
    dataset = VCGDataset(
        file_path=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        include_image=args.include_image,
        include_text=args.include_text,
        mode=args.mode,
        only_use_relevant_dets=not args.use_all_dets,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        cache_postfix=args.cache_postfix,
    )
    return dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_dir = os.path.join(args.tb_dir, "tb")
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = dataset.get_dataset('train')
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(
        train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, drop_last=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in train_iterator:
        logging.info("\n\n*** Starting Epoch: {} ***\n\n".format(epoch))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])

        for step, data_input in enumerate(epoch_iterator):
            # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, torch.clone(batch))
            if args.include_image:
                inputs, labels, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [
                    d.to(args.device) for d in data_input]
            else:
                inputs, labels = [d.to(args.device) for d in data_input]
                img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [None] * 7

            model.train()
            outputs = model(inputs, labels=labels, img_feats=img_feats, boxes=boxes, boxes_mask=boxes_mask, objects=objects, segments=segments,
                                    person_ids=person_ids, subject_ids=subject_ids)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.include_image and args.use_obj_loss and args.use_all_dets:
                loss = loss + outputs[-1]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                if args.local_rank in [-1,
                                       0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps,
                                         global_step)
                    tb_writer.flush()
                    logging_loss = tr_loss

                if args.local_rank in [-1,
                                       0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    logging.info("Evaluate epoch ... {}; iter ... {}".format(epoch, global_step))
                    results = evaluate(args, model, tokenizer, dataset, postfix='{}_{}'.format(epoch, global_step))

                    for entry in ['loss', 'perplexity']:
                        key = '{}_{}_{}'.format(entry, epoch, global_step)
                        tb_writer.add_scalar('eval_{}'.format(entry), results[key], global_step)
                        tb_writer.flush()

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break


        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer,
             dataset,
             postfix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.eval_output_dir
    eval_dataset = dataset.get_dataset('val')

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
        eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, drop_last=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(postfix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for data_input in tqdm(eval_dataloader, desc="Evaluating"):
        if args.include_image:
            inputs, labels, img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [d.to(args.device) for d in data_input]
        else:
            inputs, labels = [d.to(args.device) for d in data_input]
            img_feats, boxes, boxes_mask, objects, segments, person_ids, subject_ids = [None] * 7

        with torch.no_grad():
            outputs = model(inputs, labels=labels, img_feats=img_feats, boxes=boxes, boxes_mask=boxes_mask, objects=objects, segments=segments,
                                    person_ids=person_ids, subject_ids=subject_ids)

            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1


    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    output_eval_file = os.path.join(eval_output_dir, "metrics.json")

    if os.path.exists(output_eval_file):
        results = json.load(open(output_eval_file))
    else:
        results = {}

    if len(postfix) == 0:
        results.update({
            "perplexity": perplexity.item(),
            "eval_loss": eval_loss
        })
    else:
        results.update({
            "perplexity_{}".format(postfix): perplexity.item(),
            "loss_{}".format(postfix): eval_loss
        })

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(postfix))
        writer.write(json.dumps(results))
        writer.close()

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default="gpt2_vc", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Directory containing train, val, test files")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_output_dir", default=None, type=str, required=False,
                        help="Directory to write results to. Defaults to output_dir")
    parser.add_argument("--tb_dir", default=None, type=str, required=False,
                        help="Directory to write tensorboard to. Defaults to output_dir")

    parser.add_argument("--task", default="visual_comet", type=str,
                        help="The task to finetune the LM on. Currently only supports VisualCOMET")
    parser.add_argument("--use_all_dets", action='store_true',
                        help="Use all object detections instead of just people mentioned in the event description.")
    parser.add_argument("--use_obj_loss", action='store_true',
                        help="Use object classification loss.")
    parser.add_argument("--use_bbox", action='store_true',
                        help="Use bounding box coordinates as visual embeddings.")

    parser.add_argument("--mode", default='inference', type=str,
                        choices=['inference', 'all',],
                        help="Use which text to train on [inference, all]. "
                             "Set 'all' to generate Event, Place, and Inference text. This refers to 'EP loss' in the paper.")
    parser.add_argument("--no_image", dest='include_image', action='store_false',
                        help="Do not use image context to train the inference sentences.")
    parser.add_argument("--no_text", dest='include_text', action='store_false',
                        help="Do not use text event and place to train the inference sentences.")
    parser.add_argument('--no_person_ids', dest='use_person_ids', action='store_false',
                        help="Use person id embeddings in visual features.")
    parser.add_argument('--no_subject_ids', dest='use_subject_ids', action='store_false',
                        help="Use subejct embeddings in visual features.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")
    parser.add_argument("--max_seq_len", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", type=bool, default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same postfix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--cache_postfix', type=str, default=None,
                        help="postfix for cache")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    print(args)
    if args.eval_output_dir is None:
        args.eval_output_dir = args.output_dir
    if args.tb_dir is None:
        args.tb_dir = args.output_dir

    if args.model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling).")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_len = min(args.max_seq_len, tokenizer.max_len_single_sentence)

    # Get Dataset Loader
    dataset = load_and_cache_vcg_examples(
        args,
        tokenizer
    )

    # set up model
    config.max_event = dataset.max_event
    config.max_place = dataset.max_place
    config.max_inference = dataset.max_inference
    config.use_person_ids = args.use_person_ids
    config.use_subject_ids = args.use_subject_ids
    config.use_bbox = args.use_bbox
    print(config)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        if args.local_rank == 0:
            torch.distributed.barrier()

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Good practice: save your training arguments
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Begin actual training
        global_step, tr_loss = train(args, dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices in the end: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model in the end, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                                                    do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints: # probably not needed to run all
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        # evaluate the model saved in the end.
        for checkpoint in checkpoints:
            eval_postfix = checkpoint.split('-')[-1] + '_' + args.mode if len(checkpoints) > 1 else args.mode
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(
                args,
                model,
                tokenizer,
                dataset,
                postfix=eval_postfix
            )
            result = dict(('{}_{}'.format(k, eval_postfix), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
