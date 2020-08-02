"""
Dataloaders for VCG
"""
import json
import os
from copy import deepcopy

import pickle
import numpy as np
from tqdm import tqdm, trange

import torch

from dataloaders.tokenizers import VisualCometTokenizer
from dataloaders.mask_utils import make_mask

from utils.file_utils import read_and_parse_finetune_json, read_and_parse_generation_json

from config import VCR_IMAGES_DIR, VCR_FEATURES_DIR

# Here's an example json
# {'img_fn': 'movieclips_Batman_v_Superman_Dawn_of_Justice/7HlSKPTYZhs@40.jpg',
#  'movie': 'Batman v Superman: Dawn of Justice (2016)',
#  'metadata_fn': 'movieclips_Batman_v_Superman_Dawn_of_Justice/7HlSKPTYZhs@40.json',
#  'place': 'in a courtroom',
#  'event': '3 is a policeman watching the crowd in the courtroom closely',
#  'person': '3',
#  'xIntent': ['make sure there are not threats present',
#   'ensure that no one causes issues'],
#  'before': ['train as a police officer',
#   'stand in front of the exit door',
#   'arrive at the courthouse for work.',
#   'begin guarding the trial courtroom.'],
#  'after': ['blow his whistle to wake everyone up',
#   'whip out his nightstick',
#   'guard the courtroom until the trial is finished.',
#   'open the doors for spectators to leave.']}

def _pad_ids(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))

def _combine_and_pad_tokens(tokenizer: VisualCometTokenizer, tokens,
                            max_image, max_event, max_place, max_inference, max_seq_len):
    """
    :param tokenizer: tokenizer for the model
    :param tokens: [[image_tokens], [event_tokens], [place_tokens], [inference_tokens] ]
    :param max_seq_len: maximum sequence for concatenated tokens
    :return: Padded tokens to max length for each set (image, event, place, inference) and concatenated version of the set
    """
    new_tokens = []
    max_lens = [max_image, max_event, max_place, max_inference]
    assert len(tokens) == len(max_lens)
    for i, t in enumerate(tokens):
        max_len = max_lens[i]
        if len(t) > max_len:
            if i < 3:
                if i == 0:
                    end_token = tokenizer.end_img
                elif i == 1:
                    end_token = tokenizer.end_event
                elif i == 2:
                    end_token = tokenizer.end_place
                t = t[:max_len - 1] + [end_token]
        else:
            t.extend([tokenizer.unk_token] * (max_len - len(t)))
        new_tokens.extend(t)

    if len(new_tokens) > max_seq_len:
        new_tokens = new_tokens[:max_seq_len - 1] + [tokenizer.end_inference]
    else:
        new_tokens.extend([tokenizer.unk_token] * (max_seq_len - len(new_tokens)))

    return new_tokens

def vcg_record_to_tokens(tokenizer: VisualCometTokenizer,
                         record,
                         num_max_boxes=15,
                         ):
    event = record['event_name']
    place = record['place']
    inference = record['inference_relation']
    inference_text = record['inference_text_name']

    training_instance = [[tokenizer.begin_img] + [tokenizer.unk_token] * num_max_boxes + [tokenizer.end_img]]
    training_instance.append([tokenizer.begin_event,event,tokenizer.end_event])
    training_instance.append([tokenizer.begin_place,place,tokenizer.end_place])
    training_instance.append([tokenizer.begin_inferences[inference], inference_text, tokenizer.end_inference])

    return training_instance

def _create_partial_labels(tokenizer: VisualCometTokenizer, tokenized_text, mode):
    try:
        labels = [-1] * len(tokenized_text)

        # create lm labels for inference sentences
        possible_inferences = [tokenizer.convert_tokens_to_ids([br])[0] for br in tokenizer.begin_inferences.values()]
        begin_inference = [r for r in possible_inferences if r in tokenized_text]
        assert len(begin_inference) == 1
        inference_start_token = begin_inference[0]
        start_idx = tokenized_text.index(inference_start_token)
        inference_end_token = tokenizer.convert_tokens_to_ids([tokenizer.end_inference])[0]
        end_idx = tokenized_text.index(inference_end_token)
        labels[start_idx + 1: end_idx + 1] = tokenized_text[start_idx + 1:end_idx + 1]

        # create lm labels for event and place as well
        if mode == 'all':
            event_start_token = tokenizer.convert_tokens_to_ids([tokenizer.begin_event])[0]
            start_idx = tokenized_text.index(event_start_token)
            event_end_token = tokenizer.convert_tokens_to_ids([tokenizer.end_event])[0]
            end_idx = tokenized_text.index(event_end_token)
            labels[start_idx + 1: end_idx + 1] = tokenized_text[start_idx + 1:end_idx + 1]

            event_start_token = tokenizer.convert_tokens_to_ids([tokenizer.begin_place])[0]
            start_idx = tokenized_text.index(event_start_token)
            event_end_token = tokenizer.convert_tokens_to_ids([tokenizer.end_place])[0]
            end_idx = tokenized_text.index(event_end_token)
            labels[start_idx + 1: end_idx + 1] = tokenized_text[start_idx + 1:end_idx + 1]

        assert len(tokenized_text) == len(labels)
        return labels
    except ValueError:
        raise Exception("Failed to tokenize: {}".format(tokenized_text))

class VCGDataset:
    def __init__(self,
                 tokenizer,
                 file_path,
                 max_seq_len=256,
                 cache_dir=None,
                 overwrite_cache=False,
                 cache_postfix=None,
                 include_image=False,
                 include_text=True,
                 mode='inference',
                 num_max_boxes=15,
                 only_use_relevant_dets=True,
                 ):
        vcg_dir = os.path.dirname(file_path)
        assert os.path.isdir(vcg_dir)
        self.include_image = include_image
        self.include_text = include_text
        self.only_use_relevant_dets = only_use_relevant_dets

        self.max_seq_len = max_seq_len
        self.num_max_boxes = num_max_boxes
        self.max_image = None
        self.max_event = None
        self.max_place = None
        self.max_inference = None

        cache_name = 'cached_lm_max_seq_len_{}_mode_{}_include_text_{}'.format(max_seq_len, mode, str(include_text)).lower()
        if cache_postfix:
            cache_name += '_' + cache_postfix
        if cache_dir is None:
            cached_features_files = os.path.join(vcg_dir, cache_name)
        else:
            cached_features_files = os.path.join(cache_dir, cache_name)

        for split in ['train', 'val', 'test']:
            split_filename = '{}_annots.json'.format(split)
            assert os.path.exists(os.path.join(vcg_dir, split_filename))

        self.vcg_dataset = {}
        examples = {}
        labels = {}
        records = {}
        splits = ['train', 'val']

        if os.path.exists(cached_features_files) and not overwrite_cache:
            print("Loading features from cached file %s", cached_features_files)
            with open(cached_features_files, 'rb') as handle:
                p = pickle.load(handle)
                self.num_max_boxes = p['num_max_boxes']
                self.max_image = p['max_image']
                self.max_event = p['max_event']
                self.max_place = p['max_place']
                self.max_inference = p['max_inference']
                for split in splits:
                    examples[split], labels[split], records[split] = p['data'][split]
        else:
            print("Creating features from dataset file at {} with cache file name: {}".format(vcg_dir, cached_features_files))

            max_image = self.num_max_boxes + 2
            max_event = 0
            max_place = 0
            max_inference = 0

            for s, split in enumerate(splits):

                examples[split] = []
                labels[split] = []
                token_list = []
                text_list = []

                split_filename = '{}_annots.json'.format(split)
                records[split] = read_and_parse_finetune_json(os.path.join(vcg_dir, split_filename))

                idx = 0
                num_ex = 5

                print(split, len(records[split]))
                for record in tqdm(records[split], "Encoding Data"):
                    vcg_tokens = \
                        vcg_record_to_tokens(
                            tokenizer=tokenizer,
                            record=record,
                            num_max_boxes=num_max_boxes,
                        )

                    tokens = [tokenizer.tokenize(" ".join(vt)) for vt in vcg_tokens]
                    assert len(vcg_tokens) == 4
                    if split == 'train':
                        e_l, p_l, r_l = [len(vt) for vt in tokens[1:]]
                        max_event = max(max_event, e_l)
                        max_place = max(max_place, p_l)
                        max_inference = max(max_inference, r_l)
                    token_list.append(tokens)
                    text_list.append(vcg_tokens)
                    idx+=1

                if split == 'train':
                    print('max_image, max_event, max_place, max_inference:', max_image, max_event, max_place, max_inference)
                    self.max_image = max_image
                    self.max_event = max_event
                    self.max_place = max_place
                    self.max_inference = max_inference

                idx = 0
                for tokens in tqdm(token_list):
                    padded_tokens = _combine_and_pad_tokens(tokenizer, tokens, max_image, max_event, max_place, max_inference, max_seq_len)
                    tokenized_text = tokenizer.convert_tokens_to_ids(padded_tokens)
                    if not include_text: # mask out events and place text
                        inference_start_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.begin_event])[0]
                        start_idx = tokenized_text.index(inference_start_token_idx)
                        inference_end_token_idx = tokenizer.convert_tokens_to_ids([tokenizer.end_place])[0]
                        end_idx = tokenized_text.index(inference_end_token_idx)

                        assert end_idx > start_idx
                        unk_id = tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]
                        tokenized_text[start_idx: end_idx + 1] = [unk_id] * (end_idx - start_idx + 1)

                    examples[split].append(tokenized_text)
                    lm_labels = _create_partial_labels(tokenizer, tokenized_text, mode)
                    labels[split].append(lm_labels)

                    if idx < num_ex:
                        print("***** Example Instance for Split: {} *****".format(split))
                        print("Text: {}".format(text_list[idx]))
                        print("Tokenized Text: {}".format(tokenized_text))
                        print("Labels: {}".format(lm_labels))
                        print("********\n")
                    idx+=1

            print("Saving features into cached file %s", cached_features_files)
            with open(cached_features_files, 'wb') as handle:
                data = {s : (examples[s], labels[s], records[s]) for s in splits}
                pickle.dump(
                    {
                        'num_max_boxes' : self.num_max_boxes,
                        'max_image' : self.max_image,
                        'max_event' : self.max_event,
                        'max_place' : self.max_place,
                        'max_inference' : self.max_inference,
                        'data' : data
                    },
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

        for split in splits:
            self.vcg_dataset[split] = VCGLoader(examples[split], labels[split], records[split], split, tokenizer,
                                                include_image=self.include_image,
                                                num_max_boxes=self.num_max_boxes,
                                                only_use_relevant_dets=self.only_use_relevant_dets)

    def get_dataset(self, split):
        return self.vcg_dataset[split]

def _to_boxes_and_masks(features, boxes, obj_labels, segments, num_max_boxes):
    num_boxes = len(boxes)
    if num_boxes > num_max_boxes:
        return features[:num_max_boxes,:], boxes[:num_max_boxes,:], obj_labels[:num_max_boxes], \
               segments[:num_max_boxes,:], [1] * num_max_boxes
    d = len(features[0])
    padded_features = np.concatenate((features, np.zeros((num_max_boxes - num_boxes, d))))
    padded_boxes = np.concatenate((boxes, np.zeros((num_max_boxes - num_boxes, 4))))
    padded_obj_labels = np.concatenate((obj_labels, np.zeros(num_max_boxes - num_boxes)), axis=0)

    shape = segments.shape
    padding_segments = np.concatenate((segments, np.zeros((num_max_boxes - shape[0],) + shape[1:])),
                                      axis=0)

    mask = np.concatenate((np.ones(num_boxes), np.zeros(num_max_boxes - num_boxes)), axis=0)

    return padded_features, padded_boxes, padded_obj_labels, padding_segments, mask

class VCGLoader:
    def __init__(self, examples, labels, records, split, tokenizer,
                 include_image=True, num_max_boxes=15, only_use_relevant_dets=True, add_image_as_a_box=True):
        """

        :param split: train, val, or test
        :param mode: inference or event
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the event and inference.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        # :param embs_to_load: Which precomputed embeddings to load.
        """
        self.examples = examples
        self.labels = labels
        self.records = records
        self.split = split
        self.tokenizer = tokenizer
        self.include_image = include_image
        self.num_max_boxes = num_max_boxes
        self.add_image_as_a_box = add_image_as_a_box
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        if split not in ('train', 'val', 'test'):
            raise ValueError("Split must be in train, val, test Supplied {}".format(split))

        with open(os.path.join(os.path.dirname(__file__), 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        # self.embs_to_load = embs_to_load
        # self.h5fn = os.path.join(VCG_ANNOTS_DIR, f'{self.embs_to_load}_{self.mode}_{self.split}.h5')
        # print("Loading embeddings from {}".format(self.h5fn), flush=True)

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'inference'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    def __len__(self):
        return len(self.examples)

    def get_dets_to_use(self, item):
        """
        We might want to use fewer detections so lets do so.
        :param item:
        :param event:
        :param inferences:
        :return:
        """

        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)
        objects = metadata['names']
        people = np.array([x == 'person' for x in objects], dtype=bool)

        # choose the person listed in the event description.
        if self.only_use_relevant_dets:
            # Load events and inferences
            dets2use = np.zeros(len(objects), dtype=bool)
            subjects = np.zeros(len(objects), dtype=bool)

            for tag in list(item['person2name'].keys()):
                tag = int(tag) - 1
                if tag >= 0 and tag < len(objects):  # sanity check
                    dets2use[tag] = True

            # # Uncomment after releasing subject annotations in the event descriptions
            # for possibly_det_list in str(item['subject']).replace(',', ' ').split():
            #     for tag in possibly_det_list:
            #         if tag.isdigit():
            #             tag = int(tag) - 1
            #             if tag >= 0 and tag < len(objects):  # sanity check
            #                 dets2use[tag] = True
            #                 subjects[tag] = True
            #         elif tag.lower() in ('everyone', 'everyones', 'group', 'crowd'):
            #             dets2use |= people
            #             subjects |= people
            if not dets2use.any():
                dets2use |= people

        else:
            dets2use = np.ones(len(objects), dtype=bool)
            subjects = np.zeros(len(objects), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]
        subjects = np.where(subjects)[0]

        old_det_to_new_ind = np.zeros(len(objects), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind, subjects

    def __getitem__(self, index):
        event_inference_example = torch.tensor(self.examples[index])
        labels = torch.tensor(self.labels[index])
        record = self.records[index]
        if not self.include_image:
            return event_inference_example, labels

        #######
        # Compute Image Features. Adapted from https://github.com/rowanz/r2c/blob/master/dataloaders/vcg.py
        #######

        ###################################################################
        # Load boxes and their features.
        with open(os.path.join(VCR_IMAGES_DIR, record['metadata_fn']), 'r') as f:
            metadata = json.load(f)
        dets2use, old_det_to_new_ind, subjects = self.get_dets_to_use(record)
        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i],
                                    polygons_list=metadata['segms'][i])
                          for i in dets2use])

        # Chop off the final dimension, that's the confidence
        img_fn = record['img_fn']
        id = img_fn[img_fn.rfind('/')+1:img_fn.rfind('.')]
        with open(os.path.join(VCR_FEATURES_DIR,id)+'.pkl','rb') as p:
            features_dict = pickle.load(p)
        features = features_dict['object_features'][dets2use]
        boxes = np.array(metadata['boxes'])[dets2use, :-1]

        # create id labels to help ground person in the image
        objects = metadata['names']
        obj_labels = [self.coco_obj_to_ind[objects[i]] for i in
                      dets2use.tolist()]
        person_ids = [0] * len(obj_labels)
        for i in range(len(person_ids)):
            if obj_labels[i] == 1:
                p_id = int(dets2use[i])+1  # add 1 for person ids because it starts with 1
                person_ids[i] = self.tokenizer.convert_tokens_to_ids(['<|det%d|>' % p_id])[0]
        subject_ids = [int(dets2use[i] in subjects) for i in range(len(obj_labels))]

        # add the image in the first visual sequence
        if self.add_image_as_a_box:
            w = metadata['width']
            h = metadata['height']
            features = np.row_stack((features_dict['image_features'], features))
            boxes = np.row_stack((np.array([0, 0, w, h]), boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels
            person_ids = [self.tokenizer.convert_tokens_to_ids(['<|det0|>'])[0]] + person_ids
            subject_ids = [0] + subject_ids

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        if not np.all((boxes[:, 2] <= w)):
            boxes[:,2] = np.clip(boxes[:,2],None,w)
        if not np.all((boxes[:, 3] <= h)):
            boxes[:, 3] = np.clip(boxes[:, 3], None, h)

        padded_features, padded_boxes, padded_obj_labels, padded_segments, box_masks = \
            _to_boxes_and_masks(features, boxes, obj_labels, segms, self.num_max_boxes)
        person_ids = _pad_ids(person_ids, self.num_max_boxes)
        subject_ids = _pad_ids(subject_ids, self.num_max_boxes)

        features = torch.Tensor(padded_features)
        boxes = torch.Tensor(padded_boxes)
        boxes_mask = torch.LongTensor(box_masks)
        objects = torch.LongTensor(padded_obj_labels)
        segments = torch.Tensor(padded_segments)
        person_ids = torch.LongTensor(person_ids)
        subject_ids = torch.LongTensor(subject_ids)

        return event_inference_example, labels, features, boxes, boxes_mask, objects, segments, person_ids, subject_ids

