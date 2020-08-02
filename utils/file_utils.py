import json
import os
import random

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def write_items(items, output_file: str):
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()


def read_lines(input_file: str):
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def read_jsonl_lines(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def _map_numbers_to_names(sent, map_idx=None):
    tokens = sent.replace(',', ' , ').replace("'", " '").replace('.', ' .').split()
    person_ids = set()
    for t in tokens:
        if t.isdigit():
            person_ids.add(t)

    person_ids = list(person_ids)
    if map_idx is not None:
        available_names = [n for n in GENDER_NEUTRAL_NAMES if n not in map_idx]
        chosen_names = random.sample(available_names, len([p for p in person_ids if p not in map_idx]))
        j = 0
        for i in range(len(person_ids)):
            if person_ids[i] not in map_idx:
                map_idx[person_ids[i]] = chosen_names[j]
                j+=1
    else:
        map_idx = {}
        chosen_names = random.sample(GENDER_NEUTRAL_NAMES, len(person_ids))
        for i in range(len(person_ids)):
            map_idx[person_ids[i]] = chosen_names[i]

    tokens = [t if not t.isdigit() else map_idx[t] for t in tokens]
    sent = ' '.join(tokens).strip().replace(' .','.')
    return map_idx, sent


def _map_numbers_to_det_numbers(sent, map_idx=None):
    tokens = sent.replace(',', ' , ').replace("'", " '").replace('.', ' .').replace('?',' ?').split()
    person_ids = set()
    for t in tokens:
        if t.isdigit():
            person_ids.add(t)

    person_ids = list(person_ids)
    if map_idx is None:
        map_idx = {}
    for i in range(len(person_ids)):
        map_idx[person_ids[i]] = '<|det%d|>' % (int(person_ids[i]))

    tokens = [t if not t.isdigit() or t not in map_idx else map_idx[t] for t in tokens]
    sent = ' '.join(tokens).strip().replace("'", " '").replace(' .', '.')
    return map_idx, sent

def replace_names(sent, n2p):
    for name in n2p:
        sent = sent.replace(name, n2p[name])
    sent = sent.replace('<|det','').replace('|>','')
    return sent

def _encode_finetune_records(records):
    to_return = []
    relations = ['intent', 'before', 'after']
    for idx, record in enumerate(records):
        for r in relations:
            if r in record:
                for inference in record[r]:
                    info = {k: record[k] for k in record if k not in relations}
                    info['event_idx'] = idx
                    info['inference_relation'] = r
                    info['inference_text'] = inference

                    # replace person id token with special tokens used in tokenizer
                    map_idx, event_name = _map_numbers_to_det_numbers(info['event'])
                    info['event_name'] = event_name
                    map_idx, inference_name = _map_numbers_to_det_numbers(inference, map_idx)
                    info['inference_text_name'] = inference_name
                    info['person2name'] = map_idx
                    info['name2person'] = {v: k for k,v in map_idx.items()}
                    to_return.append(info)
    return to_return

def read_and_parse_finetune_json(input_file: str):
    with open(input_file) as f:
        records = json.load(f)
        records = _encode_finetune_records(records)
        return records

def _encode_generation_records(records):
    to_return = []
    relations = ['intent', 'before', 'after']
    for idx, record in enumerate(records):
        for r in relations:
            info = {k: record[k] for k in record if k not in relations}
            info['event_idx'] = idx
            map_idx, event_name = _map_numbers_to_det_numbers(info['event'])
            info['event_name'] = event_name
            info['inference_relation'] = r

            # if we have the ground truth annotations
            if r in record and len(record[r]) > 0:
                info['inference_relation'] = r
                info['inference_text'] = []
                info['inference_text_name'] = []
                for inference in record[r]:
                    info['inference_text'].append(inference)
                    map_idx, inference = _map_numbers_to_det_numbers(inference, map_idx)
                    info['inference_text_name'].append(inference)

            info['person2name'] = map_idx
            info['name2person'] = {v: k for k, v in map_idx.items()}
            to_return.append(info)

    return to_return

def read_and_parse_generation_json(input_file: str):
    with open(input_file) as f:
        records = json.load(f)
        records = _encode_generation_records(records)
        return records