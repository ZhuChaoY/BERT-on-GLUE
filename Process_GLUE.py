import os
import pickle
import numpy as np
import Tokenization as tkz


task_dict = \
    {'CoLA': {'label_list': ['0', '1'],
              'length': 4, 'idx_a': 3, 'idx_b': None, 'idx_label': 1},
     'MNLI': {'label_list': ['entailment', 'neutral', 'contradiction'],
              'length': 11, 'idx_a': 8, 'idx_b': 9, 'idx_label': -1},
     'MRPC': {'label_list': ['0', '1'],
              'length': 5, 'idx_a': 3, 'idx_b': 4, 'idx_label': 0},
     'QNLI': {'label_list': ['not_entailment', 'entailment'],
              'length': 4, 'idx_a': 1, 'idx_b': 2, 'idx_label': -1},
     'QQP': {'label_list': ['0', '1'],
             'length': 6, 'idx_a': 3, 'idx_b': 4, 'idx_label': -1},
     'RTE': {'label_list': ['not_entailment', 'entailment'],
             'length': 4, 'idx_a': 1, 'idx_b': 2, 'idx_label': -1},
     'SST-2': {'label_list': ['0', '1'],
               'length': 2, 'idx_a': 0, 'idx_b': None, 'idx_label': -1},
     'STS-B': {'label_list': [0.0],
               'length': 10, 'idx_a': 7, 'idx_b': 8, 'idx_label': -1},
     'WNLI': {'label_list': ['0', '1'],
              'length': 4, 'idx_a': 1, 'idx_b': 2, 'idx_label': -1}}


def Process_GLUE(dataset, len_d):
    d = task_dict[dataset]          
    p = 'GLUE/{}/_inputs-{}.data'.format(dataset, len_d)
    if os.path.exists(p):
        with open(p, 'rb') as file:
            inputs = pickle.load(file)
    else:
        tokenizer = tkz.FullTokenizer('Pretrained BERT/vocab.txt')  
        inputs = {}
        for key in ['train', 'dev']:
            inputs[key] = glue_tasks(dataset, key, len_d, tokenizer, d)
        with open(p, 'wb') as file:
            pickle.dump(inputs, file)
            
    return inputs, len(d['label_list'])


def glue_tasks(dataset, key, len_d, tokenizer, d):    
    with open('GLUE/{}/{}.txt'.format(dataset, key),
              encoding = 'mac_roman') as file:
        lines = file.readlines()
        
    inputs = []
    for line in lines[1: ]:
        line = line.split('\t')
        if len(line) != d['length']:
            continue
        text_a = tkz.convert_to_unicode(line[d['idx_a']])
        if d['idx_b']:
            text_b = tkz.convert_to_unicode(line[d['idx_b']])
        else:
            text_b = None
        label = tkz.convert_to_unicode(line[d['idx_label']].strip('\n'))
        if dataset != 'STS-B':
            if label in d['label_list']:
                inputs.append(convert_to_input(text_a, text_b, label, len_d,
                                               tokenizer, d['label_list']))
        else:
            label = float(label)
            if label >= 0.0 and label <= 5.0:
                inputs.append(convert_to_input(text_a, text_b, label, len_d,
                                               tokenizer, None))
            
    return inputs
    

def convert_to_input(text_a, text_b, label, len_d, tokenizer, label_list):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)
    if tokens_b:
        while True:
            if len(tokens_a) + len(tokens_b) <= len_d - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    else:
        if len(tokens_a) > len_d - 2:
            tokens_a = tokens_a[0: (len_d - 2)]
        
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment = [0] * (len(tokens_a) + 2)
    if tokens_b:
        tokens += (tokens_b + ['[SEP]'])
        segment += [1] * (len(tokens_b) + 1)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(ids)
    while len(ids) < len_d:
        ids.append(0)
        mask.append(0)
        segment.append(0)
    ids = np.array(ids)
    mask = np.array(mask)
    segment = np.array(segment)
        
    if label_list:
        label_map = dict(zip(label_list, range(len(label_list))))
        return ids, mask, segment, label_map[label]
    else:
        return ids, mask, segment, label


#a, b = Process_GLUE('MRPC', 128)