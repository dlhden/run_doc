from __future__ import absolute_import

import re
import os

import tqdm
import json
import pickle

import modeling
import optimization

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import collections
import tokenization
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

bert_model_dir = r'F:\NLP\BERT\bert\cased_L-12_H-768_A-12'
train_file = r'train_annotated.json'
test_file = r'dev.json'

do_lower_case = True

iterations_per_loop = 1000

train_batch_size = 2
num_train_epochs = 2
warmup_proportion = 0
learning_rate = 3e-5
predict_batch_size = 1

max_sents = 32
p = 0.4

save_checkpoints_steps = int(3000 / train_batch_size)

do_train = True
do_predict = True
do_eval = False

bert_config_file = os.path.join(bert_model_dir, 'bert_config.json')

np.random.seed(1)
master = None
num_tpu_cores = 8
dim = 768
ner_emb = 30
coref_emb = 30
sent_dist_emb = 20
dim_sent = 30

coref_maxlen = 60

sent_rel_max = 100
token_max_len = 300
max_seq_len = 511

dim_2 = 60
dim_conv = [500, 125, 105, 50]

dim_entity = 512
conv_size = c_s = 3

with open('rel_to_ind.json') as f:
    rel_to_ind = json.load(f)
with open('ind_to_rel.json') as f:
    ind_to_rel = json.load(f)
with open('type_to_ind.json') as f:
    type_to_ind = json.load(f)
with open('ind_to_type.json') as f:
    ind_to_type = json.load(f)
with open('latin_dict.json') as f:
    latin_dict = json.load(f)

vocab_file = os.path.join(bert_model_dir, 'vocab.txt')

init_checkpoint = os.path.join(bert_model_dir, 'bert_model.ckpt')
use_tpu = False
use_one_hot_embeddings = False

output_file = r'total_bert_output.pickle'
output_dir = r'./output_multi_test'

filter_label_list = ["None"]

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)


def create_vocab_dict(words):
    from collections import Counter
    word_to_count = Counter(words)
    word_and_count = sorted([(w, v) for w, v in word_to_count.items()], key=lambda x: x[1], reverse=True)
    word_to_ind = {w: i for i, (w, _) in enumerate(word_and_count)}
    ind_to_word = {i: w for i, (w, _) in enumerate(word_and_count)}
    return word_to_ind, ind_to_word


class InputExample():
    def __init__(self, index, title, sent_coref_id, sent_ner_id, sent_org_token, rels=None):
        self.index = index
        self.title = title
        self.sent_coref_id = sent_coref_id
        self.sent_ner_id = sent_ner_id
        self.sent_org_token = sent_org_token
        self.rels = rels

    def __repr__(self):
        return 'index :%d, title :%s, rel_n :%d, token_n :%d' % \
               (self.index, self.title, len(self.rels), len(self.sent_org_token))


class InputFeature():
    def __init__(self, index,
                 title,
                 tokens,
                 input_ids,
                 input_mask,
                 input_type_ids,
                 h_pos_nums=None,
                 t_pos_nums=None,
                 h_pos_stoppoint=None,
                 t_pos_stoppoint=None,
                 label_indexs=None,
                 label_stoppoint=None,
                 h_type_indexs=None,
                 t_type_indexs=None,
                 h_sent_pos=None,
                 t_sent_pos=None,
                 sent_coref_id=None,
                 sent_ner_id=None,
                 sent_dist=None,
                 orig_to_tok_map=None,
                 rel_n=None,
                 token_n=None):
        self.index = index
        self.title = title
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

        self.h_pos_nums = h_pos_nums
        self.t_pos_nums = t_pos_nums
        self.h_pos_stoppoint = h_pos_stoppoint
        self.t_pos_stoppoint = t_pos_stoppoint
        self.label_indexs = label_indexs
        self.label_stoppoint = label_stoppoint
        self.h_type_indexs = h_type_indexs
        self.t_type_indexs = t_type_indexs
        self.h_sent_pos = h_sent_pos
        self.t_sent_pos = t_sent_pos
        self.sent_coref_id = sent_coref_id
        self.sent_ner_id = sent_ner_id
        self.sent_dist = sent_dist
        self.orig_to_tok_map = orig_to_tok_map
        self.rel_n = rel_n

        self.token_n = token_n

    def __repr__(self):
        return 'index :%d, title :%s, token_n :%d' % (self.index, self.title, len(self.tokens))


def RawPos2TokenPos(raw_pos, tok_map, tok_len):
    start = tok_map[raw_pos[0]]
    if raw_pos[1] < len(tok_map):
        end = tok_map[raw_pos[1]]
    else:
        end = tok_len - 1
    return start, end


def pos_to_mat(rels):
    seq_matrix = np.zeros((token_max_len, token_max_len, max_rel_n))
    for rel in rels:
        label_index = rel_to_ind[rel['label']]
        seq_matrix[rel['em1_pos'][0]:rel['em1_pos'][1], rel['em2_pos'][0]:rel['em2_pos'][1], label_index] = 1
    ones_label = (np.sum(seq_matrix, axis=2, keepdims=True) == 0).astype(np.float)
    seq_matrix = np.concatenate([seq_matrix, ones_label], axis=-1)
    return seq_matrix


def read_examples(file, is_train=True, is_ttt=False, p=0.0):
    print('***** begin to read data *****')
    np.random.seed(12345)
    raw_datas = []
    with open(file, 'r') as f:
        data = json.load(f)
        for temp in tqdm.tqdm(data):
            if sum(len(st) for st in temp['sents']) < token_max_len:
                raw_datas.append(temp)
    print('file %s readding complete, processing raw data.' % file)

    examples = []
    label_names = []
    type_names = []

    def compute_sent_dist(en1, en2):
        return min(abs(m1['sent_id'] - m2['sent_id']) for m1 in en1 for m2 in en2)

    for index, temp in tqdm.tqdm(enumerate(raw_datas)):
        title = temp['title']
        sent_start_pos = np.cumsum([0] + [len(st) for st in temp['sents']])
        sent_org_token = [t for st in temp['sents'] for t in st]
        sent_coref_id = np.zeros((token_max_len,))
        sent_ner_id = np.zeros((token_max_len,))
        for idx, en in enumerate(temp['vertexSet']):
            for ment in en:
                em_pos = np.array(ment['pos']) + sent_start_pos[ment['sent_id']]
                sent_coref_id[em_pos[0]:em_pos[1]] = idx + 1
                sent_ner_id[em_pos[0]:em_pos[1]] = type_to_ind[ment['type']] + 1
        sent_coref_id = sent_coref_id.astype(np.int).tolist()
        sent_ner_id = sent_ner_id.astype(np.int).tolist()

        rels = []
        if is_train:
            entity_n = len(temp['vertexSet'])
            h_t_dict = {}
            for rel in temp['labels']:
                label_names.append(rel['r'])
                if (rel['h'], rel['t']) not in h_t_dict.keys():
                    h_t_dict[(rel['h'], rel['t'])] = [rel['r']]
                else:
                    h_t_dict[(rel['h'], rel['t'])].append(rel['r'])

            neg_set = []
            for i in range(entity_n):
                for j in range(entity_n):
                    if i != j and p > np.random.rand():
                        neg_set.append((i, j))
            if len(neg_set) > 400:
                np.random.shuffle(neg_set)
            neg_set = set(neg_set[:400])

            for h_i in range(entity_n):
                for t_i in range(entity_n):
                    if (h_i, t_i) in h_t_dict.keys() or (h_i, t_i) in neg_set:
                        h_pos = []
                        h_count = Counter([m['type'] for m in temp['vertexSet'][h_i]])
                        h_count_max = max(h_count.values())
                        h_type = [w for w in h_count if h_count[w] == h_count_max][0]
                        type_names.append(h_type)
                        h_names = []
                        h_count = Counter([m['sent_id'] for m in temp['vertexSet'][h_i]])
                        h_count_max = max(h_count.values())
                        h_sent_id = [w for w in h_count if h_count[w] == h_count_max][0]
                        h_sent_pos = (sent_start_pos[h_sent_id], sent_start_pos[h_sent_id + 1])

                        t_pos = []
                        t_count = Counter([m['type'] for m in temp['vertexSet'][t_i]])
                        t_count_max = max(t_count.values())
                        t_type = [w for w in t_count if t_count[w] == t_count_max][0]
                        t_names = []
                        type_names.append(t_type)
                        t_count = Counter([m['sent_id'] for m in temp['vertexSet'][t_i]])
                        t_count_max = max(t_count.values())
                        t_sent_id = [w for w in t_count if t_count[w] == t_count_max][0]
                        t_sent_pos = (sent_start_pos[t_sent_id], sent_start_pos[t_sent_id + 1])

                        sent_dist = compute_sent_dist(temp['vertexSet'][h_i], temp['vertexSet'][t_i])
                        for h in temp['vertexSet'][h_i]:
                            for t in temp['vertexSet'][t_i]:
                                em1Text = h['name']
                                em1_pos = np.array(h['pos']) + sent_start_pos[h['sent_id']]
                                em1_pos_num = list(range(em1_pos[0], em1_pos[1]))
                                h_pos.extend(em1_pos_num)
                                h_names.append(em1Text)

                                em2Text = t['name']
                                em2_pos = np.array(t['pos']) + sent_start_pos[t['sent_id']]
                                em2_pos_num = list(range(em2_pos[0], em2_pos[1]))
                                t_pos.extend(em2_pos_num)
                                t_names.append(em2Text)
                        labels = h_t_dict[(h_i, t_i)] if (h_i, t_i) in h_t_dict.keys() else ['None']
                        if labels[0] == 'None':
                            label_names.append('None')
                        temp_rel = {'h_pos': h_pos,
                                    'h_type': h_type,
                                    'h_names': h_names,
                                    'h_sent_pos': h_sent_pos,
                                    't_pos': t_pos,
                                    't_type': t_type,
                                    't_names': t_names,
                                    't_sent_pos': t_sent_pos,
                                    'sent_dist': sent_dist,
                                    'labels': labels}
                        rels.append(temp_rel)
        elif is_ttt:
            h_t_dict = {}
            for rel in temp['labels']:
                label_names.append(rel['r'])
                if (rel['h'], rel['t']) not in h_t_dict.keys():
                    h_t_dict[(rel['h'], rel['t'])] = [rel['r']]
                else:
                    h_t_dict[(rel['h'], rel['t'])].append(rel['r'])
            entity_n = len(temp['vertexSet'])
            for h_i in range(entity_n):
                for t_i in range(entity_n):
                    h_pos = []
                    h_count = Counter([m['type'] for m in temp['vertexSet'][h_i]])
                    h_count_max = max(h_count.values())
                    h_type = [w for w in h_count if h_count[w] == h_count_max][0]
                    type_names.append(h_type)
                    h_names = []
                    h_count = Counter([m['sent_id'] for m in temp['vertexSet'][h_i]])
                    h_count_max = max(h_count.values())
                    h_sent_id = [w for w in h_count if h_count[w] == h_count_max][0]
                    h_sent_pos = (sent_start_pos[h_sent_id], sent_start_pos[h_sent_id + 1])

                    t_pos = []
                    t_count = Counter([m['type'] for m in temp['vertexSet'][t_i]])
                    t_count_max = max(t_count.values())
                    t_type = [w for w in t_count if t_count[w] == t_count_max][0]
                    t_names = []
                    type_names.append(t_type)
                    t_count = Counter([m['sent_id'] for m in temp['vertexSet'][t_i]])
                    t_count_max = max(t_count.values())
                    t_sent_id = [w for w in t_count if t_count[w] == t_count_max][0]
                    t_sent_pos = (sent_start_pos[t_sent_id], sent_start_pos[t_sent_id + 1])

                    sent_dist = compute_sent_dist(temp['vertexSet'][h_i], temp['vertexSet'][t_i])
                    for h in temp['vertexSet'][h_i]:
                        for t in temp['vertexSet'][t_i]:
                            em1Text = h['name']
                            em1_pos = np.array(h['pos']) + sent_start_pos[h['sent_id']]
                            em1_pos_num = list(range(em1_pos[0], em1_pos[1]))
                            h_pos.extend(em1_pos_num)
                            h_names.append(em1Text)

                            em2Text = t['name']
                            em2_pos = np.array(t['pos']) + sent_start_pos[t['sent_id']]
                            em2_pos_num = list(range(em2_pos[0], em2_pos[1]))
                            t_pos.extend(em2_pos_num)
                            t_names.append(em2Text)
                    labels = h_t_dict[(h_i, t_i)] if (h_i, t_i) in h_t_dict.keys() else ['None']
                    temp_rel = {'h_pos': h_pos,
                                'h_type': h_type,
                                'h_names': h_names,
                                'h_sent_pos': h_sent_pos,
                                't_pos': t_pos,
                                't_type': t_type,
                                't_names': t_names,
                                't_sent_pos': t_sent_pos,
                                'sent_dist': sent_dist,
                                'labels': labels}
                    rels.append(temp_rel)
        else:
            entity_n = len(temp['vertexSet'])
            sent_dist = compute_sent_dist(temp['vertexSet'][h_i], temp['vertexSet'][t_i])
            for h_i in range(entity_n):
                for t_i in range(entity_n):
                    for h in temp['vertexSet'][h_i]:
                        for t in temp['vertexSet'][t_i]:
                            em1Text = h['name']
                            em1_pos = np.array(h['pos']) + sent_start_pos[h['sent_id']]
                            em1_pos_num = list(range(em1_pos[0], em1_pos[1]))
                            h_sent_pos = (sent_start_pos[h['sent_id']], sent_start_pos[h['sent_id'] + 1])
                            em2Text = t['name']
                            em2_pos = np.array(t['pos']) + sent_start_pos[t['sent_id']]
                            em2_pos_num = list(range(em2_pos[0], em2_pos[1]))
                            t_sent_pos = (sent_start_pos[t['sent_id']], sent_start_pos[t['sent_id'] + 1])
                            temp_rel = {'h_pos': em1_pos_num,
                                        'h_type': h['type'],
                                        'h_names': [em1Text],
                                        'h_sent_pos': h_sent_pos,
                                        't_pos': em2_pos_num,
                                        't_type': t['type'],
                                        't_names': [em2Text],
                                        't_sent_pos': t_sent_pos,
                                        'sent_dist': sent_dist,
                                        'labels': []}
                            rels.append(temp_rel)
        if len(rels) > 0:
            examples.append(InputExample(index=index,
                                         title=title,
                                         sent_org_token=sent_org_token,
                                         sent_coref_id=sent_coref_id,
                                         sent_ner_id=sent_ner_id,
                                         rels=rels))
    print('***** raw data processing complete *****')
    return examples


def convert_examples_to_features(examples, max_seq_len, output_fn, is_train=True):
    print('***** converting examples to features, examples_n : %d *****' % len(examples))
    for example in tqdm.tqdm(examples):
        index = example.index
        title = example.title
        sent_coref_id = example.sent_coref_id
        sent_ner_id = example.sent_ner_id
        sent_org_token = example.sent_org_token
        rels = example.rels
        tokens = []
        orig_to_tok_map = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sent_org_token:
            orig_to_tok_map.append(len(tokens))
            temp_token = tokenizer.tokenize(token)
            tokens.extend(temp_token)
            input_type_ids.extend([0] * len(temp_token))
        tok_len = len(tokens) + 1
        tokens = tokens[:max_seq_len - 1]
        input_type_ids = input_type_ids[:max_seq_len - 1]
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(input_type_ids) == max_seq_len

        h_pos_stoppoint = []
        h_p = 0
        h_pos_nums = []
        t_pos_stoppoint = []
        t_p = 0
        t_pos_nums = []
        l_p = 0
        label_stoppoint = []
        label_indexs = []
        sent_dist = []
        h_type_indexs = []
        t_type_indexs = []
        h_sent_pos = []
        t_sent_pos = []

        for rel in rels:
            h_pos = rel['h_pos']
            h_pos_nums.extend(h_pos)
            h_p += len(h_pos)
            h_pos_stoppoint.append(h_p)
            t_pos = rel['t_pos']
            t_pos_nums.extend(t_pos)
            t_p += len(t_pos)
            t_pos_stoppoint.append(t_p)
            h_type = rel['h_type']
            t_type = rel['t_type']
            h_sent_pos.extend(rel['h_sent_pos'])
            t_sent_pos.extend(rel['t_sent_pos'])
            if is_train:
                labels = rel['labels']
                for label in labels:
                    label_indexs.append(rel_to_ind[label])
                l_p += len(labels)
                label_stoppoint.append(l_p)
            h_type_indexs.append(type_to_ind[h_type])
            t_type_indexs.append(type_to_ind[t_type])
            sent_dist.append(rel['sent_dist'])

        rel_n = [len(rels)]

        if index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("index: %s" % (example.index))
            tf.logging.info("title: %s" % (example.title))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            if is_train:
                tf.logging.info("is train")
                tf.logging.info('relation : %s' % ','.join(labels))

        orig_to_tok_map = [i for i in orig_to_tok_map if i < max_seq_len]
        orig_to_tok_map = orig_to_tok_map[:token_max_len]
        token_n = len(orig_to_tok_map)
        while len(orig_to_tok_map) < token_max_len:
            orig_to_tok_map.append(0)
        feature = InputFeature(index=index,
                               title=title,
                               tokens=tokens,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               input_type_ids=input_type_ids,
                               h_pos_nums=h_pos_nums,
                               h_pos_stoppoint=h_pos_stoppoint,
                               t_pos_nums=t_pos_nums,
                               t_pos_stoppoint=t_pos_stoppoint,
                               label_indexs=label_indexs,
                               label_stoppoint=label_stoppoint,
                               h_type_indexs=h_type_indexs,
                               t_type_indexs=t_type_indexs,
                               h_sent_pos=h_sent_pos,
                               t_sent_pos=t_sent_pos,
                               sent_coref_id=sent_coref_id,
                               sent_ner_id=sent_ner_id,
                               sent_dist=sent_dist,
                               orig_to_tok_map=orig_to_tok_map,
                               rel_n=rel_n,
                               token_n=token_n)
        output_fn(feature)
    print('***** feature transformation complete *****')


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training, max_label_n):
        self.filename = filename
        self.is_training = is_training
        self.max_label_n = max_label_n
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        max_label_n = self.max_label_n
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["index"] = create_int_feature([feature.index])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["input_type_ids"] = create_int_feature(feature.input_type_ids)
        features["token_n"] = create_int_feature([feature.token_n])
        features["orig_to_tok_map"] = create_int_feature(feature.orig_to_tok_map)

        features['h_pos_nums'] = create_int_feature(feature.h_pos_nums)
        features['h_pos_stoppoint'] = create_int_feature(feature.h_pos_stoppoint)
        features['t_pos_nums'] = create_int_feature(feature.t_pos_nums)
        features['t_pos_stoppoint'] = create_int_feature(feature.t_pos_stoppoint)
        features['h_type_indexs'] = create_int_feature(feature.h_type_indexs)
        features['t_type_indexs'] = create_int_feature(feature.t_type_indexs)
        features['h_sent_pos'] = create_int_feature(feature.h_sent_pos)
        features['t_sent_pos'] = create_int_feature(feature.t_sent_pos)
        features['sent_coref_id'] = create_int_feature(feature.sent_coref_id)
        features['sent_ner_id'] = create_int_feature(feature.sent_ner_id)
        features['sent_dist'] = create_int_feature(feature.sent_dist)
        features['rel_n'] = create_int_feature(feature.rel_n)

        if self.is_training:
            features['label_indexs'] = create_int_feature(feature.label_indexs)
            features['label_stoppoint'] = create_int_feature(feature.label_stoppoint)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "index": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "input_type_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_n": tf.FixedLenFeature([], tf.int64),
        "orig_to_tok_map": tf.FixedLenFeature([token_max_len], tf.int64),
        "h_pos_nums": tf.VarLenFeature(tf.int64),
        "h_pos_stoppoint": tf.VarLenFeature(tf.int64),
        "t_pos_nums": tf.VarLenFeature(tf.int64),
        "t_pos_stoppoint": tf.VarLenFeature(tf.int64),
        "h_type_indexs": tf.VarLenFeature(tf.int64),
        "t_type_indexs": tf.VarLenFeature(tf.int64),
        "h_sent_pos": tf.VarLenFeature(tf.int64),
        "t_sent_pos": tf.VarLenFeature(tf.int64),
        "sent_coref_id": tf.FixedLenFeature([token_max_len], tf.int64),
        "sent_ner_id": tf.FixedLenFeature([token_max_len], tf.int64),
        "sent_dist": tf.VarLenFeature(tf.int64),
        "rel_n": tf.FixedLenFeature([], tf.int64),
    }
    if is_training:
        name_to_features["label_indexs"] = tf.VarLenFeature(tf.int64)
        name_to_features["label_stoppoint"] = tf.VarLenFeature(tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # batch_size = train_batch_size
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 h_pos_nums, h_pos_stoppoint,
                 t_pos_nums, t_pos_stoppoint,
                 h_type_indexs, t_type_indexs,
                 h_sent_pos, t_sent_pos,
                 sent_coref_id, sent_ner_id,
                 sent_dist,
                 rel_n,
                 orig_to_tok_map,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
    token_length = token_max_len
    token_n = tf.reduce_sum(tf.cast(tf.cast(orig_to_tok_map, tf.bool), tf.int32), axis=1)

    # m.shape (batch_size,tok_max_len,hidden_size)
    cond = lambda i, j, ir, m: i < batch_size
    init_ind_value = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    zero = tf.constant(0, dtype=tf.int32)

    def body(i, j, ir, m):
        start = orig_to_tok_map[i, j]
        end = tf.cond(j + 1 < token_n[i], lambda: orig_to_tok_map[i, j + 1], lambda: seq_length)
        # value = tf.reduce_max(tf.gather(final_hidden[i], tf.range(start, end)), 0)
        indice_i = tf.fill((hidden_size, 1), tf.cast(i, tf.float32))
        indice_j = tf.fill((hidden_size, 1), tf.cast(j, tf.float32))
        m_ind = tf.concat([indice_i,
                           indice_j,
                           tf.cast(tf.expand_dims(tf.range(0, hidden_size), 1), tf.float32),
                           tf.expand_dims(tf.reduce_max(tf.gather(final_hidden[i], tf.range(start, end)), 0), 1)],
                          axis=1)
        m = m.write(ir, m_ind)
        return tf.cond(j < token_n[i], lambda: (i, j + 1, ir + 1, m), lambda: (i + 1, j - j, ir + 1, m))

    _, _, _, m_ind_value = tf.while_loop(cond, body, [zero, zero, zero, init_ind_value])
    m_ind_value = m_ind_value.stack()
    m_ind_value = tf.reshape(m_ind_value, [-1, 4])
    indice = tf.cast(m_ind_value[:, :3], tf.int32)
    value = m_ind_value[:, -1]
    final_token_hidden = tf.scatter_nd(indice, value, [batch_size, token_length, hidden_size])
    coref_emb_mat = tf.Variable(tf.truncated_normal([coref_maxlen, coref_emb], stddev=0.02, dtype=tf.float32))
    ner_emb_mat = tf.Variable(tf.truncated_normal([len(rel_to_ind) + 1, ner_emb], stddev=0.02, dtype=tf.float32))

    seq_emb = keras.layers.Bidirectional(
        keras.layers.CuDNNGRU(
            dim_conv[3],
            return_sequences=True))(tf.concat([tf.gather(coref_emb_mat, sent_coref_id),
                                               tf.gather(ner_emb_mat, sent_ner_id)], axis=2))


    head_weights = tf.Variable(tf.truncated_normal([type_n,
                                                    hidden_size,
                                                    dim_conv[2]], stddev=0.02, dtype=tf.float32))
    head_bias = tf.Variable(tf.zeros([type_n, dim_conv[2]], dtype=tf.float32))
    tail_weights = tf.Variable(tf.truncated_normal([type_n,
                                                    hidden_size,
                                                    dim_conv[2]], stddev=0.02, dtype=tf.float32))
    tail_bias = tf.Variable(tf.zeros([type_n, dim_conv[2]], dtype=tf.float32))
    h_sent_weights = tf.Variable(tf.truncated_normal([hidden_size,
                                                      dim_conv[2]], stddev=0.02, dtype=tf.float32))
    h_sent_bias = tf.Variable(tf.zeros([dim_conv[2]], dtype=tf.float32))
    t_sent_weights = tf.Variable(tf.truncated_normal([hidden_size,
                                                      dim_conv[2]], stddev=0.02, dtype=tf.float32))
    t_sent_bias = tf.Variable(tf.zeros([dim_conv[2]], dtype=tf.float32))

    sent_dist_weights = tf.Variable(tf.truncated_normal([max_sents, sent_dist_emb], stddev=0.02, dtype=tf.float32))

    dim_concat = dim_conv[2] + dim_conv[3] * 4 + sent_dist_emb
    final_weights = tf.Variable(tf.truncated_normal([dim_concat, max_rel_n], stddev=0.02, dtype=tf.float32))
    final_bias = tf.Variable(tf.zeros([max_rel_n, ], dtype=tf.float32))

    cond = lambda batch_i, i, j1, j2, ir, m: batch_i < batch_size

    m_init = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def body(batch_i, i, jh, jt, ir, m):
        h_pos_st = h_pos_stoppoint[batch_i, i]
        t_pos_st = t_pos_stoppoint[batch_i, i]

        h_temp = tf.reduce_max(
            tf.gather(
                final_token_hidden[batch_i],
                tf.gather(
                    h_pos_nums[batch_i],
                    tf.range(jh, h_pos_st))), axis=0,
            keepdims=True)

        t_temp = tf.reduce_max(
            tf.gather(
                final_token_hidden[batch_i],
                tf.gather(
                    t_pos_nums[batch_i],
                    tf.range(jt, t_pos_st))), axis=0,
            keepdims=True)

        h_type_temp = tf.reduce_max(
            tf.gather(
                seq_emb[batch_i],
                tf.gather(
                    h_pos_nums[batch_i],
                    tf.range(jh, h_pos_st))), axis=0,
            keepdims=True)

        t_type_temp = tf.reduce_max(
            tf.gather(
                seq_emb[batch_i],
                tf.gather(
                    t_pos_nums[batch_i],
                    tf.range(jt, t_pos_st))), axis=0,
            keepdims=True)

        h_temp = tf.nn.selu(
            tf.nn.bias_add(
                tf.matmul(h_temp, tf.gather(head_weights, h_type_indexs[batch_i, i])),
                tf.gather(head_bias, h_type_indexs[batch_i, i])))
        t_temp = tf.nn.selu(
            tf.nn.bias_add(
                tf.matmul(t_temp, tf.gather(tail_weights, t_type_indexs[batch_i, i])),
                tf.gather(tail_bias, t_type_indexs[batch_i, i])))
        sub_temp = tf.nn.selu(h_temp - t_temp)

        h_sent_temp = tf.nn.selu(tf.nn.bias_add(tf.matmul(tf.reduce_max(tf.gather(final_token_hidden[batch_i],
                                                                                  tf.range(h_sent_pos[batch_i, 2 * i],
                                                                                           h_sent_pos[
                                                                                               batch_i, 2 * i + 1])),
                                                                        axis=0, keepdims=True), h_sent_weights),
                                                h_sent_bias))

        t_sent_temp = tf.nn.selu(tf.nn.bias_add(tf.matmul(tf.reduce_max(tf.gather(final_token_hidden[batch_i],
                                                                                  tf.range(t_sent_pos[batch_i, 2 * i],
                                                                                           t_sent_pos[
                                                                                               batch_i, 2 * i + 1])),
                                                                        axis=0, keepdims=True), t_sent_weights),
                                                t_sent_bias))
        sent_temp = tf.nn.selu(h_sent_temp - t_sent_temp)

        sent_dist_temp = tf.expand_dims(tf.gather(sent_dist_weights, sent_dist[batch_i, i]), 0)

        m_temp = tf.nn.bias_add(
            tf.matmul(
                tf.concat([sub_temp + sent_temp , sent_dist_temp, h_type_temp, t_type_temp],
                          axis=1),
                final_weights),
            final_bias)

        m = m.write(ir, m_temp[0, :])

        return tf.cond(i >= rel_n[batch_i] - 1,
                       lambda: (batch_i + 1, 0, 0, 0, ir + 1, m),
                       lambda: (batch_i, i + 1, h_pos_st, t_pos_st, ir + 1, m))

    _, _, _, _, _, vec = tf.while_loop(cond, body, [zero, zero, zero, zero, zero, m_init])
    logits = vec.stack()

    return logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        index = features["index"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["input_type_ids"]
        orig_to_tok_map = features["orig_to_tok_map"]

        h_pos_nums = tf.sparse.to_dense(features["h_pos_nums"])
        h_pos_stoppoint = tf.sparse.to_dense(features["h_pos_stoppoint"])
        t_pos_nums = tf.sparse.to_dense(features["t_pos_nums"])
        t_pos_stoppoint = tf.sparse.to_dense(features["t_pos_stoppoint"])
        h_type_indexs = tf.sparse.to_dense(features["h_type_indexs"])
        t_type_indexs = tf.sparse.to_dense(features["t_type_indexs"])
        h_sent_pos = tf.sparse.to_dense(features["h_sent_pos"])
        t_sent_pos = tf.sparse.to_dense(features["t_sent_pos"])
        sent_coref_id = features["sent_coref_id"]
        sent_ner_id = features["sent_ner_id"]
        sent_dist = tf.sparse.to_dense(features["sent_dist"])
        rel_n = features["rel_n"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        logits = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            h_pos_nums=h_pos_nums,
            h_pos_stoppoint=h_pos_stoppoint,
            t_pos_nums=t_pos_nums,
            t_pos_stoppoint=t_pos_stoppoint,
            h_type_indexs=h_type_indexs,
            t_type_indexs=t_type_indexs,
            h_sent_pos=h_sent_pos,
            t_sent_pos=t_sent_pos,
            sent_coref_id=sent_coref_id,
            sent_ner_id=sent_ner_id,
            sent_dist=sent_dist,
            rel_n=rel_n,
            orig_to_tok_map=orig_to_tok_map,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        def compute_loss(logits, rel_labels):
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)
            mul = rel_labels * log_probs * (1 - probs) ** 2
            mul = tf.where(tf.equal(rel_labels, 0.0), mul - 1e20, mul)
            loss = -tf.reduce_mean(
                tf.reduce_max(mul, axis=-1))
            return loss

        def compute_metric_intrain(logits, rel_labels):

            submit_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.arg_max(logits, 1), 0), tf.float32))
            re_len = tf.reduce_sum(tf.cast(tf.equal(rel_labels[:, 0], 0.0), tf.float32))

            true_label_bool = tf.equal(rel_labels[:, 0], 0.0)
            true_logits = tf.boolean_mask(logits, true_label_bool)
            true_probs = tf.nn.softmax(true_logits, axis=-1)
            true_preds = tf.one_hot(tf.arg_max(true_probs, 1), depth=max_rel_n, dtype=tf.float32)
            true_rellabels = tf.boolean_mask(rel_labels, true_label_bool)

            correct_re = tf.cast(tf.reduce_sum(tf.count_nonzero(true_preds * true_rellabels, axis=1)), tf.float32)

            precision = correct_re / submit_len
            recall = correct_re / re_len
            f1 = 2 * precision * recall / (precision + recall)

            precision_score = tf.reduce_mean(precision)
            recall_score = tf.reduce_mean(recall)
            f1s = tf.reduce_mean(f1)
            return f1s, precision_score, recall_score

        def my_metric_fn(labels, predictions):
            label_args = tf.arg_max(labels, 1)
            labels = tf.cast(labels, tf.float32)
            pred = tf.arg_max(predictions, 1)
            probs = tf.nn.softmax(logits, axis=-1)
            one_hot_pred = tf.one_hot(
                pred, depth=max_rel_n, dtype=tf.float32)
            weights = tf.constant(np.ones((1, max_rel_n), dtype=tf.float32) / max_rel_n)

            return {  # 'acc': tf.metrics.accuracy(label_args, pred),
                'auc': tf.metrics.auc(labels, probs),
                'precision': tf.metrics.precision(labels, one_hot_pred, weights=weights),
                'recall': tf.metrics.recall(labels, one_hot_pred, weights=weights),
                'f1 score': tf.contrib.metrics.f1_score(labels, one_hot_pred, weights=weights)}

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()

            label_indexs = tf.sparse.to_dense(features["label_indexs"])
            label_stoppoint = tf.sparse.to_dense(features["label_stoppoint"])

            cond = lambda batch_i, i, j, ir, m: batch_i < train_batch_size
            # m_init = tf.Variable(tf.zeros((0, max_rel_n)), tf.float32)
            m_init = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

            def body(batch_i, i, j, ir, m):
                l_st = label_stoppoint[batch_i, i]
                m_new = tf.scatter_nd(
                    tf.concat(
                        [tf.zeros((l_st - j, 1), dtype=tf.int32),
                         tf.expand_dims(tf.gather(label_indexs[batch_i], tf.range(j, l_st)), 1)],
                        axis=1),
                    tf.ones((l_st - j,), dtype=tf.float32), [1, max_rel_n])
                m = m.write(ir, m_new[0, :])

                return tf.cond(i >= rel_n[batch_i] - 1,
                               lambda: (batch_i + 1, 0, 0, ir + 1, m),
                               lambda: (batch_i, i + 1, l_st, ir + 1, m))

            zero = tf.constant(0, dtype=tf.int32)
            _, _, _, _, rel_labels = tf.while_loop(cond, body, [zero, zero, zero, zero, m_init])
            rel_labels = rel_labels.stack()

            loss = compute_loss(logits, rel_labels)
            f1s, precision, recall = compute_metric_intrain(logits, rel_labels)

            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            logging_hook = tf.train.LoggingTensorHook({'loss': loss,
                                                       'f1s': f1s,
                                                       'precision': precision,
                                                       'recall': recall}, every_n_iter=10)

            def host_call_fn(gs, loss, f1s, precision, recall, lr, ce):
                gs = gs[0]
                with tf.contrib.summary.create_file_writer(
                        output_dir,
                        max_queue=iterations_per_loop).as_default():
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss[0], step=gs)
                        tf.contrib.summary.scalar('f1s', f1s[0], step=gs)
                        tf.contrib.summary.scalar('precision', precision[0], step=gs)
                        tf.contrib.summary.scalar('recall', recall[0], step=gs)
                        tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
                        tf.contrib.summary.scalar('current_epoch', ce[0], step=gs)

                        return tf.contrib.summary.all_summary_ops()

            current_epoch = (tf.cast(global_step, tf.float32) /
                             epoach_steps)

            gs_t = tf.reshape(global_step, [1])
            loss_t = tf.reshape(loss, [1])
            f1s_t = tf.reshape(f1s, [1])
            precision_t = tf.reshape(precision, [1])
            recall_t = tf.reshape(recall, [1])
            lr_t = tf.reshape(learning_rate, [1])
            ce_t = tf.reshape(current_epoch, [1])

            host_call = (
                host_call_fn,
                [gs_t, loss_t, f1s_t, precision_t, recall_t, lr_t,
                 ce_t])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks={logging_hook},
                host_call=host_call,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "index": index,
                "input_ids": input_ids,
                "logits": tf.expand_dims(logits, 0),
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            label_indexs = tf.sparse.to_dense(features["label_indexs"])
            label_stoppoint = tf.sparse.to_dense(features["label_stoppoint"])

            rel_labels = tf.reshape(tf.one_hot(label_indexs, depth=max_rel_n, dtype=tf.float16), [-1, max_rel_n])

            loss = compute_loss(logits, rel_labels)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(my_metric_fn, [rel_labels, logits]),
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN , PREDICT and EVALUATE modes are supported: %s" % (mode))

        return output_spec

    return model_fn


if __name__ is '__main__':
    train_examples = read_examples(train_file, p=p)
    max_rel_n = len(rel_to_ind)
    type_n = len(type_to_ind)

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tf.gfile.MakeDirs(output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    if do_train:
        num_train_steps = int(
            len(train_examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        epoach_steps = int(
            len(train_examples) / train_batch_size * 1)
        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)
    if do_eval:
        eval_examples = read_examples(test_file, is_train=True, p=0)
        eval_steps = int(
            len(eval_examples) / predict_batch_size * 1)
    if do_predict:
        test_examples = read_examples(test_file, is_train=False, p=0, is_ttt=True)
        test_steps = int(
            len(test_examples) / predict_batch_size * 1)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        predict_batch_size=predict_batch_size)

    if do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_tfrecord = os.path.join(output_dir, "train.tf_record")
        if not os.path.exists(train_tfrecord):
            train_writer = FeatureWriter(
                filename=train_tfrecord,
                max_label_n=max_rel_n,
                is_training=True)
            convert_examples_to_features(train_examples,
                                         max_seq_len,
                                         train_writer.process_feature,
                                         is_train=True)
            train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(input_file=train_tfrecord,
                                          seq_length=max_seq_len,
                                          is_training=True,
                                          drop_remainder=True, )

        if do_eval:
            eval_tfrecord = os.path.join(output_dir, "eval.tf_record")
            if not os.path.exists(eval_tfrecord):
                eval_writer = FeatureWriter(
                    filename=eval_tfrecord,
                    max_label_n=max_rel_n,
                    is_training=True)

                convert_examples_to_features(eval_examples,
                                             max_seq_len,
                                             eval_writer.process_feature,
                                             is_train=True)
                eval_writer.close()

            tf.logging.info("***** Running eval *****")
            tf.logging.info("  Num orig examples = %d", len(eval_examples))
            tf.logging.info("  Batch size = %d", predict_batch_size)
            tf.logging.info("  Num steps = %d", eval_steps)

            eval_input_fn = input_fn_builder(input_file=eval_tfrecord,
                                             seq_length=max_seq_len,
                                             is_training=True,
                                             drop_remainder=False)

        estimator.train(input_fn=train_input_fn,
                        max_steps=int(epoach_steps * num_train_epochs))
        tf.logging.info("***** Runing eval process *****")

        if do_eval:
            def Eval_input_fn_func():
                batch_size = predict_batch_size

                def Eval_input_fn(params):
                    params["batch_size"] = batch_size
                    return eval_input_fn(params)

                return Eval_input_fn


            Eval_input_fn = Eval_input_fn_func()

            estimator.evaluate(input_fn=Eval_input_fn,
                               steps=int(eval_steps))

    tf.logging.info('***** train process complete *****')

    if do_predict:
        test_tfrecord = os.path.join(output_dir, "test.tf_record")
        if not os.path.exists(test_tfrecord):
            test_writer = FeatureWriter(
                filename=test_tfrecord,
                max_label_n=max_rel_n,
                is_training=True)

            convert_examples_to_features(test_examples,
                                         max_seq_len,
                                         test_writer.process_feature,
                                         is_train=True)
            test_writer.close()

        tf.logging.info("***** Running predict *****")
        tf.logging.info("  Num orig examples = %d", len(test_examples))
        tf.logging.info("  Batch size = %d", predict_batch_size)
        tf.logging.info("  Num steps = %d", test_steps)

        test_input_fn = input_fn_builder(input_file=test_tfrecord,
                                         seq_length=max_seq_len,
                                         is_training=True,
                                         drop_remainder=False, )


        def Test_input_fn_func():
            batch_size = predict_batch_size

            def Test_input_fn(params):
                params["batch_size"] = batch_size
                return test_input_fn(params)

            return Test_input_fn


        Test_input_fn = Test_input_fn_func()
        # If running eval on the TPU, you will need to specify the number of
        # steps.
        output_prediction_file = os.path.join(output_dir, "predictions.json")
        test_examples_dict = {ex.index: ex for ex in test_examples}
        with open(output_prediction_file, 'w') as f:
            f.truncate()
            i_ = 0
            for result in estimator.predict(
                    Test_input_fn):
                if i_ < len(test_examples):
                    if i_ % 100 == 0:
                        tf.logging.info("Processing example: %d" % (i_))

                    index = int(result["index"])
                    input_ids = result["input_ids"]
                    logits = result["logits"]
                    prob = logits - np.max(logits, axis=-1, keepdims=True)
                    prob = np.exp(prob) / np.sum(np.exp(prob), axis=-1, keepdims=True)

                    pred = np.argmax(prob, axis=-1)
                    pred = pred.astype(np.int8)

                    data = {"index": index,
                            "pred": pred.tolist()}
                    f.write(json.dumps(data))
                    f.write('\n')
                    # json.dump(data, f)
                    i_ += 1
                else:
                    break
        print('predict complete!')
        all_results = []
        with open(output_prediction_file, 'r') as f:
            for line in f:
                temp = json.loads(line)
                all_results.append(temp)

        seq_true = []
        seq_pred = []
        for r in all_results:
            if r['index'] in test_examples_dict.keys():
                t = test_examples_dict[r['index']]
                seq_true.extend([rel_to_ind[rel['labels'][0]] for rel in t.rels])
                pred = np.array(r['pred'])
                seq_pred.extend(np.ravel(pred).tolist())
        from sklearn.metrics import f1_score

        correct_re = len([i1 for i1, i2 in zip(seq_true, seq_pred) if i1 != 0 and i1 == i2])
        submission_len = len([i2 for i2 in seq_pred if i2 != 0])
        correct_len = len([i1 for i1 in seq_true if i1 != 0])
        re_p = correct_re / submission_len
        re_r = correct_re / correct_len
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        print('f1 score : %.4f' % re_f1)
