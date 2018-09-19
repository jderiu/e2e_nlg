import numpy as np
import os
import json

import _pickle as cPickle

from typing import Dict

PREPROCESSED_DATA = 'preprocessed_data'


def convert2indices(data, alphabet, dummy_word_idx, unk_word_idx, max_sent_length=140):
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence in data:
        ex = np.ones(max_sent_length)*dummy_word_idx
        max_len = max(len(sentence), max_len)
        if len(sentence) > max_sent_length:
            sentence = sentence[:max_sent_length]
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, unk_word_idx)
            ex[i] = idx
            if idx == unk_word_idx:
                unknown_words += 1
            else:
                known_words += 1
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('float32')
    return data_idx


def _load_delex_data(data_path: str, orig_fname:str) -> Dict:
    data_file = os.path.join(data_path, orig_fname.replace('csv', 'json'))
    dfile = open(data_file, 'rt', encoding='utf-8')
    data_set = json.load(dfile)
    dfile.close()

    return data_set


def _load_vectorized_data(data_path: str, orig_fname: str) -> Dict:
    data_file = os.path.join(data_path, orig_fname.replace('csv', 'pkl'))
    dfile = open(data_file, 'rb')
    data_set = cPickle.load(dfile)
    dfile.close()

    return data_set


def _save_delex_data(results, data_path, fname, tag):
    odir = os.path.join(data_path, PREPROCESSED_DATA)
    if not os.path.exists(odir):
        os.mkdir(odir)

    odir = os.path.join(data_path, PREPROCESSED_DATA, tag)
    if not os.path.exists(odir):
        os.mkdir(odir)

    ofname = os.path.join(odir, '{}.json'.format(fname.split('.')[0]))

    ofile = open(ofname, 'wt', encoding='utf-8')

    json.dump(results, ofile)


def _save_vectorized_data(results, data_path, fname, tag):
    odir = os.path.join(data_path, PREPROCESSED_DATA)
    if not os.path.exists(odir):
        os.mkdir(odir)

    odir = os.path.join(data_path, PREPROCESSED_DATA, tag)
    if not os.path.exists(odir):
        os.mkdir(odir)

    ofname = os.path.join(odir, '{}.pkl'.format(fname.split('.')[0]))

    ofile = open(ofname, 'wb')

    cPickle.dump(results, ofile, protocol=4)




