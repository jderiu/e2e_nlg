import argparse, json, os, csv, random
import _pickle as cPickle
import numpy as np
from src.data_processing.utils import PREPROCESSED_DATA
from src.data_processing.delexicalise_data import _parse_raw_mr
from src.data_processing.vectorize_data import _vectorize_mrs
from typing import List, Dict


def _read_data(fname: str) -> List[str]:
    data_reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))
    mr_raw = []
    for row in data_reader:
        mr_raw.append(row['mr'])

    return mr_raw


def _sample_surface_feautres(preocessed_mr: List[Dict[str, str]], feature_value_idx: Dict, utt_fw_vocab: Dict, follow_sent_fw_vocab: Dict, mr_data_ontology:Dict[str, Dict[str, int]], sample_const: Dict, max_nsent=4):
    sampled_surface_features = []
    for mr in preocessed_mr:
        utt_fw_samples, follow_fw_features, form_sampled_vec = _sample_surface_features_for_mr(mr, feature_value_idx, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, sample_const, max_nsent)
        d = {
            'utt_fw': utt_fw_samples,
            'follow_fw': follow_fw_features,
            'form': form_sampled_vec,
        }

        sampled_surface_features.append(d)

    return sampled_surface_features


def _sample_surface_features_for_mr(mr: Dict[str, str], feature_value_idx: Dict, utt_fw_vocab: Dict, follow_sent_fw_vocab: Dict, mr_data_ontology: Dict[str, Dict[str, int]], sample_const: Dict, max_nsent=4):
    utt_fw_nsamples = sample_const['utt_fw']
    follow_fw_nsamples = sample_const['follow_fw']
    form_nsamples = sample_const['form']

    utt_fw_samples = random.sample(list(utt_fw_vocab.values()), k=utt_fw_nsamples)
    form_sampled_vec = _sample_formulation_for_feature(feature_value_idx, mr, mr_data_ontology, form_nsamples)

    dummy_idx = max(utt_fw_vocab.values()) + 1
    utt_fw_vec = []
    for fidx in utt_fw_samples:
        v = np.zeros(shape=(dummy_idx + 1, 1))
        v[fidx] = 1.0
        utt_fw_vec.append(v)

    dummy_idx = max(follow_sent_fw_vocab.values()) + 1
    base_tag = 'follow_fw_'
    follow_fw_features = []

    for i in range(follow_fw_nsamples):
        sample = []
        if len(mr) == 3:
            choose_nsent = 0
        elif len(mr) in [4, 5]:
            choose_nsent = random.choice([0, 1])
        else:
            choose_nsent = random.choice([1, 2, 3])

        for sent_idx in range(1, max_nsent):
            if len(mr) == 3: #avoid forcing the generator to create long utterances
                fidx = dummy_idx
            elif len(mr) in [4,5]:
                if sent_idx > choose_nsent:
                    fidx = dummy_idx
                else:
                    fidx = random.choice(list(follow_sent_fw_vocab.values()))
            else:
                if sent_idx > choose_nsent:
                    fidx = dummy_idx
                else:
                    fidx = random.choice(list(follow_sent_fw_vocab.values()))

            v = np.zeros(shape=(dummy_idx + 1, 1))
            v[fidx] = 1.0
            sample.append(v)
        follow_fw_features.append(tuple(sample))

    return utt_fw_vec, follow_fw_features, form_sampled_vec


def _sample_formulation_for_feature(feature_value_idx: Dict, mr: Dict[str, str], mr_data_ontology: Dict[str, Dict[str, int]], sample_size: int):
    fromulation_for_features = []
    for i in range(sample_size):
        form_vector = []
        for attribute in mr_data_ontology.keys():
            #value -> term -> idx
            value_term_idx = feature_value_idx.get(attribute, None)
            if value_term_idx is not None:
                for value, term_idx in value_term_idx.items():
                    nterms = len(term_idx)
                    x = np.zeros(shape=(nterms, ))
                    if mr.get(attribute, None) == value:
                        idx = random.sample(list(term_idx.values()), k=1)[0]
                        x[idx] = 1.0
                    form_vector.append(x)
        v = np.concatenate(tuple(form_vector), axis=0)
        fromulation_for_features.append(v)

    return fromulation_for_features


def _save_data(results, data_path, fname, tag):
    odir = os.path.join(data_path, PREPROCESSED_DATA)
    if not os.path.exists(odir):
        os.mkdir(odir)

    odir = os.path.join(data_path, PREPROCESSED_DATA, tag)
    if not os.path.exists(odir):
        os.mkdir(odir)

    ofname = os.path.join(odir, '{}.pkl'.format(fname.split('.')[0]))

    ofile = open(ofname, 'wb')

    cPickle.dump(results, ofile, protocol=4)


def _sample_features(fname:str, feature_value_idx: Dict, utt_fw_vocab: Dict, follow_sent_fw_vocab: Dict, mr_data_ontology: Dict[str, Dict[str, int]], delex_attributes:List[str], sample_const: Dict, max_nsent):
    mr_raw = _read_data(fname)
    process_mr = _parse_raw_mr(mr_raw)
    vectorised_mrs = _vectorize_mrs(process_mr, mr_data_ontology, delex_attributes)

    sampled_surface_features = _sample_surface_feautres(process_mr, feature_value_idx, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, sample_const, max_nsent)

    result = {
        'parsed_mrs': process_mr,
        'vectorised_mrs': vectorised_mrs,
        'sample': sampled_surface_features
    }

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config_full.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']

    sample_const = config_dict['sample_constants']

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file)
    delex_attributes = config_dict['delex_attributes']

    form_vocab_fname = os.path.join(data_base_path, config_dict['form_vocab'])
    feature_value_idx = cPickle.load(open(form_vocab_fname, 'rb'))

    utt_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'utt_fw_vocab.json'), 'rt', encoding='utf-8')
    utt_fw_vocab = json.load(utt_vocab_file)

    follow_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'follow_sent_fw_vocab.json'), 'rt', encoding='utf-8')
    follow_sent_fw_vocab = json.load(follow_vocab_file)

    valid_mr_only = os.path.join(data_base_path, config_dict['valid_data_mr_only'])
    test_mr_only = os.path.join(data_base_path, config_dict['test_data_mr_only'])

    valid_sampled_data = _sample_features(valid_mr_only, feature_value_idx, utt_fw_vocab, follow_sent_fw_vocab,mr_data_ontology, delex_attributes, sample_const, 4)
    test_sampled_data = _sample_features(test_mr_only, feature_value_idx, utt_fw_vocab, follow_sent_fw_vocab,mr_data_ontology, delex_attributes, sample_const, 4)

    _save_data(valid_sampled_data, data_base_path, config_dict['valid_data_mr_only'], out_tag)
    _save_data(test_sampled_data, data_base_path, config_dict['test_data_mr_only'], out_tag)


