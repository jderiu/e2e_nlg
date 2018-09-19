import argparse
import json
import os
import logging
from typing import Dict, List, Set, Tuple
import numpy as np
import _pickle as cPickle
from src.data_processing.utils import convert2indices, PREPROCESSED_DATA, _load_delex_data, _save_vectorized_data


def _compute_vector_length(mr_ontology: Dict[str, Dict[str, int]], delex_attributes: List[str]) -> Dict[str, int]:
    vector_lengts = {}
    for attr, value_list in mr_ontology.items():
        if attr in delex_attributes:
            vector_lengts[attr] = 2 #either present or not
        else:
            vector_lengts[attr] = len(value_list) + 1 #add an extra filed if the attribute is not present
    return vector_lengts


def _vectorize_single_mr(mr: Dict[str, str], mr_ontology: Dict[str, Dict[str, int]], vector_lengts: Dict[str, int], delex_attributes: List[str]) -> Dict[str, np.array]:
    da_vector = {}
    for attr, value_list in mr_ontology.items():
        v = np.zeros(shape=(vector_lengts[attr], 1))
        if attr in mr.keys():
            value = mr[attr]

            if not attr in delex_attributes:
                vidx = value_list[value] #present and lex
            else:
                vidx = 0 #present, but delexicalised

            v[vidx] = 1.0
        else:
            v[-1] = 1.0 #if delexicalised v[1] = 1
        da_vector[attr] = v

    return da_vector


def _vectorize_mrs(preprocessed_mrs: List[Dict[str, str]], mr_ontology: Dict[str, Dict[str, int]], delex_attributes: List[str]) -> np.array:
    """
    Compute Semantic Vectors
    :param preprocessed_mrs:
    :param mr_ontology:
    :param delex_attributes:
    :return: Mapping from attribute name to numpy array of MR-vectors
    """

    vector_lengts = _compute_vector_length(mr_ontology, delex_attributes)
    full_vectors = []
    for mr in preprocessed_mrs:
        vec = _vectorize_single_mr(mr, mr_ontology, vector_lengts, delex_attributes)
        full_vectors.append(vec)

    #creat the format attribute name -> List of processed vectors
    vectorised_mrs = {}
    for attr in mr_ontology.keys():
        attr_vectors = []
        for da_vec_kv in full_vectors:
            attr_vectors.append(da_vec_kv[attr])
        vectorised_mrs[attr] = np.concatenate(attr_vectors, axis=1).T

    return vectorised_mrs


def _create_pos_vocabulary(tagged_data: List[str]) -> Dict[str, int]:
    tag_set = set()
    for tagged_utterance in tagged_data:
        tag_set.update(tagged_utterance)

    tag_vocabulary = dict()
    for i, tag in enumerate(tag_set):
        tag_vocabulary[tag] = i

    return tag_vocabulary


def vectorize_data(config_dict: Dict, fname: str, mr_data_ontology: Dict, delex_attributes: List, char_vocab: Dict):
    max_sentence_len = config_dict['max_sentence_len']
    dummy_char = max(char_vocab.values()) + 1
    unk_char = max(char_vocab.values()) + 2

    logging.info('.. Load Delexicalized Data')
    delex_data = _load_delex_data(delex_data_path, fname)
    logging.info('.. Vectorize MR Data')
    da_vecs = _vectorize_mrs(delex_data['parsed_mrs'], mr_data_ontology, delex_attributes)
    logging.info('.. Vectorize Utterances Data')
    idx_data = convert2indices(delex_data['delexicalised_texts'], char_vocab, dummy_char, unk_char, max_sentence_len)

    delex_data['char_idx'] = idx_data
    delex_data['da_act'] = da_vecs

    return delex_data


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']

    delex_attributes = config_dict['delex_attributes']

    logging.info('Load MR Data Ontology..')
    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt',encoding='utf-8')
    mr_data_ontology = json.load(mr_file)

    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)
    logging.info('Vectorize Training Data')
    train_fname = config_dict['train_data']
    train_delex = vectorize_data(config_dict, train_fname, mr_data_ontology, delex_attributes, char_vocab)
    logging.info('Vectorize Validation Data')
    valid_fname = config_dict['valid_data']
    valid_delex = vectorize_data(config_dict, valid_fname, mr_data_ontology, delex_attributes, char_vocab)
    logging.info('Vectorize Test Data')
    test_fname = config_dict['test_data']
    test_delex = vectorize_data(config_dict, test_fname, mr_data_ontology, delex_attributes, char_vocab)

    logging.info('Save Data..')
    _save_vectorized_data(train_delex, data_base_path, train_fname, out_tag)
    _save_vectorized_data(valid_delex, data_base_path, valid_fname, out_tag)
    _save_vectorized_data(test_delex, data_base_path, test_fname, out_tag)

    logging.info('Done.')
