import argparse, json, os
import argparse, json, os, itertools, random
import keras.backend as K
import numpy as np
from src.data_processing.surface_feature_vectors import _load_vectorized_data
from src.data_processing.utils import PREPROCESSED_DATA, convert2indices
from src.data_processing.delexicalise_data import _delexicalise, _get_delex_fields, _load_attributes
from src.architectures.sc_lstm_architecutre.sclstm_vanilla_architecture import generator_model
from src.data_processing.vectorize_data import _compute_vector_length
from src.training.train_classifiers import load_discriminators

from tqdm import tqdm

from typing import Dict, List, Tuple


def get_discr_ratings_single(predicted_sentences, discriminators, da_acts) -> List[List[int]]:
    discr_scores = []
    for attribute, value_vector in da_acts.items():
        ytrue_full = da_acts[attribute]

        discriminator = discriminators[attribute]

        ypred_full = discriminator.predict(predicted_sentences, batch_size=1024)
        scores = []
        for ypred, ytrue in zip(ypred_full, ytrue_full):

            pred_lbl = np.argmax(ypred, axis=0)
            true_lbl = np.argmax(ytrue, axis=0)

            if pred_lbl == true_lbl:
                scores.append(1)
            else:
                scores.append(0)
        discr_scores.append(scores)

    discr_scores = list(map(list, zip(*discr_scores)))
    return discr_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations/data_processing_config_correct_syntax', config_fname)))

    # filename
    valid_data_mr_only = config_dict['valid_data_mr_only']
    test_data_mr_only = config_dict['test_data_mr_only']

    out_tag = config_dict['processing_tag']
    data_base_path = config_dict['data_base_path']

    char_vocab_fname = config_dict['character_vocabulary']
    max_sentence_len = config_dict['max_sentence_len']

    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    dummy_char = max(char_vocab.values()) + 1
    unk_char = max(char_vocab.values()) + 2

    # path to data
    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    test_data_delex = _load_vectorized_data(delex_data_path, test_data_mr_only)#parsed_mrs, vectorised_mrs, sample (syntactic man type-> vector)
    parsed_mrs = test_data_delex['parsed_mrs']

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file) #atribute -> values -> idx
    delex_attributes = config_dict['delex_attributes'] # list of attributes to replace by tokens

    attribute_path = os.path.join(data_base_path, config_dict['attributes'])
    attribute_tokens = _load_attributes(attribute_path)

    mr_vec_lengths = _compute_vector_length(mr_data_ontology, delex_attributes) #attribute -> vector length
    discriminators = load_discriminators(config_dict, mr_vec_lengths)

    delex_fields = _get_delex_fields(attribute_tokens, delex_attributes)

    #raw_input = [x.replace('\n', '') for x in open('outputs/tgen/tgen_e2e_outputs_test.txt', 'rt', encoding='utf-8').readlines()]
    raw_input = [x.replace('\n', '') for x in open('outputs/tgen/tgen_e2e_outputs_test.txt', 'rt', encoding='utf-8').readlines()]

    delex_input = _delexicalise(parsed_mrs, raw_input, delex_fields)

    test_idx = convert2indices(delex_input, char_vocab, dummy_char, unk_char, max_sentence_len)

    discr_scores = get_discr_ratings_single(test_idx, discriminators, test_data_delex['vectorised_mrs'])

    full_scores = [np.sum(x) for x in discr_scores]

    print(np.mean(full_scores), np.mean(full_scores)/8, 1 - np.mean(full_scores)/8)

