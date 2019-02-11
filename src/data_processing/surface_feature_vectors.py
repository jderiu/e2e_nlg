import argparse, json, os, logging
import _pickle as cPickle
import numpy as np
from collections import defaultdict
from src.data_processing.utils import PREPROCESSED_DATA, _save_vectorized_data, _load_vectorized_data
from typing import Dict, List
from nltk import WordPunctTokenizer, sent_tokenize


def _utt_fw_features(sent_tok_texts: List[List[List[str]]], vocab: Dict[str, int]) -> np.ndarray:
    """
    Extracts the first word from the tokenized data, maps it to the first word index -> One Hot Encoded vector
    :param sent_tok_texts: Tokenized Sentences
    :param vocab: Vocabulary: first-word -> idx
    :return: vectorized data
    """
    dummy_idx = max(vocab.values()) + 1
    utt_fw_vec = []
    for text in sent_tok_texts:
        v = np.zeros(shape=(dummy_idx + 1, 1))
        first_word = text[0][0]
        fidx = vocab.get(first_word, dummy_idx)
        v[fidx] = 1.0
        utt_fw_vec.append(v)

    return np.concatenate(utt_fw_vec, axis=1).T


def _utterance_first_word_vocab(sent_tok_texts: List[List[List[str]]], min_freq=100) -> Dict[str, int]:
    """
    Extract the first words present in the corpus. Only use the ones that appear at more than 100 times.
    :param sent_tok_texts: Tokenized Texts
    :return: mapping: first word -> index
    """
    fw_freq = defaultdict(lambda: 0)
    for text in sent_tok_texts:
        first_word = text[0][0]
        fw_freq[first_word] += 1

    vocab = {}
    counter = 0
    for token, freq in fw_freq.items():
        if freq > min_freq:
            vocab[token] = counter
            counter += 1

    return vocab


def _follow_fw_features(sent_tok_texts: List[List[List[str]]], vocab: Dict[str, int], max_nsent=4) -> Dict[str, np.array]:
    """
    Extract the first words of the follow-up sentences in the utterance.
    :param sent_tok_texts: Tokenized Sentence
    :param vocab: Vocabulary: frst-word -> index
    :param max_nsent: maximum number of sentences to consider
    :return: mapping: sentence_number -> features
    """
    dummy_idx = max(vocab.values()) + 1
    base_tag = 'follow_fw_'
    follow_fw_features = {}
    #for each sentence (sentence number 2,3,4)
    for sent_idx in range(1, max_nsent):
        fw_vec = []
        for text in sent_tok_texts:
            v = np.zeros(shape=(dummy_idx + 1, 1))
            if len(text) > sent_idx:
                fw_i = text[sent_idx][0]
                fw_idx = vocab.get(fw_i, dummy_idx)
            else:
                fw_idx = dummy_idx #if the utterance has less sentences than the max allowed
            v[fw_idx] = 1.0
            fw_vec.append(v)

        follow_fw_features[base_tag + str(sent_idx)] = np.concatenate(fw_vec, axis=1).T
    return follow_fw_features


def _follow_sent_first_word_vocab(full_sent_toks: List[List[List[str]]]) -> Dict[str, int]:
    fw_freq = defaultdict(lambda: 0)
    for text in full_sent_toks:
        for sent in text[1:]:
            first_word = sent[0]
            fw_freq[first_word] += 1

    vocab = {}
    counter = 0
    for token, freq in fw_freq.items():
        if freq > 100:
            vocab[token] = counter
            counter += 1

    return vocab


def _sentence_tok(delex_texts: List[str]) -> List[List[List[str]]]:
    #tokenize the texts
    sentence_tok_texts = []
    tknzr = WordPunctTokenizer()
    for text in delex_texts:
        sentences = sent_tokenize(text)
        tok_sentences = []
        for sentence in sentences:
            tok_sentences.append(tknzr.tokenize(sentence))
        sentence_tok_texts.append(tok_sentences)

    return sentence_tok_texts


def _get_formulation_for_feature(vocab_fname:str, preocessed_mr: List[Dict[str, str]], outputs_raw: List[str], mr_data_ontology: Dict[str, List[str]]):
    #feature -> value -> term -> idx
    feature_value_idx = cPickle.load(open(vocab_fname, 'rb'))

    fromulation_for_features = []
    for mr, output_text in zip(preocessed_mr, outputs_raw):
        form_vector = []
        #for each attribute in the ontology
        for attribute in mr_data_ontology.keys():
            #value -> term -> idx
            value_term_idx = feature_value_idx.get(attribute, None) #check if we have special formulations for it
            if value_term_idx is not None:
                for value, term_idx in value_term_idx.items(): #iterate all the values of this attribute if there are different formulations for it
                    nterms = len(term_idx)
                    x = np.zeros(shape=(nterms, )) #prepare one hot vector -> length of the number of different terms
                    if mr.get(attribute, None) == value: #check if the value of the MR corresponds to the current value
                        for term, idx in term_idx.items(): #iterate over all terms and check if the term appears in the output text
                            if term in output_text:
                                x[idx] = 1.0
                    form_vector.append(x)
        v = np.concatenate(tuple(form_vector), axis=0)
        fromulation_for_features.append(v)

    return np.array(fromulation_for_features)


def extract_surface_level_features(tokens, delex_data, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, form_vocab_fname):
    logging.info('.. Compute Utterance First Words Features')
    utt_fw = _utt_fw_features(tokens, utt_fw_vocab)

    logging.info('.. Compute Follow Sentences First Words Features')
    follow_fw = _follow_fw_features(tokens, follow_sent_fw_vocab)

    logging.info('.. Compute Formulation Features')
    formulation_ft = _get_formulation_for_feature(form_vocab_fname, delex_data['parsed_mrs'], delex_data['outputs_raw'], mr_data_ontology)

    delex_data['utt_fw'] = utt_fw
    delex_data['follow_fw'] = follow_fw
    delex_data['form'] = formulation_ft

    surface_lengts = {
        'utt_fw': utt_fw.shape[1],
        'follow_fw': follow_fw['follow_fw_1'].shape[1],
        'form': formulation_ft.shape[1]
    }

    return delex_data, surface_lengts


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']

    form_vocab_fname = os.path.join(data_base_path, config_dict['form_vocab'])

    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file)

    logging.info('Load Data..')
    train_fname = config_dict['train_data']
    train_delex = _load_vectorized_data(delex_data_path, train_fname)

    valid_fname = config_dict['valid_data']
    valid_delex = _load_vectorized_data(delex_data_path, valid_fname)

    test_fname = config_dict['test_data']
    test_delex = _load_vectorized_data(delex_data_path, test_fname)

    logging.info('Compute First Words Vocabulary..')
    train_tok = _sentence_tok(train_delex['delexicalised_texts'])
    valid_tok = _sentence_tok(valid_delex['delexicalised_texts'])
    test_tok = _sentence_tok(test_delex['delexicalised_texts'])
    utt_fw_vocab = _utterance_first_word_vocab(train_tok + valid_tok + test_tok)
    follow_sent_fw_vocab = _follow_sent_first_word_vocab(train_tok + valid_tok + test_tok)

    logging.info('Process Train Data..')
    train_delex, surface_lengts = extract_surface_level_features(train_tok, train_delex, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, form_vocab_fname)
    logging.info('Process Valid Data..')
    valid_delex, _ = extract_surface_level_features(valid_tok, valid_delex, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, form_vocab_fname)
    logging.info('Process Test Data..')
    test_delex, _ = extract_surface_level_features(test_tok, test_delex, utt_fw_vocab, follow_sent_fw_vocab, mr_data_ontology, form_vocab_fname)

    logging.info('Save Data..')
    _save_vectorized_data(train_delex, data_base_path, train_fname, out_tag)
    _save_vectorized_data(valid_delex, data_base_path, valid_fname, out_tag)
    _save_vectorized_data(test_delex, data_base_path, test_fname, out_tag)

    utt_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'utt_fw_vocab.json'), 'wt', encoding='utf-8')
    json.dump(utt_fw_vocab, utt_vocab_file)

    follow_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'follow_sent_fw_vocab.json'), 'wt', encoding='utf-8')
    json.dump(follow_sent_fw_vocab, follow_vocab_file)

    surface_lengths_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'surface_features_lengths.json'), 'wt', encoding='utf-8')
    json.dump(surface_lengts, surface_lengths_file)

    logging.info('Done.')

