import csv
import os
import json
import argparse
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Set
from src.data_processing.utils import _save_delex_data, PREPROCESSED_DATA


def _read_data(fname: str) -> Tuple[List[str], List[str]]:
    """
    Loads the raw data from the datafile
    :param fname: path to the data file
    :return: List of raw MRs and raw Output utterances
    """
    data_reader = csv.DictReader(open(fname, encoding='utf-8', mode='rt'))
    mr_raw = []
    outputs_raw = []
    for row in data_reader:
        #there are two fields in the csv
        mr_raw.append(row['mr'])
        outputs_raw.append(row.get('ref')) #in the testset there is no ref field as it is not annotated, thus, the outputs can be None

    return mr_raw, outputs_raw


def _parse_raw_mr(mr_raw: List[str]) -> List[Dict[str, str]]:
    """
    Parses the list of MRs
    :param mr_raw: list of raw MRs
    :return: list of parsed MRs
    """
    processed_mrs = []
    for mr_string in mr_raw:
        attribute_value_pairs = _parse_single_mr(mr_string)
        processed_mrs.append(attribute_value_pairs)

    return processed_mrs


def _parse_single_mr(mr_string:str) -> Dict[str, str]:
    """
    Parse a single MR
    :param mr_string: Raw MR
    :return: dictionary of attribute -> value mapping
    """
    mr_list = mr_string.split(',')
    attribute_value_pairs = {}
    for mr in mr_list:
        kidx_begin = mr.find('[')
        kidx_end = mr.find(']')
        key = mr[:kidx_begin].strip()
        value = mr[kidx_begin + 1: kidx_end]

        attribute_value_pairs[key] = value

    return attribute_value_pairs


def _delexicalise(processed_mrs: List[Dict[str, str]], outputs_raw: List[str], delex_fields: Dict[str, str]) -> List[str]:
    delexicalised_texts = []

    for attribute_value_pair, text in zip(processed_mrs, outputs_raw):
        delex_text = _delex_single(attribute_value_pair, text, delex_fields)
        delexicalised_texts.append(delex_text)

    return delexicalised_texts


def _delex_single(attribute_value_pair: Dict[str, str], text: str, delex_fields: Dict[str, str])-> str:
    """ Delexicalises single mr - text pair where the replacement is done by simple string matching"""
    delex_attributes = set(attribute_value_pair.keys()).intersection(delex_fields.keys())

    for attribute in delex_attributes:
        value = attribute_value_pair[attribute]
        replacement_token = delex_fields[attribute]

        text = text.replace(value, replacement_token)

    return text


def _load_attributes(attribute_path: str) -> Dict[str, str]:
    """
    Loads the list of attributes with their respecive replacement tokens
    :param attribute_path:
    :return: Mapping from attribute name to replacement token
    """
    attribute_tokens = dict()

    attr_file = open(attribute_path, 'rt', encoding='utf-8')
    for line in attr_file.readlines():
        attr, replacement_token = tuple(line.replace('\n', '').split('='))
        attribute_tokens[attr] = replacement_token

    return attribute_tokens


def _get_delex_fields(attribute_tokens: Dict[str, str], delex_attributes: List[str]):
    """
    Select the attributes which we want to delexicalize
    :param attribute_tokens: Mapping from attributes to replacement tokens
    :param delex_attributes: List of attributes to be delexicalised
    :return: Mapping from attribute name to replacement tokens
    """
    delex_fields = {}
    for delex_attribute in delex_attributes:
        delex_fields[delex_attribute] = attribute_tokens[delex_attribute]
    return delex_fields


def _get_vals_for_attr(processed_mrs: List[Dict[str, str]]) -> Dict[str, Set]:
    """Returns a mapping attr -> list of values in data"""
    values_for_attribute = defaultdict(lambda: set())
    for mr in processed_mrs:
        for attr, val in mr.items():
            values_for_attribute[attr].add(val)
    return values_for_attribute


def _load_delex_fields(config_dict: Dict) -> Dict[str, str]:
    """
    Loads the attributes to delexicalize and loads the dictionary with the replacement tokens.
    :param config_dict:
    :return:
    """

    data_base_path = config_dict['data_base_path']
    attribute_path = os.path.join(data_base_path, config_dict['attributes'])
    attribute_tokens = _load_attributes(attribute_path)
    delex_attributes = config_dict['delex_attributes']

    delex_fields = _get_delex_fields(attribute_tokens, delex_attributes)
    return delex_fields


def _retrieve_mr_ontology(full_mr_list: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """
    Derive full Domain Ontology, which is represented as a mapping from attribute to the list of all possible values that appear in the corpus.
    :param full_mr_list: List of MRs
    :return: List of mappings from attribute name to indexed values.
    """
    possible_attr_values = defaultdict(lambda: set())
    for attribute_value_piars in full_mr_list:
        for attribute, value in attribute_value_piars.items():
            possible_attr_values[attribute].add(value)

    attr_value_pair_ontology = defaultdict(lambda: {})
    for attribute, value_set in possible_attr_values.items():
        value_list = list(value_set)
        for idx, value in enumerate(sorted(value_list)):
            attr_value_pair_ontology[attribute][value] = idx

    return attr_value_pair_ontology


def delex_nlg_data(fname, config_dict) -> Dict[str, object]:
    """
    Calls the delexicalization pipeline, which consists of loading the raw data, parsing the MR, delexicalizing the raw_references.
    :param fname: name of the datafile
    :param config_dict: dictionary with the configurations
    :return: results consisting of the raw mrs, parsed mrs, the raw references, and the delexicalized references
    """
    data_base_path = config_dict['data_base_path']
    delex_attributes = config_dict['delex_attributes']
    attribute_fname =  config_dict['attributes']

    return _delex_nlg_data(fname, data_base_path, delex_attributes, attribute_fname)


def _delex_nlg_data(fname, data_path, delex_attributes, attribute_fname):
    """
    Calls the delexicalization pipeline, which consists of loading the raw data, parsing the MR, delexicalizing the raw_references.
    :param fname: name of the datafile
    :param config_dict: dictionary with the configurations
    :return: results consisting of the raw mrs, parsed mrs, the raw references, and the delexicalized references
    """
    fname = os.path.join(data_path, fname)

    mr_raw, outputs_raw = _read_data(fname)
    parsed_mrs = _parse_raw_mr(mr_raw)

    attribute_path = os.path.join(data_path, attribute_fname)
    attribute_tokens = _load_attributes(attribute_path)

    # which fields to delexicalize and the token used to delexicalize.
    delex_fields = _get_delex_fields(attribute_tokens, delex_attributes)
    delexicalised_texts = _delexicalise(parsed_mrs, outputs_raw, delex_fields)

    results = {
        'mr_raw': mr_raw,
        'parsed_mrs': parsed_mrs,
        'outputs_raw': outputs_raw,
        'delexicalised_texts': delexicalised_texts,
    }

    return results


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')
    parser.add_argument('-t, --parse_type', dest='parse_type', default='train', type=str, help='train or test')
    args = parser.parse_args()

    config_fname = args.config
    parse_type = args.parse_type

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']
    logging.info('Delexicalising Training Data..')

    train_fname = config_dict['train_data']
    train_delex = delex_nlg_data(train_fname, config_dict)

    logging.info('Delexicalising Validation Data..')
    valid_fname = config_dict['valid_data']
    valid_delex = delex_nlg_data(valid_fname, config_dict)

    logging.info('Delexicalising Test Data..')
    test_fname = config_dict['test_data']
    test_delex = delex_nlg_data(test_fname, config_dict)

    full_mr_list = train_delex['parsed_mrs'] + valid_delex['parsed_mrs'] + test_delex['parsed_mrs']

    logging.info('Generating MR Ontology..')
    mr_data_ontology = _retrieve_mr_ontology(full_mr_list)

    logging.info('Saving Data..')
    _save_delex_data(train_delex, data_base_path, train_fname, out_tag)
    _save_delex_data(valid_delex, data_base_path, valid_fname, out_tag)
    _save_delex_data(test_delex, data_base_path, test_fname, out_tag)

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'wt', encoding='utf-8')
    json.dump(mr_data_ontology, mr_file)

    logging.info('Done.')