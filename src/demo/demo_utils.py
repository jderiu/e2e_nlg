import argparse, json, os, random
from src.evaluation.generate_output import load_model, prepare_input, generate_output, get_discr_ratings, convert_idx2char, _merge_str_ratings, select_top_outputs, sample_final_output, _lexicalise_output, _lexicalise_full_output
from src.data_processing.vectorize_data import _vectorize_mrs, _compute_vector_length
from src.data_processing.generate_evaluation_data import _sample_surface_feautres
from src.data_processing.delexicalise_data import _load_delex_fields
from src.data_processing.utils import PREPROCESSED_DATA

import _pickle as cPickle


class NLGModel():

    def __init__(self, config_dict):
        self.config_dict = config_dict
        data_base_path = config_dict['data_base_path']
        out_tag = config_dict['processing_tag']
        self.lex_dummies = config_dict['lex_dummies']

        mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt',
                       encoding='utf-8')
        self.mr_data_ontology = json.load(mr_file)
        self.delex_attributes = config_dict['delex_attributes']

        form_vocab_fname = os.path.join(data_base_path, config_dict['form_vocab'])
        self.feature_value_idx = cPickle.load(open(form_vocab_fname, 'rb'))

        utt_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'utt_fw_vocab.json'), 'rt',
                              encoding='utf-8')
        self.utt_fw_vocab = json.load(utt_vocab_file)

        follow_vocab_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'follow_sent_fw_vocab.json'),
                                 'rt', encoding='utf-8')
        self.follow_sent_fw_vocab = json.load(follow_vocab_file)

        self.delex_fields = _load_delex_fields(config_dict)

        self.sample_const = config_dict['sample_constants']

        self.test_model, self.discriminators = load_model(config_dict)

    def generate_utterance_for_mr(self, process_mr):
        vectorised_mrs = _vectorize_mrs(process_mr, self.mr_data_ontology, self.delex_attributes)
        sampled_surface_features = _sample_surface_feautres(
            process_mr,
            self.feature_value_idx,
            self.utt_fw_vocab,
            self.follow_sent_fw_vocab,
            self.mr_data_ontology,
            self.sample_const,
            4
        )

        result = {
            'parsed_mrs': process_mr,
            'vectorised_mrs': vectorised_mrs,
            'sample': sampled_surface_features
        }

        valid_input = prepare_input(self.config_dict, result)

        confidence_scores, output_idx = generate_output(self.test_model, valid_input)
        ratings = get_discr_ratings(output_idx, self.discriminators, vectorised_mrs)
        str_out = convert_idx2char(self.config_dict, output_idx)
        merge_out_ratings = _merge_str_ratings(str_out, confidence_scores, ratings)
        top_outputs = select_top_outputs(merge_out_ratings)
        scores = [x[1][8] for x in top_outputs[0]]

        top_out = _lexicalise_full_output(top_outputs, self.delex_fields, result, self.lex_dummies, print_score=False)
        return top_out[0], scores