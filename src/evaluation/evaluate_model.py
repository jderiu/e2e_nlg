import argparse, json, os, itertools, random
import keras.backend as K
import numpy as np
from src.data_processing.surface_feature_vectors import _load_vectorized_data
from src.data_processing.utils import PREPROCESSED_DATA
from src.architectures.sc_lstm_architecutre.sclstm_vanilla_architecture import generator_model
from src.data_processing.vectorize_data import _compute_vector_length
from src.training.train_classifiers import load_discriminators
from src.training.train_sclstm import _get_lexicalize_dict
from src.data_processing.delexicalise_data import _load_delex_fields, _load_attributes

from tqdm import tqdm

from typing import Dict, List, Tuple


def _prepare_single_input(config_dict, semantic_input: List[np.array], sampled_suface_data: Dict[str, List]):
    #create all combinations
    return_input = []

    surface_features = config_dict['surface_features']
    max_sentence_len = config_dict['max_sentence_len']
    surf_input = []
    for surface_feature, opts in sorted(list(surface_features.items()), key=lambda x: x[0]):
        #handle features with multiple elements, like follow fw
        if opts:
            surf_ft = sampled_suface_data[surface_feature]
            osurf_ft = []
            for ft in surf_ft:
                tupl = []
                for opt in opts:
                    tupl.append(ft[opt - 1])
                osurf_ft.append(tuple(tupl))
            surf_ft = osurf_ft
        else:
            surf_ft = sampled_suface_data[surface_feature]
        surf_input.append(surf_ft)

    sinputs = []
    for element in itertools.product(*surf_input):
        ielem = []
        for entry in element:
            if type(entry) == tuple:
                for i in entry:
                    ielem.append(i)
            else:
                ielem.append(entry)

        sinputs.append(semantic_input + list(ielem) + [np.zeros(shape=(max_sentence_len, 1))])

    return sinputs


def prepare_input(config_dict: Dict, data_lex: Dict) -> List[List[List[np.ndarray]]]:
    sampled_data = data_lex['sample']
    sem_input = [x[1] for x in sorted(list(data_lex['vectorised_mrs'].items()), key=lambda x: x[0])] #List[List]

    inputs = []
    for *isem, isamples in zip(*sem_input, sampled_data):
        ready_inputs = _prepare_single_input(config_dict, isem, isamples)
        tmp_inputs = list(map(list, zip(*ready_inputs)))

        tmp_inputs = [np.squeeze(np.array(x), axis=-1) if x[0].shape[-1] == 1 else np.array(x) for x in tmp_inputs]

        inputs.append(tmp_inputs)

    return inputs


def generate_output(test_model, preprocessed_data):

    confidence_scores, pred_idx = [], []
    for pdata in tqdm(preprocessed_data):
        predicted_data = test_model.predict(pdata, batch_size=256)
        pred_idx.append(predicted_data[0])
        confidence_scores.append(predicted_data[1])

    return confidence_scores, pred_idx


def convert_idx2char(config_dict, pred_idx) -> List[List[str]]:
    data_base_path = config_dict['data_base_path']
    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    inverse_vocab = {v: k for k,v in char_vocab.items()}

    mr_output_chars = []
    for mr_output_idx in pred_idx:
        mr_out = []
        for oidx in mr_output_idx:
            ochars = ''.join([inverse_vocab.get(x, '') for x in oidx])
            mr_out.append(ochars)
        mr_output_chars.append(mr_out)

    return mr_output_chars


def get_discr_ratings_single(predicted_sentences, discriminators, da_acts) -> List[List[int]]:
    discr_scores = []
    for attribute, value_vector in da_acts.items():
        ytrue = da_acts[attribute]

        discriminator = discriminators[attribute]

        ypred_full = discriminator.predict(predicted_sentences, batch_size=1024)
        scores = []
        for ypred in ypred_full:

            pred_lbl = np.argmax(ypred, axis=0)
            true_lbl = np.argmax(ytrue, axis=0)

            if pred_lbl == true_lbl:
                scores.append(1)
            else:
                scores.append(0)
        discr_scores.append(scores)

    discr_scores = list(map(list, zip(*discr_scores)))
    return discr_scores


def get_discr_ratings(pred_idx, discriminators, da_acts) -> List[List[List[int]]]:
    rated_predictions = []
    for i, predicted_sentences in tqdm(enumerate(pred_idx)):
        da_act_single = {k: v[i] for k, v in da_acts.items()}
        rated_predictions.append(get_discr_ratings_single(predicted_sentences, discriminators, da_act_single))

    return rated_predictions


def _merge_str_ratings(str_output, confidence_ratings, ratings):
    merged_output = []
    for str_out_batch, confidence_batch, rating_batch in zip(str_output, confidence_ratings, ratings):
        merged_batch = []
        for str_out_single, confidence_single, rating_single in zip(str_out_batch, confidence_batch, rating_batch):
            scores = [str(x) for x in rating_single] + [str(sum(rating_single))] + [str(confidence_single)]
            merged_batch.append((str_out_single, scores))
        merged_output.append(merged_batch)
    return merged_output


def select_top_outputs(merged_output, selection_score_idx=-1):
    filtered_output = []
    for merged_batch in merged_output:
        max_score = max([x[-1][selection_score_idx] for x in merged_batch])
        filtered_batch = [x for i, x in enumerate(merged_batch) if x[-1][selection_score_idx] == max_score]
        filtered_output.append(filtered_batch)
    return filtered_output


def sample_final_output(filtered_output):
    sampled_output = []
    for filtered_batch in filtered_output:
        x = random.choice(filtered_batch)
        sampled_output.append(x)
    return sampled_output


def _lexicalise_full_output(sampled_output: List[List[Tuple[str, List[int]]]], delex_fields: Dict, delex_data: Dict, lex_dummies: Dict):
    delex_vocab = _get_lexicalize_dict(delex_data, delex_fields)
    lexicalised_output = []
    for i, sampled_output_batch in enumerate(sampled_output):
        lexicalised_batch = []
        for oline, score in sampled_output_batch:
            for lex_key, val in delex_fields.items():
                original_token = delex_vocab[lex_key][i]
                if original_token == '':
                    original_token = lex_dummies.get(lex_key, '')

                oline = oline.replace(val, original_token)
                oline = oline.replace('The The', 'The').replace('food food', 'food').replace('the The', 'The')

            lexicalised_batch.append(oline + '\n')
        lexicalised_output.append(lexicalised_batch)
    return lexicalised_output


def _lexicalise_output(sampled_output: List[Tuple[str, List[int]]], delex_fields: Dict, delex_data: Dict, lex_dummies: Dict) -> List[str]:
    delex_vocab = _get_lexicalize_dict(delex_data, delex_fields)
    lexicalised_output = []
    for i, (oline, score) in enumerate(sampled_output):
        for lex_key, val in delex_fields.items():
            original_token = delex_vocab[lex_key][i]
            if original_token == '':
                original_token = lex_dummies.get(lex_key, '')

            oline = oline.replace(val, original_token)
            oline = oline.replace('The The', 'The').replace('food food', 'food').replace('the The', 'The')

        lexicalised_output.append(oline)
    return lexicalised_output


def load_model(config_dict:Dict):
    data_base_path = config_dict['data_base_path']
    out_tag = config_dict['processing_tag']

    surface_lengths_vocab = open(
        os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'surface_features_lengths.json'), 'rt',
        encoding='utf-8')
    surface_lengts = json.load(surface_lengths_vocab)

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file)
    delex_attributes = config_dict['delex_attributes']

    mr_vec_lengths = _compute_vector_length(mr_data_ontology, delex_attributes)

    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    step = K.variable(1., name='step_varialbe')

    train_model, test_model = generator_model(config_dict, mr_vec_lengths, surface_lengts, char_vocab, step)
    out_model_path = config_dict['sclstm_model_path']
    weights_idx = max([int(fname.split('.')[1]) for fname in os.listdir(out_model_path)])
    weights_fname = os.path.join(out_model_path, 'weights.{:02}.hdf5'.format(weights_idx))

    train_model.load_weights(weights_fname)

    discriminators = load_discriminators(config_dict, mr_vec_lengths)

    return test_model, discriminators


def _generate_full_output(merge_out_ratings):
    out_texts = []
    for merge_out_batch in merge_out_ratings:
        out_batch = []
        for text, scores in merge_out_batch:
            score_string = '\t'.join(scores)
            oline = '{}\t{}\n'.format(text, score_string)
            out_batch.append(oline)
        out_texts.append(out_batch)
    return out_texts


def _print_full_output(out_texts, ofile):
    for out_batch in out_texts:
        for oline in out_batch:
            ofile.write(oline)
        ofile.write('\n')
    ofile.close()


def _generate_output(config_dict, test_model, discriminators, data_delex, tag):
    delex_fields = _load_delex_fields(config_dict)

    valid_input = prepare_input(config_dict, data_delex)
    confidence_scores, output_idx = generate_output(test_model, valid_input)
    ratings = get_discr_ratings(output_idx, discriminators, data_delex['vectorised_mrs'])
    str_out = convert_idx2char(config_dict, output_idx)
    merge_out_ratings = _merge_str_ratings(str_out, confidence_scores, ratings)
    top_outputs = select_top_outputs(merge_out_ratings)
    sampled_top_output = sample_final_output(top_outputs)

    lex_dummies = config_dict['lex_dummies']
    final_lex_output = _lexicalise_output(sampled_top_output, delex_fields, data_delex, lex_dummies)

    opath = config_dict['output_dir']
    if not os.path.exists(opath):
        os.mkdir(opath)

    ofile_lex = open(os.path.join(opath, 'final_output_{}.txt'.format(tag)), 'wt', encoding='utf-8')
    ofile_full = open(os.path.join(opath, 'full_output_{}.txt'.format(tag)), 'wt', encoding='utf-8')
    ofile_top = open(os.path.join(opath, 'top_output_{}.txt'.format(tag)), 'wt', encoding='utf-8')

    ofile_lex.writelines([x + '\n' for x in final_lex_output])
    full_out = _generate_full_output(merge_out_ratings)
    top_out = _lexicalise_full_output(top_outputs, delex_fields, data_delex, lex_dummies)
    _print_full_output(full_out, ofile_full)
    _print_full_output(top_out, ofile_top)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    valid_data_mr_only = config_dict['valid_data_mr_only']
    test_data_mr_only = config_dict['test_data_mr_only']

    out_tag = config_dict['processing_tag']
    data_base_path = config_dict['data_base_path']

    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    valid_data_delex = _load_vectorized_data(delex_data_path, valid_data_mr_only)
    test_data_delex = _load_vectorized_data(delex_data_path, test_data_mr_only)

    test_model, discriminators = load_model(config_dict)

    _generate_output(config_dict, test_model, discriminators, valid_data_delex, 'valid')
    _generate_output(config_dict, test_model, discriminators, test_data_delex, 'test')



