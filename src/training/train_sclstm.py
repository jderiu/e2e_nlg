import argparse, os, json, logging, time
import keras.backend as K
from math import ceil
import numpy as np
from collections import defaultdict
from keras.optimizers import Adadelta
from src.data_processing.utils import PREPROCESSED_DATA
from src.data_processing.utils import _load_vectorized_data
from src.data_processing.vectorize_data import _compute_vector_length
from keras.callbacks import TensorBoard, ModelCheckpoint
from src.architectures.custom_callbacks.output_callbacks import LexFeatureOutputCallbackVanilla
from src.architectures.custom_callbacks.custom_callbacks import StepCallback, TerminateOnNaN
from src.architectures.sc_lstm_architecutre.sclstm_vanilla_architecture import generator_model
from src.training.train_classifiers import load_discriminators
from src.data_processing.delexicalise_data import _load_delex_fields


def _get_data(data_set, surface_features):
    output_data = data_set['char_idx']
    sem_input = sorted(list(data_set['da_act'].items()), key=lambda x: x[0])
    surf_input = []
    for surface_feature, opts in sorted(list(surface_features.items()), key=lambda x: x[0]):
        if opts:
            for opt in opts:
                surf_ft = data_set[surface_feature]['{}_{}'.format(surface_feature, opt)]
                surf_input.append(surf_ft)
        else:
            surf_ft = data_set[surface_feature]
            surf_input.append(surf_ft)

    input_data = [x[1] for x in sem_input] + surf_input

    return input_data + [output_data], [np.ones(len(input_data[0]))] * 3#* len(input_data)


def _get_lexicalize_dict(data_set, delex_fields):
    parsed_mrs = data_set['parsed_mrs']
    delex_vocabulary = defaultdict(lambda: [])
    for attribute, replacement_token in delex_fields.items():
        values = [x.get(attribute, '') for x in parsed_mrs]
        delex_vocabulary[attribute] = values
    return delex_vocabulary


def train_vanilla_model(config_dict, discriminators, train_delex, valid_delex, test_delex, out_model_path, log_path):
    surface_features = config_dict['surface_features']

    step = K.variable(1., name='step_varialbe')

    data_base_path = config_dict['data_base_path']
    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    delex_fields = _load_delex_fields(config_dict)

    train_input, train_output = _get_data(train_delex, surface_features)
    valid_input, valid_output = _get_data(valid_delex, surface_features)
    test_input, test_output = _get_data(test_delex, surface_features)

    test_delex_vocab = _get_lexicalize_dict(test_delex, delex_fields)

    train_model, test_model = generator_model(config_dict, mr_vec_lengths, surface_lengts, char_vocab, step)

    train_model.summary()

    batch_size = config_dict['lstm_batch_size']
    epochs = config_dict['nb_epochs']

    steps_per_epoch = ceil(train_output[0].shape[0] / config_dict['lstm_batch_size'])

    out_frequency = config_dict['out_frequency']

    terminate_on_nan = TerminateOnNaN()
    model_checkpoint = ModelCheckpoint(os.path.join(out_model_path, 'weights.{epoch:02d}.hdf5'), period=out_frequency, save_weights_only=True)

    tensorboard = TensorBoard(log_dir='logging/tensorboard', histogram_freq=0, write_grads=True, write_images=True)
    step_callback = StepCallback(step, steps_per_epoch)

    lex_output = LexFeatureOutputCallbackVanilla(config_dict, test_model, discriminators, test_input, test_delex['da_act'], delex_fields, test_delex_vocab, out_frequency, char_vocab, '', fname='{}/vanilla_valid_output'.format(log_path))
    #lex_output3 = LexFeatureOutputCallbackVanilla(config_data, test_model, discriminators, valid_dev_input3, valid_dev_lex3, 10, char_vocab, '', fname='{}/vanilla_test_output'.format(log_path))
    callbacks = [step_callback, tensorboard, model_checkpoint, terminate_on_nan, lex_output]

    logging.info('Train the Model..')

    optimizer = Adadelta(lr=1, epsilon=1e-8, rho=0.95, decay=0.0001, clipnorm=10)

    train_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
    train_model.fit(
        x=train_input,
        y=train_output,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_input, valid_output),
        callbacks=callbacks
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    logging_tag = config_dict['logging_tag']
    log_path = 'logging/sc_lstm'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path = os.path.join(log_path, logging_tag)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_path = os.path.join(log_path, 'training_log_{}'.format(str(int(round(time.time() * 1000)))))
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename='{}/sc_lstm_training.log'.format(log_path), filemode='w')

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']

    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file)
    delex_attributes = config_dict['delex_attributes']

    surface_lengths_vocab = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'surface_features_lengths.json'), 'rt', encoding='utf-8')
    surface_lengts = json.load(surface_lengths_vocab)

    mr_vec_lengths = _compute_vector_length(mr_data_ontology, delex_attributes)

    train_fname = config_dict['train_data']
    train_delex = _load_vectorized_data(delex_data_path, train_fname)

    valid_fname = config_dict['valid_data']
    valid_delex = _load_vectorized_data(delex_data_path, valid_fname)

    test_fname = config_dict['test_data']
    test_delex = _load_vectorized_data(delex_data_path, test_fname)

    discriminators = load_discriminators(config_dict, mr_vec_lengths)

    out_model_path = config_dict['sclstm_model_path']
    if not os.path.exists(out_model_path):
        os.mkdir(out_model_path)

    train_vanilla_model(config_dict, discriminators, train_delex, valid_delex, test_delex, out_model_path, log_path)







