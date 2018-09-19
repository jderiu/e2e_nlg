import argparse, os, json
from src.data_processing.utils import PREPROCESSED_DATA
from src.data_processing.surface_feature_vectors import _load_vectorized_data

import time
import logging
import numpy as np
from os.path import join
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.architectures.sc_lstm_architecutre.semantic_classifiers import get_semantic_classifier


def last_one(a):
    di = {}
    for i, j in zip(*np.where(a > 0)):
        if i in di:
            di[i] = np.max([di[i], j])
        else:
            di[i] = j
    return di


def train_discriminators(config_data, train_input, valid_input):
    optimizer = Adadelta(lr=1, epsilon=1e-8, rho=0.95)
    model_path = config_data['pretrain_model_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    disX_train = train_input['char_idx']  # take input 8 as train input and the rest as targets
    disy_train = train_input['da_act']  # take input 1-7 as targets

    sample_out_size = disX_train.shape[1]
    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    for attribute, y_train in disy_train.items():
        nlabels = y_train.shape[1]
        discriminator = get_semantic_classifier(config_data, sample_out_size, char_vocab, nlabels, attribute.replace(' ', '_'))

        discriminator.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # output of the discriminator model are the outputs -> specifiy cross_entropy as loss

        # == == == == == == == == =
        # Pretrain Discriminators
        # == == == == == == == == =
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, min_delta=1e-4)
        model_checkpoint = ModelCheckpoint(join(model_path, 'sem_classifier_weights_{}.hdf5'.format(discriminator.name)), save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        logging.info('Pretrain the {} Discriminator'.format(discriminator.name))
        valid_char = valid_input['char_idx']
        valid_label = valid_input['da_act'][attribute]

        history = discriminator.fit(
            x=disX_train,
            y=y_train,
            validation_data=(valid_char, valid_label),
            epochs=config_data['pretrain_epochs'],
            batch_size=config_data['pretrain_batch_size'],
            callbacks=[early_stopping, model_checkpoint]
        )

        losses = history.history['loss']
        val_losses = history.history['val_loss']
        val_accs = history.history['val_acc']

        for i, (loss, val_loss, val_acc) in enumerate(zip(losses, val_losses, val_accs)):
            logging.info('Epoch: {} Loss: {} Val Loss: {} Val Acc: {}'.format(i, loss, val_loss, val_acc))


def load_discriminators(config_dict, mr_vec_lengths):
    model_path = config_dict['pretrain_model_path']
    sample_out_size = config_dict['max_sentence_len']

    data_base_path = config_dict['data_base_path']
    char_vocab_fname = config_dict['character_vocabulary']
    char_fname = open(os.path.join(data_base_path, char_vocab_fname), 'rt', encoding='utf-8')
    char_vocab = json.load(char_fname)

    discriminators = {}
    for attribute, nlabels in mr_vec_lengths.items():
        nlabels = mr_vec_lengths[attribute]
        discriminator = get_semantic_classifier(config_dict, sample_out_size, char_vocab, nlabels, attribute.replace(' ', '_'))
        logging.info('Loading the {} Discriminator'.format(discriminator.name))
        model_weights = join(model_path, 'sem_classifier_weights_{}.hdf5'.format(discriminator.name))
        discriminator.load_weights(model_weights)
        discriminators[attribute] = discriminator

    return discriminators


if __name__ == '__main__':
    log_path = 'logging/semantic_classifiers'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='{}/classifier_training.log'.format(log_path), filemode='w')

    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    out_tag = config_dict['processing_tag']

    data_base_path = config_dict['data_base_path']

    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file)

    train_fname = config_dict['train_data']
    train_delex = _load_vectorized_data(delex_data_path, train_fname)

    valid_fname = config_dict['valid_data']
    valid_delex = _load_vectorized_data(delex_data_path, valid_fname)

    test_fname = config_dict['test_data']
    test_delex = _load_vectorized_data(delex_data_path, test_fname)

    train_discriminators(config_dict, train_delex, valid_delex)

