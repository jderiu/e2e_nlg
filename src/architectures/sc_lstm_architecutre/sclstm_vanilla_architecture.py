import keras.backend as K
import numpy as np
from keras.layers import Lambda, Embedding, Input, concatenate, ZeroPadding1D

from src.architectures.custom_layers.sem_recurrent import SC_LSTM
from src.architectures.custom_layers.word_dropout import WordDropout
from keras.models import Model
from typing import Dict


def sc_lstm_decoder(text_idx, text_one_hot, dialogue_act, nclasses, sample_out_size, lstm_size, inputs, step):

    def remove_last_column(x):
        return x[:, :-1, :]

    padding = ZeroPadding1D(padding=(1, 0))(text_one_hot)
    previous_char_slice = Lambda(remove_last_column, output_shape=(sample_out_size, nclasses))(padding)

    temperature = 1 / step

    lstm = SC_LSTM(
        lstm_size,
        nclasses,
        softmax_temperature=None,
        return_da=True,
        return_state=False,
        use_bias=True,
        return_sequences=True,
        implementation=2,
        dropout=0.2,
        recurrent_dropout=0.2,
        sc_dropout=0.2
    )

    recurrent_component, da_t, da_history = lstm([previous_char_slice, dialogue_act])

    decoder = Model(inputs=inputs + [text_idx], outputs=[recurrent_component, da_t, da_history], name='decoder_{}'.format('train'))
    return decoder


def generator_model(config_dict: Dict, mr_vec_lengths: Dict[str, int], surface_lengts, vocab, step):

    sample_out_size = config_dict['max_sentence_len']
    nclasses = len(vocab) + 3
    #last available index is reserved as start character
    lstm_size = config_dict['lstm_size']
    max_idx = max(vocab.values())
    dropout_word_idx = max_idx + 1

    surface_features = config_dict['surface_features']

    # == == == == == =
    # Define Inputs
    # == == == == == =
    semantic_inputs = []
    for attribute in sorted(list(mr_vec_lengths.keys())):
        vec_len = mr_vec_lengths[attribute]
        attr_idx = Input(batch_shape=(None, vec_len), dtype='float32', name='{}_idx'.format(attribute.replace(' ', '_')))
        semantic_inputs.append(attr_idx)

    surface_inputs = []
    for surface_feature, opts in sorted(list(surface_features.items()), key=lambda x: x[0]):
        dim = surface_lengts[surface_feature]
        if opts:
            for opt in opts:
                attr_idx = Input(batch_shape=(None, dim), dtype='float32', name='{}_{}'.format(surface_feature, opt))
                surface_inputs.append(attr_idx)
        else:
            attr_idx = Input(batch_shape=(None, dim), dtype='float32', name=surface_feature)
            surface_inputs.append(attr_idx)

    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    inputs = semantic_inputs + surface_inputs

    word_dropout = WordDropout(rate=0.5, dummy_word=dropout_word_idx, anneal_step=step, anneal_start=1000, anneal_end=2000)(output_idx)

    one_hot_weights = np.identity(nclasses)

    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )

    #output_one_hot_embeddings_worddrop = one_hot_out_embeddings(word_dropout)
    output_one_hot_embeddings = one_hot_out_embeddings(output_idx)

    dialogue_act = concatenate(inputs=inputs)

    decoder = sc_lstm_decoder(output_idx, output_one_hot_embeddings, dialogue_act, nclasses, sample_out_size, lstm_size, inputs, step)

    def vae_cross_ent_loss(args):
        x_truth, x_decoded_final = args
        x_truth_flatten = K.reshape(x_truth, shape=(-1, K.shape(x_truth)[-1]))
        x_decoded_flat = K.reshape(x_decoded_final, shape=(-1, K.shape(x_decoded_final)[-1]))
        cross_ent = K.categorical_crossentropy(x_truth_flatten, x_decoded_flat)
        cross_ent = K.reshape(cross_ent, shape=(-1, K.shape(x_truth)[1]))
        sum_over_sentences = K.sum(cross_ent, axis=1)
        return sum_over_sentences

    def da_loss_fun(args):
        da = args[0]
        sq_da_t = K.square(da)
        K.l2_normalize(da, axis=1)
        sum_sq_da_T = K.sum(sq_da_t, axis=1)
        return sum_sq_da_T

    def da_history_loss_fun(args):
        da_t = args[0]
        zeta = 100
        n = 1e-4
        # shape: batch_size, sample_size
        norm_of_differnece = K.sum(zeta**K.abs(da_t[:, 1:, :] - da_t[:, :-1, :]), axis=2)
        n1 = n * norm_of_differnece
        return K.sum(n1, axis=1)

    def argmax_fun(softmax_output):
        return K.argmax(softmax_output, axis=2)

    def max_avg_fun(softmax_output):
        return K.mean(K.max(softmax_output, axis=2), axis=1)

    x_p, last_da, da_history = decoder(inputs + [output_idx])

    def remove_first_column(x):
        return x[:, 1:, :]

    def get_last_column(x):
        return x[:, -1, :]

    argmax = Lambda(argmax_fun, output_shape=(sample_out_size,))(x_p)

    last_sample = Lambda(get_last_column, output_shape=(nclasses, ))(x_p)
    confidence_value = Lambda(max_avg_fun, output_shape=(1,))(x_p)

    main_loss = Lambda(vae_cross_ent_loss, output_shape=(1,), name='main')([output_one_hot_embeddings, x_p])
    da_loss = Lambda(da_loss_fun, output_shape=(1,), name='dialogue_act')([last_da])
    da_history_loss = Lambda(da_history_loss_fun, output_shape=(1,), name='dialogue_history')([da_history])

    train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss, da_loss, da_history_loss])
    #train_model = Model(inputs=inputs + [output_idx], outputs=[main_loss])
    test_model = Model(inputs=inputs + [output_idx], outputs=[argmax, confidence_value])

    return train_model, test_model



