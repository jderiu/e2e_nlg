import numpy as np
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Activation, Dense
from keras.models import Model


def conv_block_layered(input, nfilter, nlayers=2, kernel_size=3):
    conv = Conv1D(filters=nfilter, kernel_size=kernel_size, strides=2, padding='same', name='conv_layer_{}'.format(0))(input)
    tmp_relu = Activation(activation='relu', name='prelu_{}'.format(0))(conv)

    for layer in range(1, nlayers):
        # oshape = (batch_size, sample_size/2**layer+1, nkernels*2**nlayer)
        conv = Conv1D(filters=2*nfilter, kernel_size=kernel_size, strides=2, padding='same', name='conv_layer_{}'.format(layer))(tmp_relu)
        #tmp_bn = BatchNormalization(name='bn_{}'.format(layer))(conv)
        tmp_relu = Activation(activation='relu', name='prelu_{}'.format(layer))(conv)
        # oshape = (batch_size, sample_size/4*256)
    max_pool = GlobalMaxPooling1D()(tmp_relu)

    return max_pool


def get_classifier_architecture(g_in, nlabel, name, nfilter, hidden_units, kernel_size, nlayers):
    in_conv0 = conv_block_layered(g_in, nfilter, nlayers, kernel_size)

    hidden_intermediate_discr = Dense(hidden_units, activation='relu', name='discr_activation')(in_conv0)

    logits = Dense(nlabel, activation='linear', name='hidden_{}'.format(name))(hidden_intermediate_discr)

    softmax = Activation(activation='softmax', name='softmax_{}'.format(name))(logits)

    discriminator = Model(inputs=g_in, outputs=softmax, name='{}_disc'.format(name))
    return discriminator


def get_semantic_classifier(config_data, sample_out_size, vocab, nlabel, name):
    nclasses = len(vocab) + 3

    nfilter = config_data['nb_filter']
    nlayers = config_data['nlayers']
    filter_length = config_data['filter_length']
    intermediate_dim = config_data['intermediate_dim']

    one_hot_weights = np.identity(nclasses)

    output_idx = Input(batch_shape=(None, sample_out_size), dtype='int32', name='character_output')

    one_hot_out_embeddings = Embedding(
        input_length=sample_out_size,
        input_dim=nclasses,
        output_dim=nclasses,
        weights=[one_hot_weights],
        trainable=False,
        name='one_hot_out_embeddings'
    )

    text_one_hot = one_hot_out_embeddings(output_idx)

    dis_input = Input(shape=(sample_out_size, nclasses))

    discriminator = get_classifier_architecture(dis_input, nlabel, name, nfilter, intermediate_dim, kernel_size=filter_length, nlayers=nlayers)

    discr_train_losses = discriminator(text_one_hot)
    discriminator_model = Model(inputs=output_idx, outputs=discr_train_losses, name='{}'.format(name))
    return discriminator_model

