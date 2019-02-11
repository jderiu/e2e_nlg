from keras.layers import Recurrent, concatenate

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec

from src.architectures.custom_layers.recurrent_tensorflow import sc_tf_rnn


class SC_LSTM(Recurrent):
        def __init__(self, units, out_units,
                     alpha=0.2,
                     softmax_temperature=None,
                     return_da=True,
                     activation='tanh',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     out_kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     out_kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     out_kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.,
                     recurrent_dropout=0.,
                     sc_dropout=0.,
                     **kwargs):
            super(SC_LSTM, self).__init__(**kwargs)
            self.units = units
            self.alpha = alpha
            self.out_units = out_units
            self.activation = activations.get(activation)
            self.recurrent_activation = activations.get(recurrent_activation)
            self.use_bias = use_bias
            self.return_da = return_da
            self.softmax_temperature = softmax_temperature

            #different behaviour while training than from inefrence time
            self.train_phase = True

            self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]

            self.kernel_initializer = initializers.get(kernel_initializer)
            self.out_kernel_initializer = initializers.get(out_kernel_initializer)
            self.recurrent_initializer = initializers.get(recurrent_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.unit_forget_bias = unit_forget_bias

            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.out_kernel_regularizer = regularizers.get(out_kernel_regularizer)
            self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)

            self.kernel_constraint = constraints.get(kernel_constraint)
            self.out_kernel_constraint = constraints.get(out_kernel_constraint)
            self.recurrent_constraint = constraints.get(recurrent_constraint)
            self.bias_constraint = constraints.get(bias_constraint)

            self.dropout = min(1., max(0., dropout))
            self.recurrent_dropout = min(1., max(0., recurrent_dropout))
            self.sc_dropout = min(1., max(0., sc_dropout))

        def build(self, input_shape):
            assert isinstance(input_shape, list) or isinstance(input_shape, tuple)

            if isinstance(input_shape, tuple):
                main_input_shape = input_shape
            else:
                main_input_shape = input_shape[0]
            batch_size = main_input_shape[0] if self.stateful else None

            #takes orig_input while training and dialogue act for conditioning
            if self.state_spec:
                assert len(input_shape) == 4
            else:
                assert len(input_shape) == 2
            self.input_dim = main_input_shape[2]

            diact_shape = input_shape[-1]
            self.dialogue_act_dim = diact_shape[-1]

            self.input_spec[0] = InputSpec(shape=(batch_size, None, main_input_shape[2]))

            #h,c
            self.states = [None, None]
            if self.stateful:
                self.reset_states()

            self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units * 4),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint)

            ### Semantic Conditioning Weights Weights
            self.kernel_d = self.add_weight(
                shape=(self.dialogue_act_dim, self.units),
                name='diag_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )

            self.kernel_r = self.add_weight(
                shape=(self.input_dim, self.dialogue_act_dim),
                name='kernel_r',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )

            self.recurrent_kernel_r = self.add_weight(
                shape=(self.units, self.dialogue_act_dim),
                name='recurrent_kernel_r',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.kernel_constraint
            )

            self.out_kernel = self.add_weight(shape=(self.units, self.out_units),
                                              name='out_kernel',
                                              initializer=self.out_kernel_initializer,
                                              regularizer=self.out_kernel_regularizer,
                                              constraint=self.out_kernel_constraint)

            if self.use_bias:
                if self.unit_forget_bias:
                    def bias_initializer(shape, *args, **kwargs):
                        return K.concatenate([
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ])


                else:
                    bias_initializer = self.bias_initializer

                self.bias = self.add_weight(shape=(self.units * 4,),
                                            name='bias',
                                            initializer=bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

            self.kernel_i = self.kernel[:, :self.units]
            self.kernel_f = self.kernel[:, self.units * 1: self.units * 2]
            self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
            self.kernel_o = self.kernel[:, self.units * 3:]

            self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
            self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
            self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
            self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

            if self.use_bias:
                self.bias_i = self.bias[:self.units]
                self.bias_f = self.bias[self.units: self.units * 2]
                self.bias_c = self.bias[self.units * 2: self.units * 3]
                self.bias_o = self.bias[self.units * 3:]

                self.bias_r = self.add_weight(
                    shape=(self.dialogue_act_dim, ),
                    name='bias_r',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )

            else:
                self.bias_i = None
                self.bias_f = None
                self.bias_c = None
                self.bias_o = None
                self.bias_r = None
            self.built = True

        def preprocess_input(self, inputs, training=None):
                return inputs

        def compute_output_shape(self, input_shape):
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            if self.return_sequences:
                output_shape = (input_shape[0], input_shape[1], self.out_units)
            else:
                output_shape = (input_shape[0], self.out_units)

            if self.return_state:
                state_shape = [(input_shape[0], self.units) for _ in self.states]
                return [output_shape] + state_shape
            if self.return_da:
                da_state_shape = (input_shape[0], self.dialogue_act_dim)
                da_outs_shape = (input_shape[0], input_shape[1], self.dialogue_act_dim)
                return [output_shape, da_state_shape, da_outs_shape]
            else:
                return output_shape

        def compute_mask(self, inputs, mask):
            if isinstance(mask, list):
                mask = mask[0]
            output_mask = mask if self.return_sequences else None
            if self.return_state:
                state_mask = [None for _ in self.states]
                return [output_mask] + state_mask
            elif self.return_da:
                state_mask = None
                return [output_mask, state_mask, state_mask]
            else:
                return output_mask

        def get_constants(self, inputs, training=None):
            constants = []
            if self.implementation != 0 and 0 < self.dropout < 1:
                input_shape = K.int_shape(inputs)
                input_dim = input_shape[-1]
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, int(input_dim)))

                def dropped_inputs():
                    return K.dropout(ones, self.dropout)

                dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(5)]
                constants.append(dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(5)])

            if 0 < self.recurrent_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))

                def dropped_inputs():
                    return K.dropout(ones, self.recurrent_dropout)

                rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                                ones,
                                                training=training) for _ in range(5)]
                constants.append(rec_dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(5)])
            return constants

        def get_sc_constants(self, inputs, training=None):
            constants = []
            if 0 < self.sc_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.dialogue_act_dim))

                def dropped_inputs():
                    return K.dropout(ones, self.sc_dropout)

                rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                                ones,
                                                training=training) for _ in range(5)]
                constants.append(rec_dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(5)])
            return constants

        def get_initial_state(self, inputs):
            # build an all-zero tensor of shape (samples, output_dim)
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
            initial_state = [initial_state for _ in range(len(self.states))]
            return initial_state

        def get_initial_p(self, inputs):
            # build an all-zero tensor of shape (samples, output_dim)
            p_0 = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            p_0 = K.sum(p_0, axis=(1, 2))  # (samples,)
            p_0 = K.expand_dims(p_0)  # (samples, 1)
            p_0 = K.tile(p_0, [1, self.out_units])  # (samples, output_dim)
            return [p_0]

        def inference_phase(self):
            self.train_phase = False

        def __call__(self, inputs, initial_state=None, **kwargs):

            # If `initial_state` is specified,
            # and if it a Keras tensor,
            # then add it to the inputs and temporarily
            # modify the input spec to include the state.
            if initial_state is None:
                return super(Recurrent, self).__call__(inputs, **kwargs)

            if not isinstance(initial_state, (list, tuple)):
                initial_state = [initial_state]

            is_keras_tensor = hasattr(initial_state[0], '_keras_history')
            for tensor in initial_state:
                if hasattr(tensor, '_keras_history') != is_keras_tensor:
                    raise ValueError('The initial state of an RNN layer cannot be'
                                     ' specified with a mix of Keras tensors and'
                                     ' non-Keras tensors')

            if is_keras_tensor:
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(input_spec, list):
                    input_spec = [input_spec]
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = input_spec + state_spec

                # Compute the full inputs, including state
                inputs = inputs + list(initial_state)

                # Perform the call
                output = super(Recurrent, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
                return super(Recurrent, self).__call__(inputs, **kwargs)

        def call(self, inputs, mask=None, training=None, initial_state=None):
            # input shape: `(samples, time (padded with zeros), input_dim)`
            # note that the .build() method of subclasses MUST define
            # self.input_spec and self.state_spec with complete input shapes.

            # input for training [aux_softmax, ground thruth, dialogue act vector]
            input_length = K.int_shape(inputs[0])[1]
            input_list = inputs

            #takes orig_input while training and dialogue act for conditioning
            inputs = input_list[0]
            initial_state = self.get_initial_state(inputs)
            constants = self.get_constants(inputs, training=None)

            dialogue_act = input_list[-1]
            initial_state = initial_state + [dialogue_act]
            sc_constants = self.get_sc_constants(dialogue_act, training=None)
            constants = constants + sc_constants

            p0 = self.get_initial_p(inputs)
            initial_state += p0

            if isinstance(mask, list):
                mask = mask[0]

            preprocessed_input = self.preprocess_input(inputs, training=None)
            rnn_output = sc_tf_rnn(
                self.step,
                preprocessed_input,
                initial_state,
                semantic_conditioning=True,
                go_backwards=self.go_backwards,
                mask=mask,
                constants=constants)

            last_output, outputs, last_da, da_outputs, states = rnn_output

            # Properly set learning phase
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True
            da_outputs._uses_learning_phase = True
            last_da._uses_learning_phase = True

            if self.return_sequences:
                output = outputs
            else:
                output = last_output

            if self.return_state:
                if not isinstance(states, (list, tuple)):
                    states = [states]
                else:
                    states = list(states)
                output = [output] + states

            if self.return_da:
                output = [output, last_da, da_outputs]

            return output

        def step(self, inputs, states):

            h_tm1 = states[0]
            c_tm1 = states[1]
            d_tm1 = states[2]
            p_tm1 = states[3]

            inputs = K.in_train_phase(inputs, p_tm1)
            dp_mask = states[4]
            rec_dp_mask = states[5]
            sc_dp_mask = states[6]

            z = K.dot(inputs * dp_mask[0], self.kernel)
            z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)

            r = self.recurrent_activation(K.dot(inputs * dp_mask[0], self.kernel_r) + self.alpha * K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_r))
            if self.use_bias:
                r = K.bias_add(r, self.bias_r)
            d = r*d_tm1
            c = f * c_tm1 + i * self.activation(z2) + self.activation(K.dot(d * sc_dp_mask[0], self.kernel_d))

            o = self.recurrent_activation(z3)

            h = o * self.activation(c)

            #output distibution of target word prob: p in (batch_size, nclasses)
            if self.softmax_temperature is not None:
                p_softmax = K.softmax(K.dot(h, self.out_kernel)/self.softmax_temperature)
                p_ret = p_softmax
            else:
                p_softmax = K.softmax(K.dot(h, self.out_kernel))
                p_ret = K.in_train_phase(p_softmax, K.one_hot(K.argmax(p_softmax, axis=1), self.out_units))

            if 0.0 < self.dropout + self.recurrent_dropout + self.sc_dropout:
                h._uses_learning_phase = True

            return p_softmax, [h, c, d, p_ret]

        def get_config(self):
            config = {'units': self.units,
                      'activation': activations.serialize(self.activation),
                      'recurrent_activation': activations.serialize(self.recurrent_activation),
                      'use_bias': self.use_bias,
                      'kernel_initializer': initializers.serialize(self.kernel_initializer),
                      'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                      'bias_initializer': initializers.serialize(self.bias_initializer),
                      'unit_forget_bias': self.unit_forget_bias,
                      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                      'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                      'kernel_constraint': constraints.serialize(self.kernel_constraint),
                      'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                      'bias_constraint': constraints.serialize(self.bias_constraint),
                      'dropout': self.dropout,
                      'recurrent_dropout': self.recurrent_dropout,
                      'sc_dropout': self.sc_dropout}
            base_config = super(SC_LSTM, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))