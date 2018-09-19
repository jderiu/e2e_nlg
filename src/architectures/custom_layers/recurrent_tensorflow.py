import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables


def reverse(x, axes):
    """Reverse a tensor along the specified axes.

    # Arguments
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.

    # Returns
        A tensor.
    """
    if isinstance(axes, int):
        axes = [axes]
    return tf.reverse(x, axes)


def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    return tf.expand_dims(x, axis)


def sc_tf_rnn(
        step_function,
        inputs,
        initial_states,
        semantic_conditioning=True,
        go_backwards=False,
        mask=None,
        constants=None
):
    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.get_shape()) == ndim - 1:
            mask = expand_dims(mask)
        mask = tf.transpose(mask, axes)

    if constants is None:
        constants = []

    global uses_learning_phase
    uses_learning_phase = False

    if go_backwards:
        inputs = reverse(inputs, 0)

    states = tuple(initial_states)

    time_steps = tf.shape(inputs)[0]
    outputs, _ = step_function(inputs[0], initial_states + constants)
    output_ta = tensor_array_ops.TensorArray(
        dtype=outputs.dtype,
        size=time_steps,
        tensor_array_name='output_ta')
    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype,
        size=time_steps,
        tensor_array_name='input_ta')
    input_ta = input_ta.unstack(inputs)
    time = tf.constant(0, dtype='int32', name='time')

    da = states[2]
    dialogue_act_ta = tensor_array_ops.TensorArray(
        dtype=da.dtype,
        size=time_steps,
        tensor_array_name='dialogue_act_ta',
        clear_after_read=False
    )

    def _step(time, output_ta_t, dialogue_act_ta_t, *states):
        """RNN step function.

        # Arguments
            time: Current timestep value.
            output_ta_t: TensorArray.
            dialogue_act_ta_t: TensorArray.
            *states: List of states.

        # Returns
            Tuple: `(time + 1,output_ta_t, dialogue_act_ta_t) + tuple(new_states)`
        """
        current_input = input_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) +
                                           tuple(constants))
        if getattr(output, '_uses_learning_phase', False):
            global uses_learning_phase
        for state, new_state in zip(states, new_states):
            new_state.set_shape(state.get_shape())
        output_ta_t = output_ta_t.write(time, output)
        dialogue_act_ta_t = dialogue_act_ta_t.write(time, new_states[2])
        return (time + 1, output_ta_t, dialogue_act_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_step,
        loop_vars=(time, output_ta, dialogue_act_ta) + states,
        parallel_iterations=32,
        swap_memory=True)

    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    dialogue_act_ta = final_outputs[2]
    new_states = final_outputs[3:]

    da_outputs = dialogue_act_ta.stack()
    last_da = dialogue_act_ta.read(last_time - 1)

    outputs = output_ta.stack()
    last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    da_outputs = tf.transpose(da_outputs, axes)

    last_output._uses_learning_phase = uses_learning_phase
    return last_output, outputs, last_da, da_outputs, new_states


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1):
    """Decodes the output of a softmax.

    Can use either greedy search (also known as best path)
    or a constrained dictionary search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.

    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
    input_length = tf.to_int32(input_length)

    if greedy:
        (decoded, log_prob) = ctc.ctc_greedy_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            merge_repeated=False
        )
    else:
        (decoded, log_prob) = ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
            merge_repeated=False
        )

    decoded_dense = [tf.sparse_to_dense(st.indices, st.dense_shape, st.values, default_value=-1)
                     for st in decoded]
    return (decoded_dense, log_prob)