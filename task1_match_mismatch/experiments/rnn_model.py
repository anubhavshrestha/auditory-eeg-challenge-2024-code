import tensorflow as tf

def rnn_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    rnn_units=32,
    num_layers=3,
    activation="relu",
    compile=True,
    num_mismatched_segments=2
):
    eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [tf.keras.layers.Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments + 1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)

    stimuli_proj = [x for x in stimuli_input]

    if isinstance(activation, str):
        activations = [activation] * num_layers
    else:
        activations = activation

    # Spatial convolution
    rnn_proj_1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(eeg)

    # Construct RNN layers
    for layer_index in range(1, num_layers):
        rnn_proj_1 = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True if layer_index < num_layers - 1 else False,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(rnn_proj_1)

        # LSTM on envelope data, share weights
        env_proj_layer = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True if layer_index < num_layers - 1 else False,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]

    # Comparison
    cos = [tf.keras.layers.Dot(axes=1, normalize=True)([rnn_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(0.01))

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]

    # Classification
    out = tf.keras.activations.softmax(tf.keras.layers.Concatenate()(cos_proj))

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            metrics=["accuracy"],
            loss="categorical_crossentropy",
        )
        print(model.summary())

    return model
