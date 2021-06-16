# Chapter 10, pp320 Fine-Tuning Neural Network Hyperparameters
import tensorflow as tf

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def build_model(n_hidden=1, n_neurons=64, learning_rate=3e-3, input_shape=None):
    if input_shape is None:
        input_shape = [7]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        # initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)  # best performer
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu', kernel_initializer=initializer, bias_initializer='ones'))  #
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
        #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=METRICS)  # metrics=['accuracy'])
        # mse - ok
        # categorical_crossentropy : labels to be provided in one_hot representation
        #
    # print(model.summary())
    return model
