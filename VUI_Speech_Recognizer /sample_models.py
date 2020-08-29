from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional,
                          SimpleRNN, GRU, Dropout, MaxPooling1D)
from keras.regularizers import l2

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # Simple RNN
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    # softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # RNN with Batch Normalization
    rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='GRU_RNN')(input_data)
    bn_rnn = BatchNormalization(name='Batch_Norm')(rnn)
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bn_rnn)
    # softmax activation layer
    y_pred = Activation('softmax', name='Softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='Batch_Norm')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    x = input_data
    for i in range(recur_layers):
        x = GRU(units, activation='relu', return_sequences=True, implementation=2, name='GRU_RNN_{}'.format(i))(x)
        x = BatchNormalization(axis=-1, name='Batch_Norm_{}'.format(i))(x)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(x)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units=units, activation='relu', return_sequences=True, implementation=2,
                                  name='Bi_RNN_GRU'), merge_mode='concat')(input_data)
    # Add batch normalization
    bidir_rnn_bn = BatchNormalization(axis=-1, name='Batch_Norm')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn_bn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_kernel_regularized_rnn(input_dim, units, output_dim=29, l2_penalty=0.01):
    """ Build a bidirectional recurrent network for speech where,
        l2 regularizer function is applied to the kernel weights matrix
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer with kernel regularization
    bidir_rnn_kr = Bidirectional(GRU(units=units, activation='relu', return_sequences=True, implementation=2,
                                     kernel_regularizer=l2(l2_penalty), name='Bi_RNN_GRU_KrRg'),
                                     merge_mode='concat')(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn_kr)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_dropout_regularized_rnn(input_dim, units, output_dim=29, drop_prob=0.5):
    """ Build a bidirectional recurrent network for speech where,
        Dropout is applied to the output of the Bidirectional output.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer with kernel regularization
    bidir_rnn = Bidirectional(GRU(units=units, activation='relu', return_sequences=True,
                                     implementation=2, name='Bi_RNN_GRU'), merge_mode='concat')(input_data)
    # Add a Dropout Layer with externally provided rate
    bidir_rnn_dp = Dropout(rate=drop_prob, name='Dropout')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn_dp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def dilated2_cnn_bidirectional_rnn_dropout(input_dim, filters, kernel_size, conv_stride, conv_border_mode,
                                           units, drop_prob=0.5, output_dim=29):
    """ Build a dilated convolutional + batch normalization + recurrent + dropout + TimeDistributed Dense network
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # Add dilated convolutional layer
    conv_1d_dl = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=2,
                     name='conv1d_dilated2')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='batch_normalization')(conv_1d_dl)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units=units, activation='relu', return_sequences=True,
                                  implementation=2, name='Bi_RNN_GRU'), merge_mode='concat')(bn_cnn)
    # Add a Dropout Layer with externally provided rate
    bidir_rnn_dp = Dropout(rate=drop_prob, name='Dropout')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn_dp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, 2)
    print(model.summary())
    return model


def dilated2_cnn_wmaxpool_bidirectional_rnn(input_dim, filters, kernel_size, conv_stride, conv_border_mode,
                                            units, output_dim=29):
    """ 
    Build a dilated convolutional + max pool + batch normalization + recurrent network + TimeDistributed Dense network
    """

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # Add dilated convolutional layer
    conv_1d_dl = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=2,
                     name='conv1d_dilated2')(input_data)

    # Add a 1D max pool
    max_pool_1d = MaxPooling1D(pool_size=2, strides = 1, padding = 'same')(conv_1d_dl)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='batch_normalization')(max_pool_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units=units, activation='relu', return_sequences=True,
                                  implementation=2, name='Bi_RNN_GRU'), merge_mode='concat')(bn_cnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, 2)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, drop_prob=0.5, output_dim=29):
    """ 
     a dilated convolutional + max pool + batch normalization + recurrent layer + dropout + TimeDistributed Dense network
     
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Model Internal Architecture
    # Add dilated convolutional layer
    conv_1d_dl = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=2,
                     name='conv1d_dilated2')(input_data)

    # Add a 1D max pool
    max_pool_1d = MaxPooling1D(pool_size=2, strides = 1, padding = 'same')(conv_1d_dl)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='batch_normalization')(max_pool_1d)
    # Add a recurrent layer
    bidir_rnn = Bidirectional(GRU(units=units, activation='relu', return_sequences=True,
                                  implementation=2, name='Bi_RNN_GRU'), merge_mode='concat')(bn_cnn)
    # Add a Dropout Layer with externally provided rate
    bidir_rnn_dp = Dropout(rate=drop_prob, name='Dropout')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='Time_Distributed_Dense')(bidir_rnn_dp)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, 2)
    print(model.summary())
    return model