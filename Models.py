from keras.models import Sequential
from keras.layers import Dense,MaxPooling1D, Conv1D, Flatten, LSTM, BatchNormalization, Dropout, Activation, CuDNNLSTM, Embedding,Layer
from keras import regularizers

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def Convolutional_Feedforward(input_shape,output_size):
    model = Sequential()
    model.add(Conv1D(128,kernel_size=1000,strides=20,input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model

def Convolutional_Stacked(input_shape,output_size):
    model = Sequential()
    model.add(Conv1D(32,kernel_size=1000,strides=20,input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(32,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(32,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model

def Convolutional_5Stacked(input_shape,output_size):
    model = Sequential()
    model.add(Conv1D(16,kernel_size=1000,strides=20,input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(16,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(16,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(16,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(16,kernel_size=100,strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model

def Feedforward(input_shape,output_size):
    model = Sequential()
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model

def Convolutional_LSTM(input_shape,output_size):
    model = Sequential()
    model.add(Conv1D(16,kernel_size=100,strides=20,input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(32))
    #model.add(Attention(10))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model
