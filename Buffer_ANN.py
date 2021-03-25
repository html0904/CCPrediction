'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import Callback
from sklearn.metrics import f1_score,roc_auc_score, precision_score, recall_score
import Models
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        y_pred = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_true = self.validation_data[1]
        
        tp = np.sum(np.multiply(y_pred, y_true), axis=1)
        pred_p = np.sum(y_pred, axis=1)
        true_p = np.sum(y_true, axis=1)
        
        precision = np.mean(np.nan_to_num(tp/pred_p))
        recall = np.mean(tp/true_p)
        f1 = np.mean(2*np.multiply(precision,recall)/(precision+recall))

        self.val_f1s.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        f1 = f1_score(y_true,y_pred,average='weighted')
        precision = precision_score(y_true,y_pred,average='weighted')
        recall = recall_score(y_true,y_pred,average='weighted')
        try:
            roc_auc = roc_auc_score(y_true,y_pred,average='weighted')
        except:
            roc_auc = 0
        print("— val_f1: %f — val_precision: %f — val_recall %f" %(f1, precision, recall))
        return
     
metrics = Metrics()



os.chdir('.')
X = np.load('aa_seq_onehot.npy') # one hot encoding of protein seq, flattened
Y = np.load('solvents_onehot.npy') # one hot encoding of solvents, see order above

order = np.random.choice(X.shape[0], X.shape[0], replace=False)

X = X[order].reshape(X.shape[0],X.shape[1],1)
Y = Y[order]
#X = X[order]
test_cut = int(X.shape[0]*0.8)
val_cut = int(test_cut*0.9)

X_train = X[:val_cut]
X_val = X[val_cut:test_cut]
X_test = X[test_cut:]

Y_train = Y[:val_cut]
Y_val = Y[val_cut:test_cut]
Y_test = Y[test_cut:]


counts = np.sum(Y,axis=0)
max_num = np.max(counts)
class_weight = {}
for i in range(Y.shape[1]):
    class_weight[i] = max_num/counts[i]

input_size = 40000
batch_size = 64
epochs = 1000

print('Loading data...')

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('x_train shape:', X_train.shape)
print('x_test shape:', Y_test.shape)


print('y_train shape:', Y_train.shape)
print('y_test shape:', Y_test.shape)

print('Building model...')
"""
model = Sequential()
model.add(Dense(512, input_shape=(input_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(192))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])


"""
from keras import optimizers
opt = optimizers.Adam(0.0005)
model = Models.Feedforward((input_size,1),192)
model.compile(loss='binary_crossentropy',optimizer=opt,
metrics=['accuracy']) 

history = model.fit(X_train, Y_train,
                    class_weight=class_weight,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, Y_val),
                    callbacks=[metrics],shuffle=True)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
