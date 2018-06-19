import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
from transformations import get_transformations
import PIL.Image
import numpy as np
import time
import pickle

# datasets in the AutoAugment paper:
# CIFAR-10, CIFAR-100, SVHN, and ImageNet
# SVHN = http://ufldl.stanford.edu/housenumbers/

def get_dataset(dataset, reduced):
    if dataset == 'cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'reduced-cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    else:
        raise Exception('Unknown dataset %s' % dataset)
    if reduced:
        ix = np.random.choice(len(Xtr), 4000, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    ytr = utils.to_categorical(ytr)
    yts = utils.to_categorical(yts)
    return (Xtr, ytr), (Xts, yts)

(Xtr, ytr), (Xts, yts) = get_dataset('reduced-cifar10', True)
transformations = get_transformations(Xtr)

# Experiment parameters

LSTM_UNITS = 100

SUBPOLICIES = 5
SUBPOLICY_OPS = 2

OP_TYPES = 16
OP_PROBS = 11
OP_MAGNITUDES = 10

CHILD_BATCH_SIZE = 128
CHILD_BATCHES = len(Xtr) // CHILD_BATCH_SIZE
CHILD_EPOCHS = 120
CONTROLLER_EPOCHS = 500 # 15000 or 20000

class Operation:
    def __init__(self, types_softmax, probs_softmax, magnitudes_softmax):
        # Ekin Dogus says he sampled, and has not used argmax
        #self.type = types_softmax.argmax()
        #self.prob = probs_softmax.argmax() / (OP_PROBS-1)
        #m = magnitudes_softmax.argmax() / (OP_MAGNITUDES-1)
        self.type = np.random.choice(OP_TYPES, p=types_softmax)
        t = transformations[self.type]
        self.transformation = t[0]
        self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
        self.magnitude = np.random.choice(np.linspace(t[1], t[2], OP_MAGNITUDES), p=magnitudes_softmax)

    def __call__(self, X):
        _X = []
        for x in X:
            if np.random.rand() < self.prob:
                x = PIL.Image.fromarray(x)
                x = self.transformation(x, self.magnitude)
            _X.append(np.array(x))
        return np.array(_X)

    def __str__(self):
        return 'Operation %2d (P=%.3f, M=%.3f)' % (self.type, self.prob, self.magnitude)

class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, X):
        for op in self.operations:
            X = op(X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret

class Controller:
    def __init__(self):
        self.model = self.create_model()
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)
        # negative for gradient ascent
        self.grads = [g * (-self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optimizer = tf.train.GradientDescentOptimizer(0.00035).apply_gradients(self.grads)

    def create_model(self):
        input_layer = layers.Input(shape=(SUBPOLICIES, 1))
        init = initializers.RandomUniform(-0.1, 0.1)
        lstm_layer = layers.LSTM(
            LSTM_UNITS, recurrent_initializer=init, return_sequences=True,
            name='controller')(input_layer)
        outputs = []
        for i in range(SUBPOLICY_OPS):
            name = 'op%d-' % (i+1)
            outputs += [
                layers.Dense(OP_TYPES, activation='softmax', name=name + 't')(lstm_layer),
                layers.Dense(OP_PROBS, activation='softmax', name=name + 'p')(lstm_layer),
                layers.Dense(OP_MAGNITUDES, activation='softmax', name=name + 'm')(lstm_layer),
            ]
        return models.Model(input_layer, outputs)

    def fit(self, mem_softmaxes, mem_accuracies):
        session = backend.get_session()
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        dummy_input = np.zeros((1, SUBPOLICIES, 1))
        dummy_input[0, 0, 0] = 1
        dict_input = {self.model.input: dummy_input}
        # FIXME: the paper does mini-batches (10)
        for softmaxes, acc in zip(mem_softmaxes, mem_accuracies):
            scale = (acc-min_acc) / (max_acc-min_acc)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
            dict_scales = {self.scale: scale}
            session.run(self.optimizer, feed_dict={**dict_outputs, **dict_scales, **dict_input})
        return self

    def predict(self, size):
        dummy_input = np.zeros((1, size, 1), np.float32)
        dummy_input[0, 0, 0] = 1
        softmaxes = self.model.predict(dummy_input)
        # convert softmaxes into subpolicies
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*3:(j+1)*3]
                op = [o[0, i, :] for o in op]
                operations.append(Operation(*op))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies

# generator
def autoaugment(subpolicies, X, y):
    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(CHILD_BATCHES):
            _ix = ix[i*CHILD_BATCH_SIZE:(i+1)*CHILD_BATCH_SIZE]
            _X = X[_ix]
            _y = y[_ix]
            subpolicy = np.random.choice(subpolicies)
            _X = subpolicy(_X)
            _X = _X.astype(np.float32) / 255
            yield _X, _y

class Child:
    # architecture from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
        optimizer = optimizers.SGD(decay=1e-4)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)

    def fit(self, subpolicies, X, y):
        gen = autoaugment(subpolicies, X, y)
        self.model.fit_generator(
            gen, CHILD_BATCHES, CHILD_EPOCHS, verbose=0, use_multiprocessing=True)
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

mem_softmaxes = []
mem_accuracies = []

controller = Controller()

for epoch in range(CONTROLLER_EPOCHS):
    print('Controller: Epoch %d / %d' % (epoch+1, CONTROLLER_EPOCHS))

    softmaxes, subpolicies = controller.predict(SUBPOLICIES)
    for i, subpolicy in enumerate(subpolicies):
        print('# Sub-policy %d' % (i+1))
        print(subpolicy)
    mem_softmaxes.append(softmaxes)

    child = Child(Xtr.shape[1:])
    tic = time.time()
    child.fit(subpolicies, Xtr, ytr)
    toc = time.time()
    accuracy = child.evaluate(Xts, yts)
    print('-> Child accuracy: %.3f (elaspsed time: %ds)' % (accuracy, (toc-tic)))
    mem_accuracies.append(accuracy)

    if len(mem_softmaxes) > 5:
        # ricardo: I let some epochs pass, so that the normalization is more robust
        controller.fit(mem_softmaxes, mem_accuracies)
    print()

print()
print('Best policies found:')
print()
_, subpolicies = controller.predict(25)
for i, subpolicy in enumerate(subpolicies):
    print('# Subpolicy %d' % (i+1))
    print(subpolicy)
