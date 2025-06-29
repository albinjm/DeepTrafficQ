import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
from collections import deque

from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adamax

class DQNAgent:
    def __init__(self):
        self.gamma = 0.75
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.memory = deque(maxlen=200)
        self.action_size = 2
        self.model = self._build_model()

    def _build_model(self):
        input1 = Input(shape=(12, 12, 1))
        input2 = Input(shape=(12, 12, 1))
        input3 = Input(shape=(2, 1))

        x1 = Conv2D(16, (4,4), strides=(2,2), activation='relu')(input1)
        x1 = Conv2D(32, (2,2), strides=(1,1), activation='relu')(x1)
        x1 = Flatten()(x1)

        x2 = Conv2D(16, (4,4), strides=(2,2), activation='relu')(input2)
        x2 = Conv2D(32, (2,2), strides=(1,1), activation='relu')(x2)
        x2 = Flatten()(x2)

        x3 = Flatten()(input3)

        x = Concatenate()([x1, x2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=[input1, input2, input3], outputs=x)

        optimizer = Adamax(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(loss='mse', optimizer=optimizer)
        return model

    def ensure_shape(self, arr, target_shape):
        arr = np.array(arr, dtype=np.float32)
        arr = np.squeeze(arr)  
        
        if target_shape == (1, 12, 12, 1):
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            arr = np.expand_dims(arr, axis=0) 
        
        elif target_shape == (1, 2, 1):
            if arr.ndim == 1: 
                arr = np.expand_dims(arr, axis=-1) 
            arr = np.expand_dims(arr, axis=0)  
            
        return arr

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        s1 = self.ensure_shape(state[0], (1, 12, 12, 1))
        s2 = self.ensure_shape(state[1], (1, 12, 12, 1))
        s3 = self.ensure_shape(state[2], (1, 2, 1))
        state_arrays = [s1, s2, s3]

        act_values = self.model.predict(state_arrays, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            s1 = self.ensure_shape(state[0], (1, 12, 12, 1))
            s2 = self.ensure_shape(state[1], (1, 12, 12, 1))
            s3 = self.ensure_shape(state[2], (1, 2, 1))
            state_arrays = [s1, s2, s3]

            ns1 = self.ensure_shape(next_state[0], (1, 12, 12, 1))
            ns2 = self.ensure_shape(next_state[1], (1, 12, 12, 1))
            ns3 = self.ensure_shape(next_state[2], (1, 2, 1))
            next_state_arrays = [ns1, ns2, ns3]

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state_arrays, verbose=0)[0])

            target_f = self.model.predict(state_arrays, verbose=0)
            target_f[0][action] = target
            self.model.fit(state_arrays, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)