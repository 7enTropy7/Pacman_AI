import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers import GlobalMaxPool2D, BatchNormalization, Dropout
from keras.optimizers import Adam

class Pacman:
    def __init__(self,state_size,action_size):
        self.gamma = 0.9            
        self.epsilon = 1.0          
        self.epsilon_min = 0.1      
        self.epsilon_decay = 0.995  
        self.update_rate = 1000 
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)

        self.model = self.get_model()
        self.target_model = self.get_model()
        self.target_model.set_weights(self.model.get_weights())

    def get_model(self):
        model = Sequential()
        # Conv Layers
        model.add(Conv2D(64, (3,3), padding="same",input_shape = self.state_size))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(GlobalMaxPool2D())
        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='nadam')
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            else:
                target = reward
                
            target_f = self.model.predict(state)
            
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

