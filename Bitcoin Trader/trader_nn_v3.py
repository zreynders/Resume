import tensorflow as tf
import random
import numpy as np
import math
import csv
import time
import pandas as pd
from datetime import datetime, date, timedelta
import pickle
import requests
import os
#gui stuff
import pyglet

#historical data
df = pd.read_csv('data/historical.csv', skip_blank_lines=True, parse_dates=['TIMESTAMP'], index_col=['TIMESTAMP'])
df = df.sort_values(by=['TIMESTAMP'])
price1day = df.at[datetime((date.today() - timedelta(days=1)).year, (date.today() - timedelta(days=1)).month, (date.today() - timedelta(days=1)).day), 'PRICE']
price1week = df.at[datetime((date.today() - timedelta(days=7)).year, (date.today() - timedelta(days=7)).month, (date.today() - timedelta(days=7)).day), 'PRICE']

#get volume
response = requests.get('https://api.pro.coinbase.com/products/BTC-USD/stats')
data = response.json()
volume = np.float64(data.get('volume'))

#state vars
file = open(r"data/current_price.txt","r")
price = np.float64(file.readline()) * 0.00000001
file.close()

#trader state vars
short_assets = 0
short_price = 0.0
assets = 0
pool = 100000.0
cnt = 0
#unpickle simple data
if os.path.exists("data/simple_data.tr"):
    data = pickle.load(open("data/simple_data.tr", "rb"))
    cnt = data['cnt']
    pool = data['pool']
    assets = data['assets']
    short_assets = data['short_assets']
    short_price = data['short_price']
owned = pool + assets * price
state = np.array([price, assets * price, pool, price1day, price1week, volume, short_assets, short_price])

#constants
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
RANDOM_REWARD_STD = 1.0

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

#trader stuff
class BTCTrader:
    def __init__(self, sess, model, memory, max_eps, min_eps, decay, render=True):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_up = 0

    def run(self):
        #import global vars
        global assets
        global pool
        global price
        global state
        global owned
        global price1day
        global price1week
        global volume
        global short_assets
        global short_price
        #calc actions
        action = self._choose_action(state)
        """ POSSIBLE ACTIONS
            0: Stay
            1: Buy $1
            2: Buy $5
            3: Buy $10
            4: Buy $50
            5: Sell $1
            6: Sell $5
            7: Sell $10
            8: Sell All
            SHORT RELATED
            9: Sell Short: $25
            10: Sell Short: $50
            11: Sell Short: $100
            12: Buy Short

            if try to sell over owned assets, sell All
            if try to buy over pool, buy All
            if try to sell with nothing, Stay
            if try to buy with no pool, Stay
        """
        print ('Old Pool: ({0:.16f}), Old Assets: ({1:.16f})'.format(pool, assets * price))
        if action == 1:
            if pool == 0:
                action = 0
            elif pool < 1:
                action = 4
            else:
                action_label = 'Buy $1'
                assets += round(1 / price)
                pool -= (round(1 / price) * price) * 1.25
        elif action == 2:
            if pool == 0:
                action = 0
            elif pool < 5:
                action = 4
            else:
                action_label = 'Buy $5'
                assets += round(5 / price)
                pool -= (round(5 / price) * price) * 1.25
        elif action == 3:
            if pool == 0:
                action = 0
            elif pool < 10:
                action = 4
            else:
                action_label = 'Buy $10'
                assets += round(10 / price)
                pool -= (round(10 / price) * price) * 1.25
        elif action == 5:
            if assets == 0:
                action = 0
            elif assets * price < 1:
                action = 8
            else:
                action_label = 'Sell $1'
                pool += (round(1 / price) * price) * 0.75
                assets -= round(1 / price)
        elif action == 6:
            if assets == 0:
                action = 0
            elif assets * price < 5:
                action = 8
            else:
                action_label = 'Sell $5'
                pool += (round(5 / price) * price) * 0.75
                assets -= round(5 / price)
        elif action == 7:
            if assets == 0:
                action = 0
            elif assets * price < 10:
                action = 8
            else:
                action_label = 'Sell $10'
                pool += (round(10 / price) * price) * 0.75
                assets -= round(10 / price)
        elif action == 9:
            if short_assets == 0:
                action_label = 'Sell Short $25'
                short_assets -= round(25 / price)
                short_price = price
            else:
                action = 0
        elif action == 10:
            if short_assets == 0:
                action_label = 'Sell Short $50'
                short_assets -= round(50 / price)
                short_price = price
            else:
                action = 0
        elif action == 11:
            if short_assets == 0:
                action_label = 'Sell Short $100'
                short_assets -= round(100 / price)
                short_price = price
            else:
                action = 0
        elif action == 12:
            action_label = 'Buy Short'
            pool += (-short_assets * short_price) - (-short_assets * price)
            short_assets = 0
            short_price = 0.0

        if action == 4:
            action_label = 'Buy $50'
            assets += round(50 / price)
            pool -= (round(50 / price) * price) * 1.25
        if action == 8:
            action_label = 'Sell All'
            pool += (assets * price) * 0.75
            assets = 0
        if action == 0:
            action_label = 'Stay'
        print('Shorted: ({0}), At: ({1:.16f})'.format(short_assets, short_price))
        print('New Pool: ({0:.16f}), New Assets: ({1:.16f})'.format(pool, assets * price))

        #get volume
        try:
            response = requests.get('https://api.pro.coinbase.com/products/BTC-USD/stats')
            data = response.json()
            volume = np.float64(data.get('volume'))
        except Exception as e:
            print('!!!!Failed to get new volume!!!!')

        owned = pool + (assets * price)
        next_state = np.array([price, assets * price, pool, price1day, price1week, volume, short_assets, short_price])
        reward = (next_state[1] + next_state[2]) - (state[1] + state[2])
        #add memory
        self._memory.add_sample((state, action, reward, next_state))
        self._replay()
        # exponentially decay the eps value
        self._steps += 1
        self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)
        # move the agent to the next state and accumulate the reward
        state = next_state
        #set max state
        if state[1] > self._max_up:
            self._max_up = state[1]
        #add the current reward
        self._reward_store.append(reward)
        #print state changes
        print("Reward: ({0:.16f}), Action: ({1})".format(reward, action_label))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

window = pyglet.window.Window()
label = pyglet.text.Label('0',font_name='Times New Roman',font_size=36,x=window.width//2, y=window.height//2,anchor_x='center', anchor_y='center')

def main_loop():
    global BATCH_SIZE
    global MAX_EPSILON
    global MIN_EPSILON
    global LAMBDA
    global price
    global assets
    global pool
    global cnt
    global short_price
    global short_assets
    global label

    num_states = 8
    num_actions = 13

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    #unpickle memories
    if os.path.exists("data/memories.tr"):
        data = pickle.load(open("data/memories.tr", "rb"))
        mem._samples = data['samples']

    #init saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #unpickle session
        if os.path.exists("data/session.ckpt"):
            saver.restore(sess, "data/session.ckpt")
        #run session
        sess.run(model._var_init)
        tder = BTCTrader(sess, model, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        #unpickle trader
        if os.path.exists("data/trader.tr"):
            data = pickle.load(open("data/trader.tr", "rb"))
            tder._steps = data['steps']
            tder._reward_store = data['reward_store']
            tder._max_up = data['max_up']
            tder._eps = data['eps']
        starttime = time.time()
        while True:
            print('-----------------------------------------------------')
            print('Episode: ({0}), Pool & Asset Worth ({1:.16f}), Price ({2:.16f})'.format(cnt+1, pool + assets * price, price))
            #run an episode
            tder.run()
            #progress the count by 1
            cnt += 1
            label.text = str(cnt)
            #get the current BTC price
            file = open(r"data/current_price.txt","r")
            price = np.float64(file.readline())  * 0.00000001
            file.close()
            #pickle every 180 episodes
            if cnt % 10 == 0:
                print('~~~~~   pickle time!     ~~~~~')
                #dump all re-usable objects to a backup
                #simple data
                print('.simple data')
                pickle.dump({'assets':assets, 'pool':pool, 'cnt':cnt, 'short_assets':short_assets, 'short_price':short_price}, open("data/simple_data.tr", "wb"))
                #memories
                print('..memories')
                pickle.dump({'samples':mem._samples}, open("data/memories.tr", "wb"))
                #BTC trader
                print('...complex data')
                pickle.dump({'steps':tder._steps, 'reward_store':tder._reward_store, 'max_up':tder._max_up, 'eps':tder._eps}, open("data/trader.tr", "wb"))
                #session
                print('....session')
                saver.save(sess, 'data/session.ckpt')
                print('~~~~~ perfectly pickled! ~~~~~')

@window.event
def on_key_press(symbol, modifiers):
    print('A key was pressed')

@window.event
def on_draw():
    main_loop()
    window.clear()
    label.draw()
    #sleep for a little while
    time.sleep(10 - ((time.time() - starttime) % 0.1))

pyglet.app.run()
