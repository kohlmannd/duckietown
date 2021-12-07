#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import os
from datetime import datetime
##
import gym
import numpy as np
from numpy.core.fromnumeric import shape
import pyglet
import cv2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf   
import random
from pyglet import event
from pyglet.window import key
##
from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--no-img-exp", default=True, action="store_false",  help="prohibit image export")
parser.add_argument("--load-weights", default=False, action="store_true",  help="load last weights")
parser.add_argument("--test", default=False, action="store_true",  help="test run. Ignoring epsilon-greedy")
args = parser.parse_args()

#create logfolder

class Agent:
    def __init__(self):
        #Konvolúcióüs háló a cikk alapján
        model = Sequential([ 
            Conv2D(32, (3,3), input_shape=(40, 80, 15),
                   strides=(1,1),padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2, 2)),
            Conv2D(32, (3,3), activation='relu',padding='same', strides=(1,1)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(64, (3, 3), strides=(1,1),padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation="linear", name="layer1"),
        ])
        #fordítás
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001)) 
    

        self.model = model
        self.memory = [] # Ide tároljuk el minden frame-hez az infókat (state, reward stb)
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.location = 0
        self.episode = 0


    def predict(self, state): #Prediktálás a 3 kimenetre (előre, jobb, bal)
        stateConv = state
        qval = self.model.predict(np.reshape(stateConv, (1, 40, 80, 15)))
        return qval

    def act(self, state): #a kimenet alapján valószínűségi alapon választjuk a végleges döntést
        qval = self.predict(state)
        #Epsilon-Greedy actions->
        z = np.random.rand()
        epsilon = max(0.01,0.01+(1-0.01)*np.exp(-0.03*self.episode))
        if args.test:
            epsilon = 0.01
        if z > epsilon:
            return np.argmax(qval.flatten())
        else:
            return np.random.choice(range(3))
        return action

    # Ezzel mentjük frame-enként az infókat
    def remember(self, state, nextState, action, reward, done, location):
        self.location = location
        self.memory.append(np.array([state, nextState, action, reward, done]))

    #Tanulás külső függvény
    def learn(self):

        self.batchSize = 256

        if len(self.memory) < self.batchSize: #legalább egy batchet tudjunk képezni
            print(len(self.memory))
            print("Not enoguh frames")
            return  
        batch = random.sample(self.memory, self.batchSize)

        self.learnBatch(batch)


    def learnBatch(self, batch, alpha=0.9):
        batch = np.array(batch) #256*6 [actstate(40*80*15), nextstate(40*80*15), dec(1), reward(1), done(B), stepCounter(1)]
        actions = batch[:, 2].reshape(self.batchSize).tolist() #[256*1] hozott döntés
        rewards = batch[:, 3].reshape(self.batchSize).tolist() #[256*1]

        stateToPredict = batch[:, 0].reshape(self.batchSize).tolist() #állapotok amiből jóslunk (256, 40, 80, 15)

        nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist() #állapotok amiket kaptunk(256, 40, 80, 15)


        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, 40,80, 15))) #állapot jóslása az actstate értékek alapján (256, 3)
        nextStatePrediction = self.model.predict(np.reshape(
            nextStateToPredict, (self.batchSize, 40, 80, 15))) #jövőbeli jóslat (256, 3)
        # print("chosen step")
        # print(actions)
        # print("Reward")
        # print(rewards)
        # print("Choosen action q value")
        # print(statePrediction)
        # print("future state for bellman")
        # print(nextStatePrediction)
        print("Current episode:", self.episode)
        statePrediction = np.array(statePrediction) #(256, 3)
        nextStatePrediction = np.array(nextStatePrediction) #(256, 3)

        for i in range(self.batchSize):
            action = actions[i] #[1*256]
            reward = rewards[i] #[1*256]
            nextState = nextStatePrediction[i] #[3]
            qval = statePrediction[i, action] #[1] a választott döntés valószínűsége (pontosabban háló kimeneti értéke)
            if reward < -5: 
                statePrediction[i, action] = reward
            else:
                #Q-learning
                statePrediction[i, action] = (reward + 0.95 * np.max(nextState))


        early_stoping = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)
        # print("modified")
        # print(statePrediction)
        self.xTrain.append(np.reshape(
            stateToPredict, (self.batchSize, 40, 80, 15)))
        self.yTrain.append(statePrediction)
        # print("Target value")
        # print(self.yTrain)
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=32, epochs=30, validation_split=0.2, verbose=2, callbacks=[checkpointer, early_stoping])
        self.model = load_model('weights.hdf5')
        loss = history.history.get("loss")[0]
        # print("Target value")
        # print(self.model.predict(np.reshape(
        #     self.xTrain, (self.batchSize, 40,80, 15))))
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []
        self.memory = []



if(args.no_img_exp):
	logpath = "./logs/log" + datetime.now().strftime("%m_%d_%H_%M_%S")
	if not os.path.exists(logpath):
    		os.makedirs(logpath)



def prep_frame(img):
    yellow_bot = (150,140,5)
    yellow_up=(220,210,150)
    white_bot=(150,150,150)
    white_up=(230,220,220)
    img = img[150:,:,:]
    img = cv2.resize(img, dsize=(80,40), interpolation=cv2.INTER_CUBIC)
    mask_y = cv2.inRange(img,yellow_bot,yellow_up)
    mask_w = cv2.inRange(img,white_bot,white_up)
    # mask = mask_y+mask_w
    # img = cv2.bitwise_and(img, img, mask=mask)
    img[:,:,0] = mask_y
    img[:,:,1] = mask_w
    img[:,:,2] *= 0
    if(args.no_img_exp):
    	im = Image.fromarray(img)
    	global logpath
    	im.save(logpath + "/" + str(env.step_count) + ".png")
    img = img/255

    return img
    

actstate = np.ndarray((40, 80, 15))

agent = Agent()
episodeCounter = 0

def update():
    global actstate
    global agent
    global stepCounter
    global episodeCounter
    global epdone
    global epReward

    dec = agent.act(actstate) #Előző állapot alapján meghozza a döntést
    action = np.array((1,2))
    if dec == 0:
        action[0] = 1
        action[1] = 0
    elif dec == 1:
        action[0] = 0
        action[1] = 1
    else:
        action[0] = 0
        action[1] = -1

    nextframe, reward, done, info = env.step(action) #Hattatjuk a döntést a környezetre
    
    if reward ==-1000:
        reward = -40
    nextstate = np.concatenate((actstate[:,:,3:],prep_frame(nextframe)),axis=2) #dropping the oldest frame and adding the new one
    if stepCounter> 700:
            for _ in range(2):
                agent.remember(actstate, nextstate, dec, reward, done, stepCounter)
    elif stepCounter> 40:
                agent.remember(actstate, nextstate, dec, reward, done, stepCounter)                
    
    if done == True: #game ended

            print("breaking")
    epReward += reward
    actstate = nextstate
    stepCounter += 1
    #save image into the logfolder
    # if(args.no_img_exp):
    # 	im = Image.fromarray(nextframe)
    # 	global logpath
    # 	im.save(logpath + "/full_" + str(env.step_count) + ".png")
    if stepCounter > 1000:
        done=True
    if done:
        agent.remember(actstate, nextstate, dec, -10, done, stepCounter)
        env.render()
        print("done!")
        print(np.shape(agent.memory))
        plotLength.append(stepCounter)
        stepCounter = 0
        
        # pyglet.app.exit()
        
        
        #create logfolder
        logpath = "./logs/log" + datetime.now().strftime("%m_%d_%H_%M_%S")
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        epdone = True
    else:
        env.render()


stepCounter = 0
agent=Agent()
epdone = False
epReward = 0
plotLength = []
plotRew = []
while True:
    if args.env_name and args.env_name.find("Duckietown") != -1:
        env = DuckietownEnv(
            seed=np.random.randint(low=0,high=50),
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
        )
    else:
        env = gym.make(args.env_name)
    if args.load_weights:
        agent.model = load_model('weights.hdf5')
        print('Weights loaded from last run')
    epReward = 0
    while (episodeCounter<5):

        env.reset()
        img, _, _, _ = env.step([0.,0.])
        img = prep_frame(img)
        actstate = np.concatenate((img,img,img,img,img),axis=2)
        
        while epdone == False:
            update()

        episodeCounter+=1
        epdone = False
    env.render(close=True)

    agent.episode += 1

    plotRew.append(epReward)
    print("Run ",len(plotLength),"ended")
    episodeCounter=0
    # if (len(plotLength) % 10) == 0:
    #     plt.plot(range(len(plotLength)),plotLength) 
    #     plt.show() 
    #     plt.plot(range(len(plotRew)),plotRew) 
    #     plt.show() 
    agent.learn()

    
    agent.model.save_weights ("RLDuck.h5")
    print( "Saved model to disk")
        




# from PIL import Image
# import argparse
# import sys

# import os
# from datetime import datetime
# ##
# import gym
# import numpy as np
# from numpy.core.fromnumeric import shape
# import pyglet
# import cv2
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf   
# import random
# from pyglet import event
# from pyglet.window import key
# ##
# from gym_duckietown.envs import DuckietownEnv
# from matplotlib import pyplot as plt

# env = DuckietownEnv(
#             seed=np.random.randint(low=0,high=50),
#             map_name="dmad2",
#             draw_curve=False,
#             draw_bbox=False,
#             domain_rand=False,
#             frame_skip=1,
#             distortion=False,
#             camera_rand=False,
#             dynamics_rand=False,
#         )

# yellow_bot = (150,140,35)
# yellow_up=(220,210,150)
# white_bot=(175,175,175)
# white_up=(210,210,210)
# img, reward, done, info = env.step([1.,1.])
# img = img[150:,:,:]
# img = cv2.resize(img, dsize=(80,40), interpolation=cv2.INTER_CUBIC)
# mask_y = cv2.inRange(img,yellow_bot,yellow_up)
# mask_w = cv2.inRange(img,white_bot,white_up)
# mask = mask_y+mask_w
# img = cv2.bitwise_and(img, img, mask=mask)

# plt.subplot(1, 2, 1)
# plt.imshow(mask)
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()

