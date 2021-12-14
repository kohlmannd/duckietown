"""
We constructed our own RL training instead of using different RL packages. 
Our main sources which are not articles, but helped us understanding the proccess:
 - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
 - https://www.youtube.com/watch?v=nyjbcRQ-uQ8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv
 - https://github.com/SaralTayal123/ChromeDinoAI (!! It has many problems !!)
"""

"""
Importing the different environments and packages:
gym - Simulational environment
cv2 - For preprocessing the images
tensorflow - For the network (creation, training, evaluation)
"""
from PIL import Image
import argparse
import sys

import os
from datetime import datetime

import gym
import numpy as np
from numpy.core.fromnumeric import shape
import cv2
import tensorflow as tf   
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random

import pyglet
from pyglet import event
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from matplotlib import pyplot as plt


"""
These are the parameter that we can use when starting the code
Added ones:
 - --no-img-exp - You should add this if you don't want to export every frame
 - --load-weights - loads the network parameters from weights.hdf5 (for retraining and test)
 - --test - Starts the enviroment for testing. Exploring is 1% and mainly exploiting the trained network.
"""
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

"""
This is our Agent class which consists:
 - Policy networok
 - Target network
 - Methods for handling the networks (updateTarget, act)
 - Experience memory and management methods (remember)
 - Trainig methods (learn, learnBatch)
"""


if(args.no_img_exp):
	logpath = "./logs/log" + datetime.now().strftime("%m_%d_%H_%M_%S")
	if not os.path.exists(logpath):
    		os.makedirs(logpath)

class Agent:
    def __init__(self):
        #This is our neural network. Source: http://real.mtak.hu/115338/1/5-wcci_2020_duckietown.pdf
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
        #This our target network which is the exact copy of the previous one.
        target = Sequential([ 
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
    
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001)) 
    
        self.model = model
        self.target = target
        self.memory = [] # This is where we will store our experiences
        self.xTrain = [] # Side variable for training
        self.yTrain = [] # Side wariable for training
        self.loss = [] # Side variable for storing the losses
        self.location = 0 # Location refers to the run count
        self.episode = 0 # Episode refers to the episode count
        self.target.set_weights(self.model.get_weights()) 

    #Copy weights of policy network to target network
    def updateTarget(self):
        self.target.set_weights(self.model.get_weights()) 

    """
    This is our method for choosing the action based on the actual state. 
    It is based on the epsilon-greedy algorithm.
    This is controlling the ratio of exploration and exploitation.
    """
    def act(self, state): 
        qval = self.model.predict(np.reshape(state, (1, 40, 80, 15))) #Q-values based on the state
        z = np.random.rand() 
        """
        This is our epsilon function
        As our training progresses the duckiebot will rather exploit the network in the decision making process
        rather than eplore with a random action. 
        The parameters can be changed if we want to have a longer exploring period for the training. 
        """
        epsilon = max(0.01,0.01+(1-0.01)*np.exp(-0.03*self.episode)) 
        #If we start the program in test mode the exploration rate is 1%
        if args.test:
            epsilon = 0.01

        if z > epsilon:
            """
            We tried to make the decision based on a probability distribution using the soft-max of the output. 
            We experienced that the program performed better without it. You can uncomment the three 
            lines bellow to try with this solution.
            """
            # prob = tf.nn.softmax(tf.math.divide((qval.flatten()), 1)) 
            # action = np.random.choice(range(3), p=np.array(prob))
            # return action
            return np.argmax(qval.flatten())
            
        else:
            return np.random.choice(range(3))

    # This method is for storing the experience. We call this after each step.
    def remember(self, state, nextState, action, reward, done, location):
        self.location = location
        self.memory.append(np.array([state, nextState, action, reward, done]))

    #This method is for checking the memory data and picking a random batch from it.
    def learn(self):

        self.batchSize = 256

        if len(self.memory) < self.batchSize: #We are checking if there are enough experiences
            print(len(self.memory))
            print("Not enoguh frames")
            return  
        batch = random.sample(self.memory, self.batchSize) #We are sampling from the memory in a random manner.

        self.learnBatch(batch)


    def learnBatch(self, batch, alpha=0.9):
        batch = np.array(batch) #(256, [actstate(40*80*15), nextstate(40*80*15), dec(1), reward(1), done(B), stepCounter(1)] )
        actions = batch[:, 2].reshape(self.batchSize).tolist() #(256, 1) Choosen decision vector
        rewards = batch[:, 3].reshape(self.batchSize).tolist() #(256, 1) Received reward vector

        stateToPredict = batch[:, 0].reshape(self.batchSize).tolist() #States before step (256, 40, 80, 15)

        nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist() #States after the step (256, 40, 80, 15)


        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, 40,80, 15))) #(256, 3) Q-values for each action based on the state before step
        nextStatePrediction = self.target.predict(np.reshape(
            nextStateToPredict, (self.batchSize, 40, 80, 15))) #(256, 3) Q-values for each action based on the state after step

        statePrediction = np.array(statePrediction) #(256, 3)
        nextStatePrediction = np.array(nextStatePrediction) #(256, 3)

        for i in range(self.batchSize):
            action = actions[i] #(1, 1)
            reward = rewards[i] #(1, 1)
            nextState = nextStatePrediction[i] #(1, 3)
            qval = statePrediction[i, action] #(1, 1) Q-value for the choosen action
            if reward < -79: #Right now when it dies, it receives -80. If the reward funcion is modified, don't forget to modify this also.
                statePrediction[i, action] = reward
            else:
                #Target Q-value based on the Bellman equation
                statePrediction[i, action] = (reward + 0.95 * np.max(nextState))


        early_stoping = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)

        self.xTrain.append(np.reshape(
            stateToPredict, (self.batchSize, 40, 80, 15)))
        self.yTrain.append(statePrediction)
        """
            Training with a mini-batch size of 32 in maximum 30 epochs.
            We did a 20% validitional split from the 256 chosen experiences.
            Early stopping based on the validation loss.
        """
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=32, epochs=30, validation_split=0.2, verbose=2, callbacks=[checkpointer, early_stoping])  
        self.model = load_model('weights.hdf5')
        loss = history.history.get("loss")[0]

        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []
        self.memory = []


"""
This is the preprocessing of the images. We segment the yellow and white colors (middle-line and side-lines).
After segmentation we store the two type of lines in the red and green channels.
We cut off the top part (sky) that consists no useful information. Also we rescaled the images to a 80*40 size.
The blue layer does not contain any information now, but we kept it in case we want to use that for something in the future.
After the transformations we are norming the values to the 0-1 range. 
"""
def prep_frame(img):
    yellow_bot = (150,140,5)
    yellow_up=(220,210,150)
    white_bot=(150,150,150)
    white_up=(230,220,220)
    img = img[150:,:,:]
    img = cv2.resize(img, dsize=(80,40), interpolation=cv2.INTER_CUBIC)
    mask_y = cv2.inRange(img,yellow_bot,yellow_up)
    mask_w = cv2.inRange(img,white_bot,white_up)

    img[:,:,0] = mask_y
    img[:,:,1] = mask_w
    img[:,:,2] *= 0
    if(args.no_img_exp):
        im = Image.fromarray(img)
        global logpath
        global agent
        global runCounter
        logpath2 = logpath + "/episode" + str(agent.episode) + "/run" + str(runCounter)
        if not os.path.exists(logpath2):
            os.makedirs(logpath2)
        im.save(logpath2 + "/" + str(env.step_count) + ".png")
    img = img/255

    return img

"""
    This is the function that we call again and again in each run.
    It consists:
     -  Decision making
     -  Acting based on the decision
     -  Experience storing
     -  Managing other control variables
"""
def update():
    global actstate
    global agent
    global stepCounter
    global runCounter
    global rundone
    global runReward

    dec = agent.act(actstate) #Based on the actual state we choose and action
    action = np.array((1,2))
    #Convert the choosen decision (0,1,2) into actual action vector
    if dec == 0: #Stepping forward
        action[0] = 1
        action[1] = 0
    elif dec == 1: #Turning left
        action[0] = 0
        action[1] = 1
    else: #Turning right
        action[0] = 0
        action[1] = -1

    nextframe, reward, done, info = env.step(action) #Using the action vector we interact with the duckiebot and make the step. The environment returns the state and reward.
    nextstate = np.concatenate((actstate[:,:,3:],prep_frame(nextframe)),axis=2) #Dropping the oldest frame and adding the new one
        
    agent.remember(actstate, nextstate, dec, reward, done, stepCounter)  #Write the experience into the memory
    
    runReward += reward
    actstate = nextstate
    stepCounter += 1
    #Early stopping (mainly due to oscillations)
    if stepCounter > 1000:
        done=True
    
    if done:
        env.render()
        print("Run ended!")
        stepCounter = 0
        rundone = True
    else:
        env.render()

#Some global variables
stepCounter = 0
agent=Agent()
rundone = False
runReward = 0
plotRew = []
actstate = np.ndarray((40, 80, 15))
runCounter = 0


#Main loop
while True:
    #We initialize the enviroment before each episode with a random seed 
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

    #If --load-weights argument was included then we load the weights from weights.hdf5
    if args.load_weights:
        agent.model = load_model('weights.hdf5')
        agent.target = load_model('weights.hdf5')
        print('Weights loaded from last run')
    
    #In one episode we have 5 runs (respawns)
    while (runCounter<5):
        runReward = 0 
        env.reset()
        img, _, _, _ = env.step([0.,0.])
        img = prep_frame(img)
        actstate = np.concatenate((img,img,img,img,img),axis=2) #We initialize the actstate with the stationary spawn position
        
        #Let's go until we run off the map
        while rundone == False:
            update()

        runCounter+=1
        rundone = False
        plotRew.append(runReward)
    env.render(close=True) # The rendering is buggy, this is needed to close the window.

    agent.episode += 1
    runCounter=0

    #After 60 episodes we plot the reward development
    if agent.episode == 60:
        plt.plot(range(len(plotRew)),plotRew) 
        plt.show() 
    #We call the learning method to train after 5 runs (1 episode)
    agent.learn()
    #We update the target network with the policy network after 3 episodes (15 runs)
    if (agent.episode % 3) == 0:
        agent.updateTarget()
        print( "Target net parameters updated")
        
