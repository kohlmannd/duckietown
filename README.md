# First milestone

**Run:**
```
git clone https://github.com/kohlmannd/duckietown.git
cd duckietown
pip3 install -e .
python3 ./manual_control.py --env-name Duckietown-udem1-v0
```

If you want to turn off the image exporting function, use the: --no-img-exp argument

**Run with custom map:**

```python3 ./manual_control.py --env-name Duckietown --map-name dmad --no-img-exp```

The goals for the first milestone:
- reading the frames for every time step
- controlling the duckiebot
- making a custom map

We modified the original manual_control.py since it contained most of what we need.

In order to run this: the dmad.yaml file should be copied to the /usr/local/lib/python3.8/dist-packages/duckietown_world/data/gd1/maps location. (From the 'custom files' folder) We couldn't figure out how to do it otherwise. This is our custom map called 'Duckie Madness' as requested.

At this stage during the manual control it saves the images to a log folder for visualization.
Later these images will be the input layer to our neural network. (We don't actually have to export them)

The output will be the action array for the next step. Currently we modify the action by manual input.

# Second milestone

**Upgraded custom map:**

As previously you should copy it into /usr/local/lib/python3.8/dist-packages/duckietown_world/data/gd1/maps. The new map is dmad2.yaml. This version is more simple, because the last version was way too complicated for our case. (No GPU)

**Reinforcement learning:**

You can run the training with the following command:
```python3 ./rltrain.py --env-name Duckietown --map-name dmad2 --no-img-exp```
This will start the simulation. It iterates through 5 runs in 1 episode. This goes on forever.
It is clear that the model learns, but our hyperparameters and the model needs some tuning. So far we managed to avoid the "death" and the duckiebot learns to stay on the right side of the road most of the time after enough learning. But the training is far from ideal. The structure of our method:
- We create replay memory inside the agent class (This class contains the neural network and the training alg. also)
- We initialize the network
- For each episode:
    - Starting state initialization
    - For each run:
        - Selecting action (epsilon-greedy)
        - Applying the selected action
        - Reading the state (camera view) and the reward
        - Putting those into the memory
    - Training on collected data
        - Picking a batch randomly
        - State propogation based on batch
        - Passing the calculated states to the network
        - Loss: Q-Q_target
        - Updating weights

**Monitoring**

After 10 episodes you can inspect the change in run lengths and the reward. After plotting you have to close it and then the training continues. 

# Third milestone
Kohlmann Dániel - ICT3NK

Köpeczi-Bócz Ákos - AVEK7G

Széles Katalin - AC09LI

**Changes:**

We have changed and optimized several things since the last milestone. The biggest change was that we introduced a target network along the policy network in hope for stability improvements. Also we included now our best solution in best_weights.md5. This version can navigate when it is placed in the right initial position. It is able to follow the lane and take right turns, but when it meets a left turn and loses the yellow line it starts to oscillate.

**How to run:**

You can run the training with the following command after copyiing the map dmad2 in the above mentioned folder:
```python3 ./rltrain.py --env-name Duckietown --map-name dmad2 --no-img-exp```
This command will start a new training, the reward evolution will be ploted after 70 episodes.

If you want to test our best solution you should use the following command:
```python3 ./rltrain.py --env-name Duckietown --map-name dmad2 --no-img-exp --load-weights --test```





**Source:** https://github.com/duckietown/gym-duckietown
