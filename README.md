# Duckietown

**Run:**
```python3 ./manual_control.py --env-name Duckietown-udem1-v0```

At this stage during the manual control it saves the images to a log folder for visualization.
Later these images will be the input layer to our neural network.
One important hyperparameter will be the reward that is calculated in the env.step() function.

The output will be the action array for the next step. Currently we modify the action by manual input.


**Source:** https://github.com/duckietown/gym-duckietown
