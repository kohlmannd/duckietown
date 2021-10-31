# First milestone

**Run:**
```python3 ./manual_control.py --env-name Duckietown-udem1-v0```

**Run with custom map:**
```python3 ./manual_control.py --env-name Duckietown --map-name dmad```

The goals for the first milestone:
- reading the frames for every time step
- controlling the duckiebot
- making a custom map

We modified the original manual_control.py since it contained most of what we need.

In order to run this: the dmad.yaml file should be copied to the /.local/lib/python3.9/site-packages/duckietown_world/data/gd1/maps location. (From the 'custom files' folder) We couldn't figure out how to do it otherwise. This is our custom map called 'Duckie Madness' as requested.

At this stage during the manual control it saves the images to a log folder for visualization.
Later these images will be the input layer to our neural network. (We don't actually have to export them)

The output will be the action array for the next step. Currently we modify the action by manual input.


**Source:** https://github.com/duckietown/gym-duckietown
