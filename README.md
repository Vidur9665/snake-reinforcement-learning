# snake-reinforcement-learning

I made a reinforcement learning model for the “Snake” game by modifying an already provided code for the bot controlling the snake. I made a deep learning neural network with 4 layers: 1 input layer with 11 units, 2 dense layers (hidden layers) with 128 & 64 units respectively, and 1 dense layer with 4 units. This model is currently still running it’s iterations (for a total of 1000), and it’s current best score at the time of this email is 3. I believe this to be the case due to the reward system, and I also believe that an idea presented by a classmate would make this model much better. He recommended (and tested) that the model first be trained on composite so that it learns that the snake shouldn’t die, after which it will be trained on score so that it learns to consume the apples.

(s.py is the code for the snake game framework, b.py is the code for the bot. Only b.py was modified)
