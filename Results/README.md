There are 3 types of training results compared here.

One is for the fine tuning network through out all of its weights, the second is normal training with out preloaded ImageNet weights and the last is fine tuning on just the last layer of the network (mentioned as Feature Extraction). 
In the last case the network has also preloaded ImageNet weights, but the training is being done on just the layer that was changed (the output layer turned from 1024 to 3).
