There are 3 types of training results compared here.

One is for the fine tuning of the network, through out all of its weights (orange color), 
the second is normal training with out preloaded ImageNet weights (blue color)
and the third is fine tuning on just the last layer of the network (mentioned as "added layers") (green color). 

In the last case the network has also preloaded ImageNet weights, but the training is being done on just the layer that was changed (the output layer turned from 1024 to 3).
