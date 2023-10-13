Code uses PyTorch as the core library for the training. Heavily inspired by their finetuning tutorial page. 
This project uses the Deep-CNN Inception V3 to classify chest x-ray images from the Kaggle dataset covid19-image-dataset (uploaded by pranavraikokte). 
CUDA toolkit can be used (if available), for significantly faster processing in both training and evaluation. 

The dataset consists of 3 different types of x-rays, 1. chest with healthy lungs, 2. chest with viral infection on the lungs and 3. chest with sars cov 2 infection on the lungs.
The network_trainer.py file trains the CNN on these images for classification on each of the three types of lungs. 
Mulpiple approaches of training are presented, such as training of the whole network or training of the final layer, in the case of choosing to have pretrained values (from ImageNet) loaded into the network.
Images are resized to 299x299 pixels and randomly editted (resized and rotated). The values of the images are also normalized. 
Since the final layer of the network has 1024 classes, we change that to our corresponding 3 lung types.

Best results were presented with the fine tuning training process. 

The model_tester.py file can be used to load saved networks and use them for classification testing. 
(Network must be same type as Inception V3, or the code has to be tweaked to match the new network's structure)

Link for dataset: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
Link for PyTorch tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
