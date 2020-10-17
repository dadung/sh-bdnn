***************************************************************************************
***************************************************************************************

Matlab demo code for SH-BDNN

If you use/adapt our code, please appropriately cite our ECCV 2016 paper:
"Learning to Hash with Binary Deep Neural Network", Thanh-Toan Do, Anh-Dzung Doan and Ngai-Man Cheung

This code is for academic purpose only. Not for commercial/industrial activities.

***************************************************************************************
***************************************************************************************

I. PREREQUISITES
=================

1. A working version of Matlab/mex.
2. A working version of the Yael library (http://yael.gforge.inria.fr)
3. A 3rd party implementation of L-BFGS (subdirectory 'minFunc' inside https://www.cs.ubc.ca/~schmidtm/Software/code.zip)

We have included, compiled and tested all 3rd party libraries on MATLAB R2014a, OS: Ubuntu 14.04 LTS 64-bit

II. DATASET
=================

The folder './dataset/cifar' contains the mat files used for this demo code
		+ cifar_alexnet_800.mat: This is PCA-projected Alexnet 800-D representation of CIFAR-10, it contains:
				- Xtest (10000×800): 10K query images
				- Xtrain(50000×800): 50K gallery images
				- Ytest (10000×1): semantic label of 10K query images
				- Ytrain(50000×1): semantic label of 50K gallery images

III. USAGE
=================

0. Download dataset.zip [here](https://drive.google.com/file/d/1vVfG6XFG1ESzshoOA2zcLgqWbx8_lLLl/view?usp=sharing), extract and place it in the directory of source code
1. Run 'demo.m', it will visualize a comparison between our method with SDH (CVPR15) in mAP
2. If the code works properly, you will get Fig 3(a) (with 2 curves: our SH-BDNN and SDH) as our ECCV16 paper
3. It should take ~8 hours to finish, we tested on workstation Intel Xeon(R) CPU E5-1620 v2 @ 3.70GHz × 8 RAM 64GB
4. Note: Please make sure your workstation has ~40GB RAM for running


IV. CORE FUNCTIONS:
=================

1. learn_all.m: learns binary deep neural network
		- INPUT:
				- S (number of training sample × number of training sample): pairwise label matrix
				- Xtrain (number of gallery sample × dimension): gallery set
				- Xtrain_sub (dimension × number of training sample): training data
				- Xval (number of validation sample × dimension): validation data
				- val_gnd_inds (number of validation sample × number of grountruth): groundtruth indices for validation data (Note: this variable should be cell if number of groundtruth indices is different between each data point, e.g. MNIST dataset)
				- hiddenSize: array contains size of layers in all hidden layers.
				- lambda1, lambda2, lambda3, lambda4, lambda5: lambda values respectively correspond to regularization, binary constraint violation, independence, balance and semantic preserving.
				- iter_lbfgs: number of L-BFGS iteration 
				- max_iter: maximum number of iteration for alternating optimization over (W,c) and B
		- OUTPUT:
				- stack: cell contains (W,c) in our binary deep neural network.
				
2. feedForwardDeep.m: does feedforward given an input sample, output is obtained from output layer (layer n)
		- INPUT:
				- stack: cell contains (W,c) in our binary deep neural network (output of learn_all.m function)
				- data (dimension × number of sample): the input sample.
