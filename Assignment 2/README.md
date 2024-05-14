# Assignment 2

In this assignment you are asked to:

1. Implement a fully connected feed forward neural network to classify images from the 'Cats of the Wild' dataset.
2. Implement a convolutional neural network to classify images of 'Cats of the Wild'.
3. Implement transfer learning.

Both requests are very similar to what we have seen during the labs. However, you are required to follow **exactly** the assignment's specifications.
Make sure to motivate **all** your choices, parameters, losses and answers.
Once completed, please submit your solution on the iCorsi platform following the instructions below. 

It is **Highly** recommended to google colab GPUs, or you will have RAMS problems.

## Tasks

### T1. Image Classification with Fully Connected Feed Forward Neural Networks (FFNN) (35 pts)

In this task, we will try and build a classifier for the provided dataset. This task, we will use a classic Feed Forward Neural Network.
You are provided with Torch dataset class called Dataset, you can use it. 

1. Download and load the dataset using the following link 'https://drive.switch.ch/index.php/s/XSnhQDNar7y46oQ'. The dataset consist of 6 classes with a folder for each class images. The classes are 'CHEETAH' ,'OCELOT', 'CARACAL', 'LIONS', 'PUMA', 'TIGER'. Check Cell 1 in `example.ipynb` to find the ready and implemented function to load the dataset. 
2. Preprocess the data:
    - Normalize each pixel of each channel so that the range is [0, 1], by applying the min-max normlaization (division by 255);
    - One hot encode the labels (the y variable)
3. Flatten the images into 1D vectors. You can achieve that by using [torch.reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html) or by prepending a [Flatten layer](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) to your architecture; if you follow this approach this layer will not count for the rules at point 4.
4. Build a Feed Forward Neural Network of your choice, following these constraints:
    - Use only torch nn.Linear layers.
    - Use no more than 3 layers, considering also the output one.
    - Use GELU activation for all layers other than the output one.
5. Draw a plot with epochs on the x-axis and with two graphs: the training accuracy and the validation accuracy (remember to add a legend to distinguish the two graphs!).
6. Assess and comment on the performances of the network on the test set loaded in point 1, and provide an estimate of the classification accuracy that you expect on new and unseen images. 


### T2. Follow our recipe (35 pts)

Implement a multi-class classifier (CNN model) to identify the class of the images: 'CHEETAH' ,'OCELOT', 'CARACAL', 'LIONS', 'PUMA', 'TIGER'.

1. Follow steps 1 and 2 from T1 to prepare the data.
2. Build a CNN of your choice, following these constraints: 
 - use 3 convolutional layers
 - use 3 pooling layers
 - use 4 dense layers (output layer included).
3. Train and validate your model. Choose the right optimizer and loss function.
4. Follow steps 5 and 6 of T1 to assess performance.
5. Qualitatively and **statistically** compare the results obtained in T1 with the ones obtained in T2. Explain what you think the motivations for the difference in performance may be.
6. Apply image manipulation and augmentation techniques in order to improve the performance of your models. Evaluate the performance of the model using the new images and compare the results with the previous evaluation performed in part 3. Provide your observations and insights.
7. **Bonus** (Optional) Tune the model hyperparameters with a **grid search** to improve the performances (if feasible).
    - Perform a grid search on the chosen ranges based on hold-out cross-validation in the training set and identify the most promising hyper-parameter setup.
    - Compare the accuracy on the test set achieved by the most promising configuration with that of the model obtained in point 4. Are the accuracy levels **statistically** different?


### T3. Transfer Learning (30 pts)

This task involves loading the VGG19 model from PyTorch, applying transfer learning, and experimenting with different model cuts.
The VGG19 architecture have 19 layers grouped into 5 blocks, comprising 16 convolutional layers followed by 3 fully-connected layers. Its success in achieving strong performance on various image classification benchmarks makes it a well-known model.

Your task is to apply transfer learning with a pre-trained VGG19 model. A code snippet that loads the VGG19 model from PyTorch is provided. You'll be responsible for completing the remaining code sections (marked as TODO).  Specifically:

1. The provided code snippet sets param.requires_grad = False for the pre-trained VGG19 model's parameters. Can you explain the purpose of this step in the context of transfer learning and fine-tuning? Will the weights of the pre-trained VGG19 model be updated during transfer learning training?

2. We want to transfer learning with a pre-trained VGG19 model for our specific classification task. The code has sections for __init__ and forward functions, but needs to be completed to incorporate two different "cuts" from the VGG19 architecture. After each cut, additional linear layers are needed for classification (similar to Block 6 of VGG19).
implement the __init__ and forward functions to accommodate these two cuts:
- Cut 1: This cut should take the pre-trained layers up to and including the 11th convolution layer (Block 4).
- Cut 2: This cut should use all the convolutional layers from the pre-trained VGG19 model (up to Block 5).
Note after each cut take the activation function and the pooling layer associated with the convolution layer on the cut
![Alt text](cuts.png)

3. In both cases, after the cut, add a sequence of layers (of your choice) with appropriate activation functions, leading to a final output layer. For both models, train the added layers (one with Cut 1 and another with Cut 2) on the dataset. Once training is complete, statistically compare their performance.

4. Based on the performance comparison, discuss any observed differences between the two models. What could be the potential reasons behind these results?
   
5. **BONUS** (optional): Try different cuts in each block of VGG19, and plot one single figure with all the train-validation-test accuracies. Explain in detail the reasons behind the variation of results you get.


## Instructions

### Tools

Your solution must be entirely coded in **Python 3** ([not Python 2](https://python3statement.org/)).
We recommend to use torch that we seen in the labs, so that you can reuse the code in there as reference (Tensorflow and Keras are not allowed). 

All the required tasks can be completed using Torch. On the [documentation page](https://pytorch.org/docs/stable/index.html) there is a useful search field that allows you to smoothly find what you are looking for. 
You can develop your code in Colab or kaggle notebooks (recommended), where you have access to a GPU, or you can install the libraries on your machine and develop locally.


### Submission

In order to complete the assignment, you must submit a zip file named `as2_surname_name.zip` on the iCorsi platform containing: 

1. A report in `.pdf` format containing the plots and comments of the two tasks. You can use the `.tex` source code provided in the repo (not mandatory).
2. The best models you find for both the tasks (one for the first task, one or two for the second task, in case you completed the bonus point). By default, the torch function to save the model outputs a folder with several files inside. If you prefer a more compact solution, just append `.pt` or `.pth` at the end of the name you use to save the model to end up with a single file.
3. A working example for T1, T2 and T3 `tasks.ipynb` that loads the dataset, preprocesses the data, loads the trained model from file. Note that the notebook should contain a maximum of 7 cells and each cell shou be run independatly from the other cells.
 - 1 cell for loading the data (already given)
 - 1 cell for task 1
 - 1 cell for task 1 bonus (if any)
 - 1 cell for task 2
 - 1 cell for task 2 bonus (if any)
 - 1 cell for task 3
 - 1 cell for task 3 bonus (if any)
4. A folder `src` with all the source code you used to build, train, and evaluate your models.

The zip file should eventually looks like as follows

```
as2_surname_name/
    report_surname_name.pdf
    deliverable/
        example.ipynb #your solution
        # your saved files (i.e ending with pt)
```


### Evaluation criteria

You will get a positive evaluation if:

- your code runs out of the box (i.e., without needing to change your code to evaluate the assignment);
- your code is properly commented;
- the performance assessment is conducted appropriately;

You will get a negative evaluation if: 

- we realize that you copied your solution;
- your code requires us to edit things manually in order to work;
- you did not follow our detailed instructions in all the tasks.

Bonus parts are optional and are not required to achieve the maximum grade, however they can grant you extra points.
