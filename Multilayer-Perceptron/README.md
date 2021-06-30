# Multilayer Perceptron

## Conclusions:
* Classification of data taken from iris for training and testing data is very similar
* Classification of binary data once had an increase in data for cost and a slight decrease for accuracy
* Similar to the binary data classification normalized data from iris had similar graphs
* The graphs changed as the bottoms changed but always the graphs associated with cost converged down and accuracy converged up

## Starting Questions
1. how many inputs will the network have and from what does this result?
2. how many outputs will the network have and from what does this result?
3. what will be the activation function in the neurons?
4. What loss function should I choose for the task at hand?

## Answers
1. there are as many inputs as there are features (in our case 4). It always so no matter what the task, the number of inputs to the neural network, the number of features and attributes in a given set
2. outputs are as many as classes (in our case 3). We always rely on the task
3. depends on the task, and in our case was selected sigmoidal function
4. also depends on the task (our multinomial option is multiple outputs) and in our case we used Multi-Class Cross-Entropy Loss
