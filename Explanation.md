# Explanation of my code

As requested a couple of weeks ago...

## Backstory

Most of the code written in the project is based on some python code that I wrote around August of 2017. Although the file has now been 
lost in the many directories of my PC, I still remember some things about it. Specifically: 

EDIT: **FOUND IT** https://pastebin.com/iR2MW1QL

* It was a simple program that tranied a neural network to work as a logic funciton
* It worked
* It correctly implemented backpropagation

These were important because, by the time I was making my project I didn't really remeber much about how backprpagation worked. Therefore,
most of my project was just trying to convert python code into netlogo code (which was actually very very difficult)

## Code explanation

To summarize, the code can be explained in 3 parts

1. The Setup
2. The training
3. The recognition 

These are explained below. Its better to read them in order they are given because some things explained in a previous section might
be assumed in the next

### The Setup

As per most netlogo programs, the setup procedure is run when a setup button is pressed. The procedure is as follows:

```
to setup
  reset-ticks
  set theta1 (create-theta hidden-layer-size 401)
  set theta2 (create-theta 10 (hidden-layer-size + 1))
end
```

The first line resets ticks. This is fairly obvious, since we don't want to have the program display everything that's happening since
it takes away space that we can otherwise use for computation.

The next two lines create the weights for the neural network. Since our network has 3 layers (including the input and output, we only need 2 matrices of weights. For easier vectorization, these are oriented in a way that makes it very easy to multiply them with the inputs. 

In any case, the first matrix of weights has a height of the hidden layer size and the width of 401. The width remains constant due to the fact that we will always input images that have a total of 400 pixel. We add one to this number since we also need a bias. The height
of the matrix will be the size of the hidden layer. 

We can apply similar reasoning to the second matrix. The width will be the size of the hidden layer + 1 to account for bias, and the
height will be 10 (which is how many values we need to correctly predict what a digit is, given an input)

### The Training

The next part involves training the network. This is done using the ```train-once``` function whose code is as follows:

We first create a training set of a certain size using the ```create-training-set``` function
```
  create-training-set batch-size
```
This function doesn't return anything, but instead sets the values two global variables - ```inputs``` and ```outputs```

After this function is done, ```inputs``` will be a batch-size by 400 matrix, which each row respresenting the 400 pixels of a different training example. ```outputs``` will be a batch-size by 10 matrix, which each row representing the ideal output for the given input.

The next part of the code implements forward propagation:

```
  let a1 (matrix:copy (add-ones (matrix:copy inputs)))
  let z2 (a1 matrix:* (matrix:transpose theta1))
  let a2 (add-ones (sigmoid (matrix:copy z2)))
  let z3 (a2 matrix:* (matrix:transpose theta2))
  let a3 (sigmoid z3)
```

I will not go into detail about what this does, but its important to know what it produces. In the end, a3 will be a batch-size by 10 matrix. The value in the ith row and the jth column will represent the networks prediction of the probability that the ith training example is the number (j - 1). (I wrote j - 1 here because matrixes are indexed from 1).

Its also important to note that this forward propagation is vectorized, meaning it doesn't require iteration through matrixes to compute. 

The next part of the code implements backpropagation:
```
;;backpropagate
  let d3 (a3 matrix:- outputs)
  let d2 (matrix:times-element-wise (d3 matrix:* (delete-first-column theta2)) (sigmoid-gradient z2))

  ;;calculate slopes
  let delta1 (matrix:times-scalar ((matrix:transpose d2) matrix:* a1) (1 / batch-size))
  let delta2 (matrix:times-scalar ((matrix:transpose d3) matrix:* a2) (1 / batch-size))
  
  set delta1 (matrix:map apply-learning-rate delta1)
  set delta2 (matrix:map apply-learning-rate delta2)
```

This bit of code is a bit tricky to explain because I myself do not know how it works due to me not understanding the math behind it. As mentioned perviously, I simply copied this code from Python code that I knew worked.

However, I know that the variables ```delta1``` and ```delta1``` will be matrixes with the same size as ```theta1``` and ```theta2```, respectively. They will represent the slope values towards which the networks weights should move. These slopes will also be scaled with the learning rate.

These slopes are then added to ```theta1``` and ```theta2```

```
;;apply slopes
  set theta1 (theta1 matrix:- delta1)
  set theta2 (theta2 matrix:- delta2)
```

### The recognition

The code for recongising the current number on screen is actually farily simple. It is done with the ```test-cur``` function.

First, we collect the pixels of the currently drawn digit and propagate it throw the network:

```
  let cur (matrix:make-constant 1 400 0)
  matrix:set-row cur 0 get-patch-colors

  let a1 (matrix:copy (add-ones (matrix:copy cur)))
  let z2 (a1 matrix:* (matrix:transpose theta1))
  let a2 (add-ones (sigmoid (matrix:copy z2)))
  let z3 (a2 matrix:* (matrix:transpose theta2))
  let a3 (sigmoid z3)
```
We then find the largest element in a3 (since a3 contains the probabilities for the current drawing being each digit, we are just looking for the higest probability), and return its position. 

```
  let curMax (max item 0 matrix:to-row-list a3)
  let prediction position curMax item 0 matrix:to-row-list a3
  report prediction
```

