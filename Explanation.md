# Explanation of my code

As requested a couple of weeks ago...

## Backstory

Most of the code written in the project is based on some python code that I wrote around August of 2017. Although the file has now been 
lost in the many directories of my PC, I still remember some things about it. Specifically: 

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

The next two lines create the weights for the neural network. Since our network has 3 layers (including the input and output, we only
need 2 matrices of weights. For easier vectorization, these are oriented in a way that makes it very easy to multiply them with the
inputs. 

In any case, the first matrix of weights has a height of the hidden layer size and the width of 401. The width remains constant due to 
the fact that we will always input images that have a total of 400 pixel. We add one to this number since we also need a bias. The height
of the matrix will be the size of the hidden layer. 

We can apply similar reasoning to the second matrix. The width will be the size of the hidden layer + 1 to account for bias, and the
height will be 10 (which is how many values we need to correctly predict what a digit is, given an input)

### The Training
### The recognition


