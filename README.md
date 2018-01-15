# intro1-FinalProject
Final Project for intro to cs 1

## WHAT IS IT?

This is a netlogo simulation that allows you to train a neural network that can reconize and classify hand written digits. 

Read below if you don't know what neural networks are. 

## WHAT ARE NEURAL NETWORKS?

A neural network is a mathematical model that behaves like the human brain. Using a large network of neurons trained to behave in a certain way under certain conditions, a neural network can perform very complicated tasks, in this case classifying handwritten digits. 

Lets first talk about the simplest component of a neural network - a neuron. 

A neuron takes in a set of inputs (usually scaled from 0 to 1) and, from that, generates an output. This output is usally created by multiplying each input by a certain weight, summing these products together, and feeding this sum through a special function that scales it between 0 and 1. Chaining many neurons together creates a neural network:

A neural network is organized into rows of neurons called layers. Each neuron in a layer uses the outputs of the previous layer as its inputs. The only exception to this is the first layer. The inputs to this layer are controlled by the user of the network. In the case of the network used in this simulation, the first layer has 400 neurons, each representing the scaled pcolor of the 20x20 drawing that we are trying to classify. 

The last layer of the network is called the output layer. The output of neurons in this layer tells us what the neural network "thinks" about out input. The network in this model has 10 neurons in this layer, each representing a digit from 0 to 9. Once an example has been fed through a network, all of these neurons have outputs scaled from 0 to 1. The higher this ouput is, the more sure the network is about the input representing the given digit. To classify an input, the neuron with the highest value is chosen. 

The middle layer of the network is called the hidden layer. This layer is what allows the network to perform very complicated tasks, as it creates an extra transfer of information between the input and output layers. The network in this simulation only has one, but networks used for other purposes can theoretically have as many as needed.

Neural networks are trained by adjusting the weights of its neurons. This is done using a special algorithm known as backpropagation. Given a set of inputs and their correct outputs, backpropagation allows us to calculate the weights' "errors", and adjust them to be slightly more correct. A neural network is usally trained in batches, where a large number of inputs is fed through the network before the network is moved in the right direction using the average error between all those inputs. 


## HOW TO TRAIN A NETWORK

1. Press the setup button. This will initialize all internal variables, and load the network with starting weights. 
2. Configure the network parameters. Make sure to press the setup button again if you change them. The network parameters that you can change are:
    > __Hidden Layer Size__ - The size of the hidden layer of the network (read above if      you don't know what this means) 
    > __Batch Size__ - The amount of examples used per each batch from which the neural       network trains (read above if you don't know what this means)
    > __Learning Rate__ - How fast the network learns from each batch. This is the number     by which errors are multiplied when calculated from backpropagation (read above if        you don't know what this means) 

3. Either continiously press the 'train-once' button, or toggle the infinite train button. Notice that the number shown in the '% correct in last batch' window starts to go up. Note that you can still adjust the batch size and learning rate while the network is training. 
4. Once you are satisfied with the number shown in the '% correct in last batch' window (reaching around 90% usually takes less than a minute), either stop pressing the 'train-once' button, or untoggle the forever train button. 
5. Alternatively, you can press the 'load preset weights button' on the right side of the screen. These weights were generated after about 3 hours of training, and have around 95% accuracy in classifying digits. 

## USING THE NETWORK TO CLASSIFY YOUR OWN EXAMPLES

This network allows you to classify either 2 or 1 digit numbers. 

##### To classify a 1 digit number:
1. Press the 'setup one digit classification' button
2. Press the 'setup-draw' button
3. Toggle the 'draw' button
4. Draw a 1 digit number on the screen by pressing down your mouse to draw
5. Untoggle the 'draw' button
6. Press the 'classify current drawing (1 digit)' button
7. A window with the networks guess will pop up
8. If you don't feel like drawing a number, you can also load a random digit from the dataset using the corresponding button.

##### To classify a 2 digit number:
1. Press the 'setup two digit classification' button. Note that the screen will get larger.
2. Press the 'setup-draw' button
3. Toggle the 'draw' button
4. Draw a 2 digit number on the screen by pressing down your mouse to draw
5. Untoggle the 'draw' button
6. Press the 'classify current drawing (2 digit)' button
7. A window with the networks guess will pop up
8. Note that 2 digit classification simply consists of attempting to split the number into two parts to classify them separately

## BUGS AND LIMITATIONS

* There will sometimes be an error saying that netlogo has encountered number that are too large for it to handle. There is no known cause or fix to this.
* There will sometimes be errors about netlogo not finding a file in a certain directory. This only happens when you run the program directly from the provided .zip folder (without unzipping it). You can fix this by unzipping the program folder into a non temporary directory.


## CREDITS AND REFERENCES
##### Code credits:
Ivan Galakhov - python (June 2017)
##### Github link:
https://github.com/igalakhov/intro1-FinalProject
##### MNIST dataset:
http://yann.lecun.com/exdb/mnist/
##### Image credits:
http://neuralnetworksanddeeplearning.com/chap1.html


