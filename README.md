# intro1-FinalProject
Final Project for intro to cs 1

## Some documentation:

### What is this? 
This is a netlogo model that allows you to train a neural network that can reconsize handwritten digits. All data for the digits is taken from the MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/) 

### What can I control when training a neural network?
There are three parameters that you can control in when training the neural network
- hidden layer size (this is the size of the hidden layer in the network)
- batch size (the amount of examples used in every forward propagation batch)
- learning rate (the constant by which the weights found by backpropagation are multiplied)

### How do I use this model?
1. Download the zip file and unzip it into a non temporary directory (if you open the netlogo file in a temporary directory, netlogo will not be able to find the files).
2. Configure the network parameters
2. Press the setup button to load starting weights into the network
3. Press the infinite train button to begin training the network. Note that you will still be able to adjust the batch size and learning rate while the network is training
4. Notice that the number shown on the "% correct in last batch" goes up
5. Once the network finishes training, use the setup-draw and draw buttons to load your own examples into the network

