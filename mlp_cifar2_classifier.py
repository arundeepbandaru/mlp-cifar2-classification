import pickle
import sys
import numpy as np

class LinearTransform(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.grad_weights = np.zeros_like(W)
        self.grad_biases = np.zeros_like(b)
        self.velocity_weights = np.zeros_like(W)
        self.velocity_biases = np.zeros_like(b)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
        #x -> vector, W -> weight matrix, b -> bias vector
        #x -> input_dims
        #W -> input_dims x output_dims
        #b -> output_dims

    def backward(self, grad_output, x, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        self.grad_weights = np.dot(x.T, grad_output)
        #gradient wrt Weight matrix
        #shape(x) = (batch_size, input_dims) => shape(x.T) = (input_dims, batch_size)
        #shape (grad_output) = (batch_size, output_dims)
        #x_T * grad_output = (input_dims, batch_size) * (batch_size, output_dims) = (input_dims, output_dims) = dim(W)
         
        self.grad_biases = np.sum(grad_output, axis=0)
        #shape (grad_output) = (batch_size, output_dims)
        #shape (grad_biases) = (output_dims,)
        #sums the elements of grad_output across its rows, outputs an array.
        #contains the gradients of the loss with respect to each element of the bias term b

        grad_x = np.dot(grad_output, self.W.T)
        #gradient wrt input vector x
        #shape (grad_output) = (batch_size, output_dims)
        #shape(W) = (input_dims, output_dims) => shape(W.T) = (output_dims, input_dims)
        #grad_output * W.T = (batch_size, output_dims) * (output_dims, input_dims) = (batch_size, input_dims) = dim(x

        # Update weights and biases using gradient descent with momentum and L2 regularization
        self.velocity_weights = momentum * self.velocity_weights - learning_rate * (self.grad_weights + l2_penalty * self.W)        
        self.velocity_biases = momentum * self.velocity_biases - learning_rate * self.grad_biases
        #The velocity is updated by taking the previous velocity and 
        #subtracting the current gradient multiplied by the learning rate (learning_rate)

        self.W += self.velocity_weights
        self.b += self.velocity_biases
        #After updating the velocities, the weights and biases are updated by adding the velocity terms
        #The momentum term (momentum) controls how much of the previous velocity is retained and how much of the current gradient update is incorporated.

        return grad_x
    
class ReLU(object):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
        #ReLU(x) = max(x,0)
        #ReLU(x) = 0 if x < 0; x if x > 0

    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        return grad_output * (self.x > 0)
        #if x<0, the gradient will be 0
        #if x>0, the gradient will be grad_output

class SigmoidCrossEntropy(object):
    def forward(self, x):
        self.x = x
        # Apply clipping to limit the range of input values; prevent overflow from occuring
        #sig(x) = 1/(1 + e^-x)
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(
        self,
        grad_output,
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
    ):
        sigmoid_x = self.forward(self.x)
        return sigmoid_x * (1 - sigmoid_x) * grad_output
        #Derivative wrt x:
        #sig'(x) = (sig(x))*(1-sig(x))
        #check why its being multiplied by gradient of next layer: 
        #CHAIN RULE! grad_output is a constant being multiplied by the sigmoid function
class MLP(object):
    def __init__(self, input_dims, hidden_units):
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.W1 = np.random.randn(self.input_dims, self.hidden_units)
        self.b1 = np.zeros(self.hidden_units)
        self.W2 = np.random.randn(self.hidden_units, 1)
        self.b2 = np.zeros(1)

        #Initialize the first linear transform matrix with a random weight matrix & bias vector
        self.linear_transform1 = LinearTransform(
            np.random.randn(input_dims, hidden_units),
            np.random.randn(hidden_units),
        )
        
        #Initialize the second (output) linear transform matrix with a random weight matrix & bias vector
        #Desired number of output units: 1
        self.linear_transform2 = LinearTransform(
            np.random.randn(hidden_units, 1),
            np.random.randn(1),
        )

        # Initialize ReLU activation function
        self.relu = ReLU()
        
        # Initialize sigmoid loss function
        self.sigmoid_cross_entropy = SigmoidCrossEntropy()


    def train(
        self,
        x_batch,
        y_batch,
        learning_rate,
        momentum,
        l2_penalty,
    ):
        # Forward pass
        hidden_output = self.linear_transform1.forward(x_batch)
        hidden_output_relu = self.relu.forward(hidden_output)
        final_output = self.linear_transform2.forward(hidden_output_relu)
        predicted_labels = self.sigmoid_cross_entropy.forward(final_output)

        # Backward pass
        grad_output = predicted_labels - y_batch
        grad_hidden_output_relu = self.sigmoid_cross_entropy.backward(grad_output, final_output)
        grad_hidden_output = self.relu.backward(grad_hidden_output_relu, hidden_output_relu)

        # Update the second layer weights and biases
        grad_weights2 = np.dot(hidden_output_relu.T, grad_output).reshape(self.linear_transform2.W.shape)
        grad_biases2 = np.sum(grad_output, axis=0)
        self.linear_transform2.backward(grad_output, hidden_output_relu, learning_rate, momentum, l2_penalty)

        # Update the first layer weights and biases
        grad_input = self.linear_transform1.backward(grad_hidden_output, x_batch, learning_rate, momentum, l2_penalty)

        return grad_input

    def evaluate(self, x, y):
        # Forward pass
        hidden_output = self.linear_transform1.forward(x)
        hidden_output_relu = self.relu.forward(hidden_output)
        final_output = self.linear_transform2.forward(hidden_output_relu)
        predicted_labels = self.sigmoid_cross_entropy.forward(final_output)

        # Compute accuracy
        predicted_labels_binary = (predicted_labels >= 0.5).astype(int)
        accuracy = np.mean(predicted_labels_binary == y)

        return accuracy
    
    def compute_loss(self, x, y):
        linear1 = LinearTransform(self.W1, self.b1)
        relu = ReLU()
        linear2 = LinearTransform(self.W2, self.b2)
        sigmoid_ce = SigmoidCrossEntropy()

        h1 = linear1.forward(x)
        h1_relu = relu.forward(h1)
        h2 = linear2.forward(h1_relu)
        y_pred = sigmoid_ce.forward(h2)

        loss = np.mean(y_pred)
        return loss

data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

train_x = data[b'train_data']
train_y = data[b'train_labels']
test_x = data[b'test_data']
test_y = data[b'test_labels']

num_examples, input_dims = train_x.shape
hidden_units = 1000

num_epochs = 10
num_batches = 100
learning_rate = 0.092
momentum = 0.9
l2_penalty = 0.001

mlp = MLP(input_dims, hidden_units)

batch_size = num_examples // num_batches  # Calculate batch size


for epoch in range(num_epochs):
    # Shuffle the training data
    indices = np.random.permutation(num_examples)
    train_x_shuffled = train_x[indices]
    train_y_shuffled = train_y[indices]

#     total_loss = 0.0
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        # Select the current batch of examples and labels
        batch_x = train_x_shuffled[start_idx:end_idx]
        batch_y = train_y_shuffled[start_idx:end_idx]

        # Train the MLP on the current batch
        mlp.train(batch_x, batch_y, learning_rate, momentum, l2_penalty)
        loss = mlp.compute_loss(batch_x, batch_y)
#         total_loss += loss
#         total_loss /= num_batches

        print(
            '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                epoch + 1,
                batch + 1,
                loss,
            ),
            end='',
        )
        sys.stdout.flush()

    test_loss = mlp.compute_loss(test_x, test_y)
    train_loss =  mlp.compute_loss(train_x, train_y)
    train_accuracy = mlp.evaluate(train_x, train_y)
    test_accuracy = mlp.evaluate(test_x, test_y)

    print()
    print('Train Loss: {:.3f}    Train Accuracy: {:.2f}%'.format(
        train_loss,
        100. * train_accuracy,
    ))
    print('Test Loss:  {:.3f}    Test Accuracy:  {:.2f}%'.format(
        test_loss,
        100. * test_accuracy,
    ))
