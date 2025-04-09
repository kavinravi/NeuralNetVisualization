# NeuralNetVisualization
A way to visualize/animate a neural network approximating a function in realtime

## Packages/imports:
NNVis.py:
- numpy
- pytorch
- pyplot (through matplotlib)


NNVisManim.py:
- numpy
- manim
- tensorflow
- keras

Both start by randomizing a sinusoidal function, given by
```
x_data = np.random.uniform(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
x_data = np.sort(x_data)
y_data = (np.sin(0.5 * x_data) +
          np.sin(x_data) +
          0.5 * np.sin(2 * x_data) +
          np.random.uniform(-0.1, 0.1, size=x_data.shape))
```
x_data generating 100 points between -2π to 2π
y_data defines the function, which is given by y = sin(0.5x) + sin(x) + 0.5(2x), then introduces randomness 
  in the points so that the points aren't along the exact function, so that way the neural net can't just find the
  formula for this relatively simple sinusoidal function.

Simple neural network is defined, using PyTorch in NNVis.py and TensorFlow/Keras in NNVisManim.py. 

Activation function used was tanh, as it gave best performance, though not much better than ReLU. Sigmoid, LeakyReLU, etc. are also options.

Optimizer used was Adam, though SGD, RMSProp, Adagrad, etc. are also options.

Obviously, adding complexity beyond 2 100-neuron dense layers would allow for better prediction, but it seemed unnecessary given the simplicity of the function.

Finally, the network is instructed to make predictions of 200 random x-values between -2π and 2π (the domain of the input/training data).

For NNVis.py, it should open a window that essentially creates a stop-motion video showing the line (created by connecting all 200 predicted points) fitting to the data.
With 
```
if epoch % 10 == 0:
```
you can decide how "smooth" to make the animation. The lower the number, the smoother and longer the animation will be. The higher the number, the quicker but jumpier it will be.
The total # of epochs is 2000.

For NNVisManim.py, it creates a smoother animation. 
