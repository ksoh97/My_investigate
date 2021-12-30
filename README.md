# My_investigate

iNNvestigate can be installed with the following commands.
The library is based on Keras and therefore requires a supported [Keras-backend](https://keras.io/backend/)
o(Currently only the Tensorflow backend is supported. We test with Python 3.6, Tensorflow 1.12 and Cuda 9.x.):

```bash
pip install innvestigate
# Installing Keras backend
pip install [tensorflow | theano | cntk]
```

To use the example scripts and notebooks one additionally needs to install the package matplotlib:

```bash
pip install matplotlib
```

The library's tests can be executed via:
```bash
git clone https://github.com/albermax/innvestigate.git
cd innvestigate
python setup.py test
```

## Usage and Examples

The iNNvestigate library contains implementations for the following methods:

* *function:*
  * **gradient:** The gradient of the output neuron with respect to the input.
  * **smoothgrad:** [SmoothGrad](https://arxiv.org/abs/1706.03825) averages the gradient over number of inputs with added noise.
* *signal:*
  * **deconvnet:** [DeConvNet](https://arxiv.org/abs/1311.2901) applies a ReLU in the gradient computation instead of the gradient of a ReLU.
  * **guided:** [Guided BackProp](https://arxiv.org/abs/1412.6806) applies a ReLU in the gradient computation additionally to the gradient of a ReLU.
  * **pattern.net:** [PatternNet](https://arxiv.org/abs/1705.05598) estimates the input signal of the output neuron.
* *attribution:*
  * **input_t_gradient:** Input \* Gradient
  * **deep_taylor[.bounded]:** [DeepTaylor](https://www.sciencedirect.com/science/article/pii/S0031320316303582?via%3Dihub) computes for each neuron a rootpoint, that is close to the input, but which's output value is 0, and uses this difference to estimate the attribution of each neuron recursively.
  * **pattern.attribution:** [PatternAttribution](https://arxiv.org/abs/1705.05598) applies Deep Taylor by searching rootpoints along the singal direction of each neuron.
  * **lrp.\*:** [LRP](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) attributes recursively to each neuron's input relevance proportional to its contribution of the neuron output.
  * **integrated_gradients:** [IntegratedGradients](https://arxiv.org/abs/1703.01365) integrates the gradient along a path from the input to a reference.
  * **deeplift.wrapper:** [DeepLIFT (wrapper around original code, slower)](http://proceedings.mlr.press/v70/shrikumar17a.html) computes a backpropagation based on "finite" gradients.
* *miscellaneous:*
  * **input:** Returns the input.
  * **random:** Returns random Gaussian noise.

## Visualization
### MNIST examples
![Group 1628](https://user-images.githubusercontent.com/57162425/147735475-ab9a27aa-ddb3-4e82-ad02-64eda0a9d8a7.png)
 
 
### Brain examples
![Group 2045](https://user-images.githubusercontent.com/57162425/147735466-572510ad-ffc8-43a0-9ae1-1eb7dbe6152f.png)
 
 
### Counterfactual examples
![Group 1889](https://user-images.githubusercontent.com/57162425/147735471-e72a329f-948e-485e-abae-3d091c36c954.png)
