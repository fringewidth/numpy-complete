# numpy-complete
NumPy-Complete is a neural network to recognise handwritten digits, trained using only NumPy(No TensorFlow, Pytorch) on the MNIST database.

## Installation
1. Make sure the following packages are installed:
```sh
pip install numpy pandas matplotlib opencv-python
```

2. Clone the repository

```sh
git clone https://github.com/fringewidth/numpy-complete.git
```

## Usage
1. Modify `input-image.png` to a custom handwritten digit. The image must be 28px $\times$ 28px

2. Run `run-model.py`

## Features
- Recognises a handwritten digit from a 28 $\times$ 28 pixel grid using a feed-forward neural network with two hidden layers, with an accuracy of 90.41%.
- Makes use of only NumPy for numerical processing. All functions ar custom implemented and modifiable.
- Ability to run custom input.
- Training notebook available for customizing the training process.
See [train.ipynb](https://github.com/fringewidth/numpy-complete/blob/main/train.ipynb) for details on architecture and implementaion.


