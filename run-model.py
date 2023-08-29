import numpy as np
from tkinter import messagebox
import cv2

#Define Layer Class
class Layer:
    z=0
    a=0
    def __init__(self, w_path, b_path):
        self.weights=np.loadtxt(w_path, delimiter=',')
        self.biases=np.loadtxt(b_path, delimiter=',')


#Define Activation Functions
def ReLU(x):
    return np.maximum(x,0)

def softmax(x):
    m=np.max(x)
    return np.exp(x-m)/np.exp(x-m).sum()

#Initialise Layers:
l1=Layer("model-parameters/layer1-weights.csv", "model-parameters/layer1-biases.csv")
l2=Layer("model-parameters/layer2-weights.csv", "model-parameters/layer2-biases.csv")
l3=Layer("model-parameters/layer3-weights.csv", "model-parameters/layer3-biases.csv")



#Define Forward Propagation
def forward_prop(image):
    l1.z=np.dot(image,l1.weights)+l1.biases
    l1.a=ReLU(l1.z)

    l2.z=np.dot(l1.z, l2.weights)+l2.biases
    l2.a=ReLU(l2.z)

    l3.z=np.dot(l2.z, l3.weights)+l3.biases
    l3.a=softmax(l3.z)
    return l3.a

#Get the image
def get_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.array(image)

#Run the model
def run_model(image):
    predictions=forward_prop(image.flatten())
    digit=np.argmax(predictions)
    messagebox.showinfo("Prediction", "You drew the number %d"%digit)

image=get_image("input-image.png")
run_model(image)

