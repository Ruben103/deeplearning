# Deep Learning project: <br> Face Recognition and Classification.
## Robin Schut and Sharif Hamed

In this research project, we attempt to set-up a deep convolutional network that is able to classify faces from a hand-made dataset.

This software trains a LeNet network to classify images.
The sofware takes raw images and extracts faces from them. 
The extracted faces are then pre-processed for the purpose of being input to the LeNet classifier.
The classifier trains on proper photos of faces and is tested on more difficult photos. The network is based on Yann et. al (1998):

![Alt text](./Images/lenet_architecture-768x226.png?raw=True "LeNet Architecture")

### Code base
The code can be downloaded from the public git repository
[here](https://github.com/Ruben103/deeplearning)

### Run Instructions
Command to install the packages: <br>
```pip3 install -r requirements.txt```

Command to run the code <br>
```python3 main.py```

### Scripts

**main.py:** <br>
	Several controlled inputs can be set here.
	
**data.py:** <br>
	The function face_extr extracts faces from images that are specifies by a path where they are stored.
	The extracted faces are then stored in another file. 

**model.py:** <br>
The architecture of the model

**visualisation.py:** <br>
File to visualise the data.

All scripts and methods can be called in main through (out-)commenting indicated lines.

### References
* Yann LeCun, L ́eon Bottou, Yoshua Bengio, and Patrick Haffner.  Gradient-based learning appliedto document recognition.Proceedings of the IEEE, 86(11):2278–2324, 1998.
