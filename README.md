Project description:

The goal is to optimize the existing code ml-models.py. The code performs machine learning training using sklearn on different datasets and the computational time currently is too large. It's also lacking documentation, and the whole project is within one Python module. Since I'm tired of waiting for hours for the training to be completed, the goal of this project is to optimize to code, including:

1) Split the code into modules and classes
2) Add documentation 
3) Add decorators and investigate the performance of the code, then try to optimize some parts of it with respect to memory usage/computational time
4) Possibly re-organize the storage of the dataset, get rid of some loops, remove unnecessary copies of the objects
5) Use tools like CPU/GPU acceleration to get faster performance

