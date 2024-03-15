Project description:

The goal is to optimize the existing code ml-models.py. The code performs machine learning training using sklearn on different datasets and the computational time currently is too large. It's also lacking documentation, and the whole project is within one Python module. Since I'm tired of waiting for hours for the training to be completed, the goal of this project is to optimize to code, including:

Here is the summary of the work:

1) Split the code into modules and classes (Done)
2) Add documentation (Done)
3) Add decorators and investigate the performance of the code, then try to optimize some parts of it with respect to memory usage/computational time
4) Possibly re-organize the storage of the dataset, get rid of some loops, remove unnecessary copies of the objects (Done)
5) Use tools like CPU/GPU acceleration to get faster performance (Done and got much better performance! I used the ml models from cuML (cuda) and got a 3x speed for learning and testing for heart disease dataset.)
6) Add code testing (Done)

