# Deeplearning Training Template

## Overview
This repository contains a template for typical deeplearning tranining problem.
Most problems can be solved by simple deep neural network architecture and system.
The common part in this similar system can be solved by making a template.
Before using this template, everytime you have to write your own code for same thing, such as setting hyper-parameters, loss function and optimizer.
However, this template helps you to avoid boring process and save your time.

## Requirements

- python 2.7 or higher
- Keras with backend(Tensorflow or Theano)

## Simple and Naive Tutorial
Basically, There are three parts for this module. 
First, Modelbuilder and you can refer a **AbstractModelBuilder Class** in [template_trainer.py](https://github.com/kh-kim/deeplearning_training_template/blob/master/template_trainer.py). 
Second, **LearningSuite Class** in [template_trainer.py](https://github.com/kh-kim/deeplearning_training_template/blob/master/template_trainer.py).
Lastly, **Your own Iterator Class**, which inherits from [Keras Iterator Class](https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
Also, You can set a hyper-parameters in json file ([sample json](https://github.com/kh-kim/deeplearning_training_template/blob/master/hparam.json)) and read it in the program.
A first step to write your code, you need to define model architecture via your own model builder class inherits from **AbstractModelBuilder Class**.
After that, you also need to write your own iterator using Koeras Iterator Class to feed your network.
Then you can use those using LearningSuite object.
You can take a look at [sample.py](https://github.com/kh-kim/deeplearning_training_template/blob/master/sample.py) file to know further.

## Sample Usage

	$ python sample.py

## Author

Ki Hyun Kim / pointzz.ki@gmail.com