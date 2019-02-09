# DRLND_continuous-control-project-pytorch


Project 2: Continuous Learning

##Introduction

For this project, we train a two tennis agents to play a match

Trained Agent

A reward of is provided which us proportional to the distance between the robot end-effector and the target

The state space has 33 dimensions and contains the robot's position, velocity, acceleration, etc

The actions has four dimensions and consists of the torques for each of the joints in the robot

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

##Getting started

1. Download or clone the complete repositoryhttps://github.com/pablogkilroy/DRLND_continuous-control-project-pytorch from Github onto a Windows environment. 

The repository contains the following files:
- main.py: main program with episodic iterations
- ddpg_agent.py: Contains Agent class; OUNoise class and ReplayBuffer class
- Model.py: Contains Actor class and Critic class with models of the networks

The following folders are part of the repository:
- python: Contains the unity support files necessary to run the unity applications
- Reacher_Windows_x86_64: The reacher.exe unity applicaiton is used to run a single agent
- R_20: The reached.exe unity application is used to run 20 agents

2. Create a virtual environment using Anaconda prompt 
(for windows environment) 
>conda create --name drlnd-p2 python=3.6 
>activate drlnd

3. Install the files in the requirements.txt file:
>conda install --yes --file requirements.txt

4. Alternatively to 2 and 3 install the environment.yml file
conda env create -f environment.yml

##Instructions

1. Execute the main.py file. 
2. A plot of the rewards appears every 25 episodes. This can be changed in main.py in the function
def ddpg(n_episodes=200, max_t=1000, print_every=1, plot_every=25):