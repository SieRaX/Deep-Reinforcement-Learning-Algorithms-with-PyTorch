import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time

import gym
import numpy as np
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.DDPG_HER_Che import DDPG_HER_Che
from agents.actor_critic_agents.DDPG_HER import DDPG_HER
from utilities.data_structures.Config import Config
from gym.wrappers.time_limit import TimeLimit
from agents.GridTrainer import Trainer as grid_Train
from agents.Trainer import Trainer as norm_Train

from environments.ConstrainFetchPush import ConstrainFetchPush
from environments.ConstrainFetchReach import ConstrainFetchReach
from environments.ConstrainFetchSlide import ConstrainFetchSlide
from environments.ConstrainFetchPickAndPlace import ConstrainFetchPickAndPlace

constrain_encode = [None, "Fix", "Gaussian"]
env_code = ["FetchReach-v1", "FetchPush-v1", "FetchSlide-v1", "FetchPickAndPlace-v1"]
envclass = [ConstrainFetchReach, ConstrainFetchPush, ConstrainFetchSlide, ConstrainFetchPickAndPlace]

print("Starting the training>>")
print("What kind of Env would you like to train? (1: Reach, 2: Push, 3: Slide, 4: PickAndPlace): ", end="")
env_choice = int(input()) - 1
print("Do you want to make constrain? (0: original, 1: No, 2: Fixed, 3: Gaussian): ", end="")
constrain_choice = int(input()) - 1
max_step = 50
if not constrain_choice == -1:
    print("Do you want to change the maximum steps of the environment? (No: 0, Yes: positive number): ", end ="")
    max_step = int(input())
    if(max_step == 0):
        max_step = 50
print("How many episodes per one run?: ", end="")
episodes_to_run = int(input())
print("How many runs per agent?: ", end="")
runs_per_agent = int(input())
print("Name of graph png?(0: None): ", end="")
directory = input()
if directory == "0":
    directory = None
else:
    directory = "data_and_graphs/" + directory +".png"
print("Do you want it recorded at video?(N: No, Y: Yes): ", end = "")
video = input()
if video == 'N':
    video = False
else:
    video = True

config = Config()
config.seed = 1


env = gym.make(env_code[env_choice])
info = env.reset()
initial_info = {"initial_state": info["observation"], "goal": info["desired_goal"]}
env_not_wrapped = envclass[env_choice](constrain=constrain_encode[constrain_choice], initial_info=initial_info)
config.environment = TimeLimit(env_not_wrapped, max_step)
if constrain_choice == -1:
    config.environment = gym.make(env_code[env_choice])
config.num_episodes_to_run = episodes_to_run
config.file_to_save_data_results = None
# config.file_to_save_results_graph = 'data_and_graphs/FetchPush_detecting_Anamaly.png'
config.file_to_save_results_graph = directory
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = runs_per_agent
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.video_mode = video
config.max_step = max_step

config.hyperparameters = {

"Actor_Critic_Agents": {
    "Actor": {
        "learning_rate": 0.001,
        # "learning_rate": 0.1,
        "linear_hidden_units": [50, 50],
        "final_layer_activation": "TANH",
        "batch_norm": False,
        "tau": 0.01,
        "gradient_clipping_norm": 5
    },

    "Critic": {
        "learning_rate": 0.01,
        # "learning_rate": 0.1,
        "linear_hidden_units": [50, 50, 50],
        "final_layer_activation": None,
        "batch_norm": False,
        "buffer_size": 30000,
        "tau": 0.01,
        "gradient_clipping_norm": 5
    },

    "batch_size": 256,
    "discount_rate": 0.9,
    "mu": 0.0,
    "theta": 0.15,
    "sigma": 0.25,
    "update_every_n_steps": 10,
    "learning_updates_per_learning_session": 10,
    "HER_sample_proportion": 0.8,
    "clip_rewards": False
}}


if __name__== '__main__':
    #AGENTS = [DDPG_HER_Che]
    AGENTS = [DDPG, DDPG_HER, DDPG_HER_Che]

    # grid_Train = grid_Train(config, AGENTS)
    # start = time.time()
    # grid_Train.run_games_for_agents()
    # grid_time = time.time() - start

    norm_trainer = norm_Train(config, AGENTS)
    start = time.time()
    norm_trainer.run_games_for_agents()
    norm_time = time.time() - start

    # print("non-multiprocess time: ", norm_time)
    # print("multiprocess time: ", grid_time)

    # for i in range(10):
    #     trainer = Trainer(config, AGENTS)
    #     trainer.run_games_for_agents()

    # anomaly = []
    # normal = []
    # f = open("Normallist.txt", 'r')
    # lines = f.readlines()
    # for line in lines :
    #     normal.append(float(line))
    #
    # f = open("Anomaly_list.txt", 'r')
    # lines = f.readlines()
    # for line in lines :
    #     anomaly.append(float(line))
    #
    # anomaly = np.array(anomaly)
    # normal = np.array(normal)
    #
    # print("The statictical result: ")
    # print("Normal>> total:{} mean: {}, std: {}".format(normal.size, np.mean(normal), np.std(normal)))
    # print("Anomaly>> total: {}mean: {}, std: {}".format(anomaly.size, np.mean(anomaly), np.std(anomaly)))