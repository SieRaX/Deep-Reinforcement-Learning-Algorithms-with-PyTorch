import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import gym

from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.DDPG_HER_Che import DDPG_HER_Che
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer

from agents.DQN_agents.DQN_HER import DQN_HER
from environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN


config = Config()
config.seed = 1
#config.environment = Bit_Flipping_Environment(14)
config.environment = gym.make("CartPole-v1")
#config.environment = gym.make("FetchPush-v1")
config.num_episodes_to_run = 1000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
#config.runs_per_agent = 3
config.runs_per_agent = 2
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {

"Actor_Critic_Agents": {
    "Actor": {
        "learning_rate": 0.001,
        "linear_hidden_units": [50, 50],
        "final_layer_activation": "TANH",
        "batch_norm": False,
        "tau": 0.01,
        "gradient_clipping_norm": 5
    },

    "Critic": {
        "learning_rate": 0.01,
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
    AGENTS = [DDPG, DDPG_HER_Che]
    #AGENTS = [DDPG_HER_Che, DDPG]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()

