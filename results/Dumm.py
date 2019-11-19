import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import gym
import numpy as np
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.DDPG_HER_Che import DDPG_HER_Che
from agents.actor_critic_agents.DDPG_HER import DDPG_HER
from utilities.data_structures.Config import Config
from agents.Trainer import Trainer
from gym.wrappers.time_limit import TimeLimit

from environments.ConstrainFetchPush import ConstrainFetchPush
from environments.ConstrainFetchReach import ConstrainFetchReach
from environments.ConstrainFetchSlide import ConstrainFetchSlide
from environments.ConstrainFetchPickAndPlace import ConstrainFetchPickAndPlace

if __name__ == '__main__':
    env = gym.make("FetchReach-v1")
    info = env.reset()
    env.render()
    for _  in range(100):
        env.reset()
        env.render()
    # print(info["observation"])
    # # print(['a':2, 'b':2])
    initial_info_1 = {"initial_state": info["observation"], "goal": info["desired_goal"]}
    env_3 = ConstrainFetchReach(constrain = 'Gaussian', initial_info = initial_info_1)
    info = env_3.reset()
    print(env_3)
