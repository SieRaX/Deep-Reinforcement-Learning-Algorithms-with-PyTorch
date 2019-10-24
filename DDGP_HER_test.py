from agents.actor_critic_agents.DDPG import DDPG
import torch
import numpy as np
from utilities.data_structures.Replay_Buffer import Replay_Buffer

class DDPG_HER(DDPG):
    """DDPG algorithm with HER"""

    agent_name = "DDPG-HER"

    def __init__(self, config, buffersize, batchsize, HER_sample_proportion):
        DDPG.__init__(self, config)
        self.replaybuffer = Replay_Buffer(buffersize, batchsize, self.config.seed)


