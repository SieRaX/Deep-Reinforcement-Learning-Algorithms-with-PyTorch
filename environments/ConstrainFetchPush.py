import os
import gym
from gym.envs.robotics.fetch.reach import FetchReachEnv

# Ensure we get the path separator correct on windows
from gym.envs.robotics.robot_env import RobotEnv

MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

class ConstrainFetchReach(FetchReachEnv):

    environment_name = "Fetch-Reach-Constrained-v1"

    def __init__(self, reward_type = 'sparse', constrain = False, initial_info = None,):
        #first initialize the super class
        FetchReachEnv.__init__(reward_type)

        if constrain:
            self.initial_state = initial_info["initial_state"]
            self.goal = initial_info["goal"]
        self.constrian = constrain
        self.initial_info = initial_info

    def reset(self):

        self.initial_state = self.initial_info["initial_state"]

        if self.constrian:
            super(RobotEnv, self).reset()
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()
            self.goal = self.initial_info["goal"]
            obs = self._get_obs()
            return obs
        else:
            return super(ConstrainFetchReach).reset()

if __name__ == "__main__":
    env = gym.make("FetchReach-v1")
    info = env.reset()
    print(info)
    # print(info["observation"])
    # # print(['a':2, 'b':2])
    initial_info = {"initial_state": info["observation"], "goal": info["desired_goal"]}
    print(initial_info)
    env_2 = ConstrainFetchReach(constrain = True, initial_info = initial_info)
