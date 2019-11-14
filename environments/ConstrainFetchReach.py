import os
import gym
from gym.envs.robotics.fetch.reach import FetchReachEnv
import numpy as np
# Ensure we get the path separator correct on windows
from gym.envs.robotics.robot_env import RobotEnv

MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

class ConstrainFetchReach(FetchReachEnv):
    '''This class is made to get consistent initial observation every time it resets.'''

    environment_name = "Fetch-Reach-Constrained-v1"

    def __init__(self, reward_type = 'sparse', constrain = False, initial_info = None):
        #first initialize the super class
        super().__init__(reward_type)
        if constrain:
            self.initial_state_1 = initial_info["initial_state"]
            self.goal = initial_info["goal"]
        self.constrian = constrain
        self.initial_info = initial_info

    def reset(self):

        self.initial_state_1 = self.initial_info["initial_state"]
        # self.initial_state =

        if self.constrian:
            super(RobotEnv, self).reset()
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()
            self.goal = self.initial_info["goal"]
            obs = self._get_obs()
            return obs
        else:
            return super().reset()

if __name__ == "__main__":
    env = gym.make("FetchReach-v1")
    info = env.reset()
    # print(info["observation"])
    # # print(['a':2, 'b':2])
    initial_info_1 = {"initial_state": info["observation"], "goal": info["desired_goal"]}

    trigger = True
    for i in range(10):
        info_1 = info
        info = env.reset()
        check1 = np.array_equal(info["achieved_goal"], info_1["achieved_goal"])
        check2 = np.array_equal(info["observation"], info_1["observation"])
        check3 = np.array_equal(info["desired_goal"], info_1["desired_goal"])
        if not (check1 and check2 and check3):
            print("The \"FetchReach\" is different!!")
            print("The different part: observation: {}, achieved_goal: {}, desired_goal: {}".format(check2, check1, check3))
            print("In attempt ", i)
            trigger = False
            break

    print("-----------------------------")
    env_2 = ConstrainFetchReach(constrain = True, initial_info = initial_info_1)
    info = env_2.reset()
    trigger = True
    for i in range(10):
        info_1 = info
        info = env_2.reset()
        check1 = np.array_equal(info["achieved_goal"], info_1["achieved_goal"])
        check2 = np.array_equal(info["observation"], info_1["observation"])
        check3 = np.array_equal(info["desired_goal"], info_1["desired_goal"])
        if not (check1 and check2 and check3):
            print("The \"ConstrainedFetchReach\" is different!!")
            print("The different part: observation: {}, achieved_goal: {}, desired_goal: {}".format(check2, check1, check3))
            print("In attempt ", i)
            trigger = False
            break

    if(trigger):
        print("The \"ConstrainedFetchReach\" is Same!!")
