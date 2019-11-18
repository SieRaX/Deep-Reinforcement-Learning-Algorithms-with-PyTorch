import os
import gym
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
import numpy as np
# Ensure we get the path separator correct on windows
from gym.envs.robotics.robot_env import RobotEnv

MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

class ConstrainFetchPickAndPlace(FetchPickAndPlaceEnv):
    '''This class is made to get consistent initial observation every time it resets.'''

    environment_name = "Fetch-Pick_and_Place-Constrained-v1"

    def __init__(self, reward_type = 'sparse', constrain = None, initial_info = None):
        #first initialize the super class
        super().__init__(reward_type)
        assert constrain in {None, 'Fix', 'Gaussian'}, "variable constrain is not right"

        if constrain:
            self.initial_state_1 = initial_info["initial_state"]
            self.goal = initial_info["goal"]
        self.constrain = constrain
        self.initial_info = initial_info

    def reset(self):

        if not self.constrain == None:
            super(RobotEnv, self).reset()
            self.initial_state_1 = self.initial_info["initial_state"]
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()
            if self.constrain == 'Fix':
                self.goal = self.initial_info["goal"]
            elif self.constrain == 'Gaussian':
                self.goal = [np.random.normal(i, 1) for i in self.initial_info["goal"]]
            obs = self._get_obs()
            return obs
        else:
            return super().reset()

if __name__ == "__main__":
    env = gym.make("FetchPickAndPlace-v1")
    info = env.reset()
    # print(info["observation"])
    # # print(['a':2, 'b':2])
    initial_info_1 = {"initial_state": info["observation"], "goal": info["desired_goal"]}

    trigger = True
    for i in range(100):
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
    env_2 = ConstrainFetchPickAndPlace(constrain = 'Fix', initial_info = initial_info_1)
    info = env_2.reset()
    trigger = True
    for i in range(100):
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

    print("-----------------------------")
    env_3 = ConstrainFetchPickAndPlace(constrain = 'Gaussian', initial_info = initial_info_1)
    info = env_3.reset()
    trigger = True
    for i in range(100):
        info_1 = info
        info = env_3.reset()
        check1 = np.array_equal(info["achieved_goal"], info_1["achieved_goal"])
        check2 = np.array_equal(info["observation"], info_1["observation"])
        check3 = np.array_equal(info["desired_goal"], info_1["desired_goal"])
        # if not (check1 and check2 and check3):
        #     print("The \"ConstrainedFetchReach_Gaussian\" is different!!")
        #     print("The different part: observation: {}, achieved_goal: {}, desired_goal: {}".format(check2, check1, check3))
        #     print("In attempt ", i)
        #     trigger = False

    if(trigger):
        print("The \"ConstrainedFetchReach\" is Same!!")
