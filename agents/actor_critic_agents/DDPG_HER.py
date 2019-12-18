from agents.actor_critic_agents.DDPG import DDPG
from agents.HER_Base import HER_Base
import os
import numpy as np

class DDPG_HER(HER_Base, DDPG):
    """DDPG algorithm with hindsight experience replay"""
    agent_name = "DDPG-HER"

    def __init__(self, config):
        DDPG.__init__(self, config)
        HER_Base.__init__(self, self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                          self.hyperparameters["HER_sample_proportion"])

    def step(self):
        """Runs a step within a game including a learning step if required"""

        record_video = self.video_mode and self.config.num_episodes_to_run - 10 <= self.episode_number
        if record_video:
            render_list = []

        while not self.done:
            self.action = self.pick_action()
            self.conduct_action_in_changeable_goal_envs(self.action)

            # '''Saving img for the videos'''
            # img = self.environment.render('rgb_array')
            # self.render.append(img)
            # ''''''
            if record_video:
                # f = open(self.file_name, mode='wb')
                img = self.environment.render('rgb_array')
                render_list.append(img)

            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_from_HER_and_Ordinary_Buffer()  # Samples experiences from buffer
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.track_changeable_goal_episodes_data()
            self.save_experience()
            if self.done: self.save_alternative_experience()
            self.state_dict = self.next_state_dict  # this is to set the state for the next iteration
            self.state = self.next_state
            self.global_step_number += 1

        if record_video:
            render_list = np.array(render_list)
            np.save(self.file_name+'/episode'+str(self.episode_number+1), render_list)
        self.episode_number += 1

    def enough_experiences_to_learn_from(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn"""
        return len(self.memory) > self.ordinary_buffer_batch_size and len(self.HER_memory) > self.HER_buffer_batch_size