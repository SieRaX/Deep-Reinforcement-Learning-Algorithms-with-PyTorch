import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
import numpy as np
import tables
import os

import time

class DDPG(Base_Agent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)

        if self.video_mode:
            self.file_name = "DDPG_"+ self.environment_title
            for i in range(config.num_episodes_to_run):
                pathset = os.path.join(self.file_name)
                if not (os.path.exists(pathset)):
                    os.mkdir(pathset)
            # f = tables.open_file(self.file_name, mode = 'w')
            # f.close()
            # datainfo = "DDPG_"+ self.environment_title + "_info.txt"
            # f = open(self.file_name, 'w')
            # f.close()
            # f = open(datainfo, 'w')
            # f.write(str(self.height))
            # f.write(str(self.width))
            # f.write(str(self.channel))
            # f.write(str(config.max_step))
            # f.write(str(config.num_episodes_to_run))
            # f.close()


    def step(self):
        """Runs a step in the game"""
        # print("(DDPG) into the step")

        # if self.video_mode:
        #     f = open(self.file_name, mode = 'a')
            # f = open(self.file_name, 'a')
            # f.write("Episode" + str(self.episode_number) +"\n")
            # self.f = tables.open_file(self.file_name, mode='a')
            # self.atom = tables.Int64Atom()
            # self.array_c = self.f.create_earray(self.f.root, "Episode"+str(self.episode_number), self.atom, (0,self.height, self.width, self.channel))
        if self.video_mode:
            render_list = []
        while not self.done:
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            # print("picked action")

            """This is for the Cart-Pole environment"""
            if(self.get_environment_title() == "CartPole"):
                go_action = np.argmax(self.action)
                self.action = np.zeros(2)
                # print(self.action)

                self.action[go_action] = 1
                # self.action = np.put(self.action, go_action, 1)
                # print(self.action)
                self.conduct_action(go_action)

            else:
                self.conduct_action(self.action)
                # print("(DDPG) action conducted! Rendering...")
                if self.video_mode:
                    # f = open(self.file_name, mode='wb')
                    img = self.environment.render('rgb_array')
                    render_list.append(img)
                    # img = np.reshape(img, (1)).tolist()
                    # f.write(str(img))
                    # f.write('\n')
                    # img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
                    # print(type(img))
                    # print(img.shape)
                    # print(self.array_c.shape)
                    # print(img)
                    # line = '\n'
                    # f.write(img.tostring())
                    # f.write(line.encode("utf-8"))
                    # f.close()
                    # self.array_c.append(img)
                # self.render.append(img)


            # print("(DDPG)outside the loop")
            # print(self.time_for_critic_and_actor_to_learn())
            # This is the learning part
            if self.time_for_critic_and_actor_to_learn():
                # print("(DDPG) It is time to learn!!")
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    # print("(DDPG) running in range")
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
                    # print("(DDPG)running in range complete")

            # print("(DDPG) outside of critic loop")
            self.save_experience()
            # print("(DDPG) saving experience")
            ######################
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
            # print("(DDPG) incrementing step number")
        self.episode_number += 1

        if self.video_mode:
            render_list = np.array(render_list)
            np.save(self.file_name+'/episode'+str(self.episode_number), render_list)
        # print("The epsiode end! rendering!!")
        # self.environment.render()

    def sample_experiences(self):
        return self.memory.sample()

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None: state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return action.squeeze(0)

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        # print("(DDPG) inside the critic_learn()")
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        # print("(DDPG) critic learn loss: ",loss)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        # print("(DDPG) inside the compute_loss()")
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        # print("(DDPG) after torch.no_grad")
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        # print("(DDPG) inside the compute_critic_targets")
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        # print("(DDPG) inside the compute_critic_values_for_next_states")
        with torch.no_grad():
            # print("(DDPG) comput_critic_values_for_next_states()) inside the torch.no_grad()")
            # input()
            # print(self.actor_target)
            # print(next_states)
            # print(self.actor_target(next_states))
            actions_next = self.actor_target(next_states)
            # input()
            # print("(DDPG comput_critic_values_for_next_states()) after calculating actor_target")
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        # print("(DDPG compute_critic_values_for_next_states()) after torch.no_grad")
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss