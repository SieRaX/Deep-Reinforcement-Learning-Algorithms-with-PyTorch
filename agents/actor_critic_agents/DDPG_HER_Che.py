from agents.actor_critic_agents.DDPG import DDPG
from agents.HER_Base import HER_Base

class DDPG_HER_Che(HER_Base, DDPG):
	"""DDPG algorithm with hindsight experience replay"""
	agent_name = "DDPG-HER-Che"
	def __init__(self, config):
		DDPG.__init__(self, config)
		HER_Base.__init__(self, self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                          self.hyperparameters["HER_sample_proportion"])

	def step(self):
		"""Runs a step within a game including a learning step if required"""
		while not self.done:
			self.action = self.pick_action()

			#now concate the goals with next_state
			self.conduct_action_in_changeable_goal_envs(self.action)
			# img = self.environment.render('rgb_array')
			# self.render.append(img)
			if self.time_for_critic_and_actor_to_learn():
				for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
					states, actions, rewards, next_states, dones = self.sample_from_HER_and_Ordinary_Buffer()
					self.critic_learn(states, actions, rewards, next_states, dones)
					self.actor_learn(states)
			self.track_changeable_goal_episodes_data()
			self.save_experience()

			#If the episode ended, do the HER Thing
			#After this the HER algorithm saves not only the original epsiode,
			#It saves episodes that the goal is the state that made triggered the
			#done variable to true.
			if self.done: self.save_alternative_experience()
			self.state_dict = self.next_state_dict
			self.state = self.next_state
			self.global_step_number += 1
		self.episode_number += 1

	def enough_experiences_to_learn_from(self):
		return len(self.memory) > self.ordinary_buffer_batch_size and len(self.HER_memory) > self.HER_buffer_batch_size