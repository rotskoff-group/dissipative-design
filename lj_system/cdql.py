import os
import sys
import random
from simtk.openmm import OpenMMException
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from lj import LJ
from replaybuffer import ReplayBuffer
from cdqlnetwork import Model


class CDQL:
    def __init__(self, use_gpu=False, filename="temp2.h5", region_num=15, target_dist="default_gamma"):
        self.gamma = 0.95
        if use_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')
        assert (not use_gpu) or (self.device == torch.device('cuda:1'))
        self.folder_name = "./"
        self.sim_controller = LJ(filename, region_num=region_num, target_dist=target_dist)
        self.buffer = ReplayBuffer(1e6)
        self.batch_size = 32
        self.all_actions = [0.01, 0.25, 1.0]
        self.num_actions = len(self.all_actions)
        self.model = Model(self.device, num_bins=self.sim_controller.num_bins,
                           num_actions=self.num_actions)
        self.loss = []
        self.store_Q = []
        self.training_iter = 0
        self.update_freq = 2


    def _to_tensor(self, x):
        return torch.tensor(x).float().to(self.device)

    def _update(self):
        """Updates q1, q2, q1_target and q2_target networks based on clipped Double Q Learning Algorithm
        """
        if (len(self.buffer) < self.batch_size):
            return
        self.training_iter += 1
        # Make sure actor_target and critic_target are in eval mode
        assert not self.model.q_target_1.training
        assert not self.model.q_target_2.training

        assert self.model.q_1.training
        assert self.model.q_2.training
        transitions = self.buffer.sample(self.batch_size)
        batch = self.buffer.transition(*zip(*transitions))
        state_batch = self._to_tensor(batch.state)
        action_batch = self._to_tensor(
            batch.action).unsqueeze(1).to(torch.int64)
        reward_batch = self._to_tensor(batch.reward).unsqueeze(1)
        next_state_batch = self._to_tensor(batch.next_state)
        is_done_batch = self._to_tensor(batch.done).unsqueeze(1)
        with torch.no_grad():
            # Add noise to smooth out learning
            Q_next_1 = (1 - is_done_batch) * torch.min(self.model.q_target_1(
                next_state_batch), dim=1)[0].unsqueeze(1)
            Q_next_2 = (1 - is_done_batch) * torch.min(self.model.q_target_2(
                next_state_batch), dim=1)[0].unsqueeze(1)
            # Use max want to avoid underestimation bias
            Q_next = torch.max(Q_next_1, Q_next_2)
            Q_expected = reward_batch + self.gamma * Q_next  # Assumes no "Terminal State"

        Q_1 = self.model.q_1(state_batch).gather(1, action_batch)
        Q_2 = self.model.q_2(state_batch).gather(1, action_batch)
        L_1 = nn.MSELoss()(Q_1, Q_expected)
        L_2 = nn.MSELoss()(Q_2, Q_expected)
        self.loss.append([L_1.item(), L_2.item()])
        self.model.q_optimizer_1.zero_grad()
        self.model.q_optimizer_2.zero_grad()
        L_1.backward()
        L_2.backward()
        self.model.q_optimizer_1.step()
        self.model.q_optimizer_2.step()
        self.store_Q.append([Q_1.tolist(), Q_2.tolist(), Q_expected.tolist()])
        if (self.training_iter % self.update_freq) == 0:
            self.model.update_target_nn()

    def _get_action(self, state, episode, epsilon=0.0):
        """Gets action given some state
        if episode is less than 5 returns a random action for each region
        Args:
            state: List of states (corresponding to each region)
            episode: episode number
        """
        if (episode < 25):
            action = [random.choice(list(range(self.num_actions)))
                      for _ in range(len(state))]
            return action


        action = []
        self.model.q_1.eval()
        with torch.no_grad():
            for i in state:
                if i is None:
                    action.append(-1)
                else:
                    if (episode < 100 and np.random.random() < epsilon):
                        action.append(random.choice(list(range(self.num_actions))))
                    else:
                        curr_state = self._to_tensor(i)
                        curr_state = curr_state.unsqueeze(0)
                        action.append(torch.argmin(self.model.q_1(curr_state), dim=1).item())
        self.model.q_1.train()
        return action

    def _save_data(self):
        filename = self.folder_name + "replaybuffer"
        np.save(filename, np.array(self.buffer.buffer, dtype=object))

        filename = self.folder_name + "loss"
        np.save(filename, np.array(self.loss))

        filename = self.folder_name + "Q_pair.npy"
        np.save(filename, np.array(self.store_Q, dtype=object))

        self.model.save_networks(self.folder_name)

    def _save_episode_data(self, episode_folder_name):
        filename = episode_folder_name + "replaybuffer"
        np.save(filename, np.array(self.buffer.buffer, dtype=object))

        self.model.save_networks(episode_folder_name)

    def load_data(self):
        self.loss = torch.load(self.folder_name + "loss.pt").tolist()
        self.buffer.load_buffer(self.folder_name + "replaybuffer.npy")
        self.model.load_networks(self.folder_name)

    def train(self, num_decisions = 150):
        """Train q networks
        Args:
            num_decisions: Number of decisions to train algorithm for
        """
        os.system("mkdir " + self.folder_name + "Train")
        for i in range(500):
            print(i)
            episode_folder_name = self.folder_name + "Train/" + str(i) + "/"
            all_system_states = []
            all_system_rewards = []
            all_system_states_cluster = []
            all_grid_states_cluster = []

            os.system("mkdir " + episode_folder_name)
            filename = episode_folder_name + str(i) + ".h5"
            self.sim_controller.reset_context(filename)
            tag = "_train_init"
            self.sim_controller.run_decorrelation(5, tag)
            state, _, _ = self.sim_controller.get_state_reward(
                tag)  # Keep as list
            rb_episode_samples = [] #Count number of rb_samples
            for j in range(num_decisions):
                action_index = self._get_action(state, i)
                transition_to_add = [state, action_index]
                tag = "_train_" + str(j)
                actions = [self.all_actions[i] for i in action_index]
                try:
                    self.sim_controller.update_temperature(actions, tag)
                    # system_states, system_rewards, system_states_cluster = self.sim_controller.run_step(is_detailed=True, tag=tag)
                    self.sim_controller.run_step(is_detailed=False, tag=tag)
                    #
                    # all_system_states.append(system_states)
                    # all_system_rewards.append(system_rewards)
                    # all_system_states_cluster.append(system_states_cluster)

                except OpenMMException:
                    print("Broken Simulation at Episode:",
                          str(i), ", Decision:", str(j))
                    break

                state, reward, grid_states_cluster = self.sim_controller.get_state_reward(
                    tag)
                all_grid_states_cluster.append(grid_states_cluster)

                # Use len_reward for number of grids
                done = [((j + 1) == -1000)] * len(reward) #Never Done
                transition_to_add.extend([reward, state, done])
                rb_decision_samples = 0
                for rb_tuple in zip(*transition_to_add):
                    if None in rb_tuple:
                        continue
                    rb_decision_samples += 1
                    self.buffer.push(*list(rb_tuple))
                rb_episode_samples.append(rb_decision_samples)
                self._update()

                self._save_episode_data(episode_folder_name)
                np.save(episode_folder_name + "system_states",
                        np.array(all_system_states))
                np.save(episode_folder_name + "system_rewards",
                        np.array(all_system_rewards))
                np.save(episode_folder_name + "system_states_cluster",
                        np.array(all_system_states_cluster))
                np.save(episode_folder_name + "grid_states_cluster",
                        np.array(all_grid_states_cluster, dtype=object))
                np.save(episode_folder_name + "rb_episode_samples", np.array(rb_episode_samples))


            if (i % 10 == 9):
                #Save training data every 10 episodes
                self._save_data()


    def test(self):
        """Given trained q networks, generate trajectories
        Saves:
            grid_rewards: Numpy array of all the rewards of each region along traj
            grid_states: Numpy array of all the states (i.e. normalized distibution of cluster sizes)
                         of each region along traj
            grid_states_cluster: Numpy array of all the cluster sizes of each region along traj
            actions: Numpy array of actions taken along trajectory
            dissipation: Total dissipation (not average dissipation rate) along trajectory
            system_states: Numpy array of states of the system along traj:
            system_states_cluster: Numpy array of cluster sizes along traj
            system_rewards: Numpy array of reward of entire system along traj
        """
        all_grid_states = []
        all_grid_states_cluster = []
        all_grid_rewards = []
        all_system_rewards = []
        all_system_states = []
        all_system_states_cluster = []
        all_actions = []
        all_dissipation = []
        os.system("mkdir " + self.folder_name + "Test/")
        filename = self.folder_name + "Test/" + "TEST.h5"
        self.sim_controller.reset_context(filename)
        tag = "_test_init"
        self.sim_controller.run_decorrelation(5, tag)
        state, _, _ = self.sim_controller.get_state_reward(tag)  # Keep as list
        all_dissipation.append(self.sim_controller.get_dissipation())

        num_decisions = 150

        for i in range(num_decisions):
            print(i)
            action_index = self._get_action(state, episode=10000)
            tag = "_test_" + str(i)
            actions = [self.all_actions[i] for i in action_index]
            all_actions.append(actions)
            self.sim_controller.update_temperature(actions, tag)
            system_states, system_rewards, system_states_cluster = self.sim_controller.run_step(
                is_detailed=True, tag=tag)
            state, reward, grid_states_cluster = self.sim_controller.get_state_reward(
                tag)

            # The "grid states" and dissipation are recorded at the end of a decision
            # Dissipation here is total entropy production (not epr)
            # Actions are recorded at the beginning of the decision
            all_grid_states.append(state)
            all_grid_rewards.append(reward)
            all_grid_states_cluster.append(grid_states_cluster)

            all_actions.append(actions)
            all_dissipation.append(self.sim_controller.get_dissipation())

            # The "System States" are recorded every 0.25 seconds. Excludes 0th second
            all_system_states.append(system_states)
            # Just to have a 1D array use extend
            all_system_rewards.extend(system_rewards)
            all_system_states_cluster.append(system_states_cluster)

            if (i % 10 == 9):
                np.save(self.folder_name + "grid_rewards",
                        np.array(all_grid_rewards))
                np.save(self.folder_name + "grid_states",
                        np.array(all_grid_states, dtype=object))
                np.save(self.folder_name + "grid_states_cluster",
                        np.array(all_grid_states_cluster, dtype=object))
                np.save(self.folder_name + "actions", np.array(all_actions))

                np.save(self.folder_name + "dissipation",
                        np.array(all_dissipation))
                np.save(self.folder_name + "system_states",
                        np.array(all_system_states))
                np.save(self.folder_name + "system_states_cluster",
                        np.array(all_system_states_cluster))
                np.save(self.folder_name + "system_rewards",
                        np.array(all_system_rewards))


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        region_num=int(sys.argv[-2])
        target_dist=sys.argv[-1]
        c = CDQL(region_num=region_num, target_dist=target_dist)
        print(c.sim_controller.target_dist)
        print(c.sim_controller.region_num)
    else:
        c = CDQL()
    c.train()
    # c.model.load_networks(c.folder_name)
    # c.test()
