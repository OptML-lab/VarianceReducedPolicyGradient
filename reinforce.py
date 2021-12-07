import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from gym.wrappers import Monitor

import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = self.softmax(action_logits)
        dist = Categorical(action_probs)
        return action_logits, dist


class PGAgente():
    name = 'PGAgent'

    def __init__(
            self,
            env_constructor,
            gamma: float = 0.99,
            num_trajectories: int = 200,
            eval_every: int = 1,
            learning_rate: float = 0.1,
            max_steps: int = 100,
            typology: str = "reinforce"):

        assert typology in ["reinforce", "gpomdp"]

        self.env_constructor = env_constructor
        self.env = env_constructor[0](**env_constructor[1])
        self.gamma = gamma
        self.num_trajectories = num_trajectories
        self.eval_every = eval_every
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.max_steps = max_steps
        self.typology = typology

        # create network
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.policy_net = PolicyNet(obs_size, n_actions)

        # optimizer
        # self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=learning_rate, momentum=0)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def select_action(self, state, evaluation=False):
        """
        If evaluation=False, get action according to exploration policy.
        Otherwise, get action according to the evaluation policy.
        """
        tensor_state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            _, dist = self.policy_net(tensor_state)
            action = dist.sample().item()
        return action

    def fit(self, iterations):
        """
        iterations : number of updates
        """
        self.evaluation_value = []
        self.evaluation_time = []
        self.evaluation_iteration = []
        state = self.env.reset()
        done = False

        for tt in range(iterations):
            # Generate batch
            batch_losses = torch.zeros((self.num_trajectories, 1), dtype=torch.float)
            for i in range(self.num_trajectories):
                trajectory_rewards = []
                trajectory_states = []
                trajectory_actions = []

                # Collect trajectory
                state = self.env.reset()
                for _ in range(self.max_steps):
                    # Interacting with environment
                    action = self.select_action(state, evaluation=False)
                    next_state, reward, done, _ = self.env.step(action)
                    self.total_timesteps += 1

                    # Storage
                    trajectory_rewards.append(reward)
                    trajectory_states.append(state)
                    trajectory_actions.append(action)

                    # iterate
                    state = next_state
                    if done:
                        break

                # Compute the trajectory of discounted rewards
                T = len(trajectory_states)
                # Compute loss over one trajectory
                batch_state = torch.FloatTensor(
                    np.array(trajectory_states, dtype=float)
                ).to(device)
                batch_action = torch.LongTensor(
                    np.array(trajectory_actions, dtype=int)
                ).unsqueeze(1).to(device)
                action_logits, _ = self.policy_net(batch_state)
                logprobs = nn.functional.log_softmax(
                    action_logits, dim=-1).gather(1, batch_action.long())

                if self.typology == "gpomdp":
                    # GPOMDP
                    returns = torch.zeros((T, 1), dtype=torch.float)
                    returns[-1] = trajectory_rewards[-1]
                    gamma_pow = torch.zeros((T, 1), dtype=torch.float)
                    for j in range(1, T):
                        returns[T-j-1] = trajectory_rewards[T-j-1] + self.gamma * returns[T-j]
                        gamma_pow[j] = np.power(self.gamma, j)
                    batch_losses[i] = -(gamma_pow * logprobs * returns).sum()
                else:
                    # REINFORCE
                    returns = 0
                    df = 1
                    for j in range(T):
                        returns += df * trajectory_rewards[j]
                        df *= self.gamma
                    batch_losses[i] = -logprobs.sum() * returns

            # update
            loss = batch_losses.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if tt % 2 == 0:
                self.scheduler.step()

            # evaluate agent
            if tt % self.eval_every == 0:
                mean_rewards = self.eval(n_sim=5)
                print(f'[time: {self.total_timesteps}, it: {tt}] eval_rewards: {mean_rewards}')
                self.evaluation_value.append(mean_rewards)
                self.evaluation_time.append(self.total_timesteps)
                self.evaluation_iteration.append(tt)

    def eval(self, n_sim=1, **kwargs):
        rewards = np.zeros(n_sim)
        eval_env = self.env_constructor[0](**self.env_constructor[1])     # evaluation environment
        # Loop over number of simulations
        for sim in range(n_sim):
            state = eval_env.reset()
            done = False
            while not done:
                action = self.select_action(state, evaluation=True)
                next_state, reward, done, _ = eval_env.step(action)
                # update sum of rewards
                rewards[sim] += reward
                state = next_state
        return rewards.mean()


def get_env():
    """Creates an instance of a CartPole-v0 environment."""
    return gym.make('CartPole-v0')


if __name__ == "__main__":

    REINFORCE_PARAMS = dict(
        gamma=0.99,
        num_trajectories=10,            # batch size
        eval_every=1,            # evaluate every ... steps
        learning_rate=0.01,       # learning rate
        max_steps=200,
        typology="gpomdp"
    )
    agent = PGAgente(
        env_constructor=(get_env, dict()),
        **REINFORCE_PARAMS
    )
    agent.fit(100)

    plt.figure(figsize=(12, 6))
    plt.plot(agent.evaluation_time, agent.evaluation_value)
    plt.xlabel('Number of steps')
    plt.ylabel('Performance')

    plt.figure(figsize=(12, 6))
    plt.plot(agent.evaluation_iteration, agent.evaluation_value)
    plt.xlabel('Number of grad updates')
    plt.ylabel('Performance')
    plt.show()
