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
        self.fc1 = nn.Linear(obs_size, n_actions)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, n_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        # x = self.relu(self.fc1(state))
        # x = self.relu(self.fc2(x))
        action_logits = self.fc1(state)
        action_probs = self.softmax(action_logits)
        dist = Categorical(action_probs)
        return action_logits, dist


class SARAPG():
    name = 'SARAPG'

    def __init__(
            self,
            env_constructor,
            gamma: float = 0.99,
            learning_rate: float = 0.001,
            large_trajectory_batch: int = 200,
            mini_trajectory_batch: int = 100,
            eval_every: int = 250,
            max_steps: int = 100):

        self.env_constructor = env_constructor
        self.env = env_constructor[0](**env_constructor[1])
        self.gamma = gamma
        self.large_trajectory_batch = large_trajectory_batch
        self.mini_trajectory_batch = mini_trajectory_batch
        self.eval_every = eval_every
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        # create network
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.policy_net = PolicyNet(obs_size, n_actions)

        # make a copy of the reference net
        self.past_policy_net = PolicyNet(obs_size, n_actions)
        self.past_policy_net.load_state_dict(self.policy_net.state_dict())
        self.past_policy_net.eval()

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

    def fit(self, outer_loop_size, inner_loop_size):
        """
        iterations : number of updates
        """
        self.evaluation_value = []
        self.evaluation_time = []
        self.evaluation_iteration = []
        state = self.env.reset()
        done = False

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        tot_updates = 0

        for tt in range(outer_loop_size):

            for kk in range(inner_loop_size):

                if kk == 0:
                    N = self.large_trajectory_batch
                else:
                    N = self.mini_trajectory_batch

                # Generate batch
                batched_loss_tilde = torch.zeros((N, 1), dtype=torch.float)
                batched_loss = torch.zeros((N, 1), dtype=torch.float)

                for i in range(N):
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
                    returns = torch.zeros((T, 1), dtype=torch.float)
                    returns[-1] = trajectory_rewards[-1]
                    for j in range(1, T):
                        returns[T-j-1] = trajectory_rewards[T-j-1] + self.gamma * returns[T-j]

                    # Compute loss over one trajectory
                    batch_state = torch.FloatTensor(
                        np.array(trajectory_states, dtype=float)
                    ).to(device)
                    batch_action = torch.LongTensor(
                        np.array(trajectory_actions, dtype=int)
                    ).unsqueeze(1).to(device)
                    past_action_logits_g, _ = self.past_policy_net(batch_state)
                    past_logprobs = nn.functional.log_softmax(
                        past_action_logits_g, dim=-1).gather(1, batch_action.long())
                    current_action_logits_g, _ = self.policy_net(batch_state)
                    current_logprobs = nn.functional.log_softmax(
                        current_action_logits_g, dim=-1).gather(1, batch_action.long())

                    # importance weight
                    with torch.no_grad():
                        past_action_logits, _ = self.past_policy_net(batch_state)
                        # proba_past_policy = [\pi_{\theta_{t-1}}(s_i,a_i)]_{i= 1,.. ,T_i}
                        proba_past_policy = nn.functional.softmax(
                            past_action_logits, dim=-1).gather(1, batch_action.long())

                        current_action_logits, _ = self.policy_net(batch_state)
                        # proba_current_policy = [\pi_{\theta_{t}}(s_i,a_i)]_{i= 1,.. ,T_i}
                        proba_current_policy = nn.functional.softmax(
                            current_action_logits, dim=-1).gather(1, batch_action.long())
                        is_weight = torch.prod(proba_past_policy / proba_current_policy, dim=0)

                    batched_loss_tilde[i] = -(past_logprobs * returns).sum() * is_weight
                    batched_loss[i] = -(current_logprobs * returns).sum()

                if kk == 0:
                    u = {}
                    loss = batched_loss.mean()
                    loss.backward()

                    self.past_policy_net.load_state_dict(self.policy_net.state_dict())
                    self.past_policy_net.eval()

                    for name, param in self.policy_net.named_parameters():
                        param.data = param.data - self.learning_rate * param.grad
                        u[name] = param.grad.clone()
                else:
                    loss = batched_loss.mean()
                    loss_tilde = batched_loss_tilde.mean()
                    loss.backward()
                    loss_tilde.backward()

                    tmp_net = PolicyNet(obs_size, n_actions)
                    tmp_net.load_state_dict(self.policy_net.state_dict())
                    tmp_net.eval()

                    for v1, v2 in zip(self.policy_net.named_parameters(), self.past_policy_net.named_parameters()):
                        name = v1[0]
                        param = v1[1]
                        assert name == v2[0]  # same name
                        gt = v1[1].grad
                        gt_tilde = v2[1].grad
                        u[name] = u[name] + gt - gt_tilde
                        param.data = param.data - self.learning_rate * u[name]

                    self.past_policy_net = tmp_net

                tot_updates += 1

                # delete all gradients
                for name, param in self.policy_net.named_parameters():
                    param.grad.zero_()
                for name, param in self.past_policy_net.named_parameters():
                    if param.grad:
                        param.grad.zero_()

                # evaluate agent
                if tot_updates % self.eval_every == 0:
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

    SARAPG_PARAMS = dict(
        gamma=0.99,
        large_trajectory_batch=10,
        mini_trajectory_batch=10,
        eval_every=1,            # evaluate every ... steps
        learning_rate=0.01,       # learning rate
        max_steps=200
    )
    agent = SARAPG(
        env_constructor=(get_env, dict()),
        **SARAPG_PARAMS
    )
    agent.fit(outer_loop_size=100, inner_loop_size=1)

    plt.figure(figsize=(12, 6))
    plt.plot(agent.evaluation_time, agent.evaluation_value)
    plt.xlabel('Number of steps')
    plt.ylabel('Performance')

    plt.figure(figsize=(12, 6))
    plt.plot(agent.evaluation_iteration, agent.evaluation_value)
    plt.xlabel('Number of grad updates')
    plt.ylabel('Performance')
    plt.show()
