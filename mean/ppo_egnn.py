import numpy as np

import torch
import torch.nn as nn

from ppo_actor_critic import EGNNActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.terminated = [] # true only for success / "real" docking terminal states

    def store_transition(self, state, action, logprob, reward, done, state_value, terminated):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.terminated.append(terminated)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()
        self.terminated.clear()

# https://github.com/saqib1707/RL-PPO-PyTorch
class PPOAgent:
    def __init__(
            self, 
            in_node_nf,
            n_channel,
            action_dim, 
            hidden_dim, 
            lr_actor, 
            lr_critic, #TODO add back later
            gnn_layers,
            max_trans,
            continuous_action_space=False, 
            num_epochs=3, # was 10
            eps_clip=0.2, 
            action_std_init=0.6, 
            gamma=0.99,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            batch_size=256, # was 64
            max_grad_norm=0.5,
            device='cpu'
        ):
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.continuous_action_space = continuous_action_space
        self.device = device

        self.policy = EGNNActorCritic(
            in_node_nf=in_node_nf,
            n_channel=n_channel,
            action_dim=action_dim,
            hidden_nf=hidden_dim,
            gnn_layers=gnn_layers,
            max_trans=max_trans,
            device=device
        ).to(device)
        
        #TODO: split learning rate into lr_actor vs. lr_critic later
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=lr_actor
        )

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()  # Initialize MSE loss

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(self.buffer.terminated, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=self.device)

        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            # 1 for nonterminal, 0 for terminal (true success states)
            next_nonterminal = 1.0 - terminated[t]

            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return returns.detach(), advantages.detach()

    # def compute_returns(self):
    #     returns = []
    #     discounted_reward = 0

    #     for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
    #         if done:
    #             discounted_reward = 0
    #         discounted_reward = reward + self.gamma * discounted_reward
    #         returns.insert(0, discounted_reward)

    #     returns = np.array(returns, dtype=np.float32)
    #     returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
    #     return returns

    def update_policy(self):
        # Skip update if buffer is empty
        if len(self.buffer.rewards) == 0:
            self.buffer.clear()
            return
            
        last_value = torch.tensor(0.0, device=self.device)
        rewards_to_go, advantages = self.compute_gae(last_value, gamma=self.gamma, lam=0.95)

        states_list = self.buffer.states  # list of obs dicts, already on device
        
        # Convert actions, logprobs, state_vals to tensors on device
        actions_list = []
        for action in self.buffer.actions:
            if isinstance(action, torch.Tensor):
                actions_list.append(action.cpu().numpy() if action.is_cuda else action.numpy())
            else:
                actions_list.append(action)
        actions = torch.from_numpy(np.array(actions_list, dtype=np.float32)).to(self.device)
        
        logprobs_list = []
        for logprob in self.buffer.logprobs:
            if isinstance(logprob, torch.Tensor):
                logprobs_list.append(logprob.cpu().item() if logprob.is_cuda else logprob.item())
            else:
                logprobs_list.append(logprob)
        old_logprobs = torch.from_numpy(np.array(logprobs_list, dtype=np.float32)).to(self.device)
        
        # state_vals = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=self.device)
        # advantages = rewards_to_go - state_vals
        
        if len(advantages) == 1:
            advantages = advantages - advantages.mean()
        else:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std
            else:
                advantages = advantages - advantages.mean()
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_mb = 0

        for _ in range(self.num_epochs):
            # generate random indices for minibatch
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(self.buffer.states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                # Batch Data objects
                obs_mb = [states_list[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                # evaluate old actions and values
                state_values, logprobs, dist_entropy = self.policy.evaluate_actions_batch(
                    obs_mb, batch_actions
                )

                logprob_diff = logprobs - batch_old_logprobs
                ratios = torch.exp(logprob_diff)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                state_values_flat = state_values.view(-1)
                critic_loss = 0.5 * self.mse_loss(state_values_flat, batch_rewards_to_go)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # for tracking purposes:
                with torch.no_grad():
                    approx_kl = (batch_old_logprobs - logprobs).mean().item()
                    clip_frac = (torch.abs(ratios - 1.0) > self.eps_clip).float().mean().item()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
                total_kl += approx_kl
                total_clip_frac += clip_frac
                num_mb += 1
        
        if num_mb > 0:
            stats = {
                "loss/policy_loss": total_actor_loss / num_mb,
                "loss/value_loss": total_critic_loss / num_mb,
                "loss/entropy": total_entropy / num_mb,
                "ppo/approx_kl": total_kl / num_mb,
                "ppo/clip_fraction": total_clip_frac / num_mb,
                "values/advantages_mean": adv_mean,
                "values/advantages_std": adv_std,
            }
        else:
            stats = {}
        self.buffer.clear()
        return stats

