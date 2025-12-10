import os 
import yaml
import time
import pandas as pd
import numpy as np
from tqdm import tqdm 
import random 
import wandb
import tyro
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import Sequential
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from torch_geometric.data import Data

from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.spaces import Box

from Bio.PDB import PDBList, PDBParser, Select, PDBIO
from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector
from Bio.PDB.Polypeptide import three_to_index

from ppo_actor_critic import GNNActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "docking-ppo"
    wandb_entity: str = "kyz-mit"
    yamls_path: str = "/project/liulab/gkim/antigen_prediction/eval_boltz_on_sabdab/all_yaml_outdir"
    pdbs_path: str = "/project/liulab/gkim/antigen_prediction/data/renumbered_sabdab_pdb_files/pdb_files"
    sabdab_tsv: str = "/project/liulab/gkim/antigen_prediction/data/sabdab_all_fv_summary_2.tsv"


def get_chain_info_from_pdb(pdb_path, yaml_path):
    """Get chain information from YAML file."""    
    if not os.path.exists(pdb_path):
        return None, None, None, None, None
    
    if not os.path.exists(yaml_path):
        print(f"No YAML file found at {yaml_path}")
        return None, None, None, None, None
    
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract chain IDs and sequences from YAML data
        # Assume the first sequence is heavy and the second sequence is light
        # UNLESS there are more than 2 sequences
        h_chain = None
        l_chain = None
        h_seq_yaml = None
        l_seq_yaml = None

        # Look for sequences in the YAML data
        if 'sequences' in yaml_data and isinstance(yaml_data['sequences'], list):
            sequences = yaml_data['sequences']
            if len(sequences) == 2:
                h_chain = sequences[0]['protein']['id']  # First sequence is heavy
                l_chain = sequences[1]['protein']['id']  # Second sequence is light
                h_seq_yaml = sequences[0]['protein']['sequence']
                l_seq_yaml = sequences[1]['protein']['sequence']
            elif len(sequences) > 2:
                # first sequence is antigen (for multimer predictions)
                h_chain = sequences[1]['protein']['id']  # Second sequence is heavy
                l_chain = sequences[2]['protein']['id']  # Third sequence is light
                h_seq_yaml = sequences[1]['protein']['sequence']
                l_seq_yaml = sequences[2]['protein']['sequence']
        
        if 'antigen' in yaml_data and isinstance(yaml_data['antigen'], list):
            antigen = yaml_data['antigen'][0]['protein']['sequence']
        else:
            antigen = None
        
        return h_chain, l_chain, h_seq_yaml, l_seq_yaml, antigen
        
    except Exception as e:
        print(f"Error reading YAML file for {yaml_path}: {e}")
        return None, None, None, None, None
    
def featurizer(heavy_chain, ag_chain, device='cpu'):
    # https://towardsdatascience.com/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95/
    # node feature matrix with shape (number of nodes, number of features)
    # graph connectivity (how the nodes are connected) with shape (2, number of directed edges)
    # node ground-truth labels. In this problem, every node is assigned to one class (group)

    # # construct the distance matrix
    heavy_coords = np.array([res['CA'].coord for res in heavy_chain if 'CA' in res])
    antigen_coords = np.array([res['CA'].coord for res in ag_chain if 'CA' in res])
    antigen_residues = np.array([int(res.id[1]) for res in ag_chain if 'CA' in res])
    
    dist_matrix = np.linalg.norm(
        heavy_coords[:, np.newaxis, :] - antigen_coords[np.newaxis, :, :],
        axis=-1
    )
    # https://numpy.org/devdocs/reference/generated/numpy.argpartition.html
    # only sort the bottom k
    bottom_k = np.argpartition(dist_matrix.flatten(), k)[:k]
    # flatten and unravel :)
    bottom_k_indices = np.unravel_index(bottom_k, dist_matrix.shape)

    residues = bottom_k_indices
    
    def matrix_idx_to_resnum(chain, matrix_idx_list):
        ca_residues = [res for res in chain if 'CA' in res]
        return [int(ca_residues[i].id[1]) for i in matrix_idx_list]

    # Convert matrix indices → PDB residue numbers
    heavy_residues = matrix_idx_to_resnum(heavy_chain, list(dict.fromkeys(residues[0])))
    ag_residues    = matrix_idx_to_resnum(ag_chain, list(dict.fromkeys(residues[1])))
    
    # matrix indices (0..N-1) → PDB residue numbers
    heavy_id_map = [int(res.id[1]) for res in heavy_chain if 'CA' in res]
    ag_id_map    = [int(res.id[1]) for res in ag_chain if 'CA' in res]

    # pdb residue numbers of the heavy and antigen residues
    heavy_residues = [heavy_id_map[item.item()] for item in list(dict.fromkeys(residues[0]))]
    ag_residues = [ag_id_map[item.item()] for item in list(dict.fromkeys(residues[1]))]

    node_features = torch.zeros(len(heavy_residues + ag_residues), 2, device=device) # chain, residue_id
    
    heavy_idx_to_node_idx = {res_idx: i for i, res_idx in enumerate(heavy_residues)}
    ag_idx_to_node_idx = {res_idx: i + len(heavy_residues) for i, res_idx in enumerate(ag_residues)}

    # heavy chains are "0" and antigen chains are "1"
    for i, res_idx in enumerate(heavy_residues + ag_residues):
        if i < len(heavy_residues):
            res_idx = heavy_id_map[i]
            node_features[i][0] = 0
            node_features[i][1] = three_to_index(heavy_chain[res_idx].get_resname())
        else:
            res_idx = ag_id_map[i - len(heavy_residues)]
            node_features[i][0] = 1
            node_features[i][1] = three_to_index(ag_chain[res_idx].get_resname())
    
    hc_nodes = torch.tensor([heavy_idx_to_node_idx[matrix_idx_to_resnum(heavy_chain, [id.item()])[0]] 
                             for id in residues[0]], device=device)
    ag_nodes = torch.tensor([ag_idx_to_node_idx[matrix_idx_to_resnum(ag_chain, [id.item()])[0]] 
                             for id in residues[1]], device=device)

    edge_connections = torch.vstack((hc_nodes, ag_nodes))
    
    num_edges = edge_connections.T.shape[0]
    edge_features = torch.zeros(num_edges, 1, device=device)
    for j, (a, b) in enumerate(zip(residues[0], residues[1])):
        edge_features[j] = dist_matrix[a, b].item()
    # edge_features = edge_features
        
    return node_features, edge_connections, edge_features

class DockingEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, heavy_chains, antigen_chains, featurizer, k=12, device='cpu'):
        self.heavy_chains = heavy_chains
        self.antigen_chains = antigen_chains
        self.featurizer = featurizer
        self.device = device
        self.action_space = Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([10, 10, 10, 2*np.pi, 2*np.pi, 2*np.pi], dtype=np.float32))
        self.observation_space = None
        # store the starting coordinates of every antigen atom in a fixed order
        self.current_step = 0
        
    def step(self, action):
        self.current_step += 1
        node_features, edge_index, edge_attr = self.featurizer(self.heavy_chain, self.antigen_chain, device=self.device)
        state = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr) # store this as a Data type object!
        # buffer.states.append(state)
        trans = action[0:3]
        rot = action[3:]
        # print(action)
        # multiply the rotation matrices wrt each of the directions
        rotm = rotaxis2m(rot[0], Vector(1, 0, 0)) @ rotaxis2m(rot[1], Vector(0, 1, 0)) @ rotaxis2m(rot[2], Vector(0, 0, 1))
        for atom in self.antigen_chain.get_atoms():
            first_atom_coord = atom.coord
            # going into a coordinate object, convert everything to numpy
            atom.coord = rotm @ atom.coord + np.array(trans)
            node_features, edge_index, edge_attr = self.featurizer(self.heavy_chain, self.antigen_chain, device=self.device)
        next_state = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr) 
        next_reward = self._compute_reward()
        # print(next_reward)
        return next_state, next_reward, next_reward > -100 or self.current_step > 20, None
        
    def reset(self):
        complex_i = random.randrange(len(self.antigen_chains))
        print(complex_i)
        self.antigen_chain = self.antigen_chains[complex_i]
        self.heavy_chain = self.heavy_chains[complex_i]
        self.starting_ag_coords = [atom.coord.copy() for atom in self.antigen_chain.get_atoms()]
        self.current_step = 0
        # reset antigen atoms to their stored starting positions, then apply a translation
        for atom, start_coord in zip(self.antigen_chain.get_atoms(), self.starting_ag_coords):
            atom.coord = start_coord + np.array([-10, 0, 0])
        return self._get_state()

    def _get_state(self):
        node_features, edge_index, edge_attr = self.featurizer(self.heavy_chain, self.antigen_chain, device=self.device)
        # print(edge_index.shape, edge_attr.shape)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    def _compute_reward(self):
        node_features, edge_index, edge_attr = self.featurizer(self.heavy_chain, self.antigen_chain, device=self.device)
        reward = -edge_attr.mean().item()
        # print(reward)
        return reward

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def store_transition(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()

# https://github.com/saqib1707/RL-PPO-PyTorch
class PPOAgent:
    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            hidden_dim, 
            lr_actor, 
            lr_critic, 
            continuous_action_space=False, 
            num_epochs=10, 
            eps_clip=0.2, 
            action_std_init=0.6, 
            gamma=0.99,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            batch_size=64,
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

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.continuous_action_space = continuous_action_space
        self.device = device

        self.policy = GNNActorCritic(
            node_feature_dim=obs_dim,  # this is the node feature size
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            continuous_action=True  # we want continuous 6D actions
        ).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()  # Initialize MSE loss


    def compute_returns(self):
        returns = []
        discounted_reward = 0

        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = np.array(returns, dtype=np.float32)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
        return returns


    def update_policy(self):
        # Skip update if buffer is empty
        if len(self.buffer.rewards) == 0:
            self.buffer.clear()
            return
            
        rewards_to_go = self.compute_returns()

        # Handle Data objects - batch them instead of converting to numpy
        states_list = [s.to(self.device) for s in self.buffer.states]
        
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
        
        state_vals_list = []
        for sv in self.buffer.state_values:
            if isinstance(sv, torch.Tensor):
                state_vals_list.append(sv.cpu().item() if sv.is_cuda else sv.item())
            else:
                state_vals_list.append(sv)
        state_vals = torch.from_numpy(np.array(state_vals_list, dtype=np.float32)).to(self.device)

        advantages = rewards_to_go - state_vals
        
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
                batch_states = Batch.from_data_list([states_list[i] for i in batch_indices])
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                # evaluate old actions and values
                state_values, logprobs, dist_entropy = self.policy.evaluate_actions(
                    batch_states.x, batch_states.edge_index, batch_states.edge_attr, batch_states.batch, batch_actions
                )
                # print(logprobs.shape, batch_old_logprobs.shape)

                logprob_diff = torch.clamp(logprobs - batch_old_logprobs.squeeze(-1), min=-10, max=10)
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

    
if __name__ == "__main__":
    args = tyro.cli(Args)
    sd_pd = pd.read_table(args.sabdab_tsv)

    i = 0
    k = 12

    parser = PDBParser(QUIET=True)

    heavy_chains, antigen_chains = [], []
    for yaml_file in tqdm(os.listdir(args.yamls_path)[:10], desc="Processing YAML files"):
        yaml_path = os.path.join(args.yamls_path, yaml_file)
        name = yaml_file.split('.')[0]
        pdb_file = name + '.pdb'
        pdb_path = os.path.join(args.pdbs_path, pdb_file)
        # we'll use Gauen's function because it already maps from the name to the pdb that is already downloaded on the server...
        h, l, _, _, _ = get_chain_info_from_pdb(pdb_path, yaml_path)
        row = sd_pd[(sd_pd["pdb"] == name) & (sd_pd["Hchain"] == h) & (sd_pd["Lchain"] == l)]
        if row.empty:
            continue
        ag = row["antigen_chain"].values[0] # this gives us the antigen chain alone!
        structure = parser.get_structure(name, pdb_path)
        if ag not in [chain.id for chain in structure[0]]:
            continue

        # # make a distance matrix
        heavy_Cas = []
        
        # # convert this into accessing entries in a generator?
        heavy_chain = structure[0][h]
        antigen_chain = structure[0][ag]
        heavy_chains.append(heavy_chain)
        antigen_chains.append(antigen_chain)

        # node_features, edge_connections, edge_features = featurizer(heavy_chain, antigen_chain)
    print(heavy_chains, antigen_chains)

    node_feature_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DockingEnv(heavy_chains, antigen_chains, featurizer, device=device)
    
    agent = PPOAgent(
        obs_dim=node_feature_dim,
        action_dim=6,
        hidden_dim=128,
        lr_actor=1e-4,
        lr_critic=1e-4,
        continuous_action_space=True,
        device=device
    )
    if args.track:
        run_name = f"docking__{args.exp_name}__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        wandb.watch(agent.policy, log="gradients", log_freq=1000)

    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    num_episodes = 20
    rewards = []
    global_step = 0
    for ep in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        ep_reward = []
        while not done:
            action, logprob = agent.policy.select_action(state)
            
            # Compute state value
            batch = torch.zeros(state.x.size(0), dtype=torch.long, device=state.x.device)
            _, state_value = agent.policy.forward(state.x, state.edge_index, state.edge_attr, batch)
            state_value = state_value.squeeze().item()
            
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action)
            ep_reward.append(reward)
            
            # Save in buffer
            agent.buffer.states.append(state)
            agent.buffer.actions.append(action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action)
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            agent.buffer.logprobs.append(logprob.detach().cpu().item() if isinstance(logprob, torch.Tensor) else logprob)
            agent.buffer.state_values.append(state_value)
            
            state = next_state
            global_step += 1
        ep_return = sum(ep_reward)

        ep_reward_mean = ep_return / len(ep_reward)

        rewards.append(ep_reward_mean)
        if args.track:
            wandb.log(
                {
                    "rollout/ep_return": ep_return,
                    "rollout/ep_reward_mean": ep_reward_mean,
                    "rollout/ep_len": len(ep_reward),
                },
                step=global_step,
            )

        writer.add_scalar("rollout/ep_return", ep_return, global_step)
        writer.add_scalar("rollout/ep_reward_mean", ep_reward_mean, global_step)
        writer.add_scalar("rollout/ep_len", len(ep_reward), global_step)
        
        update_stats = agent.update_policy()

        if args.track and update_stats:
            wandb.log(update_stats, step=global_step)
            for key, value in update_stats.items():
                writer.add_scalar(key, value, global_step)
