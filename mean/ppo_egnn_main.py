import os 
import time
from tqdm import tqdm 
import wandb
import tyro

import torch
from torch.utils.tensorboard import SummaryWriter

from docking_args import Args
from rewards import *
from docking_env import DockingEnv
from ppo_egnn import PPOAgent

if __name__ == "__main__":
    args = tyro.cli(Args)

    env_device = torch.device("cpu")
    policy_device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = DockingEnv(args, device=env_device)

    sample_obs = env.reset() # obs dict
    # getting params from EGNN architecture
    in_node_nf = sample_obs["h"].shape[1]
    n_channel = sample_obs["x"].shape[1]

    agent = PPOAgent(
        in_node_nf=in_node_nf,
        n_channel=n_channel,
        action_dim=6,
        hidden_dim=128,
        lr_actor=1e-5,
        lr_critic=3e-5,
        gnn_layers=args.gnn_layers,
        max_trans=args.max_trans,
        continuous_action_space=True,
        num_epochs=3,
        batch_size=128, # 256 works on H200, but none open rn 
        device=policy_device
    )

    run_name = f"docking__{args.exp_name}__{args.seed}__{int(time.time())}"

    # wandb tracking setup
    if args.track:
        wandb.init( # type: ignore
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        wandb.watch(agent.policy, log="gradients", log_freq=1000) # type: ignore

    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    target_env_steps = 100_000
    max_episode_steps = args.max_episode_steps

    num_episodes = target_env_steps // max_episode_steps
    print(f"Training for ~{target_env_steps} env steps ≈ {num_episodes} episodes")

    rewards = []
    global_step = 0
    update_every_episodes = 10
    for ep in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        ep_reward = []
        while not done:
            action, logprob, state_value = agent.policy.select_action(state)
            
            # Compute state value
            next_state, reward, done, info = env.step(
                action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            )
            terminated = bool(info['terminated'])
            
            ep_reward.append(reward)
            
            # Save in buffer
            # (optional but good) detach + move state to CPU so buffer never holds CUDA tensors
            cpu_state = {
                "h": state["h"].detach().cpu(),
                "x": state["x"].detach().cpu(),
                "ctx_edges": [e.detach().cpu() for e in state["ctx_edges"]],
                "att_edges": [e.detach().cpu() for e in state["att_edges"]],
                "antigen_mask": state["antigen_mask"].detach().cpu(),
                "antibody_mask": state["antibody_mask"].detach().cpu(),
                "residues": state["residues"],  # just metadata
            }

            agent.buffer.store_transition(
                state=cpu_state,
                action=action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action,
                logprob=float(logprob.detach().cpu().item() if isinstance(logprob, torch.Tensor) else logprob),
                reward=float(reward),
                done=bool(done),
                terminated=terminated,  # from info["terminated"]
                state_value=float(state_value.detach().cpu().item() if isinstance(state_value, torch.Tensor) else state_value),
            )
            
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
        
        # update_stats = agent.update_policy()
        # Only update every K episodes
        if (ep + 1) % update_every_episodes == 0:
            update_stats = agent.update_policy()
            if args.track and update_stats:
                wandb.log(update_stats, step=global_step)
                for key, value in update_stats.items():
                    writer.add_scalar(key, value, global_step)
    # After loop, in case there are leftover episodes in the buffer:
    if len(agent.buffer.rewards) > 0:
        update_stats = agent.update_policy()

    writer.close()

    # Make sure checkpoint dir exists
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = os.path.join("checkpoints", f"{run_name}_final.pt")
    torch.save(
        {
            "model_state_dict": agent.policy.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved final checkpoint to {checkpoint_path}")
