import os
import torch
import numpy as np
import pandas as pd
import random

from gymnasium.spaces import Box

from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector

from docking_args import Args
from ab_setup import build_graph_from_pdb
from rewards import *
from constants import *


class DockingEnv():
    def __init__(self, args: Args, device: torch.device):
        self.args = args
        self.device = device

        # Load SABDAB summary
        sabdab = pd.read_table(args.sabdab_tsv)
        sabdab.dropna(inplace=True)

        self.rows = sabdab[["pdb", "Hchain", "antigen_chain"]].reset_index(drop=True)
        self.pdb_dir = args.pdbs_path

        # Action: 6D continuous (tx,ty,tz,rx,ry,rz)
        self.action_dim = 6
        self.action_space = Box(
            low=np.array([-args.max_trans]*3 + [-np.pi]*3, dtype=np.float32),
            high=np.array([ args.max_trans]*3 + [ np.pi]*3, dtype=np.float32),
        )

        self.max_episode_steps = args.max_episode_steps

        # These will be filled per-reset
        self.h0 = None
        self.x0 = None
        self.ctx_edges0 = None
        self.att_edges0 = None
        self.antigen_mask0 = None
        self.antibody_mask0 = None
        self.residues0 = None
        self.gt_ab_ca = None

        self.h = None
        self.x = None
        self.ctx_edges = None
        self.att_edges = None
        self.antigen_mask = None
        self.antibody_mask = None
        self.residues = None

        self.steps = 0

        self.prev_rmsd = None
        self.curr_rmsd = None

        # self.reset() TODO: assuming we are doing this in main!!!
    
    def _build_obs(self):
        return {
            "h": self.h,
            "x": self.x,
            "ctx_edges": self.ctx_edges,
            "att_edges": self.att_edges,
            "antigen_mask": self.antigen_mask,
            "antibody_mask": self.antibody_mask,
            "residues": self.residues,
        }
        
    def step(self, action):
        self.steps += 1
        # Ensure float tensor on device
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        else:
            action = action.to(self.device).float()

        # clamping actions to prevent coordinates from exploding in magnitude
        max_t = self.args.max_trans
        action[:3] = action[:3].clamp(-max_t, max_t)
        action[3:] = action[3:].clamp(-np.pi, np.pi)

        trans = action[:3]  # tx,ty,tz
        rot = action[3:]    # rx,ry,rz

        # Build rotation matrix
        Rx = rotaxis2m(rot[0].item(), Vector(1, 0, 0))
        Ry = rotaxis2m(rot[1].item(), Vector(0, 1, 0))
        Rz = rotaxis2m(rot[2].item(), Vector(0, 0, 1))
        R = torch.from_numpy((Rx @ Ry @ Rz).astype(np.float32)).to(self.device)

        ab_idx = torch.where(self.antibody_mask)[0]
        ab_coords = self.x[ab_idx]               # [M, C, 3]
        ab_center = ab_coords.mean(dim=(0,1))    # [3]

        centered = ab_coords - ab_center.view(1, 1, 3)
        rotated = torch.einsum("ij,...j->...i", R, centered)

        # Optional tiny per-step noise
        if self.args.step_trans_noise > 0:
            trans = trans + torch.randn_like(trans) * self.args.step_trans_noise

        self.x[ab_idx] = rotated + ab_center.view(1, 1, 3) + trans.view(1, 1, 3)

        reward, new_rmsd = self._compute_reward()
        done = False
        terminated = False
        truncated = False

        if new_rmsd.item() < self.args.rmsd_threshold:
            reward += TERMINAL_CORRECT_DOCK_REWARD
            done = True
            terminated = True
        elif self.steps >= self.max_episode_steps:
            done = True
            truncated = True

        info = {"rmsd": new_rmsd.item(), "terminated": terminated, "truncated": truncated}
        return self._build_obs(), reward, done, info

        
    def reset(self):
         # 1) sample a complex index
        complex_i = random.randrange(len(self.rows))
        row = self.rows.iloc[complex_i]
        pdb_id = row.pdb
        antibody_chain_id = row.Hchain
        antigen_chain_id = row.antigen_chain.split("|")[0].strip() #TODO: find cleaner way of handling multi-chain antigens potentially

        print("selected_complex idx =", complex_i,
              "pdb =", pdb_id,
              "heavy =", antibody_chain_id,
              "antigen =", antigen_chain_id)
        
        pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")

        # 2) build graph from this PDB
        (
            h0,
            x0,
            ctx_edges0,
            att_edges0,
            antigen_mask0,
            antibody_mask0,
            residues0,
        ) = build_graph_from_pdb(
            pdb_path,
            antigen_chain_id,
            antibody_chain_id,
            k_ctx=self.args.k_ctx,
            att_cutoff=self.args.att_cutoff,
            device=self.device,
        )

        # 3) store "ground truth" tensors
        self.h0 = h0
        self.x0 = x0
        self.ctx_edges0 = ctx_edges0
        self.att_edges0 = att_edges0
        self.antigen_mask0 = antigen_mask0
        self.antibody_mask0 = antibody_mask0
        self.residues0 = residues0

        ab_idx = torch.where(self.antibody_mask0)[0]
        self.gt_ab_ca = self.x0[ab_idx, 1, :].clone()   # CA channel

        # 4) initialize current state from ground truth
        self.h = self.h0.clone()
        self.x = self.x0.clone()
        self.ctx_edges = [e.clone() for e in self.ctx_edges0]
        self.att_edges = [e.clone() for e in self.att_edges0]
        self.antigen_mask = self.antigen_mask0.clone()
        self.antibody_mask = self.antibody_mask0.clone()
        self.residues = self.residues0

        self.steps = 0

        # 5) apply random rigid transform to antibody (for pose diversity)
        ab_idx = torch.where(self.antibody_mask)[0]

        # random translation
        rand_trans = torch.randn(3, device=self.device) * self.args.init_trans_noise

        # random rotation
        rand_rot = torch.randn(3, device=self.device) * self.args.init_rot_noise
        Rx = rotaxis2m(rand_rot[0].item(), Vector(1, 0, 0))
        Ry = rotaxis2m(rand_rot[1].item(), Vector(0, 1, 0))
        Rz = rotaxis2m(rand_rot[2].item(), Vector(0, 0, 1))
        R = torch.from_numpy((Rx @ Ry @ Rz).astype(np.float32)).to(self.device)

        ab_coords = self.x[ab_idx]               # [M, C, 3]
        ab_center = ab_coords.mean(dim=(0,1))    # [3]

        centered = ab_coords - ab_center.view(1, 1, 3)
        rotated = torch.einsum("ij,...j->...i", R, centered)
        self.x[ab_idx] = rotated + ab_center.view(1, 1, 3) + rand_trans.view(1, 1, 3)

        P = self.x[ab_idx, 1, :] # antibody CA coords
        self.curr_rmsd = rmsd(P, self.gt_ab_ca)
        self.prev_rmsd = self.curr_rmsd.detach().clone()

        # init_reward = self._compute_reward()
        print("init RMSD after reset =", self.curr_rmsd.item())

        return self._build_obs()

    def _compute_reward(self):
        """
        Reward = (previous RMSD - current RMSD) - small step cost.
        So: positive if we reduced RMSD, negative if we got worse.
        """
        ab_idx = torch.where(self.antibody_mask)[0]
        P = self.x[ab_idx, 1, :]  # [M, 3]
        new_rmsd = rmsd(P, self.gt_ab_ca)  # tensor scalar

        if self.prev_rmsd is None:
            # First step after reset: just set baseline, reward 0
            self.prev_rmsd = new_rmsd.detach().clone()
            self.curr_rmsd = new_rmsd
            return 0.0, new_rmsd

        # Improvement: positive if RMSD decreased
        improvement = (self.prev_rmsd - new_rmsd).item()

        shaped_reward = REWARD_SCALE * improvement - STEP_COST

        # Update stored rmsd
        self.prev_rmsd = new_rmsd.detach().clone()
        self.curr_rmsd = new_rmsd

        return shaped_reward, new_rmsd