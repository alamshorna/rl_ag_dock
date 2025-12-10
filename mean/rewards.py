import torch

def rmsd(P, Q, eps=1e-8):
    """
    Root-mean-square deviation between two point sets P, Q of shape [M, 3]
    (after optimal alignment is usually better, but here we assume coords already roughly aligned).
    """
    diff = P - Q
    return torch.sqrt((diff ** 2).sum(dim=-1).mean() + eps)

def compute_reward(x, gt, ab_idx):
    """
    current antibody CA = x[ab_idx, 1, :]
    gt = ground truth CA [M, 3]
    """
    P = x[ab_idx, 1, :]
    # optional: align using Kabsch before RMSD
    # R, t = kabsch_rigid_transform(P, gt)
    # P_aligned = apply_rigid_transform(P.unsqueeze(1), R, t).squeeze(1)
    return -rmsd(P, gt)