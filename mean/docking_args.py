import os
from dataclasses import dataclass

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
    
    k_ctx: int = 16
    att_cutoff: float = 12.0
    max_episode_steps: int = 50
    max_trans: float = 10.0
    init_trans_noise: float = 5.0
    init_rot_noise: float = 0.5
    step_trans_noise: float = 0.0
    rmsd_threshold: float = 2.0
    gnn_layers: int = 4