
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

AA_TO_INDEX = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4,
    "GLY": 5, "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}

ATOM_CHANNELS = ["N", "CA", "C", "O", "CB"]  # n_channel = 5

def _safe_atom_xyz(residue, atom_name, fallback_xyz):
    if atom_name in residue and residue[atom_name].is_disordered():
        atom = sorted(residue[atom_name].child_dict.values(), key=lambda a: a.get_altloc())[0]
        return atom.get_coord().astype(np.float32)
    if atom_name in residue:
        return residue[atom_name].get_coord().astype(np.float32)
    return fallback_xyz

def parse_chain_residues(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":  # Skip Hetero/Water
            continue
        if "CA" not in res:
            continue
        residues.append(res)
    return residues

def build_node_tensors(antigen_residues, antibody_residues):
    all_residues = antigen_residues + antibody_residues
    n = len(all_residues)

    # features: AA one-hot (20) + chain one-hot (2)
    h = torch.zeros((n, 22), dtype=torch.float32)

    # coordinates: [N, n_channel, 3]
    x = torch.zeros((n, len(ATOM_CHANNELS), 3), dtype=torch.float32)

    antigen_mask = torch.zeros(n, dtype=torch.bool)
    antibody_mask = torch.zeros(n, dtype=torch.bool)

    for i, res in enumerate(all_residues):
        resname = res.get_resname()
        aa_idx = AA_TO_INDEX.get(resname, None)
        if aa_idx is not None:
            h[i, aa_idx] = 1.0

        # chain indicator
        if i < len(antigen_residues):
            h[i, 20] = 1.0  # antigen
            antigen_mask[i] = True
        else:
            h[i, 21] = 1.0  # antibody heavy chain
            antibody_mask[i] = True

        ca = res["CA"].get_coord().astype(np.float32)
        for c, atom_name in enumerate(ATOM_CHANNELS):
            xyz = _safe_atom_xyz(res, atom_name, fallback_xyz=ca)
            x[i, c, :] = torch.from_numpy(xyz)

    return h, x, antigen_mask, antibody_mask, all_residues

def knn_edges(ca_xyz, k):
    # ca_xyz: [N, 3] numpy
    # returns directed edges (row -> col) excluding self
    N = ca_xyz.shape[0]
    d2 = np.sum((ca_xyz[:, None, :] - ca_xyz[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, kth=min(k, N - 1), axis=1)[:, :k]
    rows = np.repeat(np.arange(N), nn.shape[1])
    cols = nn.reshape(-1)
    return rows, cols

def build_edges(h, x, antigen_mask, antibody_mask, k_ctx=16, att_cutoff=12.0):
    # CA coordinates for distances
    ca = x[:, 1, :].cpu().numpy()  # channel 1 is "CA"
    antigen_idx = np.where(antigen_mask.cpu().numpy())[0]
    antibody_idx = np.where(antibody_mask.cpu().numpy())[0]

    # context edges: intra-chain KNN within each chain
    ctx_rows, ctx_cols = [], []
    if len(antigen_idx) > 1:
        r, c = knn_edges(ca[antigen_idx], k=min(k_ctx, len(antigen_idx) - 1))
        ctx_rows.append(antigen_idx[r])
        ctx_cols.append(antigen_idx[c])
    if len(antibody_idx) > 1:
        r, c = knn_edges(ca[antibody_idx], k=min(k_ctx, len(antibody_idx) - 1))
        ctx_rows.append(antibody_idx[r])
        ctx_cols.append(antibody_idx[c])

    if len(ctx_rows) == 0:
        ctx_row = np.zeros((0,), dtype=np.int64)
        ctx_col = np.zeros((0,), dtype=np.int64)
    else:
        ctx_row = np.concatenate(ctx_rows).astype(np.int64)
        ctx_col = np.concatenate(ctx_cols).astype(np.int64)

    # attention edges: inter-chain proximity within cutoff
    # build directed edges both ways (antigen -> antibody and antibody -> antigen)
    if len(antigen_idx) == 0 or len(antibody_idx) == 0:
        att_row = np.zeros((0,), dtype=np.int64)
        att_col = np.zeros((0,), dtype=np.int64)
    else:
        A = ca[antigen_idx]  # [Na, 3]
        B = ca[antibody_idx]  # [Nb, 3]
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        cutoff2 = float(att_cutoff ** 2)
        a_ids, b_ids = np.where(d2 <= cutoff2)

        att_row = np.concatenate([antigen_idx[a_ids], antibody_idx[b_ids]]).astype(np.int64)
        att_col = np.concatenate([antibody_idx[b_ids], antigen_idx[a_ids]]).astype(np.int64)

    ctx_edges = [torch.from_numpy(ctx_row).long(), torch.from_numpy(ctx_col).long()]
    att_edges = [torch.from_numpy(att_row).long(), torch.from_numpy(att_col).long()]

    return ctx_edges, att_edges

def kabsch_rigid_transform(P, Q, eps=1e-8):
    """
    P: [M, 3] original (antibody) points
    Q: [M, 3] target (predicted) points
    Returns R: [3, 3], t: [3]
    """
    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    X = P - P_mean
    Y = Q - Q_mean

    C = X.t() @ Y  # [3, 3]
    U, S, Vt = torch.linalg.svd(C)
    R = Vt.t() @ U.t()

    # fix reflection
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1.0
        R = Vt.t() @ U.t()

    t = (Q_mean - P_mean @ R.t()).squeeze(0)  # Q ≈ P R^T + t
    return R, t

def apply_rigid_transform(x_coords, R, t):
    """
    x_coords: [N, C, 3]
    R: [3, 3], t: [3]
    """
    return (x_coords @ R.t()) + t.view(1, 1, 3)

def build_graph_from_pdb(pdb_path, antigen_chain_id, antibody_chain_id, k_ctx=16, att_cutoff=12.0, device="cpu"):
    antigen_res = parse_chain_residues(pdb_path, antigen_chain_id)
    antibody_res = parse_chain_residues(pdb_path, antibody_chain_id)

    h, x, antigen_mask, antibody_mask, residues = build_node_tensors(antigen_res, antibody_res)
    ctx_edges, att_edges = build_edges(h, x, antigen_mask, antibody_mask, k_ctx=k_ctx, att_cutoff=att_cutoff)

    return (
        h.to(device),
        x.to(device),
        [ctx_edges[0].to(device), ctx_edges[1].to(device)],
        [att_edges[0].to(device), att_edges[1].to(device)],
        antigen_mask.to(device),
        antibody_mask.to(device),
        residues,
    )

if __name__ == "__main__":
    from mc_egnn import MCAttEGNN

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pdb_id = '1g9m'
    sabdab_path = "/project/liulab/gkim/antigen_prediction/data/sabdab_all_fv_summary_2.tsv"
    sabdab = pd.read_table(sabdab_path)

    antibody_Hchain_id = list(sabdab[sabdab.pdb == pdb_id].Hchain)[0]
    antigen_chain_id = list(sabdab[sabdab.pdb ==pdb_id].antigen_chain)[0]

    print(f"heavy chain: {antibody_Hchain_id} | antigen chain: {antigen_chain_id}")
    

    pdb_path = f"/project/liulab/gkim/antigen_prediction/data/renumbered_sabdab_pdb_files/pdb_files/{pdb_id}.pdb"

    h, x, ctx_edges, att_edges, antigen_mask, antibody_mask, residues = build_graph_from_pdb(
        pdb_path,
        antigen_chain_id,
        antibody_Hchain_id,
        k_ctx=16,
        att_cutoff=12.0,
        device=device,
    )

    # model
    gnn = MCAttEGNN(
        in_node_nf=h.shape[1],
        hidden_nf=128,
        out_node_nf=128,
        n_channel=x.shape[1],
        in_edge_nf=0,  # keep 0 unless you add ctx_edge_attr
        n_layers=4,
        residual=True,
        dropout=0.1,
        dense=False,
    ).to(device)

    # Forward
    h_out, x_out, atts = gnn(h, x, ctx_edges, att_edges, ctx_edge_attr=None, att_edge_attr=None, return_attention=True)

    # Print Per-Residue Embeddings (Example: First 5)
    for i in range(min(5, h_out.shape[0])):
        res = residues[i]
        chain = res.get_parent().id
        resid = res.id[1]
        print(f"Node {i:04d} Chain {chain} ResId {resid} Embedding[:8] = {h_out[i, :8].detach().cpu().numpy()}")

    # Compute Rigid Pose Update For Antibody Using CA Channel (Channel 1)
    ab_idx = torch.where(antibody_mask)[0]
    P = x[ab_idx, 1, :]      # Original Antibody CA [M, 3]
    Q = x_out[ab_idx, 1, :]  # Predicted Antibody CA [M, 3]

    dR, dt = kabsch_rigid_transform(P, Q)
    print("Delta Rotation (3x3):\n", dR.detach().cpu().numpy())
    print("Delta Translation (3,):\n", dt.detach().cpu().numpy())

    # Apply Pose Update To Antibody Coordinates Only (One Docking Step)
    x_new = x.clone()
    x_new[ab_idx] = apply_rigid_transform(x_new[ab_idx], dR, dt)
