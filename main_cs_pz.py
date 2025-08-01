#!/usr/bin/env python3
"""Train a CCS Counterfactual Network on News or MIMIC data.

This script reproduces the original experimental setup while improving
code organisation, readability and PEP‑8 compliance. **All default values,
random seeds, and model hyper‑parameters are kept identical to the
behaviour in the initial prototype.**

Example
-------
$ python ccs_training.py --setting news --beta 0.001 --gamma 0.1

Notes
-----
* External helpers (`get_true_y_news`, `get_true_y_mimic`, `CS`,
  `compute_mise`, `compute_pe_2`, etc.) are expected to be importable from
  the project package structure.
* The script stores a trained model under `models/ccs_counterfactual_net.pth`
  at the final epoch.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Import required for 3‑D plots
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

# Project‑local modules -------------------------------------------------------
from CCS_divergence import CS
from src.networks import CCS_Counterfactual_Net
from src.utils import (
    compute_mise,
    compute_pe_2,
    get_true_y_mimic,
    get_true_y_news,
)

# ---------------------------------------------------------------------------
# Global configuration (unchanged seeds & warnings)
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibility
SEED = 0  # Do NOT modify
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Set hyper‑parameters through CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train CCS Counterfactual Net with fixed defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--beta", type=float, default=0.001, help="Value of β")
    parser.add_argument("--gamma", type=float, default=0.1, help="Value of γ")
    parser.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include attention mechanism",
    )
    parser.add_argument(
        "--spline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use spline for treatment",
    )
    parser.add_argument("--num_epochs", type=int, default=3000, help="Epochs")
    parser.add_argument(
        "--setting", choices=("news", "mimic"), default="news", help="Dataset",
    )
    


    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_data_paths(setting: str) -> Tuple[Path, Path, Path]:
    """Return train/test/eval paths given *setting* ('news' | 'mimic')."""
    base = Path("data/mimic" if setting == "mimic" else "data/news")
    return base / "train.npy", base / "test.npy", base / "eval_test.npy"


def load_numpy_arrays(setting: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train, test and eval arrays from disk."""
    train_path, test_path, eval_path = get_data_paths(setting)
    train = np.load(train_path)
    test = np.load(test_path)
    eval_test = np.load(eval_path)
    return train, test, eval_test  # type: ignore[return‑value]


def split_arrays(
    data: np.ndarray, setting: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract *t*, *x*, *y* from raw *data* according to *setting*."""
    if setting == "mimic":
        t_arr = data[:, :2]
        x_arr = data[:, 2:-1]
        y_arr = data[:, -1]
    else:  # news
        t_arr = data[:, -2]
        x_arr = data[:, :-2]
        y_arr = data[:, -1]
    return t_arr, x_arr, y_arr.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Model / training helpers
# ---------------------------------------------------------------------------

def build_model(x_dim: int, t_dim: int, use_attention: bool, use_spline: bool) -> nn.Module:
    """Initialise CCS Counterfactual Network with fixed dimensions."""
    hidden_dim = 512  # Do NOT modify
    z_dim = 16 if t_dim > 1 else 32  # Mimic → 16, News → 32
    t_dim_latent = 8
    t_input_dim = t_dim
    return CCS_Counterfactual_Net(
        x_dim=x_dim,
        t_dim_latent=8,
        z_dim=8,
        y_dim=1,
        t_input_dim = t_dim,
        hidden_dim=hidden_dim,
        hidden_dim_t=t_dim_latent,
        attn_dim=64,
        use_attention=use_attention,
        use_spline=use_spline,
    )


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    gamma: float,
    num_epochs: int,
):
    """Training loop with unchanged loss formulation."""
    for epoch in range(1, num_epochs + 1):
        z, t_logits, y_pred = model(x, t)

        # 1) Outcome reconstruction loss
        loss_y = criterion(y_pred, y)

        # 2) Independence regularisation via Contrastive Score (CS)
        # joint_samples = torch.cat((z, t), dim=1)
        # z_perm = z[torch.randperm(z.size(0))]
        # indep_samples = torch.cat((z_perm, t), dim=1)


        loss_cs = CS(z, t_logits)
        #print(loss_cs)

        # 3) Latent space regularisation
        loss_reg = torch.norm(z, dim=0).sum()

        loss = loss_y + beta * loss_cs + gamma * loss_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == num_epochs:
            print(f"Epoch [{epoch}/{num_epochs}] — Loss: {loss.item():.4f}")
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/ccs_counterfactual_net.pth")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Path resolution & data loading ------------------------------------------------
    setting = args.setting
    train_arr, test_arr, _ = load_numpy_arrays(setting)

    t_arr, x_arr, y_arr = split_arrays(train_arr, setting)
    t_test_arr, x_test_arr, y_test_arr = split_arrays(test_arr, setting)

    # Convert to tensors ------------------------------------------------------------
    x = torch.tensor(x_arr, dtype=torch.float32)
    t = torch.tensor(t_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)
    x_test = torch.tensor(x_test_arr, dtype=torch.float32)
    t_test = torch.tensor(t_test_arr, dtype=torch.float32)
    y_test = torch.tensor(y_test_arr, dtype=torch.float32)

    # Ensure dimensionality consistency --------------------------------------------
    if setting == "news":  # Add singleton dim for scalar treatment
        t = t.unsqueeze(1)
        t_test = t_test.unsqueeze(1)

    # Optional ground‑truth check ---------------------------------------------------
    if setting == "news":
        y_true = get_true_y_news(
            t_test.detach().cpu().numpy(),
            x_test.detach().cpu().numpy(),
            noise_sd=0.2,
        )
        df = pd.DataFrame({"y_true": y_true.flatten(), "y_test": y_test.flatten()})
        print(df.head())

    # Model instantiation -----------------------------------------------------------
    model = build_model(
        x_dim=x.shape[1],
        t_dim=t.shape[1],
        use_attention=args.attention,
        use_spline=args.spline,
    )


    # Optimiser / loss --------------------------------------------------------------
    lr = 1e-4 if setting == "mimic" else 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()  # Original code used MSE for both settings

    # Training ----------------------------------------------------------------------
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        x=x,
        t=t,
        y=y,
        beta=args.beta,
        gamma=args.gamma,
        num_epochs=args.num_epochs,
    )

    # Evaluation --------------------------------------------------------------------
    mise = compute_mise(
        model=model.eval(),
        x_test=x_test,
        y_test=y_test,
        get_true_y_fn=get_true_y_mimic if setting == "mimic" else get_true_y_news,
        source="mimic/data" if setting == "mimic" else "data",
        t_dim=t.shape[1],
        x_dim=x.shape[1],
        n_dosage_points=20,
        noise_sd=0.2,
    )
    print("MISE:", mise)

    pe = compute_pe_2(
        model=model.eval(),
        x_test=x_test,
        get_true_y_fn=get_true_y_mimic if setting == "mimic" else get_true_y_news,
        source="mimic/data" if setting == "mimic" else "data",
        t_dim=t.shape[1],
        x_dim=x.shape[1],
        n_dosage_points=20,
        noise_sd=0.2,
    )
    print("PE:", pe)

    print("=" * 38)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
