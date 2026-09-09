import numpy as np
import pandas as pd
import torch
import gpytorch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def active_learning(*args, **kwargs):
    """Thin forwarder to `explainable_al.active_learning.active_learning`.

    This helper allows callers to import `active_learning` from the
    `explainable_al.utils` namespace for backwards compatibility.
    """
    from .active_learning import active_learning as _al
    return _al(*args, **kwargs)


def get_ecfp_fingerprints(smiles_list_or_df):
    """Return ECFP fingerprints for a list of SMILES or a DataFrame with 'SMILES' column."""
    if isinstance(smiles_list_or_df, (list, tuple)):
        return [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 4, nBits=4096)
            for smiles in smiles_list_or_df
        ]
    from .featuriser import smiles_to_ecfp8_df
    return smiles_to_ecfp8_df(smiles_list_or_df, "SMILES")


def get_maccs_keys(smiles_list):
    from rdkit.Chem import MACCSkeys
    return [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]


def get_chemberta_embeddings(npz_file):
    from .featuriser import load_chemberta_embeddings
    return load_chemberta_embeddings(npz_file)


def calculate_metrics(model, likelihood, test_x, test_y):
    from .metrics_plots import calculate_metrics as _calc
    return _calc(model, likelihood, test_x, test_y)
