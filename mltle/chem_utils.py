from rdkit import Chem
import numpy as np


def to_non_isomeric_canonical(smiles):
    """
    Removes isomeric features and makes RDKit canonical.

    Parameters
    ----------
    smiles: Str
        SMILES string

    Returns
    ----------
    smiles_canonical: Str
        Non-isomeric RDkit canonical SMILES string

    Example:
    ----------
    pubchem_torin1 = "CCC(=O)N1CCN(CC1)C2=C(C=C(C=C2)N3C(=O)C=CC4=CN=C5C=CC(=CC5=C43)C6=CC7=CC=CC=C7N=C6)C(F)(F)F"
    to_non_isomeric_canonical(pubchem_torin1)
    >>CCC(=O)N1CCN(c2ccc(-n3c(=O)ccc4cnc5ccc(-c6cnc7ccccc7c6)cc5c43)cc2C(F)(F)F)CC1
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles_canonical = Chem.MolToSmiles(
            mol, isomericSmiles=False, canonical=True)
        return smiles_canonical
    except Exception as e:
        # if smiles is invalid return NaN
        # do not raise error - suitable for batch processing
        print(f"Input string : {smiles} - raises ERROR : {e}")
        return np.nan
