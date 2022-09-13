from rdkit import Chem
import networkx as nx
from rdkit.Chem import MolFromSmiles
import numpy as np

# it needs to be more compact
# also need more general class


class G24:
    """
    BindingDB possible features set
    """

    def __init__(self):
        self.ATOMIC_SYMBOL = ['C', 'N', 'O', 'F', 'S',
                              'Cl', 'Br', 'P', 'I', 'B', 'Na', 'Se', 'Si']
        self.DEGREE_OF_ATOM = [2, 3, 1, 4, 0, 6]
        self.TOTAL_HYDROGRENS = [0, 1, 2, 3]
        self.IMPLICIT_HYDROGRENS = [0, 1, 2, 3]
        self.AROMTICITY = [False]

    def get_atom_features(self, atom):
        symbol = atom.GetSymbol()
        degree = atom.GetDegree()
        total_hydrogens = atom.GetTotalNumHs()
        aromaticity = atom.GetIsAromatic()

        features = [int(symbol == s) for s in self.ATOMIC_SYMBOL]
        features += [int(degree == d) for d in self.DEGREE_OF_ATOM]
        features += [int(total_hydrogens == t) for t in self.TOTAL_HYDROGRENS]
        features += [int(aromaticity == a) for a in self.AROMTICITY]

        features = np.array(features)

        return features

    def normalize_adjacency(self, adj, normalization_type='kipf'):
        """
        Kipf and laplacian normalization
        """
        if normalization_type == 'kipf':
            d = np.sum(adj, axis=1)  # degree vector
            d = 1 / np.sqrt(d)  # Invert square root degree
            D = np.diag(d)  # square root inverse degree matrix
            adj_norm = D @ adj @ D

        elif normalization_type == 'laplacian':
            adj_norm = nx.normalized_laplacian_matrix(G).toarray(adj)

        return adj_norm

    def get_graph_features(self,
                           smiles, max_drug_len=100,
                           num_features=24,
                           normalize=False,
                           normalization_type='kipf'):

        mol = Chem.MolFromSmiles(smiles)
        nodes = np.zeros(shape=(max_drug_len, num_features))

        for k, atom in enumerate(mol.GetAtoms()):
            if k == max_drug_len:
                break
            feature = self.get_atom_features(atom)
            nodes[k, :] = (feature / sum(feature))

        adj_empty = np.zeros((max_drug_len, max_drug_len))
        adj = Chem.GetAdjacencyMatrix(mol)
        np.fill_diagonal(adj, 1)  # with added self-connections

        if normalize:
            adj = self.normalize_adjacency(
                adj, normalization_type=normalization_type)

        adj_size = min(adj.shape[0], max_drug_len)
        adj_empty[:adj_size, :adj_size] = adj[:max_drug_len, :max_drug_len]
        adj = adj_empty

        nodes = nodes.astype('float32')
        adj = adj.astype('float32')

        return nodes, adj


class G78:
    """
    GraphDTA features set
    """

    def __init__(self):
        self.ATOMIC_SYMBOL = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                              'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', '*']
        self.DEGREE_OF_ATOM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.TOTAL_HYDROGRENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.IMPLICIT_HYDROGRENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.AROMTICITY = [False]

    def get_atom_features(self, atom):
        symbol = atom.GetSymbol()
        degree = atom.GetDegree()
        total_hydrogens = atom.GetTotalNumHs()
        implicit_hydrogens = atom.GetImplicitValence()
        aromaticity = atom.GetIsAromatic()

        features = [int(symbol == s) for s in self.ATOMIC_SYMBOL]
        features += [int(degree == d) for d in self.DEGREE_OF_ATOM]
        features += [int(total_hydrogens == t) for t in self.TOTAL_HYDROGRENS]
        features += [int(implicit_hydrogens == i)
                     for i in self.IMPLICIT_HYDROGRENS]
        features += [int(aromaticity == a) for a in self.AROMTICITY]

        features = np.array(features)

        return features

    def normalize_adjacency(self, adj, normalization_type='kipf'):
        """
        Kipf and laplacian normalization
        """
        if normalization_type == 'kipf':
            d = np.sum(adj, axis=1)  # degree vector
            d = 1 / np.sqrt(d)  # Invert square root degree
            D = np.diag(d)  # square root inverse degree matrix
            adj_norm = D @ adj @ D

        elif normalization_type == 'laplacian':
            adj_norm = nx.normalized_laplacian_matrix(G).toarray(adj)

        return adj_norm

    def get_graph_features(self,
                           smiles, max_drug_len=100,
                           num_features=78,
                           normalize=False,
                           normalization_type='kipf'):

        mol = Chem.MolFromSmiles(smiles)
        nodes = np.zeros(shape=(max_drug_len, num_features))

        for k, atom in enumerate(mol.GetAtoms()):
            if k == max_drug_len:
                break
            feature = self.get_atom_features(atom)
            nodes[k, :] = (feature / sum(feature))

        adj_empty = np.zeros((max_drug_len, max_drug_len))
        adj = Chem.GetAdjacencyMatrix(mol)
        np.fill_diagonal(adj, 1)  # with added self-connections

        if normalize:
            adj = self.normalize_adjacency(
                adj, normalization_type=normalization_type)

        adj_size = min(adj.shape[0], max_drug_len)
        adj_empty[:adj_size, :adj_size] = adj[:max_drug_len, :max_drug_len]
        adj = adj_empty

        nodes = nodes.astype('float32')
        adj = adj.astype('float32')

        return nodes, adj


GRAPHS = {'g24': G24(), 'g78': G78()}
