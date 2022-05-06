from tqdm.auto import tqdm
from mltle.data import maps


class MapSeq:
    def __init__(self,
                 drug_mode='smiles_1',
                 protein_mode='protein_3',
                 max_drug_len=200):


        self.drug_dict = getattr(maps, drug_mode)
        self.protein_dict = getattr(maps, protein_mode)

        self.drug_step = int(drug_mode.split('_')[-1])
        self.protein_step = int(protein_mode.split('_')[-1])

        self.max_drug_len = 200



    def create_maps(self, drug_seqs, protein_seqs):
        map_drug = {}
        for drug in tqdm(drug_seqs):
            drug_len = min(len(drug), self.max_drug_len)
            drug_vec = []

            for i in range(drug_len - self.drug_step):
                v = self.drug_dict.get(drug[i:i + self.drug_step], 0)
                drug_vec.append(v)

            map_drug[drug] = drug_vec

        map_protein = {}
        for protein in tqdm(protein_seqs):
            protein_len = len(protein)
            protein_vec = []

            for i in range(protein_len - self.protein_step):
                v = self.protein_dict.get(protein[i:i + self.protein_step].upper(), 0)
                protein_vec.append(v)

            map_protein[protein] = protein_vec

        return map_drug, map_protein