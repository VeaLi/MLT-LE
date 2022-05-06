from tqdm.auto import tqdm
from mltle.data import maps


class MapSeq:
    """
    MapSeq maps drug/protein strings to integer vectors. 

    All maps are obtained statistically by analyzing human BindingDB data.
    see more in `mltle.data.maps`, also see the descriptions below


    Parameters
    ----------

    drug_mode: Str, default='smiles_1'
        "smiles_1" - map a drug SMILES string to a vector of integers, 
        ngram=1, match every character, example: CCC -> [4,4,4],
        see `mltle.data.maps.smiles_1` for the map

        "smiles_2" - map a drug SMILES string to a vector of integers, 
        ngram=2, match every character, example: CCC -> [2,2],
        see `mltle.data.maps.smiles_2` for the map

        "selfies_1" - map a drug SELFIES string to a vector of integers, 
        ngram=1, match every character, example: CCC -> [3,3,3],
        see `mltle.data.maps.selfies_1` for the map

        "selfies_3" - map a drug SELFIES string to a vector of integers, 
        ngram=3, match every character, example: [C][C] -> [2,2],
        see `mltle.data.maps.selfies_3` for the map

    protein_mode: Str, default='protein_3'
        "protein_1" - map a protein string to a vector of integers, 
        ngram=1, match every 3 characters, example: LLLSSS -> [3, 3, 3, 5, 5, 5],
        see `mltle.data.maps.protein_1` for the map

        "protein_3" - map a protein string to a vector of integers, 
        ngram=3, match every 3 characters, example: LLLSSS -> [1, 3, 13, 2],
        see `mltle.data.maps.protein_3` for the map



    max_drug_len: Int, default=200
        shuffle data or not

    """
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
        """
        This is outer generator.
        Generates one batch
        

        Parameters
        ----------
        drug_seqs: array_like[Str]
            Iterable of drug sequences, 
            the resulting integer vectors will not exceed the maximum length,
            unknown characters will be maped to zero. Check your batch after completion

        protein_seqs: array_like[Str]
            Iterable of protein sequences, they will be automatically converted to uppercase,
            unknown characters will be maped to zero. Check your batch after completion


        Returns
        ----------
            Tuple[Dict, Dict]
            map_drug, map_protein - dictionaries that map input strings to integer vectors

        """
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