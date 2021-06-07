from typing import Dict, List

import numpy as np
import pandas as pd
import os
from pathlib import Path
import requests as re
from Bio import SeqIO
from io import StringIO
from chembl_webresource_client.new_client import new_client
import pickle as pkl
import logging
from tqdm import tqdm
from itertools import islice

# Logger Setup
log_file = 'chembl_dataset.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, filename=log_file, level=logging.DEBUG)
logger = logging.getLogger()

folder = "DEEPScreen_files"

# Command used to preprocess the split actinact file:
# cat chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt | cut -f 2- -d ',' | tr ',' '\n' > chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2_split.txt
actinact_path = Path(
    os.path.join(folder, "chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2_split.txt"))
uniprot_path = Path(os.path.join(folder, "chembl27_uniprot_mapping.txt"))
protein_file = Path(os.path.join(folder, "chembl27_training_target_list.txt"))


def get_smiles(chembl_id: str):
    """
    Take ChEMBL ID for drug molecules as input
    Use ChEMBL webresource client to obtain canonical smiles
    """
    molecule = new_client.molecule
    m: Dict = molecule.get(chembl_id)

    smiles: str = m.get('molecule_structures', None).get('canonical_smiles', None)

    if smiles is None:
        logger.debug(f"No canonical SMILES available for molecule with ChEMBL ID: {chembl_id}")
    return smiles


def get_aaseq(chembl_id):
    """
    Take UniprotID as input and obtain amino acid sequence from fastafile of uniprot
    """
    url = f"http://www.uniprot.org/uniprot/{chembl_id}.fasta"

    response = re.post(url)
    c_data = ''.join(response.text)
    seq = StringIO(c_data)
    p_seq = list(SeqIO.parse(seq, 'fasta'))
    return str(p_seq[0].seq)


def create_aa_data(data_path, uniprot_path):
    """
    Input: training target list (ChEMBL IDs for all proteins in filtered dataset)
    Creates dictionary with aa sequence for each ChEMBL ID
    """
    chembl2aaseq = {}
    df_prot_id_mapping = pd.read_csv(uniprot_path, header=None, sep="\t", skiprows=1)

    with open(file=data_path, mode='r') as f:
        for line in f.readlines():
            protein = line.strip()
            uniprot_id = df_prot_id_mapping[df_prot_id_mapping[1] == protein][0].values[0]
            aaseq = get_aaseq(uniprot_id)
            chembl2aaseq[protein] = aaseq
    return chembl2aaseq


def batch_create_smiles_data(data_path: Path, start_index=0, chunk_size=50, save_iterations=5000) -> Dict:
    """
    Input: act_inact file (all examples)
    Creates dictionary with SMILES for each ChEMBL ID in the examples using chunks of IDs
    for more efficient processing
    """
    chembl2smiles = dict()

    if not Path.exists(data_path):
        logger.error(f"File '{data_path}' does not exist, please check if you entered the right path.")
        raise FileNotFoundError(f"file {data_path} does not exist.")

    with open(file=data_path, mode='r') as file:
        print(f"Reading in data file @ '{data_path}'...")
        for line in tqdm(file.read().splitlines()):
            chembl2smiles[line] = None

    keys = list(chembl2smiles.keys())
    chembl2smiles = dict()
    print("Mapping ChEMBL ID <--> Canonical SMILES...")
    for i in range(start_index, len(keys), chunk_size):
        if i > start_index and i % save_iterations == 0:
            filename = f"chembl2smiles_{str(i - save_iterations)}-{str(i)}.pkl"
            with open(filename, mode='wb') as f:
                pkl.dump(chembl2smiles, f)
            chembl2smiles = dict()
            print(f"Saved partial mappings to \"{filename}\" and cleared dictionary")

        activities = new_client.activity.filter(molecule_chembl_id__in=keys[i:i + chunk_size]).only(
            ['molecule_chembl_id', 'canonical_smiles'])

        for act in tqdm(activities):
            chembl2smiles[act['molecule_chembl_id']] = act['canonical_smiles']

    return chembl2smiles


def create_smiles_data(data_path):
    """
    Input: act_inact file (all examples)
    Creates dictionary with SMILES for each ChEMBL ID in the examples
    """
    chembl2smiles = dict()
    unavailable = 0

    with open(file=data_path, mode='r') as f:
        for line in tqdm(f.readlines()):
            drug = line

            if drug not in chembl2smiles:
                smiles = get_smiles(drug)
                if smiles is None:
                    unavailable += 1
                chembl2smiles[drug] = smiles

    logger.debug(f"{unavailable} drugs were unavailable")
    return chembl2smiles


# Should probably be moved to utils.py later
def read_data(data_path, chembl2smiles, chembl2aaseq):
    """
    Takes chembl2smiles and chembl2aaseq dictionaries as input
    Reads interaction data from data_path
    Creates and returns list with smiles, aa-seq, label
    """
    data = []
    unavailable_smiles = []
    # line_counter = 0

    f = open(data_path, 'r')
    for line in f.readlines():

        # line_counter += 1
        # # Test run for first 20 lines
        # if line_counter > 20:
        #     break

        # To make sure only examples for which aa-seq and SMILES are available are saved
        save = True

        line_split = line.strip().split('\t')
        protein_info = line_split[0].split("_")
        protein = protein_info[0]
        active = True if protein_info[1] == "act" else False
        drugs = line_split[1].strip().split(',')

        if protein not in chembl2aaseq:
            print("Amino acid sequence not available for", protein)
            save = False
        else:
            protein_seq = chembl2aaseq[protein]

        for drug in drugs:
            if drug not in chembl2smiles:
                unavailable_smiles.append(drug)
            else:
                smiles = chembl2smiles[drug]
                # Add all smiles, protein_seq, label to list
                if save:
                    data.append(np.array((smiles, protein_seq, int(active))))
    f.close()
    print("{} drugs were not in chembl2smiles:".format(len(unavailable_smiles)))
    print(unavailable_smiles)
    return data


def check_split_file(split_file_path: Path):
    """
    Helper function that confirms if the preprocessed split file exists and creates it if it does not.
    """
    if not Path.exists(split_file_path):
        import subprocess

        try:
            rc = subprocess.call("./split_actinact.sh", shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Couldn't run script!, Error: {e}")


if __name__ == "__main__":

    with open("chembl2smiles.pkl", "rb") as f:
        chembl2smiles = pkl.load(f)
    with open("chembl2aaseq.pkl", "rb") as f:
        chembl2aaseq = pkl.load(f)

    data = read_data(
        data_path="chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt",
        chembl2smiles=chembl2smiles,
        chembl2aaseq=chembl2aaseq)

    with open("chembl_dataset.pkl", "wb") as f:
        pkl.dump(data, f)

    # check_split_file(actinact_path)  # Make the split file if it doesn't exist yet.
    # chembl2smiles = batch_create_smiles_data(actinact_path, start_index=0, chunk_size=200, save_iterations=5000)
