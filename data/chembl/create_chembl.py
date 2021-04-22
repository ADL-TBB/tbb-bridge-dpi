import numpy as np
import pandas as pd
import os
from pathlib import Path
import requests as re
from Bio import SeqIO
from io import StringIO
from chembl_webresource_client.new_client import new_client
import pickle

folder = "DEEPScreen_files"

actinact_path = Path(os.path.join(folder, "chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt"))
uniprot_path = Path(os.path.join(folder, "chembl27_uniprot_mapping.txt"))
protein_file = Path(os.path.join(folder, "chembl27_training_target_list.txt"))

def get_smiles(cID):
    '''
    Take ChEMBL ID for drug molecules as input
    Use ChEMBL webresource client to obtain canonical smiles
    '''
    molecule = new_client.molecule
    m = molecule.get(cID)
    # Check whether ChEMBL info exists and return error message if not
    if m is None:
        print("-1 error for",cID)
        return -1
    elif m['molecule_structures'] is None:
        print("-2 error for",cID)
        return -2
    elif m['molecule_structures']['canonical_smiles'] is None:
        print("-3 error for",cID)
        return -3
    else:
        smiles = m['molecule_structures']['canonical_smiles']
        return smiles

def get_aaseq(cID):
    '''
    Take UniprotID as input and obtain amino acid sequence from fastafile of uniprot
    '''
    url = "http://www.uniprot.org/uniprot/"+cID+".fasta"
    response = re.post(url)
    cData = ''.join(response.text)
    Seq = StringIO(cData)
    pSeq = list(SeqIO.parse(Seq,'fasta'))
    return str(pSeq[0].seq)


def create_aa_data(data_path, uniprot_path):
    '''
    Input: training target list (ChEMBL IDs for all proteins in filtered dataset)
    Creates dictionary with aa sequence for each ChEMBL ID
    '''
    chembl2aaseq = {}
    df_protIDmapping = pd.read_csv(uniprot_path, header=None, sep="\t", skiprows=1)

    f = open(data_path, 'r')
    for line in f.readlines():
        protein = line.strip()
        uniprotID = df_protIDmapping[df_protIDmapping[1]==protein][0].values[0]
        aaseq = get_aaseq(uniprotID)
        chembl2aaseq[protein] = aaseq
    f.close()
    return chembl2aaseq

def create_smiles_data(data_path):
    '''
    Input: act_inact file (all examples)
    Creates dictionary with SMILES for each ChEMBL ID in the examples
    '''
    chembl2smiles = {}
    total_drugs = 0
    line_counter = 0
    minone_counter = 0
    mintwo_counter = 0
    minthree_counter = 0
    f = open(data_path, 'r')
    for line in f.readlines():
        line_counter+=1
        
        # Test runs for first 20 lines
        if line_counter > 20:
            break

        print("Currently at line:",line_counter)
        if line_counter%10==0:
            print("Came across {} -1s, {} -2s and {} -3s".format(minone_counter, mintwo_counter, minthree_counter))
        line_split = line.strip().split('\t')
        drugs = line_split[1].strip().split(',')

        total_drugs += len(drugs)

        for drug in drugs:
            if drug not in chembl2smiles:
                smiles = get_smiles(drug)
                if smiles == -1:
                    minone_counter += 1
                elif smiles == -2:
                    mintwo_counter += 1
                elif smiles == -3:
                    minthree_counter += 1
                else:
                    chembl2smiles[drug] = smiles
    f.close()
    print("\nTotal seen drugs/examples:",total_drugs)
    print("{} drugs were unavailable".format(minone_counter+mintwo_counter+minthree_counter))
    return chembl2smiles

# Should probably be moved to utils.py later
def read_data(data_path, chembl2smiles, chembl2aaseq):
    '''
    Takes chembl2smiles and chembl2aaseq dictionaries as input
    Reads interaction data from data_path
    Creates and returns list with smiles, aa-seq, label
    '''
    data = []
    unavailable_smiles = []
    line_counter = 0

    f = open(data_path, 'r')
    for line in f.readlines():

        line_counter += 1
        # Test run for first 20 lines
        if line_counter > 20:
            break

        # To make sure only examples for which aa-seq and SMILES are available are saved
        save = True

        line_split = line.strip().split('\t')
        protein_info = line_split[0].split("_")
        protein = protein_info[0]
        active = True if protein_info[1]=="act" else False
        drugs = line_split[1].strip().split(',')

        if protein not in chembl2aaseq:
            print("Amino acid sequence not available for",protein)
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

'''
print("\nCreate chembl2smiles data")
chembl2smiles = create_smiles_data(actinact_path)
print("Length of dict/number of unique drugs:",len(chembl2smiles))
'''

print("\nCreate chembl2aaseq data")
# chembl2aaseq = create_aa_data(protein_file, uniprot_path)
# pickle.dump(chembl2aaseq, open("chembl2aaseq.pkl", "wb"))
chembl2aaseq = pickle.load(open("chembl2aaseq.pkl", "rb"))
print("Length of dict/number of unique proteins:",len(chembl2aaseq))

'''
print("\nCreate training data [smiles, aa-seq, label]")
data = read_data(actinact_path, chembl2smiles, chembl2aaseq)
print("Number of examples:", len(data))
print("First example:\n",data[0])
'''
