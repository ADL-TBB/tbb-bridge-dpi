import numpy as np
import gc
import os,sys,logging,random,torch
from deepchem.feat import graph_features
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
from copy import deepcopy
import pickle as pkl
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, 'smiles_transformer')
from smiles_transformer.build_vocab import WordVocab

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BaseLoader:
    def __init__(self, dataPath, device='cuda', model_name='DTI_Bridge', pSeqMaxLen=1024, dSeqMaxLen=128, seed=42, save_d_names = True):
        np.random.seed(seed)
        self.device = device
        self.model_name = model_name
        # Initialize the parameters as attributes
        self.dataPath = dataPath
        self.pSeqMaxLen = pSeqMaxLen
        self.dSeqMaxLen = dSeqMaxLen
        self.save_d_names = save_d_names
        self._create_features()
        

    def _create_features(self):
        # These data will be filled with append values in the methods called
        # down below.
        self.p2id, self.id2p = {}, []
        self.d2id, self.id2d = {}, []
        self.pSeqData = []
        self.dMolData, self.dSeqData, self.dFeaData, self.dFinData, self.dSmilesData = [], [], [], [], []
        self.pNameData, self.dNameData = {}, {}

        # Import the data as {'train'/'valid'/'test': [drug, protein, label]}
        if self.save_d_names: #If the self.save_d_names is flagged (only for bindingDB), the drug ids will as well be created 
            self.drug_names = []
            self.data, self.data_names = self.load_data(self.dataPath)
        
        self.data = self.load_data(self.dataPath)

        # Protein and drug data and their labels
        self.eSeqData, self.edgeLab = {}, {}
        self.initialize_ID_data(self.data)

        # Check how many samples you have in training, test and validation sets
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(
            self.eSeqData['train']), len(self.eSeqData['valid']), len(self.eSeqData['test'])

        # Get the boolean vector of seen and unseen proteins
        self.pSeen = self.get_seen_proteins()

        self.protein_feats = ["aminoSeq", "aminoCtr", "SeqLen", "seenbool", "pEmbeddings", "pOnehot"]
        self.drug_feats = ["atomFea", "atomFin", "atomSeq", "dSeqLen", "ST_fingerprint"]

        if self.model_name not in ['p_Embedding_Bridge', 'p_Emb_ST_Bridge', 'p_Embedding_Seq_Bridge']:
            # create features for baseline model
            # Initialize and assign each amino acid to a specific numerical id
            self.am2id, self.id2am = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]
            self.get_aminoacid_id()

            # # Initialize and assign each atom to a specific numerical id
            # self.at2id, self.id2at = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]
            # self.get_atom_id()

            print("Tokenizing proteins and drugs...")
            # Tokenize the proteins
            self.pSeqTokenized, self.pSeqLen = self.tokenize_proteins()
            self.pSeqTokenized = np.array(self.pSeqTokenized, dtype=np.int32)

            print("Creating other features...")

        if self.model_name not in ['p_Emb_ST_Bridge', 'p_Embedding_Bridge']:
            # Initialize the protein and drug kmer features
            self.pContFeat = self.get_protein_kmer_features()

        if self.model_name not in ['ST_Bridge']:
            self.dFinprFeat = np.array(self.dFinData, dtype=np.float32)

        if self.model_name not in ['p_Embedding_Bridge', 'ST_Bridge', 'p_Emb_ST_Bridge']:
            self.dGraphFeat = np.array(
                [i[:self.dSeqMaxLen] + [[0] * 75] * (self.dSeqMaxLen - len(i)) for i in self.dFeaData], dtype=np.int8)

        if self.model_name == 'DTI_Bridge':
            self.batch_dict = {
                "aminoSeq": torch.tensor(self.pSeqTokenized, dtype=torch.long),
                "atomFea": torch.tensor(self.dGraphFeat, dtype=torch.float32),
                "aminoCtr": torch.tensor(self.pContFeat, dtype=torch.float32),
                "atomFin": torch.tensor(self.dFinprFeat, dtype=torch.float32),
                "seenbool": torch.tensor(self.pSeen, dtype=torch.bool),
            }

            print("done\n")

        elif self.model_name == 'p_Embedding_Bridge':
            # create features for our model\#Create ELMO protein embeddings
            self.id2emb = torch.stack(self.load_pembeddings())

            self.batch_dict = {
                "atomFin": torch.tensor(self.dFinprFeat, dtype=torch.float32),
                "seenbool": torch.tensor(self.pSeen, dtype=torch.bool),
                "pEmbeddings": torch.tensor(self.id2emb, dtype=torch.float32),
            }

            print("done\n")

        elif self.model_name == 'p_Embedding_Seq_Bridge': # P embedding model with kmers and dseq
            # create features for our model\#Create ELMO protein embeddings
            self.id2emb = torch.stack(self.load_pembeddings())

            self.batch_dict = {
                "atomFin": torch.tensor(self.dFinprFeat, dtype=torch.float32),
                "atomFea": torch.tensor(self.dGraphFeat, dtype=torch.float32),
                "aminoCtr": torch.tensor(self.pContFeat, dtype=torch.float32),
                "pEmbeddings": torch.tensor(self.id2emb, dtype=torch.float32),
                "seenbool": torch.tensor(self.pSeen, dtype=torch.bool),
            }

            print("done\n")

        elif self.model_name == "ST_Bridge":
            self.vocab = WordVocab.load_vocab('data/smiles_trfm_model/vocab.pkl')
            self.ST_fingerprint = self.get_ST_features()

            self.batch_dict = {
                "aminoSeq": torch.tensor(self.pSeqTokenized, dtype=torch.long),
                "aminoCtr": torch.tensor(self.pContFeat, dtype=torch.float32),
                "ST_fingerprint": torch.tensor(self.ST_fingerprint, dtype=torch.float32),
                "seenbool": torch.tensor(self.pSeen, dtype=torch.bool),

            }

            print("done\n")

        elif self.model_name == "p_Emb_ST_Bridge":
            # from smiles_transformer.build_vocab import WordVocab

            self.vocab = WordVocab.load_vocab('data/smiles_trfm_model/vocab.pkl')
            self.ST_fingerprint = self.get_ST_features()
            self.id2emb = torch.stack(self.load_pembeddings())

            self.batch_dict = {
                "seenbool": torch.tensor(self.pSeen, dtype=torch.bool),
                "pEmbeddings": torch.tensor(self.id2emb, dtype=torch.float32),
                "ST_fingerprint": torch.tensor(self.ST_fingerprint, dtype=torch.float32)
            }

            print("done\n")

    def load_data(self):
        '''Returns data in format [drug, protein, label]'''
        pass

    def create_proteinID(self, protein, pCnt):
        '''
        Checks if protein is new unique one, 
        adds to pSeqData and assigns unique ID if this is the case
        '''
        if protein not in self.p2id:
            self.pSeqData.append(protein)
            self.p2id[protein] = pCnt
            self.id2p.append(protein)
            return True
        else:
            return False

    def create_drugID(self, drug, dCnt):
        '''
        Checks if drug is new unique one, 
        assigns unique ID if this is the case
        '''
        if drug not in self.d2id:
            self.d2id[drug] = dCnt
            self.id2d.append(drug)
            return True
        else:
            return False

    def get_drug_features(self, drug):
        '''
        For a unique drug (input), store:
        smiles, molecule, atomsequence, features and Morgan Fingerprint
        '''
        self.dSmilesData.append(drug)
        mol = Chem.MolFromSmiles(drug)
        self.dMolData.append(mol)
        self.dSeqData.append([a.GetSymbol() for a in mol.GetAtoms()])
        self.dFeaData.append([graph_features.atom_features(a) for a in mol.GetAtoms()])
        tmp = np.ones((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024), tmp)
        self.dFinData.append(tmp)
        return

    def initialize_ID_data(self, data):
        '''
        Assign IDs and create drugfeatures
        return data in format [protein_ID, drug_ID, label]
        '''
        print("\nCreating IDs...")
        pCnt, dCnt = 0, 0
        for sub in ['train', 'valid', 'test']:
            idx = 0
            self.pNameData[sub], self.dNameData[sub] = [], []
            id_data = []
            for drug, protein, label in data[sub]:
                if (self.create_proteinID(protein, pCnt)):
                    pCnt += 1
                if (self.create_drugID(drug, dCnt)):
                    self.get_drug_features(drug)
                    dCnt += 1
                    if self.save_d_names:
                        self.drug_names.append(self.data_names[sub][idx][0])
                idx += 1
                id_data.append([self.p2id[protein], self.d2id[drug], label])
            self.eSeqData[sub] = np.array(id_data, dtype=np.int32)

        del data
        del self.data
        gc.collect()

    def get_aminoacid_id(self):
        '''
        The function iterates through all available protein sequences and assigns each different
        amino acid to a numerical ID
        '''
        amCnt = 2  # start from 2 because the storing point is already filled with EOF and UNK
        for pSeq in self.pSeqData:
            for am in pSeq:
                if am not in self.am2id:
                    self.am2id[am] = amCnt
                    self.id2am.append(am)
                    amCnt += 1
        return amCnt

    def get_atom_id(self):
        '''
        The function iterates through all available drug sequences and assigns each different
        atom to a numerical ID
        '''
        atCnt = 2
        for dSeq in self.dSeqData:
            for at in dSeq:
                if at not in self.at2id:
                    self.at2id[at] = atCnt
                    self.id2at.append(at)
                    atCnt += 1
        return atCnt

    def tokenize_proteins(self):
        '''
        Given protein sequences as strings of characters referring to amino acids, the method
        converts all proteins to lists in which the sequence of amino acids is encoded as the 
        set of numerical ids associating to them. The sequences are clipped at a certain maximum
        length or filled with 0's if the maximum length is not reached 
        '''
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in self.pSeqData:
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append(min(len(pSeq), self.pSeqMaxLen))
            pSeqTokenized.append(pSeq[:self.pSeqMaxLen] + [1] * max(self.pSeqMaxLen - len(pSeq), 0))
        return pSeqTokenized, pSeqLen

    def tokenize_drugs(self):
        '''
        Given drug sequences as strings of characters referring to atoms, the method
        converts all proteins to lists in which the sequence of amino acids is encoded as the 
        set of numerical ids associating to them. The sequences are clipped at a certain maximum
        length or filled with 0's if the maximum length is not reached 
        '''
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in self.dSeqData:
            atoms = [self.at2id[i] for i in dSeq]
            dSeqLen.append(min(len(dSeq), self.dSeqMaxLen))
            dSeqTokenized.append(
                atoms[:self.dSeqMaxLen] + [1] * max(self.dSeqMaxLen - len(atoms), 0))
        return dSeqTokenized, dSeqLen

    def load_pretrained_smiles_trfm(self):
        """
        Load the pretrained SMILES Transformer model with pickle
        :return: SMILES transformer model
        """
        from smiles_transformer.pretrain_trfm import TrfmSeq2seq

        trfm = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        trfm.load_state_dict(torch.load('data/smiles_trfm_model/trfm_12_23000.pkl', map_location=self.device))
        return trfm

    def tokenize_smiles(self, smiles):
        """
        Tokenize SMILES as preprocesing for SMILES transfromer fingerprints
        :return: tensor with tokenized SMILES
        """
        pad_index = 0
        unk_index = 1
        eos_index = 2
        sos_index = 3
        seq_len = 220

        x_id, x_seg = [], []
        for sm in smiles:
            sm = sm.split()
            if len(sm) > 218:
                print('SMILES is too long ({:d})'.format(len(sm)))
                sm = sm[:109] + sm[-109:]
            ids = [self.vocab.stoi.get(token, unk_index) for token in sm]
            ids = [sos_index] + ids + [eos_index]
            padding = [pad_index] * (seq_len - len(ids))
            ids.extend(padding)
            x_id.append(ids)

        return torch.tensor(x_id)

    def get_ST_features(self):
        """
        Get Fingerprints from pretrained SMILES Transformer
        :return: SMILES transformer fingerprints
        """
        tokenized = self.tokenize_smiles(self.id2d)
        trfm = self.load_pretrained_smiles_trfm()
        ST_fingerprints = trfm.encode(torch.t(tokenized))

        return ST_fingerprints

    def get_protein_kmer_features(self):
        '''
        Transform proteins into k-mer vectors.
        '''
        ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
        pContFeat = ctr.fit_transform([i for i in self.pSeqData]).toarray().astype('float32')
        k1, k2, k3 = [len(i) == 1 for i in ctr.get_feature_names()], [len(i) == 2 for i in ctr.get_feature_names()], [
            len(i) == 3 for i in ctr.get_feature_names()]

        pContFeat[:, k1] = (pContFeat[:, k1] - pContFeat[:, k1].mean(
            axis=1).reshape(-1, 1)) / (pContFeat[:, k1].std(axis=1).reshape(-1, 1) + 1e-8)
        pContFeat[:, k2] = (pContFeat[:, k2] - pContFeat[:, k2].mean(
            axis=1).reshape(-1, 1)) / (pContFeat[:, k2].std(axis=1).reshape(-1, 1) + 1e-8)
        pContFeat[:, k3] = (pContFeat[:, k3] - pContFeat[:, k3].mean(
            axis=1).reshape(-1, 1)) / (pContFeat[:, k3].std(axis=1).reshape(-1, 1) + 1e-8)
        pContFeat = (pContFeat - pContFeat.mean(axis=0)) / (pContFeat.std(axis=0) + 1e-8)
        return pContFeat

    def get_seen_proteins(self):
        '''
        Create boolean vector indicating for each protein
        whether it's present in both train and test set
        '''
        train, test = self.eSeqData['train'], self.eSeqData['test']
        pSeen = [False] * (len(self.pSeqData))
        if len(test) > 0:  # Only if there is a test set
            for i in range(len(pSeen)):
                if np.any(train[:, 0] == i) and np.any(test[:, 0] == i):
                    pSeen[i] = True
        return np.array(pSeen, dtype=np.bool)


    def one_epoch_batch_data_stream(self, batchSize=32, type='valid', device='gpu'):
        edges = self.eSeqData[type]
        indexes = np.arange(len(edges))
        np.random.shuffle(indexes)
        edges = edges[indexes]
        for i in range((len(edges) + batchSize - 1) // batchSize):
            samples = edges[i * batchSize:(i + 1) * batchSize]
            pTokenizedNames, dTokenizedNames = [i[0] for i in samples], [i[1] for i in samples]
            new_batch = dict()

            for feat in self.batch_dict.keys():
                if feat in self.protein_feats:
                    new_batch[feat] = self.batch_dict[feat][pTokenizedNames].to(device)
                elif feat in self.drug_feats:
                    new_batch[feat] = self.batch_dict[feat][dTokenizedNames].to(device)

            new_batch['res'] = True
            yield new_batch, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)

    def random_batch_data_stream(self, batchSize=32, type='train', device='gpu', shuffle=True):
        edges = [i for i in self.eSeqData[type]]
        while True:
            if shuffle:
                random.shuffle(edges)
            for i in range((len(edges) + batchSize - 1) // batchSize):
                samples = edges[i * batchSize:(i + 1) * batchSize]
                pTokenizedNames, dTokenizedNames = [i[0] for i in samples], [i[1] for i in samples]
                new_batch = dict()

                for feat in self.batch_dict.keys():
                    if feat in self.protein_feats:
                        new_batch[feat] = self.batch_dict[feat][pTokenizedNames].to(device)
                    elif feat in self.drug_feats:
                        new_batch[feat] = self.batch_dict[feat][dTokenizedNames].to(device)

                new_batch['res'] = True
                yield new_batch, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)



class LoadBindingDB(BaseLoader):
    def load_data(self, dataPath):
        '''
        Read file and return data as list of [drug, protein, label]
        '''
        print('\nReading the raw data...\n')
        if self.save_d_names:
            data_ids = {'train': [], 'valid': [], 'test': []} #Only if you want to save drug labels
        data = {'train': [], 'valid': [], 'test': []}
        for folder in ['train', 'dev', 'test']:
            print("\tOpened " + folder)
            path = os.path.join(dataPath, folder)
            proteinID, proteinSequence, aminoacidID, drugID, drugSMILES = self.get_info(path)

            for type in ['edges.pos', 'edges.neg']:
                print("\tReading " + type)
                file = open(os.path.join(path, type), 'r')
                for line in file.readlines():
                    chem, dID, protein, pID = line.strip().split(',')

                    pIndex = proteinID.index(pID)  # Get index of protein ID
                    aminoacids = proteinSequence[pIndex].split()  # Get corresponding sequence of amino acid IDs
                    protein = ''
                    # Transform amino acid IDs to letters
                    for i in range(len(aminoacids)):
                        id = int(aminoacids[i])
                        protein += aminoacidID[id]
                    dIndex = drugID.index(dID)
                    drug = drugSMILES[dIndex]

                    if type == 'edges.neg':
                        label = '0'
                    else:
                        label = '1'
                    if folder != 'dev':
                        data[folder].append(np.array((drug, protein, int(label))))
                        if self.save_d_names:
                            data_ids[folder].append(np.array((dID, pID, int(label))))
                    else:
                        data['valid'].append(np.array((drug, protein, int(label))))
                        if self.save_d_names:
                            data_ids[folder].append(np.array((dID, pID, int(label))))
                file.close()
        if self.save_d_names:
            return data, data_ids
        return data

    def get_info(self, data_path):
        # protein: protein IDs (e.g. A4D1B5)
        # protein.repr: amino acid ID sequence of each protein
        # protein.vocab: the different amino acids
        # chem: drug IDs (e.g. 89659229)
        # chem.repr: SMILES
        # edges.pos: ['chem', drugID, 'protein', proteinID]
        # edges.neg: ^
        files = [os.path.join(data_path, 'protein'),
                 os.path.join(data_path, 'protein.repr'),
                 os.path.join(data_path, 'protein.vocab'),
                 os.path.join(data_path, 'chem'),
                 os.path.join(data_path, 'chem.repr')]
        proteinID = [i.strip() for i in open(files[0], 'r').readlines()]
        proteinSequence = [i.strip() for i in open(files[1], 'r').readlines()]
        aminoacidID = [i.strip() for i in open(files[2], 'r').readlines()]
        drugID = [i.strip() for i in open(files[3], 'r').readlines()]
        drugSMILES = [i.strip() for i in open(files[4], 'r').readlines()]
        return proteinID, proteinSequence, aminoacidID, drugID, drugSMILES   


    def load_pembeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path = os.path.join(data, 'embedding_files', 'prot_embedding_bindingdb.pkl')
        emb_file = open(path, 'rb')
        emb_dict = pkl.load(emb_file)
        emb_file.close()
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb     

class LoadCelegansHuman(BaseLoader):
    def load_data(self, dataPath, valid_size=0.1, test_size=0.1):
        '''
        Read file and return data as list of [drug, protein, label]
        '''
        print('\nReading the raw data...')
        temp = []
        file = open(os.path.join(dataPath, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data = self.create_sets(temp, valid_size, test_size)
        return data

    def create_sets(self, temp, valid_size, test_size):
        np.random.shuffle(temp)
        data = {'train': [], 'valid': [], 'test': []}
        samples = len(temp)
        split1 = int((1 - valid_size - test_size) * samples)
        split2 = int((1 - test_size) * samples)
        data['train'] = temp[:split1]
        data['valid'] = temp[split1:split2]
        data['test'] = temp[split2:]
        del temp
        gc.collect()
        return data
    
    def load_pembeddings(self):
        '''
        Import the ELMO protein embeddings for either human of c.elegans dataset
        '''
        data = 'data'
        if 'human' in str(self.dataPath):
            path = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        else:
            path = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')

        with open(path, 'rb') as emb_file:
            emb_dict = pkl.load(emb_file)

            id2emb = []
            for protein in self.p2id.keys():
                id2emb.append(emb_dict[protein])
        del emb_dict
        gc.collect()
        return id2emb

class LoadChembl(BaseLoader):
    """
    Placeholder class for training of the chembl model
    """

    def load_data(self, data_path, valid_size=0.1, test_size=0.1):
        '''
        Read file and return data as list of [drug, protein, label]
        Takes chembl2smiles and chembl2aaseq dictionaries as input
        Reads interaction data from data_path
        Creates and returns list with smiles, aa-seq, label
        and create train/val/test set
        '''

        data = []
        unavailable_smiles = []

        with open(os.path.join(data_path, "chembl2smiles.pkl"), mode="rb") as f:
            chembl2smiles = pkl.load(f)
        with open(os.path.join(data_path, "chembl2aaseq.pkl"), mode="rb") as f:
            chembl2aaseq = pkl.load(f)

        actinact_path = Path("data/chembl/DEEPScreen_files/chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt")
        f = open(actinact_path, mode="r")

        for line in tqdm(f.readlines()):
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
        data = self.create_sets(data, valid_size, test_size)
        return data

    def create_sets(self, temp, valid_size, test_size):
        np.random.shuffle(temp)
        data = {'train': [], 'valid': [], 'test': []}
        samples = len(temp)
        split1 = int((1 - valid_size - test_size) * samples)
        split2 = int((1 - test_size) * samples)
        data['train'] = temp[:split1]
        data['valid'] = temp[split1:split2]
        data['test'] = temp[split2:]
        return data

    def load_pembeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences (embedding file should be of the format: prot_embedding_{datasetname}.pkl
        '''
        data = 'data'
        path = os.path.join(data, 'embedding_files', f'prot_embedding_{self.dataPath.name}.pkl')
        with open(path, 'rb') as emb_file:
            emb_dict = pkl.load(emb_file)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb



class LoadSarscov2_with_Celegans(BaseLoader):
    def load_data(self, data_path):
        print('\nReading the raw data...')

        data = {'train': [], 'valid': [], 'test': []}

        temp = []
        file = open(os.path.join('data\\celegans', 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['train'] = np.array(temp)

        temp = []
        file = open(os.path.join(data_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

class LoadSarscov2_with_Human(BaseLoader):
    def load_data(self, data_path):
        print('\nReading the raw data...')

        data = {'train': [], 'valid': [], 'test': []}

        temp = []
        file = open(os.path.join('data\\human', 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['train'] = np.array(temp)

        temp = []
        file = open(os.path.join(data_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

class LoadSarscov2_with_BindingDB(BaseLoader):
    def load_data(self, data_path):
        print('\nReading the raw data...')

        data = {'train': [], 'valid': [], 'test': []}

        for folder in ['train', 'dev', 'test']:
            print("\tOpened "+folder)
            path = os.path.join("data\\bindingdb", folder)
            proteinID, proteinSequence, aminoacidID, drugID, drugSMILES = self.get_info(path)

            for type in ['edges.pos', 'edges.neg']:
                print("\tReading "+type)
                file = open(os.path.join(path, type), 'r')
                for line in file.readlines():
                    chem, dID, protein, pID = line.strip().split(',')

                    pIndex = proteinID.index(pID) # Get index of protein ID
                    aminoacids = proteinSequence[pIndex].split() # Get corresponding sequence of amino acid IDs
                    protein = ''
                    # Transform amino acid IDs to letters
                    for i in range(len(aminoacids)):
                        id = int(aminoacids[i])
                        protein += aminoacidID[id]
                    dIndex = drugID.index(dID)
                    drug = drugSMILES[dIndex]

                    if type == 'edges.neg':
                        label = '0'
                    else:
                        label = '1'
                    data['train'].append(np.array((drug, protein, int(label))))
                file.close()
        
        temp = []
        file = open(os.path.join(data_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def get_info(self, data_path):
        # protein: protein IDs (e.g. A4D1B5)
        # protein.repr: amino acid ID sequence of each protein
        # protein.vocab: the different amino acids
        # chem: drug IDs (e.g. 89659229)
        # chem.repr: SMILES
        # edges.pos: ['chem', drugID, 'protein', proteinID]
        # edges.neg: ^
        files = [os.path.join(data_path, 'protein'), 
                os.path.join(data_path, 'protein.repr'),
                os.path.join(data_path, 'protein.vocab'),
                os.path.join(data_path, 'chem'),
                os.path.join(data_path, 'chem.repr')]
        proteinID = [i.strip() for i in open(files[0], 'r').readlines()]
        proteinSequence = [i.strip() for i in open(files[1], 'r').readlines()]
        aminoacidID = [i.strip() for i in open(files[2], 'r').readlines()]
        drugID = [i.strip() for i in open(files[3], 'r').readlines()]
        drugSMILES = [i.strip() for i in open(files[4], 'r').readlines()]
        return proteinID, proteinSequence, aminoacidID, drugID, drugSMILES


class PredictInteractions(BaseLoader):
    """
    Instantly predict without training, (as for now) just for the Pembeddings model
    """
    def __init__(self, dataPath, device):
        self.dataPath = dataPath
        self.device = device
        self.records = self.load_data()
        self.emb_dict = self.open_embeddings()
        super(BaseLoader, self).__init__()
        self._create_features()

    def load_data(self):
        records = []
        file = open(os.path.join(self.dataPath, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split('\t')
            records.append([drug, protein, int(label)])
        file.close()
        return records

    def _create_features(self):
        embs = []
        dFinData = []
        for record in self.records:
            drug, protein, label = record
            tmp = np.ones((1,))
            mol = Chem.MolFromSmiles(drug)
            DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024), tmp)
            dFinData.append(tmp)
            print(type(self.emb_dict))
            embs.append(self.emb_dict[protein])

        self.id2emb = torch.stack(embs)
        self.dFinprFeat = np.array(dFinData, dtype=np.float32)

    def open_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path = os.path.join(data, 'embedding_files', f'prot_embedding_{self.dataPath.name}.pkl')
        emb_file = open(path, 'rb')
        emb_dict = pkl.load(emb_file)
        emb_file.close()
        return emb_dict

    def yield_batch(self):
        device = self.device
        samples = self.records
        yield {
                  "res": True,
                  "atomFin": torch.tensor(self.dFinprFeat, dtype=torch.float32).to(device),
                  "pEmbeddings": torch.tensor(self.id2emb, dtype=torch.float32).to(device),
              }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)

