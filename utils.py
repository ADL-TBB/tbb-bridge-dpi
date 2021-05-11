import numpy as np
import os,sys,logging,random,torch
from deepchem.feat import graph_features
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from pathlib import Path
import pickle as pkl
#Transform strings into vectors of elements and onehot encode their presence/absence in a certain string
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, 'smiles_transformer')

from smiles_transformer.pretrain_trfm import TrfmSeq2seq
from smiles_transformer.build_vocab import WordVocab

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BaseLoader():
    def __init__(self, dataPath, device='cuda', pSeqMaxLen=1024, dSeqMaxLen=128, kmers=-1, seed=42):
        np.random.seed(seed)
        self.device = device
        # Initialize the parameters as attributes
        self.dataPath = dataPath
        self.pSeqMaxLen = pSeqMaxLen
        self.dSeqMaxLen = dSeqMaxLen

        # These data will be filled with append values in the methods called
        # down below.
        self.p2id, self.id2p = {}, []
        self.d2id, self.id2d = {}, []
        self.pSeqData = []
        self.dMolData, self.dSeqData, self.dFeaData, self.dFinData, self.dSmilesData = [], [], [], [], []
        self.pNameData, self.dNameData = {}, {}

        # Import the data as {'train'/'valid'/'test': [drug, protein, label]}

        self.data = self.load_data(self.dataPath)

        # Protein and drug data and their labels
        self.eSeqData, self.edgeLab = {}, {}
        self.initialize_ID_data(self.data)

        #Create ELMO protein embeddings
        self.id2emb = torch.stack(self.create_embeddings())
        
        #Initialize and assign each amino acid to a specific numerical id
        self.am2id, self.id2am = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]
        self.amNum = self.get_aminoacid_id()

        # Initialize and assign each atom to a specific numerical id
        self.at2id, self.id2at = {"<UNK>": 0, "<EOS>": 1}, ["<UNK>", "<EOS>"]
        self.atNum = self.get_atom_id()

        print("Tokenizing proteins and drugs...")
        # Tokenize the proteins
        self.pSeqTokenized, self.pSeqLen = self.tokenize_proteins()
        self.pSeqLen = np.array(self.pSeqLen, dtype=np.int32)
        self.pSeqTokenized = np.array(self.pSeqTokenized, dtype=np.int32)

        # Tokenize the drugs
        self.dSeqTokenized, self.dSeqLen = self.tokenize_drugs()
        self.dSeqLen = np.array(self.dSeqLen, dtype=np.int32)
        self.dSeqTokenized = np.array(self.dSeqTokenized, dtype=np.int32)

        # Check how many samples you have in training, test and validation sets
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(
            self.eSeqData['train']), len(self.eSeqData['valid']), len(self.eSeqData['test'])

        print("Creating other features...")
        # Initialize the protein and drug kmer features
        self.pContFeat = self.get_protein_kmer_features()

        # Complete the feature graphs for drugs
        # self.dGraphFeat = np.array([i + [[0] * 75] * (self.dSeqMaxLen - len(i)) for i in self.dFeaData], dtype=np.int8)
        # In old utils: for bindingdb they do above and for celegans/human beneath...
        # (Beneath gives an error for human dataset but not for celegans)
        self.dGraphFeat = np.array(
            [i[:self.dSeqMaxLen] + [[0] * 75] * (self.dSeqMaxLen - len(i)) for i in self.dFeaData], dtype=np.int8)
        self.dFinprFeat = np.array(self.dFinData, dtype=np.float32)
        self.dSmilesData = np.array(self.dSmilesData)
        '''
        self.vocab = WordVocab.load_vocab('data/smiles_trfm_model/vocab.pkl')
        self.ST_fingerprint = self.get_ST_features()
        '''
        # Get the boolean vector of seen and unseen proteins
        self.pSeen = self.get_seen_proteins()

        # Get one-hot encoded proteins, i.e. for every protein a 2D array [pSeqMaxLen, n_aminoacids]
        self.pOnehot = self.get_onehot_proteins()
        self.pOneHot_unclipped = self.get_onehot_proteins_unclipped()

        print("Done\n")

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
            self.pNameData[sub], self.dNameData[sub] = [], []
            id_data = []
            for drug, protein, label in data[sub]:
                if (self.create_proteinID(protein, pCnt)):
                    pCnt += 1
                if (self.create_drugID(drug, dCnt)):
                    self.get_drug_features(drug)
                    dCnt += 1
                id_data.append([self.p2id[protein], self.d2id[drug], label])
            self.eSeqData[sub] = np.array(id_data, dtype=np.int32)

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

    def get_onehot_proteins(self):
        '''
        Create one-hot encoded proteins.
        For every protein a 2D array [protein length, amino acids]: for every row/place in the sequence
        a 1 at the index of the amino acid that's present there.
        '''
        n_proteinIDs = len(self.id2p)
        n_aminoIDs = len(self.id2am)
        pOnehot = np.zeros((n_proteinIDs, self.pSeqMaxLen, n_aminoIDs), dtype=np.int8)
        for i in range(n_proteinIDs):
            protein = self.pSeqTokenized[i]
            for j in range(self.pSeqMaxLen):
                aaID = protein[j]
                pOnehot[i, j, aaID] = 1
        return pOnehot

    def get_onehot_proteins_unclipped(self):
        '''
        Create one-hot encoded proteins without defining a maximum length.
        For every protein a 2D array [protein length, amino acids]: for every row/place in the sequence
        a 1 at the index of the amino acid that's present there.
        '''
        n_proteinIDs = len(self.id2p)
        n_aminoIDs = len(self.id2am)
        pList = []
        for i in range(n_proteinIDs):
            protein = self.id2p[i]
            pOneHot = np.zeros((len(protein), len(self.id2am)), dtype=np.int8)
            for j in range(len(protein)):
                aaID = self.am2id[protein[j]]
                pOneHot[j, aaID] = 1
            pList.append(pOneHot)
        return np.array(pList, dtype=np.object)

    def one_epoch_batch_data_stream(self, batchSize=32, type='valid', device=torch.device('cpu')):
        edges = self.eSeqData[type]
        indexes = np.arange(len(edges))
        np.random.shuffle(indexes)
        edges = edges[indexes]
        for i in range((len(edges) + batchSize - 1) // batchSize):
            samples = edges[i * batchSize:(i + 1) * batchSize]
            pTokenizedNames, dTokenizedNames = [i[0] for i in samples], [i[1] for i in samples]

            yield {
                      "res": True,
                      "aminoSeq": torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device),
                      "aminoCtr": torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device),
                      "pSeqLen": torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device),
                      "atomFea": torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device),
                      "atomFin": torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device),
                      "atomSeq": torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device),
                      "dSeqLen": torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device),
                      "seenbool": torch.tensor(self.pSeen[pTokenizedNames], dtype=torch.bool).to(device),
                      "pEmbeddings": torch.tensor(self.id2emb[pTokenizedNames], dtype=torch.float32).to(device),
                      "pOnehot": torch.tensor(self.pOnehot[pTokenizedNames], dtype=torch.int8).to(device),
                      "pOnehot_unclipped": self.pOneHot_unclipped[pTokenizedNames],
                    #   "ST_fingerprint": torch.tensor(self.ST_fingerprint[dTokenizedNames], dtype=torch.float32).to(device),
                      "dSmilesData": self.dSmilesData[dTokenizedNames]
                  }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)

    def random_batch_data_stream(self, batchSize=32, type='train', device=torch.device('cpu')):
        edges = [i for i in self.eSeqData[type]]
        while True:
            random.shuffle(edges)
            for i in range((len(edges) + batchSize - 1) // batchSize):
                samples = edges[i * batchSize:(i + 1) * batchSize]
                pTokenizedNames, dTokenizedNames = [i[0] for i in samples], [i[1] for i in samples]

                yield {
                          "res": True,
                          "aminoSeq": torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(
                              device),
                          "aminoCtr": torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device),
                          "pSeqLen": torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device),
                          "atomFea": torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device),
                          "atomFin": torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device),
                          "atomSeq": torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device),
                          "dSeqLen": torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device),
                          "seenbool": torch.tensor(self.pSeen[pTokenizedNames], dtype=torch.bool).to(device),
                          "pOnehot": torch.tensor(self.pOnehot[pTokenizedNames], dtype=torch.int8).to(device),
                          "pEmbeddings": torch.tensor(self.id2emb[pTokenizedNames], dtype=torch.float32).to(device),
                          "pOnehot_unclipped":self.pOneHot_unclipped[pTokenizedNames],
                        #   "ST_fingerprint": torch.tensor(self.ST_fingerprint[dTokenizedNames], dtype=torch.float32).to(device),
                          "dSmilesData":self.dSmilesData[dTokenizedNames]
                      }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)
 
    def unshuffled_data_stream(self, batchSize=32, type='test', device=torch.device('cpu')):
        edges = self.eSeqData[type]
        for i in range((len(edges) + batchSize - 1) // batchSize):
            samples = edges[i * batchSize:(i + 1) * batchSize]
            pTokenizedNames, dTokenizedNames = [i[0] for i in samples], [i[1] for i in samples]
           
            yield {
                      "res": True,
                      "aminoSeq": torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device),
                      "aminoCtr": torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device),
                      "pSeqLen": torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device),
                      "atomFea": torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device),
                      "atomFin": torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device),
                      "atomSeq": torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device),
                      "dSeqLen": torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device),
                      "seenbool": torch.tensor(self.pSeen[pTokenizedNames], dtype=torch.bool).to(device),
                      "pOnehot": torch.tensor(self.pOnehot[pTokenizedNames], dtype=torch.int8).to(device),
                      "pEmbeddings": torch.tensor(self.id2emb[pTokenizedNames], dtype=torch.float32).to(device),
                      "pOnehot_unclipped": self.pOneHot_unclipped[pTokenizedNames],
                    #   "ST_fingerprint": torch.tensor(self.ST_fingerprint[dTokenizedNames], dtype=torch.float32).to(device),
                      "dSmilesData": self.dSmilesData[dTokenizedNames]
                  }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)


class LoadBindingDB(BaseLoader):
    def load_data(self, dataPath):
        '''
        Read file and return data as list of [drug, protein, label]
        '''
        print('\nReading the raw data...\n')
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
                    else:
                        data['valid'].append(np.array((drug, protein, int(label))))
                file.close()
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        emb_file = open(path, 'rb')
        emb_dict = pkl.load(emb_file)
        emb_file.close()
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb     

class LoadCelegansHuman(BaseLoader):
    def load_data(self, data_path, valid_size=0.1, test_size=0.1):
        '''
        Read file and return data as list of [drug, protein, label]
        '''
        print('\nReading the raw data...')
        temp = []
        file = open(os.path.join(data_path, 'data.txt'), 'r')
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
        return data
    
    def create_embeddings(self):
        '''
        Import the ELMO protein embeddings for either human of c.elegans dataset
        '''
        data = 'data'
        if 'human' in str(self.dataPath):
            path = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        else:
            path = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        emb_file = open(path, 'rb')
        emb_dict = pkl.load(emb_file)
        emb_file.close()
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class LoadChEMBL(BaseLoader):
    def load_data(self, data_path, valid_size=0.1, test_size=0.1):
        '''
        Define in chembl_set which dataset (.pkl file) has to be read
        Read file and return data as list of [drug, protein, label]
        '''
        print(f'\nReading the raw data (from {data_path})...')
        temp = []
        with open(data_path, 'rb') as f:
            temp = pkl.load(f)
        print(f"Length of dataset: {len(temp)}")
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
        return data
    
    def create_embeddings(self):
        '''
        Import the ELMO protein embeddings for either human of c.elegans dataset
        '''
        data = 'data'
        path = os.path.join(data, 'embedding_files','prot_embedding_chembl.pkl')
        emb_file = open(path, 'rb')
        emb_dict = pkl.load(emb_file)
        emb_file.close()
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb


# Classes for SARS-CoV-2 case study
class LoadSarscov2_with_Celegans(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')

        celegans_path = data_path[0]
        sarscov_path = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        temp = []
        file = open(os.path.join(celegans_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = temp[:split]
        data['valid'] = temp[split:]

        temp = []
        file = open(os.path.join(sarscov_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_sarscov.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class LoadSarscov2_with_Human(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')
        
        human_path = data_path[0]
        sarscov_path = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        temp = []
        file = open(os.path.join(human_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = temp[:split]
        data['valid'] = temp[split:]

        temp = []
        file = open(os.path.join(sarscov_path, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            protein, drug, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_sarscov.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class LoadSarscov2_with_BindingDB(BaseLoader):
    def load_data(self, data_path):
        print('\nReading the raw data...')

        bindingdb_path = data_path[0]
        sarscov_path = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        for folder in ['train', 'dev', 'test']:
            print("\tOpened "+folder)
            path = os.path.join(bindingdb_path, folder)
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
                    if folder=='dev':
                        data['valid'].append(np.array((drug, protein, int(label))))
                    else: # Add both training and test set to training data
                        data['train'].append(np.array((drug, protein, int(label))))
                file.close()
        
        temp = []
        file = open(os.path.join(sarscov_path, 'data.txt'), 'r')
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_sarscov.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb


# Classes to load data for cross-dataset experiments
class Load_trainBDB_testCElegans(BaseLoader):
    def load_data(self, data_path):    
        print('\nReading the raw data...')
        path_bindingdb = data_path[0]
        path_celegans = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        for folder in ['train', 'dev', 'test']:
            path = os.path.join(path_bindingdb, folder)
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
                    
                    if folder == 'dev':
                        data['valid'].append(np.array((drug, protein, int(label))))                    
                    else: # train and test in training set
                        data['train'].append(np.array((drug, protein, int(label))))
                file.close()

        # Read CElegans data into test set
        temp = []
        file = open(os.path.join(path_celegans, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def get_info(self, data_path):
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb 

class Load_trainBDB_testHuman(BaseLoader):
    def load_data(self, data_path):    
        print('\nReading the raw data...')
        path_bindingdb = data_path[0]
        path_human = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        for folder in ['train', 'dev', 'test']:
            path = os.path.join(path_bindingdb, folder)
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
                    
                    if folder == 'dev':
                        data['valid'].append(np.array((drug, protein, int(label))))                    
                    else: # train and test in training set
                        data['train'].append(np.array((drug, protein, int(label))))
                file.close()

        # Read Human data into test set
        temp = []
        file = open(os.path.join(path_human, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data
 
    def get_info(self, data_path):
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class Load_trainHuman_testCElegans(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')

        path_human = data_path[0]
        path_celegans = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        # Read Human data into training set
        temp = []
        file = open(os.path.join(path_human, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = np.array(temp[:split])
        data['valid'] = np.array(temp[split:])

        # Read CElegans data into test set
        temp = []
        file = open(os.path.join(path_celegans, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb  

class Load_trainCElegans_testHuman(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')

        path_celegans = data_path[0]
        path_human = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        # Read CElegans data into training set
        temp = []
        file = open(os.path.join(path_celegans, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = temp[:split]
        data['valid'] = temp[split:]

        # Read Human data into test set
        temp = []
        file = open(os.path.join(path_human, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        data['test'] = np.array(temp)

        return data

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class Load_trainHuman_testBDB(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')

        path_human = data_path[0]
        path_bindingdb = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        # Read Human data into training set
        temp = []
        file = open(os.path.join(path_human, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = np.array(temp[:split])
        data['valid'] = np.array(temp[split:])

        # Read BDB data into test set
        for folder in ['test']:
            path = os.path.join(path_bindingdb, folder)
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
                    
                    data['test'].append(np.array((drug, protein, int(label))))
                file.close()
        return data

    def get_info(self, data_path):
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_human.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb

class Load_trainCElegans_testBDB(BaseLoader):
    def load_data(self, data_path, valid_size=0.1):
        print('\nReading the raw data...')

        path_celegans = data_path[0]
        path_bindingdb = data_path[1]

        data = {'train': [], 'valid': [], 'test': []}

        # Read Human data into training set
        temp = []
        file = open(os.path.join(path_celegans, 'data.txt'), 'r')
        for line in file.readlines():
            if line == '':
                break
            drug, protein, label = line.strip().split(' ')
            temp.append(np.array((drug, protein, int(label))))
        file.close()
        # np.random.shuffle(temp)
        samples = len(temp)
        split = int((1 - valid_size) * samples)
        data['train'] = np.array(temp[:split])
        data['valid'] = np.array(temp[split:])

        # Read BDB data into test set
        for folder in ['test']:
            path = os.path.join(path_bindingdb, folder)
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
                    
                    data['test'].append(np.array((drug, protein, int(label))))
                file.close()
        return data

    def get_info(self, data_path):
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

    def create_embeddings(self):
        '''
        For all the proteins of the dataset, obtain the ELMO embeddings
        for the sequences
        '''
        data = 'data'
        path1 = os.path.join(data, 'embedding_files','prot_embedding_celegans.pkl')
        path2 = os.path.join(data, 'embedding_files','prot_embedding_bindingDB.pkl')
        with open(path1, 'rb') as f:
            emb_dict = pkl.load(f)
        with open(path2, 'rb') as f:
            emb_dict2 = pkl.load(f)
        emb_dict.update(emb_dict2)
        id2emb = []
        for protein in self.p2id.keys():
            id2emb.append(emb_dict[protein])
        return id2emb



