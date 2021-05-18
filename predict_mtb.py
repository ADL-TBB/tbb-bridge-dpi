# Initialize connection with rest of lib
from utils import *
from DL_ClassifierModel import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
data = 'data/mtb/protcase_1a.txt'
model_path = 'pEmb_model_958.pkl'
#
# if data=="celegans" or data=="human":
#     data_class = LoadCelegansHuman(dataPath=data_path)
# else: #bindingdb
#     data_class = LoadBindingDB(dataPath=data_path)
with open('mtb_indices.txt', 'r') as infile:
    mtb_rows = infile.read().splitlines(keepends=False)


model = p_Embedding_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024,
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cpu'))

mtb_headers = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'reference']

interactions = []
for root, dirs, files in os.walk("./data/mtb", topdown=True):
    for name in sorted(dirs):
        print(os.path.join(root, name))
        mtb_file = os.path.join(root, name)

        data_class = PredictInteractions(mtb_file, device='cpu')
        model.load(path=model_path, map_location="cpu", dataClass=data_class)
        model.to_eval_mode()
        Ypre,Y = model.calculate_y_prob_by_iterator(data_class.yield_batch())

        # print(Ypre)
        interactions.append(Ypre)

interactions = np.array(interactions).transpose()
df = pd.DataFrame(interactions, columns=mtb_headers, index=mtb_rows)
fig, ax = plt.subplots(figsize=(11, 9))

# sns.set_palette(sns.color_palette("Spectral", as_cmap=True))
sns.heatmap(df, cmap='viridis')

plt.savefig('interactions.png')
plt.show()
#
# model = DTI_Bridge(outSize=128,
#                 cHiddenSizeList=[1024],
#                 fHiddenSizeList=[1024, 256],
#                 fSize=1024, cSize=data_class.pContFeat.shape[1],
#                 gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
#                 hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
#
# model.load(path=model_path, map_location="cuda", dataClass=data_class)
# model.to_eval_mode()
#
# stream = data_class.unshuffled_data_stream(batchSize=12, type='test', device=torch.device('cuda'))
#
# YArr, Y_preArr = [], []
# # results = {}
# i = 1
# while True:
#     try:
#         X, Y = next(stream)
#         # print(X['pSeqLen']) # Check if they're in correct order
#     except:
#         break
#     Y_pre = model.calculate_y_prob(X, mode='predict').cpu().data.numpy()
#     Y_preArr.append(Y_pre)
#     i += 1
# results = np.array(Y_preArr).T # Transpose to get rows=drugs, columns=proteins
#
# drugnames = ["Hydroxychloroquine", "Chloroquine", "Dexamethasone", "Remdesivir", "Nafamostat", "Camostat",
#             "Pepcid", "Arbidol", "Nitazoxanide", "Ivermectin", "Fluvoxamine", "EIDD-2801"]
#
# proteinnames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"]
#
# plt.figure(figsize=(10,10))
# ax = sns.heatmap(results, xticklabels=proteinnames, yticklabels=drugnames)
# plt.show()