import numpy as np
import torch
import time
import os
from torch import nn as nn
from nnLayer import *
from metrics import *
from sklearn.model_selection import KFold
from torch.backends import cudnn
from Others import *


class BaseClassifier:
    def __init__(self):
        pass

    def calculate_y_logit(self, X, XLen):
        pass

    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1,
                 optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, kFold=5,
                 isHigherBetter=True, metrics="AUC", report=["ACC", "AUC"],
                 savePath='model', seed=9527, loc=-1):

        skf = KFold(n_splits=kFold, random_state=seed, shuffle=True)

        data = np.concatenate((dataClass.eSeqData['train'], dataClass.eSeqData['valid']))

        results = []
        for i, (train_index, test_index) in enumerate(skf.split(data)):
            dataClass.eSeqData['train'] = data[train_index]
            dataClass.eSeqData['valid'] = data[test_index]
            dataClass.trainSampleNum = len(dataClass.eSeqData['train'])
            dataClass.validSampleNum = len(dataClass.eSeqData['valid'])
            dataClass.pSeen = dataClass.get_seen_proteins()
            print(f'CV_{i + 1}:')
            if loc > 0 and i + 1 != loc:
                print(f'Pass CV_{i + 1}')
                continue
            self.reset_parameters()
            res = self.train(dataClass, trainSize, batchSize, epoch, stopRounds, earlyStop, saveRounds, optimType,
                             preheat, lr1, lr2, momentum, weightDecay,
                             isHigherBetter, metrics, report, f"{savePath}_cv{i + 1}")
            results.append(res)
        Metrictor.table_show(results, report)
        if dataClass.testSampleNum > 0:
            print("(Results on test set)")
        else:
            print("(Results on validation set)")
        
    def cv_train_by_protein(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10,
                            saveRounds=1,
                            optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, kFold=5,
                            isHigherBetter=True, metrics="AUC", report=["ACC", "AUC"],
                            savePath='model', seed=9527, loc=-1):
        kf = KFold(n_splits=kFold, random_state=seed, shuffle=True)
        validRes = []
        # dataClass.trainIdList+dataClass.validIdList
        proteins = list(range(len(dataClass.p2id)))
        # self._save_emb('cache/_preEmbedding.pkl')
        for i, (trainProteins, validProteins) in enumerate(kf.split(proteins)):
            print(f'CV_{i + 1}:')
            if loc > 0 and i + 1 != loc:
                print(f'Pass CV_{i + 1}')
                continue
            self.reset_parameters()
            # self._load_emb('cache/_preEmbedding.pkl')

            dataClass.trainIdList = [i for i in range(
                len(dataClass.eSeqData)) if dataClass.eSeqData[i, 0] in trainProteins]
            dataClass.validIdList = [i for i in range(
                len(dataClass.eSeqData)) if dataClass.eSeqData[i, 0] in validProteins]
            dataClass.trainSampleNum, dataClass.validSampleNum = len(
                dataClass.trainIdList), len(dataClass.validIdList)

            res = self.train(dataClass, trainSize, batchSize, epoch, stopRounds, earlyStop, saveRounds, optimType,
                             preheat, lr1, lr2, momentum, weightDecay,
                             isHigherBetter, metrics, report, f"{savePath}_cv{i + 1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)

    def get_optimizer(self, optimType, lr, weightDecay, momentum):
        if optimType == 'Adam':
            return torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType == 'AdamW':
            return torch.optim.AdamW(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType == 'SGD':
            return torch.optim.SGD(self.moduleList.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)

    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1,
              optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, isHigherBetter=True,
              metrics="AUC", report=["ACC", "AUC"],
              savePath='model'):
        assert batchSize % trainSize == 0
        metrictor = Metrictor()
        self.stepCounter = 0
        self.stepUpdate = batchSize // trainSize
        self.preheat()

        # Initialize optimizer and LR scheduler
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.moduleList.parameters()), lr=lr1, weight_decay=weightDecay)
        schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=4, verbose=True)

        # Get random training stream
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum + trainSize - 1) // trainSize
        mtc, bestMtc, stopSteps = 0.0, 0.0, 0

        # Get random validation stream
        if dataClass.validSampleNum > 0:
            validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)

        st = time.time()
        print('Start pre-heat training:')
        for e in range(epoch):
            if e == preheat:
                if preheat > 0:
                    self.load(savePath + '.pkl')
                self.normal()
                optimizer = self.get_optimizer(
                    optimType=optimType, lr=lr2, weightDecay=weightDecay, momentum=momentum)
                # self.schedulerWU = ScheduledOptim(optimizer, lr2, 1000)
                schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=30, verbose=True)
                print('Start normal training: ')
            for i in range(itersPerEpoch):
                # Training mode
                self.to_train_mode()
                # Next stream
                X, Y = next(trainStream)
                # Calculate loss
                if X['res']:
                    loss = self._train_step(X, Y, optimizer)
                # ...
                if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
                    # Validation mode
                    self.to_eval_mode()
                    print(f"After iters {e * itersPerEpoch + i + 1}: [train] loss= {loss:.3f};", end='')
                    # If there are validation examples
                    if dataClass.validSampleNum > 0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X, Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch - i - 1) + (epoch - e - 1)
                               * itersPerEpoch) * trainSize
                    speed = (e * itersPerEpoch + i + 1) * trainSize / (time.time() - st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;" % (speed, restNum / speed))

            # Epoch is over, so check results of training and validation set
            if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
                self.to_eval_mode()
                print(f'========== Epoch:{e + 1:5d} ==========')

                Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(
                    trainSize, type='train', device=self.device))
                    
                metrictor.set_data(Y_pre, Y)
                print(f'[Total Train]', end='')
                metrictor(report)
                print(f'[Total Valid]', end='')

                Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(
                    trainSize, type='valid', device=self.device))

                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                schedulerRLR.step(mtc)
                print('=================================')

                if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl" % savePath, e + 1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps >= earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e + 1}, stop training.')
                        break
        self.load("%s.pkl" % savePath)
        self.to_eval_mode()
        os.rename("%s.pkl" % savePath, "%s_%s.pkl" % (savePath, ("%.3lf" % bestMtc)[2:]))

        print(f'============ Result ============')

        print(f'[Total Train]', end='')
        Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(
            trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        metrictor(report)

        print(f'[Total Valid]', end='')
        Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(
            trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        res = metrictor(report)

        if dataClass.testSampleNum > 0:
            print(f'[Total Test]', end='')
            Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(
                trainSize, type='test', device=self.device))
            metrictor.set_data(Y_pre, Y)
            metrictor(report)
            res = metrictor(report) # Report test scores if there is a test set
        print(f'================================')
        return res

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            # stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            if 'am2id' in stateDict:
                stateDict['am2id'], stateDict['id2am'] = dataClass.am2id, dataClass.id2am
            if 'go2id' in stateDict:
                stateDict['go2id'], stateDict['id2go'] = dataClass.go2id, dataClass.id2go
            if 'at2id' in stateDict:
                stateDict['at2id'], stateDict['id2at'] = dataClass.at2id, dataClass.id2at
        torch.save(stateDict, path)
        print('Model saved in "%s".' % path)

    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            # if "trainIdList" in parameters:
            #     dataClass.trainIdList = parameters['trainIdList']
            # if "validIdList" in parameters:
            #     dataClass.validIdList = parameters['validIdList']
            # if "testIdList" in parameters:
            #     dataClass.testIdList = parameters['testIdList']
            if 'am2id' in parameters:
                dataClass.am2id, dataClass.id2am = parameters['am2id'], parameters['id2am']
            if 'go2id' in parameters:
                dataClass.go2id, dataClass.id2go = parameters['go2id'], parameters['id2go']
            if 'at2id' in parameters:
                dataClass.at2id, dataClass.id2at = parameters['at2id'], parameters['id2at']
        print("%d epochs and %.3lf val Score 's model load finished." %
              (parameters['epochs'], parameters['bestMtc']))

    def _save_emb(self, path):
        stateDict = {}
        for module in self.embModuleList:
            stateDict[module.name] = module.state_dict()
        torch.save(stateDict, path)
        print('Pre-trained Embedding saved in "%s".' % path)

    def _load_emb(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.embModuleList:
            module.load_state_dict(parameters[module.name])
        print('Pre-trained Embedding loaded in "%s".' % path)

    def preheat(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = False

    def normal(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = True

    def calculate_y_prob(self, X, mode):
        Y_pre = self.calculate_y_logit(X, mode)['y_logit']
        return torch.sigmoid(Y_pre)

    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X, 'predict')
        Y_logit = out['y_logit']

        addLoss = 0.0
        if 'loss' in out:
            addLoss += out['loss']
        return self.criterion(Y_logit, Y) + addLoss

    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)

    def calculate_y_prob_by_iterator(self, dataStream):
        YArr, Y_preArr = [], []
        while True:
            try:
                X, Y = next(dataStream)
            except:
                break
            Y_pre, Y = self.calculate_y_prob(
                X, mode='predict').cpu().data.numpy(), Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr, Y_preArr = np.hstack(YArr).astype(
            'int32'), np.hstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

    # New function to calculate scores for seen/unseen proteins separately
    def calculate_y_with_seenbool(self, dataStream):
        YArr, Y_preArr, seenbool = [], [], []

        while True:
            try:
                X, Y = next(dataStream)
            except:
                break
            Y_pre, Y = self.calculate_y_prob(X, mode='predict').cpu().data.numpy(), Y.cpu().data.numpy()
            
            seenbool.append(X['seenbool'].cpu())

            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr, Y_preArr, seenbool = np.hstack(YArr).astype('int32'), np.hstack(Y_preArr).astype('float32'), np.hstack(seenbool).astype(bool)
        return Y_preArr, YArr, seenbool

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter < self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y) / self.stepUpdate
        loss.backward()

        if p:
            nn.utils.clip_grad_norm_(
                self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            # self.schedulerWU.step_and_update_lr()
            # self.schedulerWU.zero_grad()
        return loss * self.stepUpdate


class DTI_Bridge(BaseClassifier):
    def __init__(self, outSize,
                 cHiddenSizeList,
                 fHiddenSizeList,
                 fSize=1024, cSize=8422,
                 gcnHiddenSizeList=[], fcHiddenSizeList=[], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2, device=torch.device('cuda'),
                 useFeatures={"kmers": True, "pSeq": True,
                              "FP": True, "dSeq": True},
                 maskDTI=False):
        self.nodeEmbedding = TextEmbedding(torch.tensor(np.random.normal(size=(max(
            nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout, name='nodeEmbedding').to(device)

        self.amEmbedding = TextEmbedding(
            torch.eye(24), dropout=hdnDropout, freeze=True, name='amEmbedding').to(device)
        self.pCNN = TextCNN(24, 64, [25], ln=True, name='pCNN').to(device)
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True,
                             dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='pFcLinear').to(device)

        self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN').to(device)
        self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True,
                             dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='dFcLinear').to(device)

        self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
            device)
        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
            device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout,
                           dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout,
                            bnEveryLayer=True, dpEveryLayer=True).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList(
            [self.nodeEmbedding, self.cFcLinear, self.fFcLinear, self.nodeGCN, self.fcLinear,
             self.amEmbedding, self.pCNN, self.pFcLinear, self.dCNN, self.dFcLinear])
        self.device = device
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def calculate_y_logit(self, X, mode='train'):
        Xam = (self.cFcLinear(X['aminoCtr']).unsqueeze(1) if self.useFeatures['kmers'] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(X['aminoSeq']))).unsqueeze(
                  1) if self.useFeatures['pSeq'] else 0)  # => batchSize × 1 × outSize
        Xat = (self.fFcLinear(X['atomFin']).unsqueeze(1) if self.useFeatures['FP'] else 0) + \
              (self.dFcLinear(self.dCNN(X['atomFea'])).unsqueeze(
                  1) if self.useFeatures['dSeq'] else 0)  # => batchSize × 1 × outSize

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(
                self.nodeEmbedding.embedding.weight)).repeat(len(Xat), 1, 1)
            # => batchSize × nodeNum × outSize
            node = torch.cat([Xam, Xat, node], dim=1)
            # => batchSize × nodeNum × 1
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)

            cosNode = torch.matmul(node, node.transpose(
                1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
            # cosNode = cosNode*0.5 + 0.5
            cosNode = F.relu(cosNode)  # => batchSize × nodeNum × nodeNum
            # => batchSize × nodeNum × nodeNum
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskDTI:
                cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(
                len(Xam), 1, 1)  # => batchSize × nodeNum × nodeNum
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / \
                                                               (torch.sum(cosNode, dim=2) ** 0.5)
            # => batchSize × batchnodeNum × nodeNumSize
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            # => batchSize × nodeNum × outSize
            node_gcned = self.nodeGCN(node, pL)

            node_embed = node_gcned[:, 0, :] * \
                         node_gcned[:, 1, :]  # => batchSize × outSize
        else:
            node_embed = (Xam * Xat).squeeze(dim=1)  # => batchSize × outSize
        # if self.resnet:
        #    node_gcned += torch.cat([Xam[:,0,:],Xat[:,0,:]],dim=1)
        # , "loss":1*l2}
        return {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}


class ST_Bridge(BaseClassifier):
    def __init__(self, outSize,
                 cHiddenSizeList,
                 fHiddenSizeList,
                 fSize=1024, cSize=8422,
                 gcnHiddenSizeList=[], fcHiddenSizeList=[], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2, device=torch.device('cuda'), sampleType='CEL',
                 useFeatures={"kmers": True, "pSeq": True,
                              "FP": True, "dSeq": True},
                 maskDTI=False):
        self.nodeEmbedding = TextEmbedding(torch.tensor(np.random.normal(size=(max(
            nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout, name='nodeEmbedding').to(device)

        self.amEmbedding = TextEmbedding(
            torch.eye(24), dropout=hdnDropout, freeze=True, name='amEmbedding').to(device)
        self.pCNN = TextCNN(24, 64, [25], ln=True, name='pCNN').to(device)
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True,
                             dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='pFcLinear').to(device)

        self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN').to(device)

        self.STLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear',
                              dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
             device) # altered MLP layer for SMILES transformer

        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
            device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout,
                           dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout,
                            bnEveryLayer=True, dpEveryLayer=True).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList(
            [self.nodeEmbedding, self.cFcLinear, self.STLinear, self.nodeGCN, self.fcLinear,
             self.amEmbedding, self.pCNN, self.pFcLinear, self.dCNN])
        self.sampleType = sampleType
        self.device = device
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def calculate_y_logit(self, X, mode='train'):
        Xam = (self.cFcLinear(X['aminoCtr']).unsqueeze(1) if self.useFeatures['kmers'] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(X['aminoSeq']))).unsqueeze(
                  1) if self.useFeatures['pSeq'] else 0)  # => batchSize × 1 × outSize
        Xat = self.STLinear(X['ST_fingerprint']).unsqueeze(1)  # changed to fit the transformer fingerprint MLP

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(
                self.nodeEmbedding.embedding.weight)).repeat(len(Xat), 1, 1)
            # => batchSize × nodeNum × outSize
            node = torch.cat([Xam, Xat, node], dim=1)
            # => batchSize × nodeNum × 1
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)

            cosNode = torch.matmul(node, node.transpose(
                1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
            # cosNode = cosNode*0.5 + 0.5
            cosNode = F.relu(cosNode)  # => batchSize × nodeNum × nodeNum
            # => batchSize × nodeNum × nodeNum
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskDTI:
                cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(
                len(Xam), 1, 1)  # => batchSize × nodeNum × nodeNum
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / \
                                                               (torch.sum(cosNode, dim=2) ** 0.5)
            # => batchSize × batchnodeNum × nodeNumSize
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            # => batchSize × nodeNum × outSize
            node_gcned = self.nodeGCN(node, pL)

            node_embed = node_gcned[:, 0, :] * \
                         node_gcned[:, 1, :]  # => batchSize × outSize
        else:
            node_embed = (Xam * Xat).squeeze(dim=1)  # => batchSize × outSize
        # if self.resnet:
        #    node_gcned += torch.cat([Xam[:,0,:],Xat[:,0,:]],dim=1)
        # , "loss":1*l2}
        return {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}
        
        
class p_Ebedding_bridge(BaseClassifier):
    def __init__(self, outSize,
                 cHiddenSizeList,
                 fHiddenSizeList,
                 fSize=1024, cSize=8422,
                 gcnHiddenSizeList=[], fcHiddenSizeList=[], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2, device=torch.device('cuda'),
                 useFeatures={"pEmbeddings": True, "pSeq": True,
                              "FP": True, "dSeq": True},
                 maskDTI=False):
        self.nodeEmbedding = TextEmbedding(torch.tensor(np.random.normal(size=(max(
            nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout, name='nodeEmbedding').to(device)

        self.amEmbedding = TextEmbedding(
            torch.eye(24), dropout=hdnDropout, freeze=True, name='amEmbedding').to(device)
            
        self.pCNN = TextCNN(24, 64, [25], ln=True, name='pCNN').to(device)

                             
        self.pFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='pFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)

        self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN').to(device)
        
        self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True,
                             dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='dFcLinear').to(device)

        self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
            device)
        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear',
                             dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(
            device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout,
                           dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout,
                            bnEveryLayer=True, dpEveryLayer=True).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList([self.nodeEmbedding, self.fFcLinear, self.nodeGCN, self.fcLinear, self.pFcLinear, self.dCNN, self.dFcLinear])
        self.device = device
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def calculate_y_logit(self, X, mode='train'):
        Xam = (self.pFcLinear(X['pEmbeddings']).unsqueeze(1) if self.useFeatures['pEmbeddings'] else 0)  # => batchSize × 1 × outSize
        Xat = (self.fFcLinear(X['atomFin']).unsqueeze(1) if self.useFeatures['FP'] else 0) + \
              (self.dFcLinear(self.dCNN(X['atomFea'])).unsqueeze(
                  1) if self.useFeatures['dSeq'] else 0)  # => batchSize × 1 × outSize

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(
                self.nodeEmbedding.embedding.weight)).repeat(len(Xat), 1, 1)
            # => batchSize × nodeNum × outSize
            node = torch.cat([Xam, Xat, node], dim=1)
            # => batchSize × nodeNum × 1
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)

            cosNode = torch.matmul(node, node.transpose(
                1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)  # => batchSize × nodeNum × nodeNum
            # cosNode = cosNode*0.5 + 0.5
            cosNode = F.relu(cosNode)  # => batchSize × nodeNum × nodeNum
            # => batchSize × nodeNum × nodeNum
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskDTI:
                cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(
                len(Xam), 1, 1)  # => batchSize × nodeNum × nodeNum
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / \
                                                               (torch.sum(cosNode, dim=2) ** 0.5)
            # => batchSize × batchnodeNum × nodeNumSize
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            # => batchSize × nodeNum × outSize
            node_gcned = self.nodeGCN(node, pL)

            node_embed = node_gcned[:, 0, :] * \
                         node_gcned[:, 1, :]  # => batchSize × outSize
        else:
            node_embed = (Xam * Xat).squeeze(dim=1)  # => batchSize × outSize
        # if self.resnet:
        #    node_gcned += torch.cat([Xam[:,0,:],Xat[:,0,:]],dim=1)
        # , "loss":1*l2}
        return {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}

