import numpy as np
import pickle
import seaborn as sns
from sklearn import metrics as skmetrics
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1, len(Y))
    Y_pre = np.argmax(preds, axis=0)
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True


def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1, len(Y))
    Y_pre = np.argmax(preds, axis=0)
    return 'precision', float(Counter(Y == Y_pre)[True]/len(Y)), True


id2lab = [[-1, -1]]*20
for a in range(1, 11):
    for s in [1, 2]:
        id2lab[a-1+(s-1)*10] = [a, s]


class Metrictor:
    def __init__(self):
        self._reporter_ = {"ACC": self.ACC, "AUC": self.AUC, "Precision": self.Precision,
                           "Recall": self.Recall, "F1": self.F1, "LOSS": self.LOSS}

    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res

    def set_data(self, Y_prob_pre, Y, threshold=0.5):
        self.Y = Y.astype('int')
        if len(Y_prob_pre.shape) > 1:
            self.Y_prob_pre = Y_prob_pre[:, 1]
            self.Y_pre = Y_prob_pre.argmax(axis=-1)
        else:
            self.Y_prob_pre = Y_prob_pre
            self.Y_pre = (Y_prob_pre > threshold).astype('int')

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i, res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" +
                  "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" +
              "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))

    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')

    def ACC(self):
        return ACC(self.Y_pre, self.Y)

    def AUC(self):
        return AUC(self.Y_prob_pre, self.Y)

    def Precision(self):
        return Precision(self.Y_pre, self.Y)

    def Recall(self):
        return Recall(self.Y_pre, self.Y)

    def F1(self):
        return F1(self.Y_pre, self.Y)

    def LOSS(self):
        return LOSS(self.Y_prob_pre, self.Y)


def calc_ROC(y_test, y_score, savePath, timestamp, plot=True):
    picklefile = f'logs/ROC_{savePath}_{timestamp}.pkl'  # create log with unique timestamp
    plotfile = f'logs/plot_ROC_{savePath}_{timestamp}.png'  # create log with unique timestamp
    all_metrics = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr['class'], tpr['class'], _ = skmetrics.roc_curve(y_test, y_score)
    roc_auc['class'] = skmetrics.auc(fpr['class'], tpr['class'])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = skmetrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = skmetrics.auc(fpr["micro"], tpr["micro"])

    all_metrics['fpr'] = fpr
    all_metrics['tpr'] = tpr
    all_metrics['roc_auc'] = roc_auc

    pickle.dump(all_metrics, open(picklefile, 'wb'))

    if plot:
        plt.figure()
        lw = 2
        plt.plot(fpr['class'], tpr['class'], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['class'])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver operating characteristic', fontsize=16)
        plt.legend(loc="lower right", fontsize=8)
        plt.savefig(plotfile, dpi=300)
        plt.clf() # clear the plot object


def calc_conf_matrix(y_true, y_pred, savePath, timestamp, plot=True):
    logfile = f'logs/CM_{savePath}_{timestamp}.txt'  # create log with unique timestamp
    plotfile = f'logs/plot_CM_{savePath}_{timestamp}.png'  # create log with unique timestamp
    y_pred = np.round(np.clip(y_pred, 0, 1)) # predicted values from continuous to 0,1
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()

    header = ['TN', 'FP', 'FN', 'TP', '\n']
    with open(logfile, 'a') as out:
        out.write(','.join(header))
        out.write(f'{tn},{fp},{fn},{tp}\n')

    if plot:
        sns.set(rc={'figure.figsize':(8,6), 'axes.labelsize': 14})
        y_pred = np.round(np.clip(y_pred, 0, 1))
        cm = skmetrics.confusion_matrix(y_true, y_pred, normalize=None)
        ax = sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.cividis)
        ax.set(xlabel='Actual', ylabel='Predicted')
        plt.savefig(plotfile, dpi=300)
        plt.clf()  # clear the plot object

class MetricLog:
    """
    log train and validation loss
    """
    def __init__(self, savePath, timestamp, to_report):
        Path("logs").mkdir(parents=True, exist_ok=True)
        self.logger = f'logs/train_val_{savePath}_{timestamp}.txt' # create log with unique timestamp
        self.plot_log = f'logs/learn_curve_{savePath}_{timestamp}.png' # create plot file with unique timestamp
        self.to_report = to_report
        self.save_train = list()
        self.save_val = list()
        header = [f'{mtc}_train' for mtc in to_report] + [f'{mtc}_valid' for mtc in to_report]
        self.write_header(header)

    def log_train_val(self, train, val):
        train_temp = [train[mtc] for mtc in self.to_report] # log LOSS and additional params in to_report param
        val_temp = [val[mtc] for mtc in self.to_report] # log LOSS and additional params in to_report param
        self.save_train.append(train_temp)
        self.save_val.append(val_temp)

        train_formatted = [f'{train[mtc]:.3f}' for mtc in self.to_report] # format to 3 digit floats
        val_formatted = [f'{val[mtc]:.3f}' for mtc in self.to_report] # format to 3 digit floats
        self.write_log(train_formatted, val_formatted)

    def write_header(self, header):
        with open(self.logger, 'a') as out:
            out.write(f'{",".join(header)}\n')

    def write_log(self, train_mtc, val_mtc):
        """
        write all metrics in to_report for train and test
        """
        with open(self.logger, 'a') as out:
            out.write(f'{",".join(train_mtc)},{",".join(val_mtc)}\n')

    def plot_curve(self):
        """
        default learn curve plotting with just LOSS
        """
        idx = self.to_report.index('LOSS')
        x = [i for i in range(1, len(self.save_train)+1)]

        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()
        ax.plot(x, [item[idx] for item in self.save_train], label='train loss', c='blue')
        ax.plot(x, [item[idx] for item in self.save_val],label='validation loss', c='orange')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.xticks(np.arange(0, max(x)+1, 16))

        plt.legend(fontsize=10)
        plt.title('learning curve', fontsize=16)
        plt.xlabel('epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)

        plt.savefig(self.plot_log, dpi=300)


def ACC(Y_pre, Y):
    return (Y_pre == Y).sum() / len(Y)


def AUC(Y_prob_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_prob_pre)


def Precision(Y_pre, Y):
    return skmetrics.precision_score(Y, Y_pre)


def Recall(Y_pre, Y):
    return skmetrics.recall_score(Y, Y_pre)


def F1(Y_pre, Y):
    return skmetrics.f1_score(Y, Y_pre)


def LOSS(Y_prob_pre, Y):
    Y_prob_pre, Y = Y_prob_pre.reshape(-1), Y.reshape(-1)
    Y_prob_pre[Y_prob_pre > 0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre < 0.01] += 1e-3
    return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))
