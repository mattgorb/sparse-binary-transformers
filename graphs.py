import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

dataset='MSL'
if dataset == 'SMD':
    entities = 28
elif dataset == 'SMAP':
    entities = 55
elif dataset == 'MSL':
    entities = 27

scores={}
labels={}
model_type='Sparse'
for ent in range(entities):
    filename=f'scores_{dataset}_entity_{ent}_type_{model_type}.csv'
    #filename=f'scores_{dataset}_entity_{ent}_type_{model_type}_pr_{pr}.csv'
    df=pd.read_csv(f'results/ad_results/{filename}')
    scores[ent]=df['scores'].values
    labels[ent]=df['labels'].values
    #print(df.head())
msl_thresholds=[1.544992272372864,4.050923752652735,3.281812778620262,0.8393471723900429,0.4740399300231022,0.503962932915515,0.17607586507098508,0.45531278719104884
                   ,3.013816104199144,1.8041341717051032,2.0346874132463,2.10990155323170,0.79844429181294,0.2136289496458972,1.1149105418467369,1.2484911285622893
                   ,0.48835518668833827,1.1354279453185692,1.9610125523434965,1.140732284018575,0.11866892129182816,2.0582447609779604,1.6623414861993189,0.6707647878471543
                   ,1.2953062022739934,1.3951475272135772,1.4823189840617592,]
plt.clf()
'''for ent in range(entities):
    preds=(scores[ent]>=msl_thresholds[ent]).astype(int)
    #preds=scores[ent]
    #fpr, tpr, threshold = metrics.roc_curve(labels[ent], preds)
    precision, recall, thresholds = precision_recall_curve(labels[ent], preds)
    #roc_auc = metrics.auc(fpr, tpr)
    #print(roc_auc)

    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, )#'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(recall, precision, )
plt.show()'''
scores_list=[]
labels_list=[]
for ent in range(entities):
    preds=(scores[ent]>=msl_thresholds[ent]).astype(int)
    scores_list.extend(preds)
    labels_list.extend(labels[ent])
precision, recall, thresholds = precision_recall_curve(labels_list, scores_list)
#fpr, tpr, threshold = metrics.roc_curve(labels_list, scores_list)
#roc_auc = metrics.auc(fpr, tpr)
#print(fpr)
#print(tpr)
#print(roc_auc)
#plt.plot(fpr, tpr, )#'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot(recall, precision)
plt.legend(loc = 'lower right')
plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


'''model_type='Dense'
for ent in range(entities):
    filename=f'scores_{dataset}_entity_{ent}_type_{model_type}.csv'
    #filename=f'scores_{dataset}_entity_{ent}_type_{model_type}_pr_{pr}.csv'
    df=pd.read_csv(f'results/ad_results/{filename}')
    print(df.head())'''