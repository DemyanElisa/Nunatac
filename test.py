import numpy as np
from AD_preprocessing import *
from AD_training import *
from AD_evaluation import *

adpr = Preprocessing()

path_norm = './data/pump/id_00/normal/'
path_abnorm = './data/pump/id_00/abnormal/'

norm_signals, norm_srs = adpr.load_files_from_dir(path_norm)
abnorm_signals, abnorm_srs = adpr.load_files_from_dir(path_abnorm)
all_signals = np.vstack([norm_signals, abnorm_signals])
all_srs = norm_srs+abnorm_srs
all_labels = np.hstack([np.zeros(len(norm_srs)), np.ones(len(abnorm_srs))])
inds = list(range(len(all_labels)))
np.random.shuffle(inds)


print(all_signals.shape)
#all_signals = np.reshape(all_signals[inds], (len(inds), -1))
all_signals = all_signals[inds]
all_labels = all_labels[inds]

all_features = adpr.extract_features_from_all_signals(all_signals, all_srs)
all_features = np.concatenate([block[..., np.newaxis] for block in all_features],
                     axis=2)
print('generated features shape is', all_features.shape)


all_features = np.reshape(all_features, (-1, all_features.shape[0]*all_features.shape[1]))
models_list = ['SVM']


ev = AD_evaluation(models_list, all_signals, all_labels)
ev.evaluate_models_on_split_sets()

#for m in ['KNN', 'DBSCAN', 'ISOF', 'LOF', 'SVM']:
#    adinf = adi.Inference(model_choice = m)
#    adinf.fit_model(all_features, all_labels)

#for m in ['RNN', 'AUTOENCODER']:
#    adinf = adi.Inference(all_features, model_choice = m)
#    adinf.fit_model(all_features, all_labels)
