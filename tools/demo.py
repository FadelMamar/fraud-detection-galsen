from fraudetect.dataset import load_data, data_loader, train_test_split
from fraudetect.features import perform_feature_engineering, transform_data
from fraudetect.config import COLUMNS_TO_DROP, COLUMNS_TO_ONE_HOT_ENCODE, COLUMNS_TO_SCALE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

df_data = load_data(r"D:\fraud-detection-galsen\data\training.csv")

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None, dtype=np.float64)
scaler = StandardScaler()

X_train, y_train = perform_feature_engineering(transactions_df=df_data, 
                                           columns_to_drop=COLUMNS_TO_DROP,
                                           columns_to_onehot_encode=COLUMNS_TO_ONE_HOT_ENCODE,
                                           columns_to_scale=COLUMNS_TO_SCALE,
                                           onehot_encoder=onehot_encoder,
                                           scaler=scaler,
                                           mode='train',
                                           windows_size_in_days=[1,7,30]
                                           )



#%% Outliers detectors
from fraudetect.features import concat_decision_scores_pyod
from fraudetect.detectors import get_detector, instantiate_detector
from fraudetect import import_from_path, sample_cfg

configs = import_from_path('hyp_search_conf',
                           r'D:\fraud-detection-galsen\tools\hyp_search_conf.py')

model_list = list()
for name in configs.outliers_detectors.keys():
    detector, cfg = get_detector(name=name, config=configs.outliers_detectors)
    cfg = sample_cfg(cfg)
    detector = instantiate_detector(detector, cfg)
    model_list.append(detector)
    
X_t = concat_decision_scores_pyod(model_list, X_train)


#%%  Undersampling
from imblearn.under_sampling import (TomekLinks, 
                                     RandomUnderSampler, 
                                     AllKNN, 
                                     NearMiss, 
                                     EditedNearestNeighbours,
                                     CondensedNearestNeighbour,
                                     OneSidedSelection,
                                     ClusterCentroids,
                                     NeighbourhoodCleaningRule
                                     )
from sklearn.cluster import MiniBatchKMeans


# under_sampler = TomekLinks(sampling_strategy='majority',n_jobs=4)
# X_us,  y_us = under_sampler.fit_resample(X_train, y_train)

# under_sampler_rus = RandomUnderSampler(sampling_strategy='majority',random_state=41,replacement=False)
# X_rus,  y_rus = under_sampler_rus.fit_resample(X_train, y_train)

# under_sampler_knn = AllKNN(sampling_strategy='majority',n_neighbors=5,kind_sel='all',allow_minority=False,n_jobs=4)
# X_knn,  y_knn = under_sampler_knn.fit_resample(X_train, y_train)

frac = float(2*y_train.sum()/(1-y_train).sum())
under_sampler_nm = NearMiss(sampling_strategy=frac,n_neighbors=5,version=1,n_jobs=4)
X_nm,  y_nm = under_sampler_nm.fit_resample(X_train, y_train)

# under_sampler_enn = EditedNearestNeighbours(sampling_strategy='majority',n_neighbors=5,kind_sel='mode',n_jobs=4)
# X_enn,  y_enn = under_sampler_enn.fit_resample(X_train, y_train)

# under_sampler_cnn = CondensedNearestNeighbour(sampling_strategy='majority',n_neighbors=5,random_state=41,n_seeds_S=5,n_jobs=4)
# X_cnn,  y_cnn = under_sampler_cnn.fit_resample(X_train, y_train)

# under_sampler_oss = OneSidedSelection(sampling_strategy='majority',n_neighbors=5,random_state=41,n_seeds_S=2,n_jobs=4)
# X_oss,  y_oss = under_sampler_oss.fit_resample(X_train, y_train)

# Too slow
# under_sampler_cc = ClusterCentroids(sampling_strategy=frac,
#                                     estimator=MiniBatchKMeans(max_iter=100,batch_size=2048,tol=1e-6)
#                                     ,voting='auto',random_state=41)
# X_cc,  y_cc = under_sampler_cc.fit_resample(X_train, y_train)

# under_sampler_ncr = NeighbourhoodCleaningRule(sampling_strategy='majority',n_neighbors=5,threshold_cleaning=0.2)
# X_ncr,  y_ncr = under_sampler_ncr.fit_resample(X_train, y_train)


#%% Oversampling

from imblearn.over_sampling import (SMOTE,
                                    ADASYN,
                                    BorderlineSMOTE,
                                    KMeansSMOTE,
                                    SVMSMOTE)


frac = float(2*y_train.sum()/(1-y_train).sum())

oversampler_smote = SMOTE(sampling_strategy=frac,random_state=41, k_neighbors=5,)
X_ste, y_ste = oversampler_smote.fit_resample(X_train, y_train)
print(y_ste.sum()/y_train.sum())

oversampler_ada = ADASYN(sampling_strategy=frac,random_state=41, n_neighbors=5,)
X_ada, y_ada = oversampler_ada.fit_resample(X_train, y_train)
print(y_ada.sum()/y_train.sum())

oversampler_bste = BorderlineSMOTE(sampling_strategy=frac,random_state=41,m_neighbors=5,k_neighbors=10,kind='borderline-1')
X_bste, y_bste = oversampler_bste.fit_resample(X_train, y_train)
print(y_bste.sum()/y_train.sum())

oversampler_sste = SVMSMOTE(sampling_strategy=frac,random_state=41,out_step=0.5,m_neighbors=5,k_neighbors=3)
X_sste, y_sste = oversampler_sste.fit_resample(X_train, y_train)
print(y_sste.sum()/y_train.sum())






#%% Test of code
from fraudetect.features import data_resampling
from fraudetect.sampling import sample_cfg, get_sampler
from hyp_search_conf import under_sampler, over_sampler, combined_sampler, outliers_detectors


sampler_names=['nearmiss',] #'nearmiss','SMOTE']

sampler_cfgs = list()

for name in sampler_names:
    
    if name in under_sampler.keys():
        cfg = under_sampler[name]
        
    elif name in over_sampler.keys():
        cfg = over_sampler[name]
        
    elif name in combined_sampler.keys():
        cfg = combined_sampler[name]
    
    cfg = sample_cfg(cfg) # for test purposes
    sampler_cfgs.append({name:cfg})

X_t, y_t = data_resampling(X=X_train,
                            y=y_train,
                            sampler_names=sampler_names,
                            sampler_cfgs=sampler_cfgs)








