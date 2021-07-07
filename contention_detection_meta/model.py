# from contention_detection_meta.utils import belongs, compute_f1_score, compute_precision_recall
import numpy as np
from sklearn.mixture import BayesianGaussianMixture,GaussianMixture
import matplotlib.pyplot as plt
from utils import *
from data_loader import groupby_cpu

'''
This module includes two major classes: LLC_contention_detector and baseline, along with the 
GMM selection-of-k utility functions. The APIs are like scikit-learn: initialize the learner
with some key parameters, train and predict with the learner with input data.
'''

# best_score_threshold = 0.6
# max_components = 10
default_cpu_range=[(0.95,1.01)]
supported_baselines = ['p99','p95','p90','ur10','ur5','ur2','ur1']

def select_component_num_BIC(X,max_components):
    '''
    X: Nx1 array, MPKI/CPI value in each time interval
    '''
    bic = []
    for i in range(max_components):
        gmm = GaussianMixture(n_components=i)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            best_component_num=i
        
    return best_gmm, best_component_num

def select_component_num_VBEM(X,max_components):
    vGMM = BayesianGaussianMixture(n_components=max_components)
    vGMM.fit(X)

    return vGMM 

def visualize_model(X,model):
    min_val=np.min(X)
    max_val=np.max(X)
    padding=(max_val-min_val)/20
    x_pos=np.linspace(min_val-padding,max_val+padding)
    y_pos=model.score_samples(X)

    plt.plot(x_pos,y_pos,color='black',linewidth=3)
    for i in range(model.n_components):
        u = model.means_[i]
        v = model.covariances_[i]
        y = np.exp(-(x_pos - u) ** 2 / (2 * v ** 2)) / (v * np.sqrt(2 * np.pi))
        plt.plot(x_pos,y,linewidth=1)
    
    plt.show()


class LLC_contention_detector():
    def __init__(self,best_score_threshold=0.6,max_components=10):
        self.threshold = best_score_threshold
        self.max_components = max_components
        self.select_func_map = {
            'BIC': select_component_num_BIC,
            'VBEM': select_component_num_VBEM
        }
        self.target_idx_map = {
            'MPKI': 1,
            'CPI': 0
        }
        self.rules = {} # [l,u) -> theta_OCC, theta_MPKI, theta_CPI

    def train(self,X,method='BIC',range_list=default_cpu_range):
        '''
        X: training set, expected to be Nx4 array, dimensions by order are CPI, MPKI, OCC, CPU
        Group by is expected to be done by data_loader, returning a dict [l,u) -> N_ix3 list [CPI, MPKI, OCC]
        '''
        X_=groupby_cpu(X,range_list)
        for r in X_.keys():
            self.training_set=X_[r]
            MPKI_candidates=self.generate_threshold('MPKI',method)
            rule={}
            theta_m,theta_o=self.select_MPKI_threshold(MPKI_candidates)
            rule['MPKI']=theta_m
            rule['OCC']=theta_o

            CPI_candidates=self.generate_threshold('CPI',method)
            rule['CPI']=self.select_CPI_threshold(theta_m,CPI_candidates)

            self.rules[r]=rule

        return self.rules

    def predict(self,X,Y):
        '''
        X: test set, expected to be Nx4 array, dimensions by order are CPI, MPKI, OCC, CPU
        Y: test set label, expected to be Nx1 0/1 array, 0 for non-contention and 1 for contention
        '''
        X_predict = np.zeros_like(Y)
        for i in range(X.shape[0]):
            r = belongs(X[i][3],self.rules.keys())
            if r:
                if X[i][2] < self.rules[r]['OCC'] and X[i][1] > self.rules[r]['MPKI'] and X[i][0] > self.rules[r]['CPI']:
                    X_predict[i]=1

        evaluate_classification(X_predict,Y)
        return X_predict

    def fit_predict(self,X,Y,method='BIC',range_list=default_cpu_range):
        self.train(X,method,range_list)
        return self.predict(X,Y)

    def generate_threshold(self,target='MPKI',method='BIC'):
        candidates=[]
        X_=self.training_set[:,self.target_idx_map[target]]
        model = self.select_func_map[method](X_,self.max_components)
        # visualize_model(X_, model)

        for i in range(model.n_components):
            u = model.means_[i]
            v = model.covariances_[i]
            candidates.append(u+3*v)

        return candidates

    def select_MPKI_threshold(self,MPKI_candidates):
        target_idx = self.target_idx_map['MPKI']
        X_ = self.training_set[:,target_idx:target_idx+2]
        OCC_candidates=np.sort(X_[:,1])

        f_avg = 0
        for m in MPKI_candidates:
            for o in OCC_candidates:
                p, r = compute_precision_recall(X_, m, o)
                f1_1 = compute_f1_score(p, r)
                p, r = compute_precision_recall(X_, m, o, 'le', 'ge')
                f1_2 = compute_f1_score(p, r)
                f1 = (f1_1+f1_2)/2
                if f_avg < f1:
                    f_avg = f1
                    best_MPKI = m
                    best_OCC = o

        if f_avg < self.threshold:
            best_MPKI = MPKI_candidates[-1]
            best_OCC = np.max(OCC_candidates)

        return best_MPKI, best_OCC

    def select_CPI_threshold(self,MPKI,CPI_candidates):
        target_idx = self.target_idx_map['CPI']
        X_ = self.training_set[:,target_idx:target_idx+2]
        CPI_candidates=np.sort(X_[:,0])

        f_avg = 0
        for c in CPI_candidates:
            p, r = compute_precision_recall(X_, MPKI, c, 'g', 'g')
            f1_1 = compute_f1_score(p, r)
            p, r = compute_precision_recall(X_, MPKI, c, 'le', 'le')
            f1_2 = compute_f1_score(p, r)
            f1 = (f1_1+f1_2)/2
            if f_avg < f1:
                f_avg = f1
                best_CPI = c

        if f_avg < self.threshold:
            best_CPI = np.max(CPI_candidates)

        return best_CPI

class baseline():
    '''
    Currently implemented baseline methods are those in Table II of the original paper. That is:
    99%ile threshold, 95%ile threshold, 90%ile threshold;
    Usage ratio 10, Usage ratio 5, Usage ratio 2, Usage ratio 1.
    Note that the Usage ratio class also require cpu usage data of the stressors.
    '''
    def __init__(self, method='p99'):
        if method not in supported_baselines:
            print(self.__doc__)
            print("Specify one of the method keys: ",supported_baselines)
            exit()

        self.method=method
        self.rules={}

    def train(self,X,range_list=default_cpu_range,stressor_data=None):
        '''
        If the method is chosen to be one of the 'Usage ratio' class, the stressor_data must be provided,
        and expect to be of the same shape with X in dim 0. Note that it should be the total ratio rather than
        average, e.g. allocated 6 cores in total to the hogs and the legal range should be (0,600).
        '''
        if 'ur' in self.method and (stressor_data == None or stressor_data.shape[0] != X.shape[0]):
            print(self.train.__doc__)
            exit()

        X_=groupby_cpu(X,range_list)

        if 'p' in self.method:
            pct=int(self.method[1:])
            for r in X_.keys():
                rule={}
                rule['MPKI']=np.percentile(X_[r][:,1],pct)
                rule['CPI']=np.percentile(X_[r][:,0],pct)
                self.rules[r]=rule
        else:
            ratio=int(self.method[-1])
            for r in X_.keys():
                filtered_X=[]
                rule={}
                for i in range(X_[r].shape[0]):
                    if X_[r][i][3] / stressor_data >= ratio:
                        filtered_X.append(X_[r][i])

                filtered_X=np.array(filtered_X)

                rule['MPKI']=np.max(filtered_X[:,1])
                rule['CPI']=np.max(filtered_X[:,0])
                self.rules[r]=rule

        return self.rules

    def predict(self,X,Y):
        X_predict = np.zeros_like(Y)
        for i in range(X.shape[0]):
            r = belongs(X[i][3],self.rules.keys())
            if r:
                if X[i][1] > self.rules[r]['MPKI'] and X[i][0] > self.rules[r]['CPI']:
                    X_predict[i]=1

        evaluate_classification(X_predict,Y)
        return X_predict   

    def fit_predict(self,X,Y,range_list=default_cpu_range,stressor_data=None):
        self.train(X,range_list,stressor_data)
        return self.predict(X,Y)
