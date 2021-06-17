# from contention_detection_meta.utils import belongs, compute_f1_score, compute_precision_recall
import numpy as np
from sklearn.mixture import BayesianGaussianMixture,GaussianMixture
import matplotlib.pyplot as plt
from utils import *
from data_loader import groupby_cpu

# best_score_threshold = 0.6
# max_components = 10
default_cpu_range=[(0.95,1.01)]

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


class LLC_contention_detector(best_score_threshold=0.6,max_components=10):
    def __init__(self):
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
        for range in X_.keys():
            self.training_set=X_[range]
            MPKI_candidates=self.generate_threshold('MPKI',method)
            rule={}
            theta_m,theta_o=self.select_MPKI_threshold(MPKI_candidates)
            rule['MPKI']=theta_m
            rule['OCC']=theta_o

            CPI_candidates=self.generate_threshold('CPI',method)
            rule['CPI']=self.select_CPI_threshold(theta_m,CPI_candidates)

            self.rules[range]=rule

        return self.rules

    def predict(self,X,Y):
        '''
        X: test set, expected to be Nx4 array, dimensions by order are CPI, MPKI, OCC, CPU
        Y: test set label, expected to be Nx1 0/1 array, 0 for non-contention and 1 for contention
        '''
        TP, TN, FP, FN = 0, 0, 0, 0
        X_predict = np.zeros_like(Y)
        for i in range(X.shape[0]):
            r = belongs(X[i][3],self.rules.keys())
            if r:
                if X[i][2] < self.rules[r]['OCC'] and X[i][1] > self.rules[r]['MPKI'] and X[i][0] > self.rules[r]['CPI']:
                    X_predict[i]=1

        for i in range(X_predict.shape[0]):
            if X_predict[i] == 0:
                if Y[i] == 0:
                    TN += 1
                else:
                    FN += 1
            elif Y[i] == 0:
                FP += 1
            else:
                TP += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = compute_f1_score(precision,recall)

        print(precision,recall,F1)
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

        if f_avg < self.best_score_threshold:
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

        if f_avg < self.best_score_threshold:
            best_CPI = np.max(CPI_candidates)

        return best_CPI

