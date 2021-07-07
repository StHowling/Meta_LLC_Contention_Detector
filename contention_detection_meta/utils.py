import numpy as np

# Ugly counting functions

def count_gt(X,theta,col=0):
    count=0
    for i in range(X.shape[0]):
        if X[i][col]>theta:
            count+=1    
    return count

def count_lt(X,theta,col=0):
    count=0
    for i in range(X.shape[0]):
        if X[i][col]<theta:
            count+=1
    return count

def count_ge(X,theta,col=0):
    count=0
    for i in range(X.shape[0]):
        if X[i][col]>=theta:
            count+=1
    return count

def count_le(X,theta,col=0):
    count=0
    for i in range(X.shape[0]):
        if X[i][col]<=theta:
            count+=1
    return count

def count_joint_gl(X,theta_1,theta_2):
    count=0
    for i in range(X.shape[0]):
        if X[i][0]>theta_1 and X[i][1]<theta_2:
            count+=1
    return count

def count_joint_lege(X,theta_1,theta_2):
    count=0
    for i in range(X.shape[0]):
        if X[i][0]<=theta_1 and X[i][1]>=theta_2:
            count+=1
    return count

def count_joint_gg(X,theta_1,theta_2):
    count=0
    for i in range(X.shape[0]):
        if X[i][0]>theta_1 and X[i][1]>theta_2:
            count+=1
    return count

def count_joint_lele(X,theta_1,theta_2):
    count=0
    for i in range(X.shape[0]):
        if X[i][0]<=theta_1 and X[i][1]<=theta_2:
            count+=1
    return count

count_func_map = {
    'g': count_gt,
    'l': count_lt,
    'ge': count_ge,
    'le': count_le,
    'gl': count_joint_gl,       #OCC
    'lege': count_joint_lege,   #OCC
    'gg': count_joint_gg,       #CPI
    'lele': count_joint_lele    #CPI
}

def compute_precision_recall(X,theta_1,theta_2,metric1='g',metric2='l'):
    '''
    X: subset of raw data, expected to be Nx2 np.array
    theta_1: MPKI threshold to verify
    theta_2: LLC OCC or CPI threshold to verify
    metricx: in {'l','g','le','ge'}, i.e., boolean operators
    '''
    count_theta_1 = count_func_map[metric1](X,theta_1)
    count_theta_2 = count_func_map[metric2](X,theta_2,1)
    count_joint = count_func_map[metric1+metric2](X,theta_1,theta_2)

    if count_theta_1==0 or count_theta_2==0:
        precision, recall = 0, 0
    else:
        precision = count_joint/count_theta_2
        recall = count_joint/count_theta_1

    return precision,recall

def compute_f1_score(p,r):
    if p==0 or r==0:
        return 0
    return 2*p*r/(p+r)
    
def belongs(val,range_list):
    for r in range_list:
        if val >= r[0] and val < r[1]:
            return r
    
    return False

def evaluate_classification(prediction,label):
    assert prediction.shape == label.shape
    TN, FN, FP, TP = 0, 0, 0, 0

    for i in range(prediction.shape[0]):
        if prediction[i] == 0:
            if label[i] == 0:
                TN += 1
            else:
                FN += 1
        elif label[i] == 0:
            FP += 1
        else:
            TP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1=compute_f1_score(precision,recall)
    print("Presicion: %.2f\tRecall: %.2f\tF1-score: %.2f"%(precision,recall,F1))


