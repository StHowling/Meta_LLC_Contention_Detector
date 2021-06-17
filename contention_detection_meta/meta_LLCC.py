import json
from model import *
from data_loader import *

range_list=[]
for i in range(7):
    range_list.append((0.6+i*0.05,0.65+i*0.05))
    range_list.append((0.95,1.01))

if __name__=='__main__':
    config_file="settings.json"
    config=json.load(config_file)

    X, Y=load_data(config['data_file'])
    X_train, X_test = train_test_split(X,config['split_ratio'])
    Y_train, Y_test = train_test_split(Y,config['split_ratio'])

    model=LLC_contention_detector(config['best_score_threshold'],config['max_components'])
    print("Training model:")
    model.fit_predict(X_train,Y_train,config['method'])
    # model.fit_predict(X_train,Y_train,config['method'],range_list)

    print("Testing model")
    model.predict(X_test,Y_test)
