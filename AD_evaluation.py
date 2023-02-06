from AD_training import *:x
from AD_preprocessing import *
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import mlflow
import mlflow.sklearn
#from mlflow import MlflowClient
import logging
from mlflow import log_metric, log_param, log_artifacts

class AD_evaluation:

          """
          Initialises AD evaluation class
  
          PARAMETERS
          -------
              data (array)
              models_list (list of strings)
              labels (array) - true labels   
     
          """
    def __init__(self, models_list, data, labels):
        self.data = data
        self.labels = labels
        self.models_list = models_list


    
    def split_dataset(self, random_state = 42):
          """
          Splits data into train, test and val sets 

          PARAMETERS
          -------
              random state (int) - random seed for splitting procedure         

          """
        if self.labels is None:
            X_train, X_test = train_test_split(self.data, test_size=0.2, random_state=random_state)
            X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=random_state)
            return X_train, X_test, X_val, None, None, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2)#, random_state=random_state)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, 'x tr y tr x te y te shapes')
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)#, random_state=random_state)
            print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, 'x tr y tr x val y val shapes')
            return X_train, X_test, X_val, y_train, y_test, y_val


    def prediction_metrics(self, data, preds, targets, threshold=0.5):
        """
        Computes performance metrices for a given predictions and true labels
        
        PARAMETERS
        -------
            data (array)
            preds (array) - model predictions
            targets (array) - true labels
            threshold (float) - threshold for converting soft predictions to hard ones (ints)
        """

        if targets is not None:
            preds = preds > threshold
            acc = accuracy_score(targets, preds)
            f1_sc  = f1_score(targets, preds)
            cohen_kappa = cohen_kappa_score(targets, preds)
            matthews_corr = matthews_corrcoef(targets, preds)
            roc_auc = roc_auc_score(targets, preds)
            return acc, f1_sc, cohen_kappa, matthews_corr, roc_auc
        else:
            ari = adjusted_rand_score(targets, preds)
            ss = silhouette_score(data.T, preds)
            dbs = davies_bouldin_score(data.T, preds)
            return ari, ss, dbs



    def print_auto_logged_info(self, r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))
    


    def evaluate_models_on_split_sets(self):
        
        """
       
        Trains models and evaluates performance metrices on the given dataset with logging using MlFlow writer
        
        """


        X_train, X_test, X_val, y_train, y_test, y_val = self.split_dataset(self.data)
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("my-experiment")
        for model in self.models_list:
            print("experiment on model {}".format(model))
            mlflow.autolog()
            with mlflow.start_run() as run:
                if model in ['KNN', 'DBSCAN', 'ISOF', 'LOF', 'SVM']:
                    X_train, X_test, X_val = np.reshape(X_train, (X_train.shape[1], -1)), np.reshape(X_test, (X_test.shape[1], -1)), np.reshape(X_val, (X_val.shape[1], -1))
                t = Training(data=self.data, model_choice=model, labels=self.labels)
                t.fit_model(X_train, y_train)
                if model in ['SVM']:
                    train_preds = t.model.predict(X_train.T)
                    test_preds = t.model.predict(X_test.T)
                    val_preds = t.model.predict(X_val.T)
                    print('shape train preds {}, shape test preds {}'.format(train_preds.shape, test_preds.shape))
                elif model in ['KNN', 'DBSCAN', 'ISOF', 'LOF', 'SVM', 'RBM']:
                    train_preds = t.model.predict_proba(X_train)
                    test_preds = t.model.predict_proba(X_test)
                    val_preds = t.model.predict_proba(X_val)
                elif model in ['RNN', 'CNN', 'AUTOENCODER']:
                    train_preds = t.model(X_train)
                    test_preds = t.model(X_test)
                    val_preds = t.model(X_val)
                if self.labels is None:
                    ari, ss, dbs = self.prediction_metrics(X_train, train_preds, y_train)
                    mlflow.log_metric("ari", ari)
                    mlflow.log_metric("ss", ss)
                    mlflow.log_metric("dbs", dbs)
                else:
                    acc, f1_score, cohen_kappa, matthews_corr, roc_auc = self.prediction_metrics(X_train, train_preds, y_train) 
                    mlflow.log_metric("acc", acc)  
                    mlflow.log_metric("f1_score", f1_score)
                    mlflow.log_metric("cohen_kappa", cohen_kappa)
                    mlflow.log_metric("matthews_corr", matthews_corr)
                    mlflow.log_metric("roc_auc", roc_auc)
                self.print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
                #self.print_auto_logged_info(mlflow.get_run(run_id=0))#run_id=run.info.run_id))
    

        





