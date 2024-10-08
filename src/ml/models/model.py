from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

import torch.nn.functional as F
import tensorflow as tf
import torch
from shutil import copytree, rmtree

from copy import deepcopy

from typing import Tuple
class Model:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._name = 'model'
        self._notation = 'm'
        self._settings = dict(settings)
        self._experiment_root = settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._n_classes = settings['experiment']['nclasses']
        self._random_seed = settings['seeds']['model']

        self._gs_fold = 0

    def get_name(self):
        """Returns the name of the model, useful when debugging
        """
        return self._name

    def get_best_epochs(self):
        return self._best_epochs

    def get_notation(self):
        """Shorter name of the model. Especially used when saving files in patah which
        contain the name of the models.
        """
        return self._notation

    def _set_seed(self):
        """Set the seed for the parameters initialisation or anything else
        """
        torch.manual_seed(self._settings['seeds']['model'])

    def set_gridsearch_parameters(self, params, combinations):
        """When using a gridsearch, the model uses this function to update
        its arguments. 
        """
        logging.debug('Gridsearch params: {}'.format(params))
        logging.debug('Combinations: {}'.format(combinations))
        print('    ', params, combinations)
        for i, param in enumerate(params):
            logging.debug('  index: {}, param: {}'.format(i, param))
            self._model_settings[param] = combinations[i]

    def set_gridsearch_fold(self, fold:int):
        """Used to save the model under a specific name
        """
        self._gs_fold = fold

    def set_outer_fold(self, fold:int):
        self._outer_fold = fold
            
    def get_settings(self):
        return dict(self._model_settings)

    
    def _format_final_simulation(self, x, y):
        raise NotImplementedError

    def _choose_format_final(self) -> Tuple[list, list]:
        """Returns the features and the target for the final objective
        Args:
            x (list): _description_

        Returns:
            Tuple[list, list]:
                features
                targets
                lengths (in some datasets, the sequences are of different lengths)
        """
        if self._settings['data']['type'] == 'simulation':
            self._format_final = self._format_final_simulation

    def _format_features_simulation(self, x, y):
        raise NotImplementedError

    def _choose_format_features(self) -> Tuple[list, list]:
        """Returns the features and the target for the final objective
        Args:
            x (list): _description_

        Returns:
            Tuple[list, list]:
                features
                targets
                lengths (in some datasets, the sequences are of different lengths)
        """
        if self._settings['data']['type'] == 'simulation':
            self._format_features = self._format_features_simulation

    def _format(self, x: list, y: list) -> Tuple[list, list]:
        """formats the data into list or numpy array according to the library the model comes from

        Args:
            x (list): features
            y (list): labels

        Returns:
            x: formatted features
            y: formatted labels
        """
        raise NotImplementedError

    def _categorical_vector(self, class_idx: int):
        vector = list(np.zeros(self._settings['experiment']['nclasses']))
        vector[class_idx] = 1
        return vector
        
    def _format_label_categorical(self, y:list):
        new_y = [self._categorical_vector(idx) for idx in y]
        return new_y
    
    def _format_features(self, x: list) -> list:
        """formats the data into list or numpy array according to the library the model comes from

        Args:
            x (list): features

        Returns:
            x: formatted features
        """
        raise NotImplementedError

    def _get_model_checkpoint_path(self) -> str:
        _, checkpoint_path = self._get_csvlogger_path()
        return checkpoint_path

    def load_model_weights(self, x:np.array, checkpoint_path:str):
        """Given a data point x, this function sets the model of this object
        Args:
            x ([type]): [description]
        Raises:
            NotImplementedError: [description]
        """
        x = self._format_features(x) 
        self._init_model(x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
            copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)

    def load_priormodel_weights(self, x:np.array, checkpoint_path:str):
        """Given a data point x, this function sets the model of this object
        Args:
            x ([type]): [description]
        Raises:
            NotImplementedError: [description]
        """
        priors, x = self._format_prior_features(x) 
        self._init_model(priors, x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
            copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)

    def _init_model(self):
        """Initiates a model with self._model
        """
    
    def fit(self, x_train: list, y_train: list, x_val: list, y_val:list):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        raise NotImplementedError
    
    def predict(self, x: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        raise NotImplementedError

    def _inpute_full_prob_vector(self, y_pred:list, y_probs:list) -> list:
        """Sometimes, during nested cross validation, samples from minority classes are missing. The probability vector is thus one cell too short. However, we can recover the mapping position -> original label via the predict function

        Returns:
            list: new probability vector, where the number of cell is the 
        """
        if len(y_probs[0]) == self._n_classes:
            return y_probs

        label_map = {cl:[] for cl in range(self._n_classes)}
        prob_index = [np.argmax(y) for y in y_probs]
        prob_value = [max(y) for y in y_probs]
        
        for index in range(len(y_probs)):
            if prob_value[index] > 0.5:
                label_map[prob_index[index]].append(y_pred[index])

        # print(label_map)
        # print(y_pred)
        # print(y_probs)
                
        new_map = {cl:np.unique(label_map[cl]) for cl in range(self._n_classes)}
        new_probs = np.zeros((len(y_probs), self._n_classes))
        for label in new_map:
            assert len(new_map[label]) <= 1
            
        for index, prob in enumerate(y_probs):
            for i in range(len(prob)):
                new_probs[index][new_map[i]] = prob[i]
            
        return new_probs

    ############ SKLEARN
    def predict_sklearn(self, x:list) -> list:
        x_predict = self._format_features(x)
        return self._model.predict(x_predict)
    
    def predict_proba_sklearn(self, x:list) -> list:
        x_predict = self._format_features(x)
        probs = self._model.predict_proba(x_predict)
        if len(probs[0]) != self._n_classes:
            preds = self._model.predict(x_predict)
            probs = self._inpute_full_prob_vector(preds, probs)
        return probs

    def save_sklearn(self):
        path = '{}/models/'.format(self._experiment_name)
        os.makedirs(path, exist_ok=True)
        path +=  '{}_l{}_f{}.pkl'.format(
            self._name, self._settings['data']['adjuster']['limit'], self._fold
        )
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path

    def get_path_sklearn(self, fold:int) -> str:
        path = '{}/models/'.format(self._experiment_name)
        path +=  '{}_l{}_f{}.pkl'.format(
            self._name, self._settings['data']['adjuster']['limit'], self._fold
        )
        return path

    def save_fold_sklearn(self, fold: int) -> str:
        path = '{}/models/'.format(self._experiment_name)
        os.makedirs(path, exist_ok=True)
        path +=  '{}_l{}_f{}.pkl'.format(
            self._name, self._settings['data']['adjuster']['limit'], self._fold
        )
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path

    def save_fold_early_sklearn(self, fold: int) -> str:
        path = '{}/models/'.format(self._experiment_name)
        path +=  '{}_l{}_f{}_l{}'.format(
            self._name, self._settings['data']['adjuster']['limit'], self._fold, self._maxlen
        )
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path

    ############ TENSORFLOW
    def predict_tensorflow(self, x:list) -> list:
        x_predict = self._format_features(x)
        predictions = self._model.predict(x_predict)
        predictions = [np.argmax(x) for x in predictions]
        print(predictions)
        return predictions
    
    def predict_proba_tensorflow(self, x:list) -> list:
        x_predict = self._format_features(x)
        probs = self._model.predict(x_predict)
        if len(probs[0]) != self._n_classes:
            preds = self._model.predict(x_predict)
            preds = [np.argmax(x) for x in preds]
            probs = self._inpute_full_prob_vector(preds, probs)
        return probs
    
    def save_tensorflow(self) -> str:
        path = '{}/models/{}/'.format(
            self._experiment_name, self._notation
        )
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        self._model = path
        path = '{}/lstm.history.pkl'.format(self._experiment_name)
        with open(path, 'wb') as fp:
            pickle.dump(self._history.history, fp)
        return path
    
    def get_path_tensorflow(self, fold: int) -> str:
        path = '{}/models/{}/'.format(
            self._experiment_name, self._notation
        )
        return path
            
    def save_fold_tensorflow(self, fold: int) -> str:
        path = '{}/models/{}_f{}/'.format(
            self._experiment_name, self._notation, fold
        )
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path

    def save_fold_early_tensorflow(self, fold: int) -> str:
        path = '{}/models/{}_f{}_l{}/'.format(
            self._experiment_name, self._notation, fold, self._maxlen
        )
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path

    ############ PYTROCH
    # def predict_torch(self, x:list) -> list:
    #     probabilities = []
    #     x_tensor = self._format_features(x)
    #     test_loader = DataLoader(
    #         TensorDataset(x_tensor, x_tensor), shuffle=self._model_settings['shuffle'],
    #         batch_size=self._model_settings['batch_size']
    #     )
    #     self._model = self._model.eval()
    #     for batch_idx, (features, labels) in enumerate(test_loader):
    #         with torch.no_grad():
    #             val_probs = self._model(features)
    #             probabilities = probabilities + [val_probs]
    #     print('predictions', len(probabilities), probabilities[0].shape)
    #     predictions = [np.argmax(prob) for prob in probabilities]
    #     return predictions
        
    def predict_proba_torch(self, x:list) -> list:
        probabilities = []
        x_tensor, ls = self._format_features(x)
        self._model = self._model.eval()
        probabilities = self._model(x_tensor, ls)
        return probabilities

    def _load_weights_torch(self, path):
        self.model = deepcopy(torch.load(path, map_location=torch.device('cpu')))

    def predict_proba(self, x:list) -> list:
        """Predict the probabilities of each label for x

        Args:
            x (list): features

        Returns:
            list: list of probabilities for each data point
        """
        raise NotImplementedError
    
    def save(self) -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
    
    def save_fold(self, fold) -> str:
        """Saving the model for a specific fold in the following path:

        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError

        
