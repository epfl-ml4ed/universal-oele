import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.xvalidators.xvalidator import XValidator
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch
from sklearn.model_selection import train_test_split

# from utils.config_handler import ConfigHandler

class TransferMixXVal(XValidator):
    """Implements nested cross validation: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                perform gridsearch on the chosen model, and choose the best model according to the validation set
                Predict the test set on the best model according to the gridsearch
            => Outer loop computes the performances on the test set
            => Inner loop selects the best model for that fold

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(self, settings:dict, gridsearch:GridSearch, gridsearch_splitter: Splitter, outer_splitter: Splitter, sampler:Sampler, model:Model, scorer:Scorer):
        super().__init__(settings, model, scorer)
        self._name = 'nested cross validator'
        self._notation = 'nested_xval'
        
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        self._init_gs(0)

    def _init_gs(self, fold):
        self._scorer.set_optimiser_function(self._xval_settings['nested_xval']['optim_scoring'])
        self._gs = self._gridsearch(
            model=self._model,
            grid=self._xval_settings['nested_xval']['paramgrid'],
            scorer=self._scorer,
            splitter = self._gs_splitter,
            settings=self._settings,
            outer_fold=fold,
        )
        self._combinations = self._gs._combinations
        self._parameters = self._gs._parameters

        
    def xval(
        self, x_primary:list, y_primary:list, demographics_primary:list, indices_primary:list,
        x_secundary:list, y_secundary:list, demographics_secundary:list, indices_secundary:list) -> dict:
        self._settings['ml']['models']['early_stopping'] = False
        results = {}
        results['x_primary'] = x_primary
        results['demographics_primary'] = demographics_primary
        results['y_primary'] = y_primary
        results['indices_primary'] = indices_primary
        results['x_secundary'] = x_secundary
        results['demographics_secundary'] = demographics_secundary
        results['y_secundary'] = y_secundary
        results['indices_secundary'] = indices_secundary

        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x_secundary, y_secundary, demographics_secundary)):
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            primary_model = self._model(self._settings)

            x_train = [x_secundary[ti] for ti in train_index] + x_primary
            x_test = [x_secundary[ti] for ti in test_index]
            y_train = [y_secundary[ti] for ti in train_index] + y_primary
            y_test = [y_secundary[ti] for ti in test_index]
            
            demo_train = [demographics_secundary[tr]['dataset_label'] for tr in train_index]
            demo_test = [demographics_secundary[te]['dataset_label'] for te in test_index]
            print('train sep: {}, test sep: {}'.format(Counter(demo_train), Counter(demo_test)))

            primary_model.fit(x_train, y_train, x_test, y_test) # validations are not used
            primary_model.save('primary')
            test_predictions, _ = primary_model.predict(x_test, y_test)
            test_probas = primary_model.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, test_predictions, test_probas)

            results[f]['y_pred'] = test_predictions
            results[f]['y_proba'] = test_probas
            results[f].update(test_results)
            
            self.save_results(results)
        return results
    
    def save_results(self, results):
        path = '{}/results/'.format(
            self._settings['experiment']['name']
        )
        os.makedirs(path, exist_ok=True)

        path += '{}_modelseeds{}_all_folds.pkl'.format(
            self._notation,
            self._settings['seeds']['model']
        )
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            