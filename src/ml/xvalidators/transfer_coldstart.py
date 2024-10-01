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

class TransferColdStartXVal(XValidator):
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
        self._settings['ml']['models']['early_stopping'] = True
        results = {0:{}}
        results['x_primary'] = x_primary
        results['demographics_primary'] = demographics_primary
        results['y_primary'] = y_primary
        results['indices_primary'] = indices_primary
        results['x_secundary'] = x_secundary
        results['demographics_secundary'] = demographics_secundary
        results['y_secundary'] = y_secundary
        results['indices_secundary'] = indices_secundary
        primary_model = self._model(self._settings)
        train_x, val_x, train_y, val_y = train_test_split(x_primary, y_primary, test_size=0.1, random_state=129)
        results[0]['model_train_x'] = train_x
        results[0]['model_train_y'] = train_y
        results[0]['model_val_x'] = val_x
        results[0]['model_val_y'] = val_y


        primary_model.fit(train_x, train_y, val_x, val_y) # validations are not used
        primary_model.save('primary')
        coldstarted_predictions, _ = primary_model.predict(x_secundary, y_secundary)
        coldstarted_probas = primary_model.predict_proba(x_secundary)
        coldstarted_results = self._scorer.get_scores(y_secundary, coldstarted_predictions, coldstarted_probas)

        results[0]['y_pred'] = coldstarted_predictions
        results[0]['y_proba'] = coldstarted_probas
        results[0].update(coldstarted_results)
        
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
            
            