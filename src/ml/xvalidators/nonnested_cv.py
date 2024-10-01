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


class NonNestedRankingXVal(XValidator):
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
        self._name = 'nonnested cross validator'
        self._notation = 'nonnested_xval'

        
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def xval(self, x:list, y:list, demographics:list) -> dict:
        results = {}
        results['x'] = x
        # results['y'] = y
        logging.debug('x:{}, y:{}'.format(x, y))
        results['optim_scoring'] = self._xval_settings['nested_xval']['optim_scoring'] #debug
        self._outer_splitter.set_indices(range(len(y)))
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x, y, demographics)):
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            
            # division train / test
            x_train = [x[xx] for xx in train_index]
            y_train = [y[yy] for yy in train_index]
            x_test = [x[xx] for xx in test_index]
            y_test = [y[yy] for yy in test_index]
            print('train: {}, test: {}'.format(Counter(y_train), Counter(y_test)))
            
            # Inner loop
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            sampler_indices = self._sampler.get_indices()
            results[f]['oversample_index'] = [train_index[s_idx] for s_idx in sampler_indices]
            
            # Train
            model = self._model(self._settings)
            if self._settings['ml']['models']['save_best_model'] or self._settings['ml']['models']['early_stopping']:
                train_x, val_x, train_y, val_y = train_test_split(results[f]['oversample_index'], y_resampled, test_size=0.1, random_state=129)
                results[f]['train_index'] = [tx for tx in train_x]
                results[f]['validation_index'] = [vs for vs in val_x]
                train_x = [x[t_idx] for t_idx in train_x]
                val_x = [x[v_idx] for v_idx in val_x]
            else:
                train_x, train_y = x_resampled, y_resampled
                val_x, val_y = x_test, y_test

            model.set_outer_fold(f)
            history = model.fit(train_x, train_y, x_val=val_x, y_val=val_y)
            results[f]['best_params'] = model.get_settings()
            results[f]['loss_history'] = history

            if self._settings['ml']['models']['save_best_model']:
                results[f]['best_epochs'] = model.get_best_epochs()

            results[f]['best_estimator'] = model.save_fold(f)

            # Predict
            y_pred, y_test = model.predict(x_test, y_test)
            y_proba = model.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba)
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f]['y_test'] = y_test
            results[f].update(test_results)
            
            print(' Best Results on outer fold: {}'.format(test_results))
            self._model_notation = model.get_notation()
            self.save_results(results)
        return results
    
    def save_results(self, results):
        path = '../experiments/{}/results/'.format(
            self._experiment_name
        )
        os.makedirs(path, exist_ok=True)
        path += '{}_m{}_modelseeds{}.pkl'.format(
            self._notation, self._model_notation,
            self._settings['seeds']['model']
        )
        
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            