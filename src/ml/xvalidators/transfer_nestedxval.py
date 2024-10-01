import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from ml.samplers.sampler import Sampler
from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.xvalidators.xvalidator import XValidator
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch

# from utils.config_handler import ConfigHandler

class TransferNestedXVal(XValidator):
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
        
    def _init_gs(self, fold, oversampled_indices):
        self._scorer.set_optimiser_function(self._xval_settings['nested_xval']['optim_scoring'])
        self._gs = self._gridsearch(
            model=self._model,
            grid=self._xval_settings['nested_xval']['paramgrid'],
            scorer=self._scorer,
            splitter = self._gs_splitter,
            settings=self._settings,
            outer_fold=fold,
        )

        
    def xval(
        self, x_primary:list, y_primary:list, demographics_primary:list, indices_primary:list,
        x_secundary:list, y_secundary:list, demographics_secundary:list, indices_secundary:list) -> dict:
        # indices will refer to the actual indices from id _dictionary
        # index are the indices from the splits
        results = {}
        results['x_primary'] = x_primary
        results['demographics_primary'] = demographics_primary
        results['y_primary'] = y_primary
        results['indices_primary'] = indices_primary
        results['x_secundary'] = x_secundary
        results['demographics_secundary'] = demographics_secundary
        results['y_secundary'] = y_secundary
        results['indices_secundary'] = indices_secundary
        logging.debug('x:{}, y:{}'.format(x_primary, y_primary))
        results['optim_scoring'] = self._xval_settings['nested_xval']['optim_scoring'] #debug
        self._outer_splitter.set_indices(indices_secundary)
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x_secundary, y_secundary, demographics_secundary)):
            logging.debug('outer fold, length train: {}, length test: {}'.format(len(train_index), len(test_index)))
            logging.debug('outer fold: {}'.format(f))
            logging.info('- ' * 30)
            logging.info('  Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['train_indices'] = [indices_secundary[idx] for idx in train_index]
            results[f]['test_index'] = test_index
            results[f]['test_indices'] = [indices_secundary[idx] for idx in test_index]
            
            # division train / test
            x_train = [x_secundary[xx] for xx in train_index]
            y_train = [y_secundary[yy] for yy in train_index]
            demographics_train = [demographics_secundary[dd] for dd in train_index]
            x_test = [x_secundary[xx] for xx in test_index]
            y_test = [y_secundary[yy] for yy in test_index]
            
            # Inner loop
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            results[f]['oversample_indexes'] = self._sampler.get_indices()
            results[f]['oversample_indices'] = [results[f]['train_indices'][idx] for idx in results[f]['oversample_indexes']]
            demographics_resampled = [demographics_train[idx] for idx in results[f]['oversample_indexes']]
    
            # Train
            self._init_gs(f, results[f]['oversample_indices'])
            #debuf
            self._gs.fit(
                x_primary, y_primary, demographics_primary,
                x_resampled, y_resampled, demographics_resampled, f
            )
            
            # Predict
            y_pred, y_test_score = self._gs.predict(x_test, y_test)
            y_proba = self._gs.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test_score, y_pred, y_proba)
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            results[f]['best_params'] = self._gs.get_best_model_settings()
            best_estimator = self._gs.get_best_model()
            results[f]['best_estimator'] = best_estimator.save_fold(f)
            results[f]['gridsearch_object'] = self._gs.get_path(f)
            logging.info('    best parameters: {}'.format(results[f]['best_params']))
            logging.info('    estimator path: {}'.format(results[f]['best_estimator']))
            logging.info('    gridsearch path: {}'.format(results[f]['gridsearch_object']))
            
            print('Best Results on outer fold: {}'.format(test_results))
            logging.info('Best Results on outer fold: {}'.format(test_results))
            self._model_notation = best_estimator.get_notation()
            self.save_results(results)
        return results
    
    def save_results(self, results):
        path = '{}/results/'.format(
            self._settings['experiment']['name']
        )
        os.makedirs(path, exist_ok=True)

        path += '{}_m{}_l{}_modelseeds{}_all_folds.pkl'.format(
            self._notation, self._model_notation, self._settings['data']['cropper'],
            self._settings['seeds']['model']
        )
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            