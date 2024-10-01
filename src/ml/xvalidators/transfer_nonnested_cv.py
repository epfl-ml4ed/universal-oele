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


class TransferNonNestedRankingXVal(XValidator):
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
        print('HEYA')
        self._name = 'transfernonnested cross validator'
        self._notation = 'trnonnested_xval'

        
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._sampler = sampler()
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def xval(self, x_primary:list, y_primary:list, demographics_primary:list, indices_primary:list,
        x_secundary:list, y_secundary:list, demographics_secundary:list, indices_secundary:list) -> dict:
        
        results = {}
        results['settings'] = self._settings
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

        primary_model = self._model(self._settings)
        primary_model.update_settings(self._settings['ml']['transfer']['primary'])
        primary_model.fit(x_primary, y_primary, x_primary, y_primary) # validations are not used
        primary_model.save('primary')
        primary_weights = primary_model.get_model_path()

        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x_secundary, y_secundary, demographics_secundary)):
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            
            # division train / test
            x_train = [x_secundary[xx] for xx in train_index]
            y_train = [y_secundary[yy] for yy in train_index]
            x_test = [x_secundary[xx] for xx in test_index]
            y_test = [y_secundary[yy] for yy in test_index]
            print('train: {}, test: {}'.format(Counter(y_train), Counter(y_test)))
            
            # Inner loop
            x_resampled, y_resampled = self._sampler.sample(x_train, y_train)
            sampler_indices = self._sampler.get_indices()
            results[f]['oversample_index'] = [train_index[s_idx] for s_idx in sampler_indices]
            
            # Train
            model = self._model(self._settings)
            model.set_outer_fold(f)
            model.update_settings(self._settings['ml']['transfer']['secundary'])
            model.init_model()
            naked_predictions, y_val_pred = model.predict(x_test, y_test)
            naked_probas = model.predict_proba(x_test)
            naked_score = self._scorer.get_scores(y_val_pred, naked_predictions, naked_probas)
            print('no-training: {}'.format(naked_score))
            results[f]['naked_training'] = {
                'predictions': naked_predictions,
                'probabilities': naked_probas
            }

            model.load_weights(primary_weights, transfer=True)
            transfer_predictions, y_val_pred = model.predict(x_test, y_test)
            transfer_probas = model.predict_proba(x_test)
            transfer_score = self._scorer.get_scores(y_val_pred, transfer_predictions, transfer_probas)
            results['transfer_training'] = {
                'predictions': transfer_predictions,
                'probabilities': transfer_probas,
                'score': transfer_score
            }
            print('pretrained scores: {}'.format(transfer_score))
            model.transfer()
            results[f]['loss_history'] = model.fit(x_train, y_train, x_val=x_test, y_val=y_test)

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
            
            