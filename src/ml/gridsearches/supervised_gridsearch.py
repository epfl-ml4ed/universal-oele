import os
import copy
import pickle
import logging
import itertools

import numpy as np
import pandas as pd
from six import b

from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch

class SupervisedGridSearch(GridSearch):
    """
    Gridsearch where the folds are stratified by the label

    """
    def __init__(self, model:Model, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(model, grid, scorer, splitter, settings, outer_fold)
        self._name = 'supervised gridsearch'
        self._notation = 'supgs'

        self._folds = {}
        
    def fit(self, x_train:list, y_train:list, demographics: list, fold:int):
        self.loss_histories = {}
        for _, combination in enumerate(self._combinations):
            self.loss_histories['_'.join([str(c) for c in combination])] = {}
            logging.info('  Testing parameters: {}'.format(combination))
            folds = []
            fold_indices = {}
            splitter = self._splitter(self._settings)
            for f, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train, demographics)):
                self.loss_histories['_'.join([str(c) for c in combination])][f] = []
                logging.debug('    inner fold, train length: {}, test length: {}'.format(len(train_index), len(validation_index)))
                x_val = [x_train[xx] for xx in validation_index]
                y_val = [y_train[yy] for yy in validation_index]
                xx_train = [x_train[xx] for xx in train_index]
                yy_train = [y_train[yy] for yy in train_index]

                model = self._model(self._settings)
                model.set_outer_fold(self._outer_fold)
                model.set_gridsearch_parameters(self._parameters, combination)
                model.set_gridsearch_fold(f)
                history = model.fit(xx_train, yy_train, x_val=x_val, y_val=y_val)
                self.loss_histories['_'.join([str(c) for c in combination])][f].append(history)
                
                y_pred, y_val = model.predict(x_val, y_val)
                y_proba = model.predict_proba(x_val)
                
                score = self._scoring_function(y_val, y_pred, y_proba)
                logging.info('    Score for fold {}: {} {}'.format(f, score, self._scoring_name))
                folds.append(score)
                fold_indices[f] = {
                    'train': train_index,
                    'validation': validation_index
                }
            self._add_score(combination, folds, fold_indices)
            self.save(fold)
            
        best_parameters = self.get_best_model_settings()
        combinations = []
        for param in self._parameters:
            combinations.append(best_parameters[param])
            
        config = copy.deepcopy(self._settings)
        model = self._model(config)
        model.set_outer_fold(self._outer_fold)
        model.set_gridsearch_parameters(self._parameters, combinations)
        model.fit(x_train, y_train, x_val, y_val)
        model.save(extension='best_model_f{}'.format(fold))
        self._best_model = model
        
            
    def predict(self, x_test: list, y_test) -> list:
        return self._best_model.predict(x_test, y_test)
        
        
    def predict_proba(self, x_test:list) -> list:
        return self._best_model.predict_proba(x_test)
