import logging
from xml.dom.expatbuilder import makeBuilder

import numpy as np

from ml.models.model import Model
from ml.splitters.splitter import Splitter
from ml.scorers.scorer import Scorer
from ml.gridsearches.gridsearch import GridSearch
from sklearn.model_selection import train_test_split

class TransferSupervisedGridSearch(GridSearch):
    """
    Gridsearch where the folds are stratified by the label

    """
    def __init__(self, model:Model, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(model, grid, scorer, splitter, settings, outer_fold)
        self._name = 'transfersupervised gridsearch'
        self._notation = 'tsupgs'

        self._folds = {}
        
    def fit(
            self, x_primary:list, y_primary:list, demographics_primary: list,
            x_secundary:list, y_secundary:list, demographics_secundary: list, fold:int
        ):
        for _, combination in enumerate(self._combinations):
            combination_string = '_'.join([str(comb) for comb in self._combinations])
            print('Testing parameters: {}'.format(combination))
            folds = []
            fold_indices = {}
            splitter = self._splitter(self._settings)

            primary_model = self._model(self._settings)
            primary_model.set_outer_fold(self._outer_fold)
            primary_model.set_gridsearch_parameters(self._parameters, combination)
            primary_model.set_gridsearch_fold(fold)
            primary_model.update_settings(self._settings['ml']['transfer']['primary'])
            if self._settings['ml']['transfer']['primary']['early_stopping']:
                x_primary_train, x_primary_validation, y_primary_train, y_primary_validation = train_test_split(
                    x_primary, y_primary, test_size=0.2, random_state=self._settings['seeds']['splitter']
                )
            else:
                x_primary_train, y_primary_train, x_primary_validation, y_primary_validation = x_primary, y_primary, x_primary, y_primary
            primary_model.fit(x_primary_train, y_primary_train, x_primary_validation, y_primary_validation) # validations are not used
            primary_model.save('primary')
            primary_weights = primary_model.get_model_path()
            for f, (train_index, validation_index) in enumerate(splitter.split(x_secundary, y_secundary, demographics_secundary)):
                fold_indices[f] = {}

                logging.debug('    inner fold, train length: {}, test length: {}'.format(len(train_index), len(validation_index)))
                x_val = [x_secundary[xx] for xx in validation_index]
                y_val = [y_secundary[yy] for yy in validation_index]
                x_train = [x_secundary[xx] for xx in train_index]
                y_train = [y_secundary[yy] for yy in train_index]

                model = self._model(self._settings)
                model.set_outer_fold(f)
                model.set_gridsearch_parameters(self._parameters, combination)
                model.update_settings(self._settings['ml']['transfer']['secundary'])
                model.init_model()
                naked_predictions, y_val_pred = model.predict(x_val, y_val)
                naked_probas = model.predict_proba(x_val)
                naked_score = self._scoring_function(y_val_pred, naked_predictions, naked_probas)
                fold_indices[f]['naked_training'] = {
                    'predictions': naked_predictions,
                    'probabilities': naked_probas
                }
                fold_indices[f]['naked_training']['score'] = naked_score
                print('Naked score: {}'.format(naked_score))

                # Score when pre-trainged with another dataset
                model.load_weights(primary_weights, transfer=True)
                transfer_predictions, y_val_pred = model.predict(x_val, y_val)
                transfer_probas = model.predict_proba(x_val)
                transfer_score = self._scoring_function(y_val_pred, transfer_predictions, transfer_probas)
                fold_indices[f]['transfer_training'] = {
                    'predictions': transfer_predictions,
                    'probabilities': transfer_probas
                }
                fold_indices[f]['transfer_training']['score'] = transfer_score
                print('transfer Scores: {}'.format(transfer_score))
                
                if model.get_settings()['save_best_model'] or model.get_settings()['early_stopping']:
                    train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.1, random_state=129)
                    fold_indices[f]['model_train_x'] = train_x
                    fold_indices[f]['model_train_y'] = train_y
                    fold_indices[f]['model_val_x'] = val_x
                    fold_indices[f]['model_val_y'] = val_y
                else:
                    train_x, train_y = x_train, y_train
                    val_x, val_y = x_val, y_val

                # Score when fine tuned 
                model.set_outer_fold(f)
                model.transfer()
                model.fit(train_x, train_y, x_val=val_x, y_val=val_y)

                y_pred, y_val_pred = model.predict(x_val, y_val)
                y_proba = model.predict_proba(x_val)
                
                score = self._scoring_function(y_val_pred, y_pred, y_proba)
                print('Fine tuned scores: {}'.format(score))
                logging.info('    Score for fold {}: {} {}'.format(f, score, self._scoring_name))
                folds.append(score)
                fold_indices[f]['train'] =  train_index
                fold_indices[f]['validation'] = validation_index

            self._add_score(combination, folds, fold_indices)
            self.save(fold)
            
        best_parameters = self.get_best_model_settings()
        combinations = []
        for param in self._parameters:
            combinations.append(best_parameters[param])
            
        config = dict(self._settings)
        model = self._model(config)
        model.set_gridsearch_parameters(self._parameters, combinations)
        model.update_settings(self._settings['ml']['transfer']['secundary'])
        model.init_model()
        model.load_weights(primary_weights, transfer=True)
        model.set_outer_fold(self._outer_fold)
        model.transfer()
        if model.get_settings()['save_best_model'] or model.get_settings()['early_stopping']:
            x_train, x_val, y_train, y_val = train_test_split(x_secundary, y_secundary, test_size=0.1, random_state=129)
        else:
            x_train, x_val, y_train, y_val = x_secundary, x_secundary, y_secundary, y_secundary

        self.loss_history = model.fit(x_train, y_train, x_val, y_val)

        # model.save(extension='best_model_f{}'.format(fold))
        self._best_model = model
        
            
    def predict(self, x_test:list, y_test:list) -> list:
        return self._best_model.predict(x_test, y_test)
        
        
    def predict_proba(self, x_test:list) -> list:
        return self._best_model.predict_proba(x_test)
