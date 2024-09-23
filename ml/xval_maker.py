import yaml
import logging
# from ml.gridsearches.transfer_gridsearch import TransferSupervisedGridSearch
from ml.models.attentionrnn import AttentionRNNClassifier

from ml.models.lstm_transfer import LSTMTransferTorchModel
from ml.models.attentionrnn import AttentionRNNClassifier
from ml.models.rnn_attention import RNNAttentionClassifier

from ml.samplers.no_sampler import NoSampler

from ml.scorers.binaryclassification_scorer import BinaryClfScorer
from ml.scorers.multiclassification_scorer import MultiClfScorer

from ml.splitters.splitter import Splitter
from ml.splitters.stratified_kfold import StratifiedKSplit

from ml.xvalidators.nonnested_cv import NonNestedRankingXVal
from ml.xvalidators.nested_cv import NestedXVal
from ml.xvalidators.transfer_mix import TransferMixXVal
from ml.xvalidators.transfer_nestedxval import TransferNestedXVal
from ml.xvalidators.transfer_nonnested_cv import TransferNonNestedRankingXVal
from ml.xvalidators.transfer_coldstart import TransferColdStartXVal

from ml.gridsearches.transfer_gridsearch import TransferSupervisedGridSearch
from ml.gridsearches.supervised_gridsearch import SupervisedGridSearch

from ml.splitters.flat_stratified import FlatStratified
from ml.splitters.one_fold import OneFoldSplit

class XValMaker:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._pipeline_settings = self._settings['ml']['pipeline']
        
        self._build_pipeline()
        

    def get_gridsearch_splitter(self):
        return self._gs_splitter

    def get_sampler(self):
        return self._sampler

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self, splitter:str) -> Splitter:
        if splitter == 'stratkf':
            return StratifiedKSplit
        if splitter == 'flatstrat':
            return FlatStratified
        if splitter == '1kfold':
            return OneFoldSplit
    
    def _choose_inner_splitter(self): # only for nested xval
        self._inner_splitter = self._choose_splitter(self._pipeline_settings['inner_splitter'])

    def _choose_outer_splitter(self):
        self._outer_splitter = self._choose_splitter(self._pipeline_settings['outer_splitter'])

    def _choose_gridsearch_splitter(self):
        self._gs_splitter = self._choose_splitter(self._pipeline_settings['gs_splitter'])
            
    def _choose_sampler(self):
        if self._pipeline_settings['sampler'] == 'nosplr':
            self._sampler = NoSampler
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['model']))
        if self._pipeline_settings['model'] == 'lstm_transfer':
            self._model = LSTMTransferTorchModel
            gs_path = './configs/gridsearch/gs_lstm.yaml'

        if self._pipeline_settings['model'] == 'attentionrnn':
            self._model = AttentionRNNClassifier
            gs_path = './configs/gridsearch/gs_attentionrnn.yaml'

        if self._pipeline_settings['model'] == 'rnn_attention':
            self._model = RNNAttentionClassifier
            gs_path = './configs/gridsearch/gs_attentionrnn.yaml'
        
            
        if self._settings['ml']['pipeline']['gridsearch'] != 'nogs':
            with open(gs_path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['ml']['xvalidators']['nested_xval']['paramgrid'] = gs
                print(gs)
                    
    def _choose_scorer(self):
        if self._pipeline_settings['scorer'] == '2clfscorer':
            self._scorer = BinaryClfScorer
        elif self._pipeline_settings['scorer'] == 'multiclfscorer':
            self._scorer = MultiClfScorer

    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'supgs' and self._settings['transfer']:
            self._gridsearch = TransferSupervisedGridSearch
        else:
            self._gridsearch = SupervisedGridSearch
            
    def _choose_xvalidator(self):
        self._choose_gridsearcher()
        if self._pipeline_settings['xvalidator'] == 'nonnested_xval' and self._settings['baseline']:
            self._xval = NonNestedRankingXVal
            # self._xval = TransferNonNestedRankingXVal

        elif self._pipeline_settings['xvalidator'] == 'nested_xval' and self._settings['baseline']:
            self._xval = NestedXVal

        elif self._settings['coldstart']:
            self._xval = TransferColdStartXVal

        elif self._settings['mix']:
            self._xval = TransferMixXVal
        
        elif self._pipeline_settings['xvalidator'] == 'nested_xval' and self._settings['transfer']:
            self._xval = TransferNestedXVal

        elif self._pipeline_settings['xvalidator'] == 'nonnested_xval' and self._settings['transfer']:
           
            self._xval = TransferNonNestedRankingXVal
        

        self._xval = self._xval(self._settings, self._gridsearch, self._gs_splitter, self._outer_splitter, self._sampler, self._model, self._scorer)
    
    def _train_non_gen(self, X:list, y:list, demographics:list, indices:list):
        results = self._xval.xval(X, y, demographics)
        return results

    def _train_transfer(self, 
        X_primary:list, y_primary:list, demographics_primary:list, indices_primary:list,
        X_secundary:list, y_secundary:list, demographics_secundary:list, indices_secundary:list,
    ):
        results = self._xval.xval(
            X_primary, y_primary, demographics_primary, indices_primary,
            X_secundary, y_secundary, demographics_secundary, indices_secundary
        )
        return results

    def _choose_train(self):
        if self._settings['transfer']:
            self.train = self._train_transfer
        else:
            self.train = self._train_non_gen

    def _build_pipeline(self):
        # self._choose_splitter()
        # self._choose_inner_splitter()
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_sampler()
        self._choose_model()
        self._choose_scorer()
        self._choose_xvalidator()
        self._choose_train()
        
    
        