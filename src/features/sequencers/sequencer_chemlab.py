import pickle
import yaml
import numpy as np
import pandas as pd
import copy

from sklearn.preprocessing import MinMaxScaler
from features.sequencers.sequencer import Sequencer
from collections import Counter

class ChemLabSequencer(Sequencer):
    """Generate a matrix where the interaction is represented according to the sequencer's type.
    This one in particular handles the data parsed in Switzerland in Chemistry, Biology, Pharmacy and Chemistry and Textiles
    Vocational schools, with the beer's law lab.

    Args:
        Sequencer (_type_): main class
    """

    def __init__(self, settings:dict):
        super().__init__(dict(settings))
        self._name = 'chemlab_seq'
        self._notation = 'chemseq'
        self._dataset = 'chemlab_beerslaw'

        self._load_settings()
        self._get_label_map()

    def _load_settings(self):
        path = './configs/datasets/chemlab_config.yaml'
        with open(path) as fp:
            setts = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['chemlab'] = setts

        main_path = './configs/datasets/global_config.yaml'
        with open(main_path) as fp:
            main_setts = yaml.load(fp, Loader=yaml.FullLoader)
        self._settings['chemlab'].update(main_setts)


    def _get_label_map(self):
        root_path = '{}/experiment_keys/chemlab_beerslaw_'.format(self._settings['paths']['data'])
        if self._settings['chemlab']['data']['label_map'] == 'binconcepts':
            path = '{}binconcepts.yaml'.format(root_path)

        with open(path) as fp:
            self._label_map = yaml.load(fp, Loader=yaml.FullLoader)

        vb_path = '{}/experiment_keys/chemlab_beerslaw_vector_binary.yaml'.format(self._settings['paths']['data'])
        with open(vb_path, 'r') as f:
            self._vector_binary_map = yaml.load(f, Loader=yaml.FullLoader)

    def load_all_sequences(self):
        """Functions returning information about the ChemLab experiment conducted in vocational schools with the
        Beer's Law Lab Phet Simulation.

        Return:
            states (list): processed interaction, only the state features
            actions (list): processed interaction list, only the action features
            demographics (list<dict>): list of demographical dictionaries 


        """
        path = '{}{}sim_dictionary.pkl'.format(self._settings['paths']['data'], self._settings['chemlab']['paths']['root'])
        with open(path, 'rb') as fp:
            sim_dictionary = pickle.load(fp)
        states = []
        actions = []
        actions_scale = []
        labels = []
        demographics = []
        indices = []


        for idx in sim_dictionary['sequences']:
            indices.append(idx)
            s, a = self._create_sequence(**sim_dictionary['sequences'][idx])
            l = self._get_label(sim_dictionary['sequences'][idx]['permutation'])
            d = self._get_demographics(**sim_dictionary['sequences'][idx])
            d['dataset'] = 'chemlab'
            d['dataset_label'] = '{}-{}'.format(int(l), d['dataset'])

            states.append(s)
            actions.append(a)
            actions_scale = [*actions_scale, *a]
            labels.append(l)
            demographics.append(copy.deepcopy(d))

        sequences = [
            [[*states[student][ts], *actions[student][ts]] for ts in range(len(states[student]))] 
        for student in range(len(states))]
        sequences = self._scale_features(states, actions, actions_scale)
        return sequences, labels, demographics, indices

    def _get_label(self, permutation:str):
        """Returns the processed label based on the ranking selected
        """
        return self._label_map['map'][permutation]

    def _get_demographics(self, **kwargs):
        """Returns the permutation for the permutation stratification
        """
        demog = {demo: kwargs[demo] for demo in self._settings['chemlab']['data']['available_demographics']}
        demog['vector_binary'] = self._vector_binary_map['map'][kwargs['permutation']]
        return demog

        
    def _create_sequence(self, **kwargs):
        """Function to create the sequences under the appropriate format

        Args:
            actions (_type_): chronological list of actions in the interaction
            begins (_type_): beginning timestamp of each of the action
            ends (_type_): end timestamp of each of the action
            simulation_values (_type_): non interaction values taken throughout the simulation (state, tec.)
            xaxis (_type_): _description_
            yaxis (_type_): _description_
            trials (_type_): _description_
        """

        raise NotImplementedError



        



