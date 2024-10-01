import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from features.sequencers.sequencer import Sequencer
from collections import Counter

class AssistmentSequencer(Sequencer):
    """Generate a matrix where the interaction is represented according to the sequencer's type.
    This one in particular handles the data parsed from the EDM2023 Cup

    Args:
        Sequencer (_type_): main class
    """

    def __init__(self, settings:dict):
        super().__init__(dict(settings))
        self._name = 'assistment_seq'
        self._notation = 'assissseq'
        self._dataset = 'edmcup_assistment'

        self._load_settings()
        self._get_label_map()

    def _load_settings(self):
        path = './configs/datasets/assistment.yaml'
        with open(path) as fp:
            setts = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['assistment'] = setts

        main_path = './configs/datasets/global_config.yaml'
        with open(main_path) as fp:
            main_setts = yaml.load(fp, Loader=yaml.FullLoader)
        self._settings['assistment'].update(main_setts)

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
        state_actions = []
        labels = []
        demographics = []
        indices = []

        for idx in sim_dictionary['sequences']:
            indices.append(idx)
            s, a = self._create_sequence(**sim_dictionary['sequences'][idx])
            l = self._get_label(sim_dictionary['sequences'][idx]['permutation'])
            d = self._get_demographics(**sim_dictionary['sequences'][idx])

            states.append(s)
            actions.append(a)
            labels.append(l)
            demographics.append(d)
            state_actions.append({'state': s, 'action': a})
        return state_actions, labels, demographics, indices

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



        



