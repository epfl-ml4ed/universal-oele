import os
import numpy as np
import copy
import pickle
import yaml
from sklearn.preprocessing import MinMaxScaler
from features.sequencers.sequencer import Sequencer

class InstructionBeerslawSequencer(Sequencer):
    def __init__(self, settings):
        self._settings = dict(settings)
        self._name = 'instructbeerbeersseq'
        self._notation = 'ibseq'

        self._load_settings()

        self._demographics = [
            'lid', 'gender', 'year', 'label', 'post_test', 'instruction_type', 's', 'a'
        ]

    def _load_settings(self):
        print('STARTING')
        path = './configs/datasets/instruction_beerslaw.yaml'
        with open(path) as fp:
            setts = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['instruction_beerslaw'] = setts

        main_path = './configs/datasets/global_config.yaml'
        with open(main_path) as fp:
            main_setts = yaml.load(fp, Loader=yaml.FullLoader)
        self._settings.update(main_setts)

    def load_all_sequences(self):
        """Functions returning information about the ChemLab experiment conducted in vocational schools with the
        Beer's Law Lab Phet Simulation.

        Return:
            states (list): processed interaction, only the state features
            actions (list): processed interaction list, only the action features
            demographics (list<dict>): list of demographical dictionaries 


        """
        print(os.listdir('{}{}'.format(self._settings['paths']['data'], self._settings['instruction_beerslaw']['paths']['root'])))
        path = '{}{}sim_dict.pkl'.format(self._settings['paths']['data'], self._settings['instruction_beerslaw']['paths']['root'])
        with open(path, 'rb') as fp:
            sim_dictionary = pickle.load(fp)

        states = []
        actions = []
        actions_scale = []
        labels = []
        demographics = []
        indices = []

        for idx in sim_dictionary['index']['index_lid']:
            # with open(sim_dictionary['sequences'][idx]['path'].replace('../data', self._settings['paths']['data']), 'rb') as fp:
            #     sim_beerslaw = pickle.load(fp)
            # s, a = self.process_sequence(sim_beerslaw['logs'])
            s = sim_dictionary['sequences'][idx]['s']
            a = sim_dictionary['sequences'][idx]['a']
            merged_s = sim_dictionary['sequences'][idx]['merged_s']
            merged_a = sim_dictionary['sequences'][idx]['merged_a']
            s, a = self._create_vector_sequence(s, a, merged_s, merged_a)
            if len(s) == 0:
                continue
            indices.append(idx)
            l = sim_dictionary['sequences'][idx]['label']
            d = self._get_demographics(**sim_dictionary['sequences'][idx])
            d['dataset'] = 'vetbeer'
            d['dataset_label'] = '{}-{}'.format(int(l), d['dataset'])

            states.append(s)
            actions.append(a)
            actions_scale = [*actions_scale, *a]
            labels.append(l)
            demographics.append(copy.deepcopy(d))
        
        assert len(labels) == len(demographics) and len(demographics) == len(indices)
        sequences = self._scale_features(states, actions, actions_scale)
        print(max([len(alls) for alls in sequences]))
        return sequences, labels, demographics, indices

    def _get_demographics(self, **kwargs):
        """Returns the permutation for the permutation stratification
        """
        return {d: kwargs[d] for d in self._demographics}
        
    def process_sequence(self, **kwargs):
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



        



