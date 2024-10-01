import pickle
import yaml
import numpy as np
import pandas as pd

from features.sequencers.sequencer import Sequencer
from features.sequencers.sequencer_chemlab import ChemLabSequencer

from sklearn.preprocessing import MinMaxScaler

class EDM2022ChemLabSequencer(ChemLabSequencer):
    """
    This class creates sequences such that the first cells implement the states:
    - green green
    - green red
    - not green not red
    - not observed
    Actions:
    - other
    - concentration
    - width
    - pdf
    - break

    Args:
        ColoradoSequencer (_type_): _description_
    """

    def __init__(self, settings):
        super().__init__(dict(settings))
        self._name = 'state_phase_action_colorado_seq'
        self._notation = 'spa_coloseq'
        self._dataset = 'colorado_capacitor'

        self._states = [
            'greengreen',
            'greenred',
            'notgreennotred',
            'noobserved',
            'other',
            'concentration',
            'width',
            'pdf',
            'break'
        ]

        self._old_states = [
            'greengreen',
            'greenred',
            'notgreennotred',
            'noobserved',
            'other', # 0
            'concentration', #1
            'width', #2
            'solution', #3
            'wavelength', #4
            'tools', #5
            'concentrationlab', #6
            'pdf', #7
            'break' #8
        ]

        self._action_converter = {
            0: 0,
            1: 1,
            2: 2,
            3: 0,
            4: 0,
            5: 0,
            6: -1, #should be filtered
            7: 3,
            8: 4
        }


        self._break_threshold = self._settings['chemlab']['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['chemlab']['sequence_parameters']['merge_time']
        self._n_actions = 5
        self._n_states = 4
        self._n_old_states = 4

    def _create_action_break_vector(self, length:float):
        action_vector = np.zeros(5)
        action_vector[4] = length
        return action_vector

    def _new_action_vector(self, vector):
        current_action = vector[self._n_old_states:]
        current_action_idx = np.argmax(current_action)
        new_action = [0 for _ in range(self._n_actions)]
        new_action[self._action_converter[current_action_idx]] = max(current_action)

        # print(current_action, new_action)
        return new_action


    def _create_sequence(self, **kwargs):
        assert len(kwargs['sequence']) == len(kwargs['begin']) and len(kwargs['begin']) == len(kwargs['end'])
        break_threshold = self.get_threshold(kwargs['begin'], kwargs['end'], 0.6)
        
        new_states = []
        new_actions = []

        for i_a, action in enumerate(kwargs['sequence']):
            action_vector = action[self._n_old_states:]
            new_states.append(action[0:self._n_states])
            new_actions.append(self._new_action_vector(action_vector))

            if i_a < len(kwargs['sequence'])-1:
                if kwargs['begin'][i_a+1] - kwargs['end'][i_a] > break_threshold:
                    new_states.append(new_states[-1])
                    new_actions.append(self._create_action_break_vector(
                        kwargs['begin'][i_a+1] - kwargs['end'][i_a]
                    ))


        assert len(new_states) == len(new_actions)
        return new_states, new_actions

                

if __name__ == '__main__': 
    settings = {}
    sequencer = EDM2022ChemLabSequencer(settings)
    s, a, l, d = sequencer.load_all_sequences()

    print('Process Over')
            


            






