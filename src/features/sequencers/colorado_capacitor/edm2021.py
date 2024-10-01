import pickle
import yaml
import numpy as np
import pandas as pd

from features.sequencers.sequencer import Sequencer
from features.sequencers.sequencer_colorado import ColoradoSequencer

from sklearn.preprocessing import MinMaxScaler

class EDM2021ColoradoSequencer(ColoradoSequencer):
    """
    This class creates sequences such that the first cells implement the states:
    - non observed
    - not optimal
    - easy mode (easily observable)
    - difficult mode (not easily observable)
    The middle cells implement the phases:
    - explore (always the case)
    - record
    - analysis
    Actions:
    - Break
    - other
    - battery
    - area
    - separation



    Args:
        ColoradoSequencer (_type_): _description_
    """

    def __init__(self, settings):
        super().__init__(dict(settings))
        self._name = 'state_phase_action_colorado_seq'
        self._notation = 'spa_coloseq'
        self._dataset = 'colorado_capacitor'

        self._states = [
            'stored_energy',
            'closed_circuit',
            'other', #0
            'voltage', #1
            'plateseparation', #2
            'platearea', #3
            'break' #4
        ]

        self._break_threshold = self._settings['colorado']['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['colorado']['sequence_parameters']['merge_time']
        self._n_states = 2
        self._n_actions = 5
        
    def _create_action_break_vector(self, length:float):
        action_vector = np.zeros(5)
        action_vector[4] = length
        return action_vector


    def _create_sequence(self, **kwargs):
        assert len(kwargs['sequence']) == len(kwargs['begin']) and len(kwargs['begin']) == len(kwargs['end'])
        break_threshold = self.get_threshold(kwargs['begin'], kwargs['end'], 0.6)
        new_states = []
        new_actions = []
        # first break
        if kwargs['begin'][0] > break_threshold:
            new_states.append([0, 1])
            new_actions.append(self._create_action_break_vector(kwargs['begin'][0]))

        for i_a, action in enumerate(kwargs['sequence']):
            if action[-1] >0:
                print('BREAKS ARE ALREADY IN')
            new_states.append(action[0:self._n_states:])
            new_actions.append(action[self._n_states::])

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
    sequencer = EDM2021ColoradoSequencer(settings)
    s, a, l, d = sequencer.load_all_sequences()

    print('Process Over')
            


            






