import pickle
import yaml
import numpy as np
import pandas as pd

from features.sequencers.sequencer import Sequencer
from features.sequencers.sequencer_colorado import ColoradoSequencer

class StatePhaseActionColoradoSequencer(ColoradoSequencer):
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
            'nonobserved', #no stored energy ticked on
            'notoptimal', #not played with voltage yet
            'easymode', #closed circuit and stored energy
            'difficultmode', #open circuit and stored energy
            'explore',
            'analysis',
            'record',
            'break', #0
            'battery', #1
            'separation', #2
            'area', #3
            'other' #4
        ]

        self._old_states = [
            'stored_energy',
            'closed_circuit',
            'other', #0
            'voltage', #1
            'plateseparation', #2
            'platearea', #3
            'break' #4
        ]
        self._old_to_new = {
            0: 4,
            1: 1,
            2: 2,
            3: 3,
            4: 0
        }

        self._break_threshold = self._settings['colorado']['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['colorado']['sequence_parameters']['merge_time']
        self._n_states = 7
        self._n_actions = 5
        self._old_break_index = 6
        self._new_break_index = 0
        self._explore_index = 4
        self._voltage_index = 1

    def _assign_state(self, state_vector:list, voltage_boolean:bool):
        """Assigns not observed, not optimal, easy mode and difficult mode

        Args:
            state_vector (list): current state vector (state_vector[0]: stored energy, state_vector[1]: closed circuit)
            voltage_boolean (bool): whether the student has interacted with the voltage already

        Returns:
            _type_: _description_
        """
        if not state_vector[0]:
            return 0
        
        if not voltage_boolean: #no voltage, closed circuit
            return 1
        
        if voltage_boolean and state_vector[1]: #voltage and closed circuit
            return 2

        if voltage_boolean and not state_vector[1]: #voltage and open circuit
            return 3

    def  _create_state_vector(self, new_state:int):
        new_vector = [0 for _ in range(self._n_states)]
        new_vector[new_state] = 1
        new_vector[self._explore_index] = 1 # explore
        return new_vector

    def _create_action_vector(self, action_vector:list, voltage_boolean:bool):
        a = np.argmax(action_vector)
        new_vector = np.zeros(self._n_actions)
        new_vector[self._old_to_new[a]] = 1
        if a == self._voltage_index:
            voltage_boolean = True
        return new_vector, voltage_boolean


    def _create_state_phase_action_vector(self, action:list, voltage_boolean:bool):
        """Reformats the vector

        Args:
            action (list): action vector as previous encoding
        """
        state_vector = action[0:2]
        action_vector = action[2:]
            
        new_state = self._assign_state(state_vector, voltage_boolean)
        new_state_vector = self._create_state_vector(new_state)
        new_action_vector, voltage_boolean = self._create_action_vector(action_vector, voltage_boolean)
        return new_state_vector, new_action_vector, voltage_boolean

    def _create_action_break_vector(self):
        action_vector = [0 for _ in range(self._n_actions)]
        action_vector[self._new_break_index] = 1
        return action_vector

    def _filter_sequences_for_old_breaks(self, sequences, begins, ends):
        ns, nb, ne = [], [], []
        for i in range(len(sequences)):
            if sequences[i][self._old_break_index] == 0:
                    ns.append(sequences[i])
                    nb.append(begins[i])
                    ne.append(ends[i])

        return ns, nb, ne

    def _create_sequence(self, **kwargs):
        assert len(kwargs['sequence']) == len(kwargs['begin']) and len(kwargs['begin']) == len(kwargs['end'])
        sequences = [s for s in kwargs['sequence']]
        begins = [b for b in kwargs['begin']]
        ends = [e for e in kwargs['end']]
        sequences, begins, ends = self._filter_sequences_for_old_breaks(sequences, begins, ends)
        voltage_boolean = False

        new_states, new_actions, new_begins, new_ends = [], [], [], []
        break_time = 0
        n_breaks = np.floor(begins[0] / self._break_threshold)
        initial_state = self._create_state_vector(0)
        for _ in range(int(n_breaks)):
            initial_action = self._create_action_break_vector()
            new_states.append(initial_state)
            new_actions.append(initial_action)

            new_begins.append(break_time)
            break_time += self._break_threshold
            new_ends.append(break_time)

        sv_0, av_0, _ = self._create_state_phase_action_vector(sequences[0], False)
        av_0 = self._create_action_break_vector()
        new_states.append(sv_0)
        new_actions.append(av_0)
        new_begins.append(begins[0])
        new_ends.append(ends[0])

        for i_a in range(1, len(sequences)):
            break_time = (begins[i_a] - ends[i_a-1]) / self._break_threshold
            n_breaks = int(np.floor(break_time))
            for i in range(int(n_breaks)):
                new_states.append([s for s in new_states[-1]])
                new_actions.append(self._create_action_break_vector())
                new_begins.append(ends[i_a-1] + (self._break_threshold * i))
                new_ends.append(ends[i_a-1] + (self._break_threshold * (i+1)))
            # Actions handling
            new_state_vector, new_action_vector, voltage_boolean = self._create_state_phase_action_vector(sequences[i_a], voltage_boolean)
            if np.argmax(new_action_vector) == np.argmax(new_actions[-1]) and new_state_vector  == new_states[-1]:
                if begins[i_a] - new_ends[-1] < self._merge_threshold:
                    new_ends[-1] = ends[i_a]
                    continue

            new_states.append(new_state_vector)
            new_actions.append(new_action_vector)
            new_begins.append(begins[i_a])
            new_ends.append(ends[i_a])

        new_actions = [(np.array(a)) * (new_ends[i_a] - new_begins[i_a]) for i_a, a in enumerate(new_actions)]
        new_actions = [[aa for aa in a] for a in new_actions]

        # print()
        # print("CHECK")
        # for i_a, a in enumerate(new_actions):
        #     print('new: {} old: {}'.format(a, kwargs['sequence'][i_a]))

        assert len(new_states) == len(new_actions) and len(new_actions) == len(new_begins) 
        return new_states, new_actions

                

if __name__ == '__main__': 
    settings = {}
    sequencer = StatePhaseActionColoradoSequencer(settings)
    s, a, l, d = sequencer.load_all_sequences()

    print('Process Over')
            


            






