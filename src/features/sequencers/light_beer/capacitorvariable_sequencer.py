import numpy as np
from features.sequencers.light_beer.light_beer_sequencer import LightBeerSequencer
from features.sequencers.sequencer_lightbeercapacitor import LightBeerCapacitorSequencer
from features.parsers.parser import Parser

class UniversalLightCapacitorSequencer(LightBeerSequencer, LightBeerCapacitorSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'capacitor variable sequencer'
        self._notation = 'capvar'
        self._dataset = 'lightbeer_capacitor'

        self._states = [
            'nonobserved', #no stored energy ticked on - 0
            'notoptimal', #not played with voltage yet - 1
            'easymode', #closed circuit and stored energy - 2
            'difficultmode', #open circuit and stored energy - 3
            'explore' # 4,
            'analysis' # 5,
            'record' # 6,
            'break', # 7
            'battery', # 8
            'separation', # 9
            'area', # 10
            'other' # 11
        ]

        self._voltage_notplayed = False
        self._n_states = 7

    def _get_state(self, voltage_values, voltage_timestamps, connected_begin, connected_end, timestamp, voltage_once):
        v = self._get_value_timestep(
            voltage_timestamps, voltage_values, timestamp
        )

        c, connected_begin, connected_end = self._state_return(
            connected_begin, connected_end, timestamp
        )

        
        if (v == 0 or v == -1) and voltage_once:
            state = 1 
        elif v == 0 or v == -1:
            state = 0
        elif v != 0 and c:
            state = 2
        elif v != 0 and not c:
            state = 3
        else:
            state = 0
        if state != 0:
            voltage_once = True
        return connected_begin, connected_end, state, voltage_once

    def _categorise_analysis_action(self, action:str, begin:float, end:float, xaxis:dict, yaxis:dict) -> str:
        """categorise the graph and table actions into groups such that depending on the axis, the graph actions are categorised
        as analysis + the right variable, or as analysis + table if an action is made on the table, or as analysis_else for any other
        graph action not categorised.
        Returns the original action if it does not correpond to the above case

        Args:
            action (str): the name of the action
            begin (float): beginning timestamp of that action
            end (float): end timestamp of that action
            xaxis (dict): values on the xaxis across time
            yaxis (dict): values on the yaxis across time

        Returns:
            str: the new name of the action
        """
        if action == 'graph':
            x_value, _ = self._find_axis_value(xaxis['values'], xaxis['timestamps'], begin)
            y_value, _ = self._find_axis_value(yaxis['values'], yaxis['timestamps'], begin)

            if (x_value == 'charge' and y_value == 'separation') or (x_value == 'separation' and y_value == 'charge'):
                return 'analysis_separation'

            elif (x_value == 'charge' and y_value == 'voltage') or (x_value == 'voltage' and y_value == 'charge'):
                return 'analysis_voltage'

            elif (x_value == 'charge' and y_value == 'area') or (x_value == 'area' and y_value == 'charge'):
                return 'analysis_area'

            else:
                return 'analysis_other'

        elif action == 'table':
            return 'analysis_other'
        
        else:
            return action

    def _create_vector_sequence(self, sim:Parser):
        actions, begins, ends = self._create_action_sequences(sim)
        if len(actions) == 0:
            return [], []
        sequences = sim.get_sequences()
        vector_actions = [self._qualify_action(a) for a in actions]

        # print(sequences.keys())
        circuit_state = sequences['battery']['connected']
        circuit_begin, circuit_end = [b for b in circuit_state['begins']], [e for e in circuit_state['ends']]
        voltage_timsetamps = [ts for ts in sequences['voltage']['timestamps']]
        voltage_values = [v for v in sequences['voltage']['values']]

        states = []
        voltage_once = False
        for i in range(len(begins)):
            circuit_begin, circuit_end, s, voltage_once = self._get_state(
                voltage_values, voltage_timsetamps, circuit_begin, circuit_end, begins[i], voltage_once
            )
            states.append(s)

        vector_features = [
            self._qualify_state(states[ii], vector_actions[ii]) for ii in range(len(states))
        ]

        states = [vf[:self._n_states] for vf in vector_features]
        actions = [vf[self._n_states:] for vf in vector_features]
        actions = [(ends[i] - begins[i]) * np.array(actions[i]) for i in range(len(actions))]
        actions = [[aaa for aaa in aa] for aa in actions]
        return states, actions

        




        