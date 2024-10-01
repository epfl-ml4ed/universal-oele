from hashlib import new
import numpy as np
import pandas as pd
from typing import Tuple
from features.parsers.parser import Parser
from features.sequencers.sequencer import Sequencer

class LightBeerSequencer(Sequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._break_threshold = self._settings['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['sequence_parameters']['merge_time']

        self._dimension = 12


    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation

    def _load_labelmap(self):
        raise NotImplementedError

    def _get_value_timestep(self, timestamps: list, values: list, current_ts: float):
        """For a given variable, returns what the value of that variable was at a particular timestep. 
        Args:
            timestamps ([list]): list of timestamps of when the variable was changed
            values ([list]): list of the values of when the variable was changed
            timestep ([float]): timestep we want the value at
        Returns:
            value: value of that variable at that timestep
        """
        begins = timestamps[:-1]
        ends = timestamps[1:]
        for i in range(len(begins)):
            if begins[i] <= current_ts and current_ts < ends[i]:
                return values[i]

        if current_ts == timestamps[-1]:
            return values[-1]
        else:
            print(current_ts)
            print(begins)
            print(ends)
            exit(1)

    def _state_return(self, begin: list, end: list, timestep: float) -> Tuple[bool, list, list]:
        if begin == [] or end == []:
            return False, begin, end

        elif timestep >= begin[0] and timestep < end[0]:
            return True, begin, end
        
        elif timestep < begin[0]:
            return False, begin, end

        elif timestep >= end[0]:
            begin = begin[1:]
            end = end[1:]
            return self._state_return(begin, end, timestep)

    def _create_breaks(self, actions:list, begins:list, ends:list):
        new_actions = []
        new_begins = []
        new_ends = []

        if begins[0] > self._break_threshold:
            n_breaks = begins[0] // self._break_threshold
            b = 0
            for _ in range(int(n_breaks)):
                new_actions.append('break')
                new_begins.append(b)
                b += self._break_threshold
                new_ends.append(b)

        for i in range(len(actions) - 1):
            new_actions.append(actions[i])
            new_begins.append(begins[i])
            new_ends.append(ends[i])

            if begins[i+1] - ends[i] > self._break_threshold:
                n_breaks = (begins[i+1] - ends[i]) // self._break_threshold
                b = ends[i]
                for _ in range(int(n_breaks)):
                    new_actions.append('break')
                    new_begins.append(b)
                    b += self._break_threshold
                    b = min([b, begins[i+1]])
                    new_ends.append(b)
                
                    
        new_actions.append(actions[-1])
        new_begins.append(begins[-1])
        new_ends.append(ends[-1])

        assert len(new_actions) == len(new_begins) and len(new_begins) == len(new_ends)
        return new_actions, new_begins, new_ends

    def _mergeactions(self, actions, begins, ends):
        """Merge actions which are separated by less than a specific threshold

        Args:
            actions (_type_): actions
            begins (_type_): beginning ts
            ends (_type_): ending ts

        Returns:
            _type_: _description_
        """
        new_actions = [actions[0]]
        new_begins = [begins[0]]
        new_ends = [ends[0]]
        for i in range(1, len(actions)):
            if (('analysis' in actions[i] and 'analysis' in new_actions[-1]) or \
                    ('record' in actions[i] and 'record' in new_actions[-1]) or\
                        (actions[i] in ['voltage', 'circuit', 'area', 'separation'] and new_actions[-1] in ['voltage', 'circuit', 'area', 'separation'])) and\
                         actions[i] != 'break':
                if begins[i] - new_ends[-1] < self._merge_threshold:
                    new_ends[-1] = ends[i]
                    if actions[i] not in new_actions[-1]:
                        new_actions[-1] = '{}_{}'.format(new_actions[-1], actions[i].replace('record_', '').replace('analysis_', ''))
                else:
                    new_actions.append(actions[i])
                    new_begins.append(begins[i])
                    new_ends.append(ends[i])
            else:
                new_actions.append(actions[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])

        assert len(new_actions) == len(new_begins) and len(new_begins) == len(new_ends)
        return new_actions, new_begins, new_ends

    def _find_axis_value(self, axis_values:list, axis_timestamps: list, ts: float):
        """_summary_

        Args:
            axis_values (list): values of the axis values throughout time
            axis_timestamps (list): values of the axis timestamps throughout time
            ts (float): at what timestamp to do this
        """
        value = ''
        if len(axis_timestamps) == 1 and ts == axis_timestamps[0]:
            value = axis_values[0]
        elif ts < axis_timestamps[0]:
            value = 'linear'
        elif ts >= axis_timestamps[0] and ts < axis_timestamps[1]:
            value = axis_values[0]
        elif ts >= axis_timestamps[1]:
            return self._find_axis_value(axis_values[1:], axis_timestamps[1:], ts)


        if value == -1:
            value = 'width'
        if 'width' in value.lower():
            return 'width', 2
        if 'absorbance' in value.lower():
            return 'absorbance', 3
        if 'wavelength' in value.lower():
            return 'wavelength', 780
        if 'concentration' in value.lower():
            return 'concentration', 200
        if 'trialnumber' in value.lower():
            return 'other', 67

        if 'separation' in value.lower():
            return 'separation', 10
        if 'area' in value.lower():
            return 'area', 400
        if 'voltage' in value.lower():
            return 'voltage', 1.5
        if 'charge' in value.lower():
            return 'charge', 200
        if 'trialnumber' in value.lower():
            return 'other',  3

    def _categorise_analysis(self, actions:list, begins:list, ends:list, xaxis:dict, yaxis:dict):
        """categorise the graph and table actions into groups such that depending on the axis, the graph actions are categorised
        as analysis + the right variable, or as analysis + table if an action is made on the table, or as analysis_else for any other
        graph action not categorised.
        Returns the original action if it does not correpond to the above case

        Args:
            action (list): actions
            begin (list): beginning timestamps of that action
            end (list): end timestamps of that action
            xaxis (dict): values on the xaxis across time
            yaxis (dict): values on the yaxis across time

        Returns:
            str: the new name of the action
        """
        
        return [self._categorise_analysis_action(actions[idx], begins[idx], ends[idx], xaxis, yaxis) for idx in range(len(actions))]

    def _categorise_record_based_on_previous_record(self, begin:float, trials:dict):
        """Looks into the current and the previous 

        Args:
            begin (float): timestamp for now
            trials (dict): trials dictionary
        """

        """Looks into the current and the previous 

        Args:
            begin (float): timestamp for now
            trials (dict): trials dictionary
        """
        index = trials['timestamps'].index(begin)
        current_trial_index = trials['additions'][index]

        insertion_indices = [i for i in range(len(trials['additions'])) if trials['additions'][i]!=-1]
        find_current_index = insertion_indices.index(index)
        if find_current_index == 0:
            return 'record'
        previous_index = insertion_indices[find_current_index - 1]
        previous_trial_index = trials['additions'][previous_index]

        current_trial = trials['values'][index][current_trial_index]
        previous_trial = trials['values'][previous_index][previous_trial_index]

        record_name = 'record_'
        if 'Width' in current_trial:
            if current_trial['Width'] != previous_trial['Width']:
                record_name += 'width_'
            if current_trial['Concentration'] != previous_trial['Concentration']:
                record_name += 'concentration_'
            if current_trial['Wavelength'] != previous_trial['Wavelength']:
                record_name += 'wavelength_'
            

        if 'Area' in current_trial:
            if current_trial['Area'] != previous_trial['Area']:
                record_name += 'area_'
            if current_trial['Separation'] != previous_trial['Separation']:
                record_name += 'separation_'
            if current_trial['Battery voltage'] != previous_trial['Battery voltage']:
                record_name += 'voltage_'
            if current_trial['Connection'] != previous_trial['Connection']:
                record_name += 'circuit_'
        
        record_name = record_name[:-1]
        return record_name
        
    def _recording_based_on_previous_record(self, actions:list, begins:list, ends:list,  trials:dict):
        """Labels the records based on the previous recording item. Namely, the label will have the format
        record_{r} where r denotes the amount of variable changed since last trials, giving the possible names:
        - record_0
        - record_1
        - record_2
        - record_3 

        Args:
            actions (list): _description_
            begins (list): _description_
            ends (list): _description_
            trials (dict): _description_
        """
        new_actions = [self._categorise_record_based_on_previous_record(begins[i], trials) if 'record' in actions[i] else actions[i] for i in range(len(actions))]
        assert len(new_actions) == len(begins) and len(begins) and len(ends)
        return new_actions, begins, ends

    def _qualify_action(self, action):
        vector = [0 for _ in range(self._dimension)]
        if 'analysis' in action:
            vector[5] = 1
        elif 'record' in action:
            vector[6] = 1 
        else:
            vector[4] = 1

        if 'width' in action or 'area' in action:
            vector[10] = 1
        if 'concentration' in action or 'separation' in action:
            vector[9] = 1
        if 'voltage' in action or 'wavelength' in action:
            vector[8] = 1
        if 'circuit' in action:
            vector[10] = 1
            vector[9] = 1
            vector[8] = 1
        if 'break' in action:
            vector[7] = 1
        if 'other' in action:
            vector[11] = 1
        # else:
        #     vector[11] = 1
        return vector

    def _qualify_state(self, state, vector):
        vector[state] = 1
        return vector

    def _create_action_sequences(self, sim:Parser):
        # Cleaning the data
        sequences = sim.get_sequences()
        actions = [s for s in sequences['sequence']]
        begins = [b for b in sequences['begins']]
        ends = [e for e in sequences['ends']]
        if len(actions) != len(begins) or len(actions) != len(ends):
            croissant = True
            smaller_length = min(len(actions), len(begins), len(ends))
            for i in range(smaller_length):
                if ends[i] < begins[i]:
                    croissant = False

            if not croissant:
                for i in range(smaller_length):
                    try:
                        if ends[i] < begins[i]:
                                actions = actions[:i]
                                begins = begins[:i]
                                ends = ends[:i]
                    except IndexError:
                        actions, begins, ends = [], [], []
            else:
                if len(begins) == len(actions) and len(ends) < len(begins):
                    while len(ends) < len(begins):
                        ends.append(begins[len(ends)-1] + 0.05)
                else:
                    print(len(ends), len(begins), len(actions))
        if len(actions) == 0:
            return [], [], []

        # Extract all values
        simulation_values = sim.get_simulation_values()
        xaxis = dict(simulation_values['xaxis'])
        yaxis = dict(simulation_values['yaxis'])
        trials = sim.get_table_data()
        assert len(actions) == len(begins) == len(ends)

        actions, begins, ends = self._recording_based_on_previous_record(actions, begins, ends, trials)
        actions = self._categorise_analysis(actions, begins, ends, xaxis, yaxis)
        actions = [self._label_map[a] if a in self._label_map else a for a in actions]
        if self._settings['data']['merge']:
            actions, begins, ends = self._mergeactions(actions, begins, ends)
        actions, begins, ends = self._create_breaks(actions, begins, ends)

        return actions, begins, ends

    def _create_vector_sequence(self, sim:Parser):
        raise NotImplementedError


    

