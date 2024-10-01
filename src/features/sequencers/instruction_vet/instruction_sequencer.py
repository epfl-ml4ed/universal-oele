import numpy as np
import pandas as pd
from typing import Tuple

import datetime
from features.parsers.parser import Parser
from features.sequencers.sequencer import Sequencer

class InstructionSequencer(Sequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._break_threshold = self._settings['sequence_parameters']['break_time']
        self._merge_threshold = self._settings['sequence_parameters']['merge_time']

        self._dimension = 12


    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation

    # def _load_labelmap(self):
    #     raise NotImplementedError

    def get_timestamp(self, begin_timestamp, now_timestamp):
        """Get the time difference between the start of the activity and 'now_timestamp'

        Args:
            begin_timestamp (_type_): beginning timestamp of the interaction on the simulation
            now_timestamp (_type_): timestamp you want the difference between 

        Returns:
            timestamp difference
        """
        bt = datetime.datetime.fromtimestamp(begin_timestamp/1e3)
        nt = datetime.datetime.fromtimestamp(now_timestamp/1e3)
        timest = nt - bt
        return timest.total_seconds()

    def get_value_at_timestamp(self, timestamps: list, values: list, current_ts: float):
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

    def get_relevant_parameters(self, params):
        """Returns the parameters from the log
        """
        new_params = {}
        for param in params:
            if param not in ['index', 'action']:
                new_params[param] = params[param]
        return new_params

    def merge_values(self, current_values, current_timestamps, updated_values, updated_timestamps):
        """Merge dependent variables with updated values

        Args:
            current_values (_type_): values now
            current_timestamps (_type_): ts now
            updated_values (_type_): values to merge
            updated_timestamps (_type_): ts to merge
        """
        merged_timestamps = current_timestamps + updated_timestamps
        merged_values = current_values + updated_values
        merged_indices = np.argsort(merged_timestamps)

        new_values = [merged_values[m_idx] for m_idx in merged_indices]
        new_timestamps = [merged_timestamps[m_idx] for m_idx in merged_indices]

        return new_values, new_timestamps

    def process_action(self, previous_log: dict, current_log: dict, next_log: dict, actions: list, begins: list, 
                                                            ends: list, parameters: dict, variable: str, click_start:bool, begin_timestamp:float):
        """Inserts the action correctly into the actions, begins, ends and parameters list, based on whether something is being dragged, an action should be merged, etc.

        Args:
            previous_log (dict): dictionary grouping the logs from the previous action
            current_log (dict): dictionary grouping the logs from the current action
            next_log (dict): dictionary grouping the logs from the next action
            actions (list): list of current actions
            begins (list): list of the beginning timestamps
            ends (list): list of the end timestamps
            parameters (dict): parameters of all the logs so far
            variable (str): name of the variable being interacted with in the current log
            click_start (bool): whether something is being dragged right now
            begin_timestamp (float): beginning timestamp of the simulation

        Returns:
            _type_: actions, begins, ends, parameters (same as args, but updated)
        """
        if previous_log == []:
            begins.append(0)
            actions.append(variable)
            parameters.append([self.get_relevant_parameters(current_log)])
            if self.get_timestamp(current_log['timestamp'], next_log['timestamp']) > 1:
                ends.append(begins[-1] + 1)
            else:
                ends.append(self.get_timestamp(begin_timestamp, next_log['timestamp']))

        else:
            # Part of a dragging movement
            if click_start:
                parameters[-1].append(self.get_relevant_parameters(current_log))
                ends[-1] = self.get_timestamp(begin_timestamp, current_log['timestamp'])
                if actions[-1] == 'click_start':
                    actions[-1] = variable
                elif actions[-1] != variable:
                    raise KeyError
                
            # Sequence of clicks
            elif len(actions) > 0:
                if actions[-1] == variable:
                    timediff = self.get_timestamp(previous_log['timestamp'], current_log['timestamp'])
                    if timediff <= self._merge_threshold:
                        ends[-1] = self.get_timestamp(begin_timestamp, current_log['timestamp'])
                        parameters[-1].append(self.get_relevant_parameters(current_log))
                    else:
                        begins.append(self.get_timestamp(begin_timestamp, current_log['timestamp']))
                        actions.append(variable)
                        parameters.append([self.get_relevant_parameters(current_log)])
                        if self.get_timestamp(current_log['timestamp'], next_log['timestamp']) > 1:
                            ends.append(begins[-1] + 1)
                        else:
                            ends.append(self.get_timestamp(begin_timestamp, next_log['timestamp']))
                
                else:
                    begins.append(self.get_timestamp(begin_timestamp, current_log['timestamp']))
                    actions.append(variable)
                    parameters.append([self.get_relevant_parameters(current_log)])
                    if next_log != []:
                        if self.get_timestamp(current_log['timestamp'], next_log['timestamp']) > 1:
                            ends.append(begins[-1] + 1)
                        else:
                            ends.append(self.get_timestamp(begin_timestamp, next_log['timestamp']))
                    else:
                        ends.append(begins[-1] + 1)

        return actions, begins, ends, parameters




    

