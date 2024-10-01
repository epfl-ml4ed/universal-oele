from turtle import width
import numpy as np
from features.sequencers.instruction_vet.instruction_sequencer import InstructionSequencer
from features.sequencers.sequencer_instructionbeerslaw import InstructionBeerslawSequencer
from features.parsers.parser import Parser

class UniversalInstructionBeerslawSequencer(InstructionSequencer,InstructionBeerslawSequencer):

    def __init__(self, settings):
        super().__init__(settings)
        self._name = 'beerslaw variable sequencer'
        self._notation = 'blvar'

        self._states = [
            'nonobserved', # cannot see when it is doubled
            'notoptimal', # nothing is irrelevant here
            'easymode', # anything else
            'difficultmode', # can see it linearly, but not mathematically very precise (max absorbance at 0.2)
            'explore', # 4
            'analysis', # 5
            'record', # 6
            'break', # 7
            'colour', # 8
            'concentration', # 9
            'width', # 10
            'other', # 11
        ]
        
        self._simulation_states = {
            'notobserved': 0,
            'easy': 2,
            'difficult': 3
        }
        self._phase_state = {
            'explore': 0,
            'analysis': 1,
            'record': 2
        }
        self._actions = {
            'break': 0,
            'colour': 1,
            'concentration': 2,
            'width': 3,
            'other': 4
        }

        self._dataset = 'instruction_beerslaw'
        self._n_states = 7

    def _get_current_variable_state(self, solution:str, wavelength:float, old_state:str) -> str:
        """Based on the solution and wavelength in the simulation, returns whether the state is:
            not observed: cannot see when it is doubled
            difficult: can see it linearly, but not mathematically very precise (max absorbance at 0.2)
            easy: anything else
        Args:
            solution (str): absorbing solution
            wavelength (float): wavelength solution

        Returns:
            str: state (notobserved, easy or difficult)
        """
        if solution == 'ERROR' or wavelength == 'ERROR':
            return old_state
        elif solution == 'drink Mix (red)':
            if 578 <= wavelength and wavelength:
                return 'difficult'
            else:
                return 'easy'
        elif solution == 'Cobalt nitrate (red)':
            if (380 <= wavelength and wavelength <= 384) or (428 <= wavelength and wavelength <= 488) and (591 <= wavelength and wavelength <= 721):
                return 'difficult'
            elif (385 <= wavelength and wavelength <= 427) or (722 <= wavelength):
                return 'notobserved'
            else:
                return 'easy'
        elif solution == 'Cobalt chloride (red)':
            if 760 <= wavelength:
                return 'difficult'
            else:
                return 'easy'
        elif solution == 'Potassium dichromate (orange)':
            if (429 <= wavelength and wavelength <= 564):
                return 'difficult'
            elif 565 <= wavelength:
                return 'notobserved' 
            else:
                return 'easy'
        elif solution == 'Potassium chromate (yellow)':
            if (380 <= wavelength and wavelength <= 383) or (441 <= wavelength and wavelength <= 507):
                return 'difficult'
            elif 508 <= wavelength:
                return 'notobserved'
            else:
                return 'easy'
        elif solution == 'Nickel chloride (green)':
            if (380 <= wavelength and wavelength <= 408) or (464 < wavelength and wavelength <= 513) or (589 <= wavelength and wavelength <= 731):
                return 'difficult'
            elif (514 <= wavelength and wavelength <= 588):
                return 'notobserved'
            else:
                return 'easy'
        elif solution == 'Copper sulfate (blue)':
            if (380 <= wavelength and wavelength <= 595):
                return 'notobserved'
            elif (596 <= wavelength and wavelength <= 708):
                return 'difficult'
            else:
                return 'easy'
        elif solution == 'Potassium permanganate (purple)':
            if (380 <= wavelength and wavelength <=640):
                return 'difficult'
            elif (641 <= wavelength):
                return 'notobserved'

    def _process_axis(self, acts, begs, ens, params, axis, begin_ts):
        """Process what values are on each axis of the plot

        Args:
            acts (_type_): list of all actions
            begs (_type_): list of all beginning timestamps
            ens (_type_): list of all end timestamps
            params (_type_): list of all parameters
            variable (_type_): variable to process
            begin_ts (_type_): beginning timestamp

        Returns:
            _type_: _description_
        """
        axis_indices = [i for i in range(len(acts)) if acts[i] == axis]
        axis_params = [params[idx] for idx in axis_indices]
        
        if len(axis_indices) == 0:
            variable_updates = ['none', 'none']
            variable_timestamps = [begs[0], ens[-1]]
            return variable_updates, variable_timestamps

        axis_updates = [
            'none', *[axup[0]['new_value'] for axup in axis_params]
        ]
        axis_timestamps = [
            begs[0], *[self.get_timestamp(begin_ts, axup[0]['timestamp']) for axup in axis_params]
        ]
        axis_updates.append(axis_updates[-1])
        axis_timestamps.append(ens[-1])
        assert len(axis_updates) == len(axis_timestamps) 
        return axis_updates, axis_timestamps
    
    def _process_variable(self, acts, begs, ens, params, variable, begin_ts):
        """Extracts the information related to `variable`

        Args:
            acts (list): list of all actions
            begs (list): list of all beginning timestamps
            ens (list): list of all end timestamps
            params (list): list of all parameters
            variable (str): variable to process
            begin_ts (float): beginning timestamp

        Returns:
            the values when the simulation was updated
            the timestamps when the simulation was updated
            how it affected the absorbance
        """
        var_indices = [i for i in range(len(acts)) if acts[i] == variable]
        var_params = [params[idx] for idx in var_indices]

        if len(var_indices) == 0:
            initial_map = {
                'wavelength': 508,
                'width': 1,
                'concentration': 0.1,
                'xaxis': 'none',
                'yaxis': 'none'
            }
            variable_updates = [initial_map[variable], initial_map[variable]]
            variable_timestamps = [begs[0], ens[-1]]
            return variable_updates, variable_timestamps, [], []

        absorbance_updates = []
        absorbance_timestamps = []
        variable_updates = [var_params[0][0]['old_value']]
        variable_timestamps = [begs[0]]
        for var_drag in var_params:
            variable_updates = [
                *variable_updates,
                *[var_drag[i]['new_value'] for i in range(len(var_drag)) if 'new_value' in var_drag[i]]
            ]
            variable_timestamps = [
                *variable_timestamps,
                *[self.get_timestamp(begin_ts, var_drag[i]['timestamp']) for i in range(len(var_drag)) if 'new_value' in var_drag[i]]
            ]

            absorbance_updates = [
                *absorbance_updates,
                *[var_drag[i]['new_abs'] for i in range(len(var_drag)) if 'new_abs' in var_drag[i]]
            ]
            absorbance_timestamps = [
                *absorbance_timestamps,
                *[self.get_timestamp(begin_ts, var_drag[i]['timestamp']) for i in range(len(var_drag)) if 'new_abs' in var_drag[i]]
            ]

        variable_updates.append(variable_updates[-1])
        variable_timestamps.append(ens[-1])
        assert len(variable_updates) == len(variable_timestamps) and len(absorbance_updates) == len(absorbance_timestamps)
        return variable_updates, variable_timestamps, absorbance_updates, absorbance_timestamps

    def _process_solution(self, acts:list, begs:list, ens:list, params:list, begin_ts:list):
        """Process the solution name throughout the simulation

        Args:
            acts (list): list of all actions
            begs (list): list of all beginning timestamps
            ens (list): list of all end timestamps
            params (list): list of all parameters
            begin_ts (float): beginning timestamp

        Returns:
            the values when the solution was updated
            the timestamps when the solution was updated
            how it affected the absorbance
        """
        sol_indices = [i for i in range(len(acts)) if acts[i] == 'solution']
        sol_params = [params[idx] for idx in sol_indices]
        if len(sol_indices) == 0:
            solution_updates = ['drink Mix (red)', 'drink Mix (red)']
            solution_timestamps = [begs[0], ens[-1]]
            return solution_updates, solution_timestamps, [], [], [], [], [], []

        absorbance_updates = []
        absorbance_timestamps = []
        wavelength_updates = []
        wavelength_timestamps = []
        concentration_updates = []
        concentration_timestamps = []
        solution_updates = ['drink Mix (red)']
        solution_timestamps = [begs[0]]

        for solution_selection in sol_params:
            if len(solution_selection) > 0:        
                solution_updates = [
                    *solution_updates,
                    *[solution_selection[i]['new_value'] for i in range(len(solution_selection)) if 'new_value' in solution_selection[i]]
                ]
                solution_timestamps = [
                    *solution_timestamps,
                    *[self.get_timestamp(begin_ts, solution_selection[i]['timestamp']) for i in range(len(solution_selection)) if 'new_value' in solution_selection[i]]
                ]
        
                absorbance_updates = [
                    *absorbance_updates,
                    *[solution_selection[i]['new_abs'] for i in range(len(solution_selection)) if 'new_abs' in solution_selection[i]]
                ]
                absorbance_timestamps = [
                    *absorbance_timestamps,
                    *[self.get_timestamp(begin_ts, solution_selection[i]['timestamp']) for i in range(len(solution_selection)) if 'new_abs' in solution_selection[i]]
                ]
                wavelength_updates = [
                    *wavelength_updates,
                    *[solution_selection[i]['new_wave'] for i in range(len(solution_selection)) if 'new_wave' in solution_selection[i]]
                ]
                wavelength_timestamps = [
                    *wavelength_timestamps,
                    *[self.get_timestamp(begin_ts, solution_selection[i]['timestamp']) for i in range(len(solution_selection)) if 'new_wave' in solution_selection[i]]
                ]
                concentration_updates = [
                    *concentration_updates,
                    *[solution_selection[i]['new_conc'] for i in range(len(solution_selection)) if 'new_conc' in solution_selection[i]]
                ]
                concentration_timestamps = [
                    *concentration_timestamps,
                    *[self.get_timestamp(begin_ts, solution_selection[i]['timestamp']) for i in range(len(solution_selection)) if 'new_conc' in solution_selection[i]]
                ]
        solution_updates.append(solution_updates[-1])
        solution_timestamps.append(ens[-1])
        assert len(solution_updates) == len(solution_timestamps) and len(absorbance_updates) == len(absorbance_timestamps)
        assert len(wavelength_updates) == len(wavelength_timestamps) and len(concentration_updates) == len(concentration_timestamps)
        return solution_updates, solution_timestamps, absorbance_updates, absorbance_timestamps, wavelength_updates, wavelength_timestamps, concentration_updates, concentration_timestamps

    def _log_sequences(self, individual_logs):
        """Processed the logs parsed by Kate

        Args:
            individual_logs (_type_): parsed log from Kate Kutsenok

        Returns:
            actions: list of things student interacted with on the simulations
            begins: beginning timestamp of all actions
            ends: end timestamp of all actions
            parameters: parameters of the values changed during the interaction
        """
        click_start = False
        open_solution_menu = False
        open_solution_menu_click = False
        reading_instructions = False
        
        begin_timestamp = individual_logs[0]['timestamp']
        
        actions = []
        parameters = []
        begins = []
        ends = []
        
        previous_log = []
        for i_log, klog in enumerate(individual_logs):
            if klog['action'] == 'click_sum':
                continue
            if i_log == len(individual_logs) - 1:
                next_log = []
            else:
                next_log = individual_logs[i_log + 1]
        
            ### Dealing with the solution
            if klog['action'] == 'open_solution_menu' or open_solution_menu or klog['action'] == 'solution':
                if klog['action'] == 'open_solution_menu':
                    open_solution_menu = True
                    actions.append('solution')
                    parameters.append([])
                    begins.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                    ends.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                elif klog['action'] in ['click_sum', 'click_start', 'click_end']:
                    if open_solution_menu_click:
                        parameters[-1].append('no-change')
                        open_solution_menu_click = False
                        open_solution_menu = False
                    else:
                        ends[-1] = self.get_timestamp(begin_timestamp, klog['timestamp'])
                        if klog['action'] == 'click_end':
                            open_solution_menu_click = True
        
                elif klog['action'] == 'solution':
                    if open_solution_menu:
                        parameters[-1].append(self.get_relevant_parameters(klog))
                        ends[-1] = self.get_timestamp(begin_timestamp, klog['timestamp'])
                        open_solution_menu = False
                    else:
                        actions, begins, ends, parameters = self.process_action(
                            previous_log, klog, next_log, actions, begins, ends, parameters, 'solution', click_start, begin_timestamp
                        )
                else:
                    parameters[-1].append('no-change')
                    open_solution_menu = False
                
            if not open_solution_menu and klog['action'] != 'solution':
                if klog['action'] == 'open_instruction':
                    assert not reading_instructions
                    reading_instructions = True
                    begins.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                    actions.append('reading_instruction')
                    parameters.append([])
                    ends.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                elif klog['action'] == 'close_instruction':
                    if i_log == 0:
                        actions, begins, ends, parameters = self.process_action(
                            previous_log, klog, next_log, actions, begins, ends, parameters, 'reading_instruction', click_start, begin_timestamp
                        )
                    else:
                        assert reading_instructions
                        parameters[-1].append(self.get_relevant_parameters(klog))
                        ends[-1] = self.get_timestamp(begin_timestamp, klog['timestamp'])
                        reading_instructions = False
                    
                ##### Dealing with dragging
                elif klog['action'] == 'click_start':
                    if click_start:
                        if actions[-1] == 'click_start':
                            continue
                        if actions[-1] == individual_logs[i_log+1]['action'] or individual_logs[i_log+1]['action'] == 'click_end':
                            continue
                    assert not click_start
                    assert len(actions) == len(parameters) and len(parameters) == len(begins) and len(begins) == len(ends)
                    click_start = True
                    begins.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                    actions.append('click_start')
                    parameters.append([])
                    ends.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                elif klog['action'] == 'click_end': 
                    if not click_start:
                        click_start = False
                        continue
                    assert len(actions) == len(parameters) and len(parameters) == len(begins) and len(begins) == len(ends)
                    click_start = False

                    if actions[-1] == 'click_start':
                        begins.pop()
                        ends.pop()
                        actions.pop()
                        parameters.pop()
                    else:
                        ends[-1] = self.get_timestamp(begin_timestamp, klog['timestamp'])
        
                ##### Dealing with action
                elif klog['action'] in [
                    'wavelength', 'concentration', 'width', 'record', 'delete_point_from_table', 'ruler', 'plot', 'remove_from_graph', 
                    'yaxis_scale', 'xaxis_scale', 'restore_sim', 'yaxis', 'xaxis', 'move_up_point', 'move_down_point'
                ]:
                    actions, begins, ends, parameters = self.process_action(
                        previous_log, klog, next_log, actions, begins, ends, parameters, klog['action'], click_start, begin_timestamp
                    )

                elif klog['action'] == 'end':
                    begins.append(ends[-1])
                    ends.append(self.get_timestamp(begin_timestamp, klog['timestamp']))
                    actions.append('break')
                    parameters.append([])
                else:
                    print('New log')
                    print(klog)
        
            previous_log = dict(klog)

        return actions, begins, ends, parameters, begin_timestamp
    
    def _process_record(self, actions:list, bts:float, parameters:list, wavelengths:dict, concentrations:dict, widths:dict, solutions:dict):
        """Stores sim state for each record

        Args:
            actions (list): list of actions in the interaction
            begins (list): beginning timestamp of each action
            ends (list): ending timestamp of each action
            parameters (list): updated parameters for each action
            wavelengths (dict): all wavelength updates (value, timestamp)
            concentrations (dict): concentration updates (value, timestamp)
            widths (dict): width updates (value, timestamp)
            solutions (dict): solution updates (value, timestamp)
        """
        record_indices = [i for i in range(len(actions)) if actions[i] == 'record']
        record_parameters = [parameters[ridx] for ridx in record_indices]

        record_table = {}
        for r_param_index in record_parameters:
            r_param = r_param_index[0]
            i_record = r_param['table_point']
            current_solution = self.get_value_at_timestamp(solutions['timestamps'], solutions['values'], self.get_timestamp(bts, r_param['timestamp']))
            current_concentration = self.get_value_at_timestamp(concentrations['timestamps'], concentrations['values'], self.get_timestamp(bts, r_param['timestamp']))
            current_width = self.get_value_at_timestamp(widths['timestamps'], widths['values'], self.get_timestamp(bts, r_param['timestamp']))
            current_wavelength = self.get_value_at_timestamp(wavelengths['timestamps'], wavelengths['values'], self.get_timestamp(bts, r_param['timestamp']))

            differences = 0
            if len(record_table) > 0:
                previous_record = max(record_table.keys())
                if record_table[previous_record]['solution'] != current_solution:
                    differences += 1
                if record_table[previous_record]['concentration'] != current_concentration:
                    differences += 1
                if record_table[previous_record]['width'] != current_width:
                    differences += 1
                if record_table[previous_record]['wavelength'] != current_wavelength:
                    differences += 1

                record_table[i_record] = {
                    'solution': current_solution,
                    'concentration': current_concentration,
                    'width': current_width,
                    'wavelength': current_wavelength,
                    'timestamp': self.get_timestamp(bts, r_param['timestamp']),
                    'differences': differences,
                    'solution_change': record_table[previous_record]['solution'] != current_solution,
                    'concentration_change': record_table[previous_record]['concentration'] != current_concentration,
                    'width_change': record_table[previous_record]['width'] != current_width,
                    'wavelength_change': record_table[previous_record]['wavelength'] != current_wavelength
                }
            
            else:
                record_table[i_record] = {
                    'solution': current_solution,
                    'concentration': current_concentration,
                    'width': current_width,
                    'wavelength': current_wavelength,
                    'timestamp': self.get_timestamp(bts, r_param['timestamp']),
                    'differences': 0,
                    'solution_change': 0,
                    'concentration_change': 0,
                    'width_change': 0,
                    'wavelength_change': 0
                }

        if len(record_table.keys()) > 0:
            if min(record_table.keys()) == 2:
                for i in range(1, max(record_table.keys())):
                    record_table[i] = record_table[i+1]
        return record_table

    def _process_restore_sim(self, actions:list, parameters: list, record_table:dict):
        """Restores the sim according to the record table

        Args:
            actions (list): list of actions in the interaction
            parameters (list): updated parameters for each action
            record_table (dict): state of the simulation at each record
        """
        restore_indices = [i for i in range(len(actions)) if actions[i] == 'restore_sim']
        restore_parameters = [parameters[ridx] for ridx in restore_indices]

        timestamps = []
        wavelengths = {
            'values': [],
        }
        concentrations = {
            'values': [],
        }
        widths = {
            'values': [],
        }
        solutions = {
            'values': [],
        }
        for r_param in restore_parameters:
            record_point = r_param[0]['table_point']
            wavelengths['values'].append(record_table[record_point]['wavelength'])
            concentrations['values'].append(record_table[record_point]['concentration'])
            widths['values'].append(record_table[record_point]['width'])
            solutions['values'].append(record_table[record_point]['solution'])
            timestamps.append(record_table[record_point]['timestamp'])
        wavelengths['timestamps'] = timestamps
        concentrations['timestamps'] = timestamps
        widths['timestamps'] = timestamps
        solutions['timestamps'] = timestamps
        return wavelengths, concentrations, widths, solutions

    def _process_variables(self, a:list, b:list, e:list, p:list, bts:list):
        """Extracts all the variable updates (when they changed and how they changed) 

        Args:
            actions (list): list of actions in the interaction
            begins (list): beginning timestamp of each action
            ends (list): ending timestamp of each action
            parameters (list): updated parameters for each action
            bts (dict): beginning timestamp
        """
        absorbance = {
            'values': [],
            'timestamps': []
        }
        solutions = {
            'values': [],
            'timestamps': []
        }
        wavelengths = {
            'values': [],
            'timestamps': []
        }
        concentrations = {
            'values': [],
            'timestamps': []
        }
        widths = {
            'values': [],
            'timestamps': []
        }
        xaxis = {
            'values': [],
            'timestamps': []
        }
        yaxis = {
            'values': [],
            'timestamps': []
        }
        ### Fetching all sequence variables
        # wavelength
        wl_updates, wl_timestamps, abs_updates, abs_updates_ts = self._process_variable(a, b, e, p, 'wavelength', bts)
        wavelengths['values'] = wl_updates
        wavelengths['timestamps'] = wl_timestamps
        absorbance['values'] = abs_updates
        absorbance['timestamps'] = abs_updates_ts
        # concentration
        con_updates, con_timestamps, abs_updates, abs_updates_ts = self._process_variable(a, b, e, p, 'concentration', bts)
        concentrations['values'] = con_updates
        concentrations['timestamps'] = con_timestamps
        merged_abs, merged_abs_ts = self.merge_values(absorbance['values'], absorbance['timestamps'], abs_updates, abs_updates_ts)
        absorbance['values'] = merged_abs
        absorbance['timestamps'] = merged_abs_ts
        # widths
        wid_updates, wid_timestamps, abs_updates, abs_updates_ts = self._process_variable(a, b, e, p, 'width', bts)
        widths['values'] = wid_updates
        widths['timestamps'] = wid_timestamps
        merged_abs, merged_abs_ts = self.merge_values(absorbance['values'], absorbance['timestamps'], abs_updates, abs_updates_ts)
        absorbance['values'] = merged_abs
        absorbance['timestamps'] = merged_abs_ts
        # solutions
        sol_vals, sol_ts, abs_updates, abs_updates_ts, wl_updates_val, wl_updates_ts, con_updates_val, con_updates_ts = self._process_solution(a, b, e, p, bts)
        solutions['values'] = sol_vals
        solutions['timestamps'] = sol_ts
        merged_abs, merged_abs_ts = self.merge_values(absorbance['values'], absorbance['timestamps'], abs_updates, abs_updates_ts)
        absorbance['values'] = merged_abs
        absorbance['timestamps'] = merged_abs_ts
        merged_wl, merged_wl_ts = self.merge_values(wavelengths['values'], wavelengths['timestamps'], wl_updates_val, wl_updates_ts)
        wavelengths['values'] = merged_wl
        wavelengths['timestamps'] = merged_wl_ts
        merged_con, merged_con_ts = self.merge_values(concentrations['values'], concentrations['timestamps'], con_updates_val, con_updates_ts)
        concentrations['values'] = merged_con
        concentrations['timestamps'] = merged_con_ts
        # recording
        record_table = self._process_record(a, bts, p, wavelengths, concentrations, widths, solutions)
        wl_update, con_update, wid_update, sol_update = self._process_restore_sim(a, p, record_table)
        merged_wl, merged_wl_ts = self.merge_values(wavelengths['values'], wavelengths['timestamps'], wl_update['values'], wl_update['timestamps'])
        wavelengths['values'] = merged_wl
        wavelengths['timestamps'] = merged_wl_ts
        merged_con, merged_con_ts = self.merge_values(concentrations['values'], concentrations['timestamps'], con_update['values'], con_update['timestamps'])
        concentrations['values'] = merged_con
        concentrations['timestamps'] = merged_con_ts
        merged_wid, merged_wid_ts = self.merge_values(widths['values'], widths['timestamps'], wid_update['values'], wid_update['timestamps'])
        widths['values'] = merged_wid
        widths['timestamps'] = merged_wid_ts
        merged_sol, merged_sol_ts = self.merge_values(solutions['values'], solutions['timestamps'], sol_update['values'], sol_update['timestamps'])
        solutions['values'] = merged_sol
        solutions['timestamps'] = merged_sol_ts
        
        # axes
        xax_updates, xax_timestamps = self._process_axis(a, b, e, p, 'xaxis', bts)
        xaxis['values'] = xax_updates
        xaxis['timestamps'] = xax_timestamps
        yax_updates, yax_timestamps = self._process_axis(a, b, e, p, 'yaxis', bts)
        yaxis['values'] = yax_updates
        yaxis['timestamps'] = yax_timestamps

        return absorbance, widths, concentrations, solutions, wavelengths, xaxis, yaxis, record_table
        
    def _create_simulation_state_vector(self, state):
        """create the state vector (4 cells) based on whether it's in an easy, difficult or not observed state
        """
        new_states = [0 for _ in range(4)]
        new_states[self._simulation_states[state]] = 1
        return new_states

    def _impute_simulation_states(self, actions:list, begins:list, solutions: list, wavelengths: list):
        """impute state vectors for each action
        """
        assert len(actions) == len(begins)
        simulation_states = []
        state_name = 'notobserved'
        for i_s in range(len(actions)):
            solution = self.get_value_at_timestamp(solutions['timestamps'], solutions['values'], begins[i_s])
            wl = self.get_value_at_timestamp(wavelengths['timestamps'], wavelengths['values'], begins[i_s])
            state_name = self._get_current_variable_state(solution, wl, state_name)
            simulation_states.append(self._create_simulation_state_vector(state_name))
        return simulation_states

    def _create_phase_vector(self, phase):
        """create the phase vector (3 cells) based on whether it's explore, record or analysis
        """
        new_phases = [0 for _ in range(3)]
        new_phases[self._phase_state[phase]] = 1
        return new_phases

    def _infer_phase(self, action):
        if action in ['concentration', 'ruler', 'restore_sim', 'solution', 'wavelength', 'width', 'reading_instruction', 'break']:
            return 'explore'
        elif action in ['delete_point_from_table', 'move_down_point', 'move_up_point', 'plot', 'remove_from_graph', 'xaxis_scale', 'xaxis', 'yaxis', 'yaxis_scale']:
            return 'analysis'
        elif action in ['record']:
            return 'record'
    
    def _impute_phase_vectors(self, actions):
        simulation_phases = []
        for i_sp in range(len(actions)):
            phase = self._infer_phase(actions[i_sp])
            simulation_phases.append(self._create_phase_vector(phase))
        return simulation_phases
        
    def _create_action_vector(self, actions, begin, end):
        """create the action vectors (5 cells) based on whether it's colour, concentration, width, colour
        """
        new_actions = [0 for _ in range(5)]
        for i_a, action in enumerate(actions):
            new_actions[self._actions[action]] = end - begin
        return new_actions

    def _characterise_record(self, begin, record_table):
        record_actions = []
        current_record = [i_record for i_record in record_table if record_table[i_record]['timestamp'] == begin]
        if len(current_record) == 2:
            assert record_table[current_record[0]]['solution'] == record_table[current_record[1]]['solution']
            assert record_table[current_record[0]]['concentration'] == record_table[current_record[1]]['concentration']
            assert record_table[current_record[0]]['wavelength'] == record_table[current_record[1]]['wavelength']
            assert record_table[current_record[0]]['differences'] == record_table[current_record[1]]['differences']
            assert record_table[current_record[0]]['solution_change'] == record_table[current_record[1]]['solution_change']
            assert record_table[current_record[0]]['concentration_change'] == record_table[current_record[1]]['concentration_change']
            assert record_table[current_record[0]]['width_change'] == record_table[current_record[1]]['width_change']
            assert record_table[current_record[0]]['wavelength_change'] == record_table[current_record[1]]['wavelength_change']
            current_record = [current_record[0]]
        assert len(current_record) == 1
        current_record = record_table[current_record[0]]
        if current_record['solution_change'] or current_record['wavelength_change']:
            record_actions.append('colour')
        if current_record['concentration_change']:
            record_actions.append('concentration')
        if current_record['width_change']:
            record_actions.append('width')
        return record_actions

    def _characterise_analysis(self, timestamp, xaxis, yaxis): 
        x_value = self.get_value_at_timestamp(xaxis['timestamps'], xaxis['values'], timestamp)
        y_value = self.get_value_at_timestamp(yaxis['timestamps'], yaxis['values'], timestamp)
        axis_values = [x_value, y_value]
        if 'absorbance' in axis_values and 'concentration' in axis_values:
            return ['concentration']
        elif 'absorbance' in axis_values and 'cuvetteWidth' in axis_values:
            return ['width']
        elif 'absorbance' in axis_values and 'trialNumber' in axis_values:
            return ['other']
        else:
            return ['other']

    def _characterise_variable(self, action):
        if action == 'concentration':
            return ['concentration']
        elif action == 'reading_instruction':
            return ['other']
        elif action == 'ruler':
            return ['other']
        elif action == 'solution':
            return ['colour']
        elif action == 'wavelength':
            return ['colour']
        elif action == 'width':
            return ['width']
        elif action == 'restore_sim':
            return ['other']
        elif action == 'break':
            return ['break']

    def _characterise_explore(self, action, timestamp, record_table, xaxis, yaxis):
        if action in ['record']:
            return self._characterise_record(timestamp, record_table)
        elif action in ['delete_point_from_table', 'move_down_point', 'move_up_point', 'plot', 'remove_from_graph', 'xaxis', 'xaxis_scale', 'yaxis', 'yaxis_scale']:
            return self._characterise_analysis(timestamp, xaxis, yaxis)
        elif action in ['break', 'concentration', 'reading_instruction', 'restore_sim', 'ruler', 'solution', 'wavelength', 'width']:
            return self._characterise_variable(action)

    def _impute_explore(self, actions, begins, ends, record_table, xaxis, yaxis):
        assert len(actions) == len(begins)
        variable_vectors = []
        for i in range(len(actions)):
            action_names = self._characterise_explore(actions[i], begins[i], record_table, xaxis, yaxis)
            variable_vectors.append(self._create_action_vector(action_names, begins[i], ends[i]))
        return variable_vectors

    def _impute_breaks(self, begins, ends, simulation_states, simulation_phases, simulation_variables):
        break_vector = np.array([1, 0, 0, 0, 0])
        new_states = []
        new_phases = []
        new_variables = []

        for i in range(len(simulation_states) - 1):
            new_states.append(simulation_states[i])
            new_phases.append(simulation_phases[i])
            new_variables.append(simulation_variables[i])

            time_diff = begins[i+1] - ends[i]
            for _ in range(int(time_diff // self._break_threshold)):
                new_states.append(simulation_states[i])
                new_phases.append(simulation_phases[i])
                new_variables.append(break_vector * self._break_threshold)

        time_diff = ends[-1] - begins[-1]
        for _ in range(int(time_diff // self._break_threshold)):
            new_states.append(simulation_states[-1])
            new_phases.append(simulation_phases[-1])
            new_variables.append(break_vector * self._break_threshold)
        return new_states, new_phases, new_variables

    def _mergeactions(self, states, phases, actions, begins, ends):
        """Merge actions which are separated by less than a specific threshold

        Args:
            actions (_type_): actions
            begins (_type_): beginning ts
            ends (_type_): ending ts

        Returns:
            _type_: _description_
        """
        assert len(states) == len(actions) and len(actions) == len(begins) and len(begins) == len(ends)
        new_states = [states[0]]
        new_phases = [phases[0]]
        new_actions = [actions[0]]
        new_begins = [begins[0]]
        new_ends = [ends[0]]
        for i in range(1, len(actions)):
            if (states[i][0] > 0 and new_states[-1][0] > 0) or \
                    (states[i][1] > 0 and new_states[-1][1] > 0) or\
                        (states[i][2] > 0 and new_states[-1][2] > 0) and\
                         actions[0] == 0:
                if begins[i] - new_ends[-1] < self._merge_threshold:
                    new_ends[-1] = ends[i]
                    new_actions[-1] = [element for element in (np.array(new_actions[-1]) + np.array(actions[i]))]
                else:
                    new_states.append(states[i])
                    new_phases.append(phases[i])
                    new_actions.append(actions[i])
                    new_begins.append(begins[i])
                    new_ends.append(ends[i])
            else:
                new_states.append(states[i])
                new_phases.append(phases[i])
                new_actions.append(actions[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])

        assert len(new_states) == len(new_phases) and len(new_phases) == len(new_states) and len(new_actions) == len(new_begins) and len(new_begins) == len(new_ends)
        return new_states, new_phases, new_actions, new_begins, new_ends

    def process_sequence(self, individual_logs):
        """Extracts all of the data from the sequence

        Args:
            individual_logs (_type_): file parsed by Kate

        Returns:
            xx
        """
        
        a, b, e, p, bts = self._log_sequences(individual_logs)
        absorbance, widths, concentrations, solutions, wavelengths, xaxis, yaxis, record_table  = self._process_variables(a, b, e, p, bts)
        simulation_states = self._impute_simulation_states(a, b, solutions, wavelengths)
        simulation_phases = self._impute_phase_vectors(a)
        simulation_variables = self._impute_explore(a, b, e, record_table, xaxis, yaxis)
    
        assert len(a) == len(b) and len(b) == len(e) and len(e) == len(simulation_states) 
        assert len(simulation_states) == len(simulation_phases) and len(simulation_phases) == len(simulation_variables)
        if self._settings['data']['merge']:
            simulation_states, simulation_phases, simulation_variables, b, e = self._mergeactions(simulation_states, simulation_phases, simulation_variables, b, e)
        new_states, new_phases, new_variables = self._impute_breaks(b, e, simulation_states, simulation_phases, simulation_variables)
        all_states = [new_states[i] + new_phases[i] for i in range(len(new_states))]

        assert len(all_states) == len(new_variables)
        return all_states, new_variables 

    def _create_vector_sequence(self, s, a, merged_s, merged_a):
        return s, a

