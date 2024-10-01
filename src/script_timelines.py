import os
import yaml

from features.parsers.vet_instruction.beerslaw_parser import BeerslawParser
from features.sequencers.colorado_capacitor.variable_generalised import StatePhaseActionColoradoSequencer

def capacitor_timelines():
    settings = {}
    sequencer = StatePhaseActionColoradoSequencer(settings)
    s, a, l, d = sequencer.load_all_sequences()

    print('labels', l)

    print('Length of sequence')
    lens = [len(ss) for ss in s]
    print('Maximum length: ', max(lens))

    print('Process Over')

def beerslaw_parser(settings):
    bl_parser = BeerslawParser(settings)
    bl_parser.initial_parse()

    path = '{}{}'.format(settings['paths']['data'], '/vet_instruction/beerslaw/logs/')
    files = os.listdir(path)
    files = [f for f in files if '']



if __name__ == '__main__': 
    with open('./configs/parsing_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # capacitor_timelines()
    beerslaw_parser(settings)
            
