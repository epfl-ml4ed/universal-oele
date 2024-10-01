import pickle
import yaml
import numpy as np
import argparse

from utils.config_handler import ConfigHandler
from features.pipeline_maker import PipelineMaker

from ml.xval_maker import XValMaker

def create_features(settings):
    pipeline = PipelineMaker(settings)
    primary_state_actions, primary_labels, primary_demographics, primary_indices, \
        secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
            settings = pipeline.load_sequences()
    print('Test processed')
    print(
        'states: {}\nlabels: {}\ndemographics: {}\nindices= {}'.format(
            np.array(primary_state_actions).shape, np.array(primary_labels).shape, np.array(primary_demographics).shape, np.array(primary_indices).shape
        )
    )

    print(
        'states: {}\nlabels: {}\ndemographics: {}\nindices= {}'.format(
            np.array(secundary_state_actions).shape, np.array(secundary_labels).shape, np.array(secundary_demographics).shape, np.array(secundary_indices).shape
        )
    )

    lens_primary = [len(sa['action']) for sa in primary_state_actions]
    print('max len primary: {}'.format(max(lens_primary)))
    lens_secundary = [len(sc['action']) for sc in secundary_state_actions]
    print('max len secundary: {}'.format(max(lens_secundary)))


def seeds_train(settings):
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    
    pipeline = PipelineMaker(settings)
    primary_state_actions, primary_labels, primary_demographics, primary_indices, \
        secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
            settings = pipeline.load_sequences()

    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = XValMaker(settings)
        ml_pipeline.train(primary_state_actions, primary_labels, primary_demographics, primary_indices)

def baseline(settings):
    with open('./configs/gridsearch/gs_coldstart.yaml', 'r') as f:
        coldstart = yaml.load(f, Loader=yaml.FullLoader)
        the_root = settings['experiment']['root_name']

    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    settings['transfer'] = False
    pipeline = PipelineMaker(settings)
    primary_state_actions, primary_labels, primary_demographics, primary_indices, \
        _, _, _, _, \
            settings = pipeline.load_sequences()
    settings['ml']['pipeline']['xvalidator'] = 'nonnested_xval'
    settings['ml']['pipeline']['model'] = 'attentionrnn'
    settings['ml']['models']['attention_rnn']['secundary_attention_hidden_size'] = 1
    settings['ml']['models']['attention_rnn']['secundary_rnn_ncells'] = 32
    settings['ml']['models']['attention_rnn']['secundary_epochs'] = 0
    settings['ml']['models']['attention_rnn']['transfer_model'] = 'model1'
    
    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = XValMaker(settings)
        ml_pipeline.train(
            primary_state_actions, primary_labels, primary_demographics, primary_indices, 
        )

def transfer_learning(settings):
    with open('./configs/gridsearch/gs_coldstart.yaml', 'r') as f:
        coldstart = yaml.load(f, Loader=yaml.FullLoader)
        the_root = settings['experiment']['root_name']
    
    settings['ml']['splitters']['stratifier_col'] = 'dataset_label'
    settings['ml']['models']['attention_rnn']['primary_epochs'] = int(settings['primary'])
    settings['ml']['models']['attention_rnn']['secundary_epochs'] = int(settings['secundary'])

    for expe_name in coldstart:
        print('{}{}{}'.format('*'*20, expe_name, '*'*20))
        settings['experiment']['root_name'] = '{}/{}'.format(the_root, expe_name)
        settings['data']['primary'] = coldstart[expe_name]['primary']
        settings['data']['secundary'] = coldstart[expe_name]['secundary']
        handler = ConfigHandler(settings)
        settings = handler.get_experiment_name()
        pipeline = PipelineMaker(settings)
        primary_state_actions, primary_labels, primary_demographics, primary_indices, \
            secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
                settings = pipeline.load_sequences()

        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = XValMaker(settings)
        ml_pipeline.train(
            primary_state_actions, primary_labels, primary_demographics, primary_indices, 
            secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices
        )
        print()
    # handler = ConfigHandler(settings)
    # settings = handler.get_experiment_name()
    
    # pipeline = PipelineMaker(settings)
    # primary_state_actions, primary_labels, primary_demographics, primary_indices, \
    #     secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
    #         settings = pipeline.load_sequences()


    # for _ in range(settings['experiment']['model_seeds_n']):
    #     if settings['test']:
    #         primary_test_indices = np.random.choice(
    #             range(len(primary_state_actions)), replace=False, size=int(len(primary_state_actions)/2)
    #         )
    #         primary_state_actions = [primary_state_actions[idx] for idx in primary_test_indices]
    #         primary_labels = [primary_labels[idx] for idx in primary_test_indices]
    #         primary_demographics = [primary_demographics[idx] for idx in primary_test_indices]
    #         primary_indices = [primary_indices[idx] for idx in primary_test_indices]

    #         secundary_test_indices = np.random.choice(
    #             range(len(secundary_state_actions)), replace=False, size=int(len(secundary_state_actions)/2)
    #         )
    #         secundary_state_actions = [secundary_state_actions[idx] for idx in secundary_test_indices]
    #         secundary_labels = [secundary_labels[idx] for idx in secundary_test_indices]
    #         secundary_demographics = [secundary_demographics[idx] for idx in secundary_test_indices]
    #         secundary_indices = [secundary_indices[idx] for idx in secundary_test_indices]

    #     seed = np.random.randint(settings['experiment']['max_seed'])
    #     settings['seeds']['model'] = seed
    #     ml_pipeline = XValMaker(settings)
    #     ml_pipeline.train(
    #         primary_state_actions, primary_labels, primary_demographics, primary_indices, 
    #         secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices
        # )

def coldstart_learning(settings):
    settings['transfer'] = True
    settings['ml']['models']['attention_rnn']['primary_epochs'] = int(settings['primary'])
    settings['ml']['splitters']['stratifier_col'] = 'dataset_label'
    print('settings:', settings['ml']['models']['attention_rnn'])
    with open('./configs/gridsearch/gs_coldstart.yaml', 'r') as f:
        coldstart = yaml.load(f, Loader=yaml.FullLoader)
        the_root = settings['experiment']['root_name']

    for expe_name in coldstart:
        if settings['secundary'] in expe_name:
            print('{}{}{}'.format('*'*20, expe_name, '*'*20))
            settings['experiment']['root_name'] = '{}/{}'.format(the_root, expe_name)
            settings['data']['primary'] = coldstart[expe_name]['primary']
            settings['data']['secundary'] = coldstart[expe_name]['secundary']
            handler = ConfigHandler(settings)
            settings = handler.get_experiment_name()
        
            pipeline = PipelineMaker(settings)
            primary_state_actions, primary_labels, primary_demographics, primary_indices, \
                secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
                    settings = pipeline.load_sequences()

            seed = np.random.randint(settings['experiment']['max_seed'])
            settings['seeds']['model'] = seed
            ml_pipeline = XValMaker(settings)
            ml_pipeline.train(
                primary_state_actions, primary_labels, primary_demographics, primary_indices, 
                secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices
            )
            print()

def mix_learning(settings):
    settings['transfer'] = True
    settings['ml']['models']['attention_rnn']['primary_epochs'] = int(settings['primary'])
    settings['ml']['splitters']['stratifier_col'] = 'dataset_label'
    print('settings:', settings['ml']['models']['attention_rnn'])
    with open('./configs/gridsearch/gs_coldstart.yaml', 'r') as f:
        coldstart = yaml.load(f, Loader=yaml.FullLoader)
        the_root = settings['experiment']['root_name']

    for expe_name in coldstart:
        if settings['secundary'] in expe_name:
            print('{}{}{}'.format('*'*20, expe_name, '*'*20))
            settings['experiment']['root_name'] = '{}/{}'.format(the_root, expe_name)
            settings['data']['primary'] = coldstart[expe_name]['primary']
            settings['data']['secundary'] = coldstart[expe_name]['secundary']
            handler = ConfigHandler(settings)
            settings = handler.get_experiment_name()
        
            pipeline = PipelineMaker(settings)
            primary_state_actions, primary_labels, primary_demographics, primary_indices, \
                secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices, \
                    settings = pipeline.load_sequences()

            seed = np.random.randint(settings['experiment']['max_seed'])
            settings['seeds']['model'] = seed
            ml_pipeline = XValMaker(settings)
            ml_pipeline.train(
                primary_state_actions, primary_labels, primary_demographics, primary_indices, 
                secundary_state_actions, secundary_labels, secundary_demographics, secundary_indices
            )
            print()

def test(settings):

    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()

    feature_pipeline = PipelineMaker(settings)
    state_actions, labels, demographics, indices, settings = feature_pipeline.load_sequences()

    lengths = [len(sa['state']) for sa in state_actions]
    print('Longer sequence: {}'.format(max(lengths)))
    print('done')

    # ml_pipeline = XValMaker(settings)
    # ml_pipeline.train(state_actions, labels, demographics, indices)


def handle_arparse(settings):
    settings['data']['primary'] = settings['primary']
    settings['data']['secundary'] = settings['secundary']
    settings['ml']['pipeline']['model'] = settings['model']
    settings['experiment']['root_name'] = '{}/{}'.format(settings['experiment']['root_name'], settings['name'])
    if settings['test']:
        settings['ml']['splitters']['nfolds'] = 2
    settings['ml']['models']['attention_rnn']['transfer_model'] = settings['transfermodel']
    return settings

def main(settings):
    settings = handle_arparse(settings)
    if settings['datapipeline']:
        create_features(settings)
    if settings['transfer']:
        transfer_learning(settings)
    if settings['baseline']:
        baseline(settings)
    if settings['coldstart']:
        coldstart_learning(settings)
    if settings['mix']:
        mix_learning(settings)

if __name__ == '__main__': 
    with open('./configs/classification_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')
    # Tasks
    parser.add_argument('--testing', dest='testing', default=False, action='store_true')
    parser.add_argument('--datapipeline', dest='datapipeline', default=False, action='store_true')
    parser.add_argument('--seeds', dest='seedstrain', default=False, action='store_true')
    parser.add_argument('--transfer', dest='transfer', default=False, action='store_true')
    parser.add_argument('--coldstart', dest='coldstart', default=False, action='store_true')
    parser.add_argument('--mix', dest='mix', default=False, action='store_true')
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')

    parser.add_argument('--primary', dest='primary', default='universal_capcol', action='store')
    parser.add_argument('--secundary', dest='secundary', default='universal_chem', action='store')
    parser.add_argument('--model', dest='model', default='attentionrnn', action='store')
    parser.add_argument('--transfermodel', dest='transfermodel', default='switch', action='store')
    parser.add_argument('--name', dest='name', default='helloworld', action='store')
    
    settings.update(vars(parser.parse_args()))
    main(settings)