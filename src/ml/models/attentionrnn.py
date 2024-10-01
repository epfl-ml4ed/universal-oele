import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple
from shutil import copytree, rmtree 

from ml.models.model import Model

from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
torch.autograd.set_detect_anomaly(True)

################################## Pytorch Model ##################################
class RNNAttention(nn.Module):
    def __init__(self, settings):
        super(RNNAttention, self).__init__()
        self._settings = copy.deepcopy(settings)
        # print('INIT', settings['ml']['models']['attention_rnn'])
        self._model_settings = copy.deepcopy(settings['ml']['models']['attention_rnn'])
        self._n_classes = settings['experiment']['nclasses']
        self.gru_hidden_size = self._model_settings['rnn_ncells']
        self.attention_hidden_size = self._model_settings['attention_hidden_size']
        if self._model_settings['transfer_model'] == 'datt':
            self.attention_hidden_size = int(self.attention_hidden_size*2)

        ########## Primary Model
        self.primary_hidden = None
        self.forward = self._primary_forward
        self.epochs = self._model_settings['primary_epochs']
        # attention layers
        self.key = torch.nn.Linear(self._settings['features']['dimension'], self.attention_hidden_size)
        self.value = torch.nn.Linear(self._settings['features']['dimension'], self.attention_hidden_size)
        self.query = torch.nn.Linear(self._settings['features']['dimension'], self.attention_hidden_size)
        self.softmax_attention = torch.nn.Softmax(dim=-1)
        # Gru layers
        self.num_directions = 2 if 'bi' in self._model_settings['rnn_cell_type'] == True else 1
        self.gru = torch.nn.GRU(
            self.attention_hidden_size,
            self.gru_hidden_size,
            num_layers=self._model_settings['rnn_nlayers'],
            bidirectional=bool(self.num_directions-1),
            batch_first=True
        )
        # Dense layers
        self.dropout = nn.Dropout(p=self._model_settings['classifier_dropout'])
        self.linear = nn.Linear(self.gru_hidden_size, self._n_classes)
        self._init_cuda()

        ########## Secundary Model
        self.secundary_hidden = None
        # print('TRANSFER', self._model_settings['transfer_model'])

        if self._model_settings['transfer_model'] in ['da', 'avegru', 'fratt', 'datt']:
            # attention
            self.secundary_attention_hidden_size = self._model_settings['attention_hidden_size']
            if self._model_settings['transfer_model'] == 'datt':
                self.secundary_attention_hidden_size = self.attention_hidden_size
            self.secundary_key = torch.nn.Linear(self._settings['features']['dimension'], self.secundary_attention_hidden_size)
            self.secundary_value = torch.nn.Linear(self._settings['features']['dimension'], self.secundary_attention_hidden_size)
            self.secundary_query = torch.nn.Linear(self._settings['features']['dimension'], self.secundary_attention_hidden_size)

            # gru
        if self._model_settings['transfer_model'] in ['avegru', 'datt']:
            self.secundary_gru_hidden_size = self.gru_hidden_size
            if self._model_settings['transfer_model'] == 'avegru':
                self.secundary_gru_input_size = self.attention_hidden_size
            if self._model_settings['transfer_model'] == 'datt':
                self.secundary_gru_input_size = 2 * self.attention_hidden_size
                self._secundary_linear = nn.Linear(self.gru_hidden_size, self._n_classes)
            self.secundary_gru = torch.nn.GRU(
                self.secundary_gru_input_size,
                self.secundary_gru_hidden_size,
                num_layers=self._model_settings['rnn_nlayers'],
                bidirectional=bool(self.num_directions-1),
                batch_first=True
            )
        else:
            self.secundary_gru_hidden_size = 1


        
    def _init_cuda(self):
        self.device = torch.device('cuda' if (torch.cuda.is_available() and os.path.isdir('../cluster_config/')) else 'cpu')
        # print('device: {}'.format(self.device))

    def update_model_settings(self, settings):
        self._model_settings.update(settings)
        
    def get_attention_weights(self):
        self.model.eval()
        with torch.no_grad():
            queries = list(self.query.parameters())[0].detach().numpy()
            keys = list(self.key.parameters())[0].detach().numpy()
            values = list(self.value.parameters())[0].detach().numpy()
        return queries, keys, values

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def transfer(self):
        self.epochs = self._model_settings['secundary_epochs']
        if self._model_settings['transfer_model'] == 'switch':
            self.forward = self._primary_forward
        elif self._model_settings['transfer_model'] == 'frattgru':
            self.freeze_layer(self.key)
            self.freeze_layer(self.value)
            self.freeze_layer(self.query)
            self.freeze_layer(self.gru)
            self.forward = self._primary_forward
        elif self._model_settings['transfer_model'] == 'fratt':
            self.freeze_layer(self.key)
            self.freeze_layer(self.value)
            self.freeze_layer(self.query)
            self.forward = self._primary_forward
        elif self._model_settings['transfer_model'] == 'da':
            self.freeze_layer(self.key)
            self.freeze_layer(self.value)
            self.freeze_layer(self.query)
            self.forward = self._primary_forward
            self.forward = self._da_forward
        elif self._model_settings['transfer_model'] == 'datt':
            self.freeze_layer(self.key)
            self.freeze_layer(self.value)
            self.freeze_layer(self.query)
            self.forward = self._primary_forward
        elif self._model_settings['transfer_model'] == 'avegru':
            self.freeze_layer(self.key)
            self.freeze_layer(self.value)
            self.freeze_layer(self.query)
            self.freeze_layer(self.gru)
            self.forward = self._primary_forward

        
        
    def init_hidden(self, batch_size):
        return torch.nn.init.orthogonal_(torch.zeros(
            self._model_settings['rnn_nlayers'] * self.num_directions,
            batch_size,
            self.gru_hidden_size
        )).to(self.device)

    def secundary_init_hidden(self, batch_size):
        return torch.nn.init.orthogonal_(torch.zeros(
            self._model_settings['rnn_nlayers'] * self.num_directions,
            batch_size,
            self.secundary_gru_hidden_size
        )).to(self.device)

    ########## Primary Model
    def _attention(self, sequences, lengths):
        queries = self.query(sequences)
        keys = self.key(sequences)
        values = self.value(sequences)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (10 ** 0.5)
        self._attention_weights = self.softmax_attention(scores)
        attention_outputs = torch.bmm(self._attention_weights, values)
        return attention_outputs, self._attention_weights

    def _gru(self, sequences, lengths):
        # Keras
        attention_outputs = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        rnn_x, lasth = self.gru(attention_outputs, self.primary_hidden)
        rnn_x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_x, batch_first=True)
        return rnn_x, lasth[0]

    def _classifier(self, concatenated_attention):
        clf_outputs = self.dropout(concatenated_attention)
        clf_outputs = self.linear(clf_outputs)
        clf_outputs = F.log_softmax(clf_outputs, dim=1)
        return clf_outputs

    ########## Secundary Model 
    def _secundary_attention(self, sequences, lengths):
        queries = self.secundary_query(sequences)
        keys = self.secundary_key(sequences)
        values = self.secundary_value(sequences)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (10 ** 0.5)
        self._secundary_attention_weights = self.softmax_attention(scores)
        attention_outputs = torch.bmm(self._secundary_attention_weights, values)
        return attention_outputs, self._secundary_attention_weights

    def _secundary_gru(self, sequences, lengths):
        # Keras
        attention_outputs = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        rnn_x, lasth = self.secundary_gru(attention_outputs, self.secundary_hidden)
        rnn_x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_x, batch_first=True)
        return rnn_x, lasth[0]

    def _secundary_classifier(self, concatenated_attention):
        clf_outputs = self.dropout(concatenated_attention)
        clf_outputs = self._secundary_linear(clf_outputs)
        clf_outputs = F.log_softmax(clf_outputs, dim=1)
        return clf_outputs

    ########## Forward
    def _da_forward(self, sequences, lens):
        # primary
        attention_outputs, attention_weights = self._attention(sequences, lens)

        # secundary
        sec_attention_outputs, sec_attention_weights = self._secundary_attention(sequences, lens)
        averaged_attention = torch.cat([attention_outputs, sec_attention_outputs])
        averaged_attention = torch.reshape(averaged_attention, (2, *attention_outputs.size()))
        averaged_attention = torch.mean(averaged_attention, axis=0)

        _, final_rnn = self._gru(averaged_attention, lens)
        output = self._classifier(final_rnn)
        return output

    def _datt_forward(self, sequences, lens):
        # primary
        attention_outputs, attention_weights = self._attention(sequences, lens)

        # secundary
        sec_attention_outputs, sec_attention_weights = self._secundary_attention(sequences, lens)
        averaged_attention = torch.cat([attention_outputs, sec_attention_outputs])

        _, final_rnn = self._secundary_gru(averaged_attention, lens)
        output = self._secundary_classifier(final_rnn)
        return output

    def _avegru_forward(self, sequences, lens):
        # primary
        attention_outputs, attention_weights = self._attention(sequences, lens)
        _, final_rnn = self._gru(attention_outputs, lens)

        # secundary
        sec_attention_outputs, sec_attention_weights = self._secundary_attention(sequences, lens)
        _, sec_final_rnn = self._secundary_gru(sec_attention_outputs, lens)

        averaged_gru = torch.cat([final_rnn, sec_final_rnn])
        averaged_gru = torch.reshape(averaged_gru, (2, *sec_final_rnn.size()))
        averaged_gru = torch.mean(averaged_gru, axis=0)

        output = self._classifier(averaged_gru)
        return output

    def _primary_forward(self, sequences, lens):
        attention_outputs, attention_weights = self._attention(sequences, lens)
        _, final_rnn = self._gru(attention_outputs, lens)
        output = self._classifier(final_rnn)
        return output

class RNNAtt_loss(torch.nn.Module):
    def __init__(self):
        super(RNNAtt_loss, self).__init__()
        self._loss = torch.nn.NLLLoss(reduction='sum')

    def forward(self, ypreds, ybatch):
        return self._loss(ypreds, ybatch)

################################## Framework Model ##################################
class AttentionRNNClassifier(Model):
    """This class implements an LSTM Torch
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'long short term memory transfer'
        self._notation = 'lstm_tr'
        self._model_settings = settings['ml']['models']['attention_rnn']
        self._maxlen = self._settings['ml']['models']['maxlen']
        self._padding_value = 0
        self._fold = -1
        self._best_epochs = self._model_settings['secundary_epochs']
        self._outer_fold = 'pretrain'
        self._validation_path = '{}/temp_weights.pt'.format(
            self._experiment_name
        )
        self._transfer_boolean = False
        self._choose_fit()
        self._init_cuda()

    def _init_cuda(self):
        self.device = torch.device('cuda' if (torch.cuda.is_available() and os.path.isdir('../cluster_config/')) else 'cpu')
        # print('device: {}'.format(self.device))

    def update_settings(self, settings):
        self._model_settings.update(settings)
        self._settings['ml']['models']['lstm'].update(settings)
        self._choose_fit()

    def transfer(self):
        self.model.transfer()
        self._transfer_boolean = True
        # self.fit = self._fit_early_stopping

    def get_attention(self, x):
        self.model.eval()
        with torch.no_grad():
            x_vec, x_len = self._format_features(x)
            att_outputs, _ =  self.model._attention(x_vec, x_len)
        return att_outputs

    def get_attention_weights(self):
        self.model.eval()
        with torch.no_grad():
            queries = list(self.model.query.parameters())[0].detach().numpy()
            keys = list(self.model.key.parameters())[0].detach().numpy()
            values = list(self.model.value.parameters())[0].detach().numpy()
        return queries, keys, values

    def _choose_fit(self):
        if self._model_settings['early_stopping']:
            self.fit = self._fit_early_stopping
        else:
            self.fit = self._fit_no_early_stopping

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        x_vector, lengths = self._format_features(x)
        tensor_y = torch.LongTensor(y)
        return x_vector.to(self.device), tensor_y.to(self.device), lengths.to(self.device)
    
    def _format_features(self, x:list) -> list:
        lengths = [len(xm) for xm in x]
        lengths = torch.Tensor(lengths)
        x_vector = pad_sequences(x, padding="post", value=self._padding_value, maxlen=self._maxlen, dtype=float)
        x_vector = torch.Tensor(x_vector)
        return x_vector, lengths
    
    def _init_model(self):
        self._set_seed()
        self.model = RNNAttention(self._settings)
        self.model = self.model.to(self.device)
        self._optimiser = optim.Adam(self.model.parameters(), eps=1e-07)
        self._loss_fn = torch.nn.NLLLoss(reduction='sum')

    def init_model(self):
        self._init_model()

    def _save_best_validation_model(self):
        torch.save(self.model, self._validation_path)

    def _fit_no_early_stopping(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train, lengths = self._format_final(x_train, y_train)
        x_val, y_val, _ = self._format_final(x_val, y_val)

        loader = DataLoader(
            TensorDataset(x_train, y_train, lengths), shuffle=True,
            batch_size=self._model_settings['batch_size']
        )
        if not self._transfer_boolean:
            self._init_model()
        print('MAXEPOCH', self.model.epochs)
        for epoch in range(self.model.epochs):
            print(epoch)
            self.model.train()
            for X_batch, y_batch, l_batch in loader:
                self._optimiser.zero_grad()
                self.model.primary_hidden = self.model.init_hidden(X_batch.size(0))
                self.model.secundary_hidden = self.model.secundary_init_hidden(X_batch.size(0))
                ypreds = self.model(X_batch, l_batch)
                loss = self._loss_fn(ypreds, y_batch)
                loss.backward()
                self._optimiser.step()

        # self.save()
        self._fold += 1

    def _fit_early_stopping(self, x_train:list, y_train:list, x_val:list, y_val:list):
        print('WHY AM I HERE')
        x_train, y_train, lengths = self._format_final(x_train, y_train)
        x_val, y_val, lengths_val = self._format_final(x_val, y_val)
        loader = DataLoader(
            TensorDataset(x_train, y_train, lengths), shuffle=self._model_settings['shuffle'],
            batch_size=self._model_settings['batch_size']
        )
        if not self._transfer_boolean:
            self._init_model()
        

        validation_loader = DataLoader(
            TensorDataset(x_val, y_val, lengths_val), shuffle=self._model_settings['shuffle'],
            batch_size=self._model_settings['batch_size']
        )
        loss_history = {'train': [], 'validation': []}

        best_loss = np.inf
        counter_patience = 0
        patience = self._model_settings['patience']
        for ep in range(self.model.epochs):
            print('   epoch {}'.format(ep))
            self.model.train()
            for X_batch, y_batch, l_batch in loader:
                # Train
                self._optimiser.zero_grad()
                # self.model.primary_hidden = self.model.init_hidden(X_batch.size(0))
                # self.model.secundary_hidden = self.model.secundary_init_hidden(X_batch.size(0))
                ypreds = self.model(X_batch, l_batch)
                # self.model.primary_hidden.detach()
                # self.model.secundary_hidden.detach()
                # ypreds = ypreds[:, 1]
                loss = self._loss_fn(ypreds, y_batch)
                loss.backward()
                self._optimiser.step()
                loss_history['train'].append(loss.detach().item())

            # Early Stopping
            self.model.eval()
            val_loss = 0
            for xval_batch, yval_batch, lval_batch in validation_loader:
                # self.model.primary_hidden = self.model.init_hidden(X_batch.size(0))
                # self.model.secundary_hidden = self.model.secundary_init_hidden(X_batch.size(0))
                val_outputs = self.model(xval_batch, lval_batch)
                # self.model.primary_hidden.detach()
                # self.model.secundary_hidden.detach()
                val_outputs = val_outputs[:, 1]
                curr_val_loss = self._loss_fn(val_outputs, yval_batch)
                val_loss += curr_val_loss.item()
            val_loss /= len(validation_loader)
            loss_history['validation'].append(val_loss)
            # print('VAL LOSS', val_loss, np.abs(val_loss - best_loss), np.abs(val_loss - best_loss) < self._model_settings['epsilon'])
            if val_loss >= best_loss or np.abs(val_loss - best_loss) <= self._model_settings['epsilon']:
                counter_patience += 1
                if counter_patience >= patience:
                    print('        Early Intermediate Stopping after {} epochs!'.format(ep))
                    if self._model_settings['save_best_model']:
                        self.load_weights(self._validation_path)
                    self._best_epochs = ep
                    break
            else:
                best_loss = val_loss
                counter_patience = 0
                # self._save_best_validation_model()
                

        # self.save()
        self._fold += 1
        return loss_history

    def predict(self, x:list, y:list) -> list:
        test_x, test_lens = self._format_features(x)
        test_x, test_lens = test_x.to(self.device), test_lens.to(self.device)
        dataset = TensorDataset(test_x, test_lens)
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.model.eval()
        predictions = torch.Tensor().to(self.device)
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                scores = self.model(batch_x, batch_len)
                probas = scores.cpu().detach().numpy()
                probas = [np.argmax(yp) for yp in probas]
                probas = torch.Tensor(probas).to(self.device)
                predictions = torch.cat((predictions, probas))
        print('prediction sizes', predictions.size())
        predictions = predictions.cpu().detach().numpy()
        return predictions, y
    
    def predict_proba(self, x:list) -> list:
        test_x, test_lens = self._format_features(x)
        test_x, test_lens = test_x.to(self.device), test_lens.to(self.device)
        dataset = TensorDataset(test_x, test_lens)
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.model.eval()
        predictions = torch.Tensor().to(self.device)
        with torch.no_grad():
            for batch_x, batch_len in testloader:
                scores = self.model(batch_x, batch_len)
                predictions = torch.cat((predictions, scores))
        predictions = predictions.cpu().detach().numpy()
        return predictions

    def save(self, extension='') -> str:
        path = '{}/{}/seed{}_nc{}_dropout{}_opadam_bs{}_ep{}_fold{}/'.format(
            self._experiment_name, self._outer_fold, self._settings['seeds']['model'], 
            self._model_settings['rnn_ncells'], 
            self._model_settings['rnn_dropout'], self._model_settings['batch_size'],
            self._model_settings['secundary_epochs'], self._gs_fold
        )
        os.makedirs(path, exist_ok=True)
        path += '{}_torch_object.pt'.format(extension)
        torch.save(self.model, path)
        self.model_path = path
        path = path.replace('_torch_object.pt', 'model.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path
    
    def get_model_path(self):
        return self.model_path

    def load_weights(self, path, transfer=False) -> str:
        self._load_weights_torch(path)
        self.model.update_model_settings(self._model_settings)
        if transfer:
            self.transfer()

    def get_path(self, fold: int) -> str:
        return self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        return self.save(extension='fold_{}'.format(fold))

    def save_fold_early(self, fold: int) -> str:
        return self.save(extension='fold_{}_len{}'.format(
            fold, self._maxlen
        ))