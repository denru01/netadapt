from .network_utils_abstract import NetworkUtilsAbstract
import os
import sys
import torch
import copy
import pickle
import warnings
sys.path.append(os.path.abspath('../'))
from constants import *


class networkUtils_helloworld(NetworkUtilsAbstract):

    def __init__(self, model, input_data_shape, dataset_path=None, finetune_lr=1e-3):
        super(networkUtils_helloworld).__init__()
        '''
            4 conv layers:
                conv1: 3, 16
                conv2: 16, 32
                conv3: 32, 64
                conv4: 64, 10
            
            Input:
                `model`: model from which we will get network_def.
                `input_data_shape`: (list) [C, H, W].
                `dataset_path`: (string) path to dataset.
                `finetune_lr`: (float) short-term fine-tune learning rate.        
        '''
        
        self.input_data_shape = input_data_shape
        self.lookup_table = None
        

    def get_network_def_from_model(self, model):    
        '''
            return network def (list) of the input model containing layerwise info
            
            Input:
                `model`: model we will get network_def from
            
            Output:
                `network_def`: (list) each element corresponds to one layer and 
                is a tuple (num_input_channels, num_output_channels)
        '''
        network_def = list()
        for idx in range(4):
            layer = getattr(model.features, str(idx * 2))
            network_def.append((layer.in_channels, layer.out_channels))
        return network_def
    
    
    def simplify_network_def_based_on_constraint(self, network_def, block, constraint, resource_type,
                                                 lookup_table_path=None):
        '''
            Derive how much a certain block of layers ('block') should be simplified 
            based on resource constraints.
            
            Input:
                `network_def`: (list) simplifiable network definition (conv). 
                defined in self.get_network_def_from_model(...)
                `block`: (int) index of block to simplify
                `constraint`: (float) representing the FLOPs/weights constraint the simplied model should satisfy
                `resource_type`: (string) `FLOPS`, `WEIGHTS`
                `lookup_table_path`: (string) path to lookup table. Here we construct lookup table for FLOPS and it is needed only when resource_type == 'FLOPS'
        
            Output:
                `simplified_network_def`: (list) simplified network_def whose resource is `simplified_resource`
                `simplified_resource`: (float) resource comsumption of `simplified_network_def`
        '''
        
        assert block < self.get_num_simplifiable_blocks()

        # Determine the number of filters and the resource consumption.
        simplified_network_def = copy.deepcopy(network_def)
        simplified_resource = None
        return_with_constraint_satisfied = False
        num_out_channels_try = list(range(network_def[block][1], 0, -1))
  
        for current_num_out_channels in num_out_channels_try:  
            simplified_network_def[block] = (simplified_network_def[block][0], current_num_out_channels)
            simplified_network_def[block+1] = (current_num_out_channels, simplified_network_def[block+1][1])
            simplified_resource = self.compute_resource(simplified_network_def, resource_type, lookup_table_path)
            if simplified_resource <= constraint:
                return_with_constraint_satisfied = True
                break
        if not return_with_constraint_satisfied:
            warnings.warn('Constraint not satisfied: constraint = {}, simplified_resource = {}'.format(constraint,
                                                                                         simplified_resource))
        return simplified_network_def, simplified_resource


    def simplify_model_based_on_network_def(self, simplified_network_def, model):
        '''
            Choose which filters to perserve (Here only the first `num_filters` filters will be perserved)
            
            Input:
                `simplified_network_def`: (list) network_def shows how each layer should be simplified.
                `model`: model to be simplified.
            
            Output:
                `simplified_model`: simplified model
        '''
        simplified_model = copy.deepcopy(model)
        
        for idx in range(self.get_num_simplifiable_blocks()):
            layer = getattr(simplified_model.features, str(idx * 2))            
            num_filters = simplified_network_def[idx][1] 
            # Here we keep the first `num_filters` weights
            # Not based on magnitude
            
            # update output channel weight 
            setattr(layer, WEIGHTSTRING, torch.nn.Parameter(getattr(layer, WEIGHTSTRING)[0:num_filters, :, :, :]))
            layer.out_channels = num_filters
            # update input channel weight (next layer)
            layer = getattr(simplified_model.features, str((idx+1)*2))
            setattr(layer, WEIGHTSTRING, torch.nn.Parameter(getattr(layer, WEIGHTSTRING)[:, 0:num_filters, :, :]))
            layer.in_channels = num_filters
        return simplified_model
    

    def extra_history_info(self, network_def):
        '''
            Output num of channels layerwise
            
            Input:
                `network_def`: (list) defined in self.get_network_def_from_model()
                
            Output:
                `num_filters_str`: (string) representing num of output channels
        '''
        num_filters_str = []
        for layer_idx in range(len(network_def)):
            num_filters_str.append(str(network_def[layer_idx][1]))
        num_filters_str = ' '.join(num_filters_str)
        return num_filters_str


    def _compute_weights(self, network_def):
        '''
            Compute the number of parameters of a whole network.
            (considering only weights)
            
            Input: 
                `network_def`: (list) defined in get_network_def_from_model()
            
            Output:
                `total_num_weights`: (float) num of weights
        '''
        total_num_weights = 0.0
        for layer_idx, layer_properties in enumerate(network_def):
            layer_num_weights = network_def[layer_idx][0] * network_def[layer_idx][1] * 3 * 3
            total_num_weights += layer_num_weights
        return total_num_weights
   
    
    def _compute_flops_from_lookup_table(self, network_def, lookup_table_path):
        # Note that it return FLOPs
        '''
            Compute FLOPs from a lookup table.
            
            Although num of FLOPs can be calculated, 
            we use lookup table here to show how NetAdapt framework uses lookup tables for resource estimation.
            
            Input:
                `network_def`: (list) defined in get_network_def_from_model()
                `lookup_table_path`: (string) path to lookup table
            
            Output:
                `resource`: (float) num of flops
        '''
        resource = 0
        if self.lookup_table == None:
            with open(lookup_table_path, 'rb') as file_id:
                self.lookup_table = pickle.load(file_id)
        for layer_idx in range(len(network_def)):
            if (network_def[layer_idx][0], network_def[layer_idx][1]) in self.lookup_table[layer_idx].keys():
                resource += self.lookup_table[layer_idx][(network_def[layer_idx][0], network_def[layer_idx][1])]
        return resource
    
    
    def build_lookup_table(self, network_def_full, resource_type, lookup_table_path):
        '''
            Build lookup table
            Here we only build a lookup table for FLOPs
        
            Input: 
                `network_def_full`: (list) defined in get_network_def_from_model()
                `resource_type`: not used here as we build 'FLOPS' here
                `lookup_table_path`: (string) path to save lookup table
        '''
        
        lookup_table = []
        print("Building lookup table.")
        for i in range(4):
            feature_map_resource = dict()
            for num_in_channels in range(network_def_full[i][0], 0, -1):
                for num_out_channels in range(network_def_full[i][1], 0, -1):
                    feature_map_resource[(num_in_channels, num_out_channels)] = num_in_channels*num_out_channels*32*32*3*3
            lookup_table.append(feature_map_resource)
        with open(lookup_table_path, 'wb') as file_id:
            pickle.dump(lookup_table, file_id)      
        return 

        
    def compute_resource(self, network_def, resource_type, lookup_table_path=None):
        '''
            Input: 
                `network_def`: (list) defined in get_network_def_from_model()
                `resource_type`: (string) 'FLOPS'/'WEIGHTS'
                `lookup_table_path`: (string) path to lookup table
                
            Output:
                resource: (float) num of flops or weights
        '''
        if resource_type == 'FLOPS':
            return self._compute_flops_from_lookup_table(network_def, lookup_table_path)
        else:
            return self._compute_weights(network_def)
    
    
    def get_num_simplifiable_blocks(self):
        '''
            4 conv layers
            
            the # of output channel of the last layer is not reducible
        '''
        return 3


    def fine_tune(self, model, iterations, print_frequency=100):
        '''
            do not finetune in this example
            
            please specify data loader, loss function, optimizer to customize finetuning in your case
            
            Input:
                `model`: model whose weights will be modified
                `iterations`: (int) num of iteration to change model weights
                `print_frequency`: (int) how often to print log info
                
            Output:
                `finetune_model`: model whose weights have been modified
        '''
        
        finetune_model = copy.deepcopy(model)
        for i in range(iterations):
            for idx in range(4):
                layer = getattr(finetune_model.features, str(idx * 2))
                layer.weight.data = layer.weight.data + idx
        return finetune_model


    def evaluate(self, model):
        '''
            for simplicity, we return a value determined by the network architecture
            
            please specify evaluate function in your case
            
            Input:
                `model`: model whose architecture will determine the output value
                
            Output:
                `acc`: (int) value depends on the input model architecture
        '''
        
        network_def = self.get_network_def_from_model(model)
        acc = 0
        if network_def[0][1] != 16 and network_def[1][1] == 32 and network_def[2][1] == 64:
            acc = 1
        elif network_def[0][1] == 16 and network_def[1][1] != 32 and network_def[2][1] == 64:
            acc = 5
        elif network_def[0][1] == 16 and network_def[1][1] == 32 and network_def[2][1] != 64:
            acc = 80
        elif network_def[0][1] != 16 and network_def[1][1] == 32 and network_def[2][1] != 64:
            acc = 85
        elif network_def[0][1] == 16 and network_def[1][1] != 32 and network_def[2][1] != 64:
            acc = 10
        elif network_def[0][1] != 16 and network_def[1][1] == 32 and network_def[2][1] != 64:
            acc = 12
        elif network_def[0][1] != 16 and network_def[1][1] != 32 and network_def[2][1] != 64:
            acc = 90
        else:
            acc = 95
            
        return acc


def helloworld(model, input_data_shape, dataset_path=None, finetune_lr=1e-3):
    return networkUtils_helloworld(model, input_data_shape, dataset_path, finetune_lr)