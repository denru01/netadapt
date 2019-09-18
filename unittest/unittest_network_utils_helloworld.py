import torch
import os
import sys
import pickle
sys.path.append(os.path.abspath('../'))

import network_utils as networkUtils
import unittest
import nets as models
from constants import *
import copy

MODEL_ARCH = 'helloworld'
INPUT_DATA_SHAPE = (3, 32, 32)
LOOKUP_TABLE_PATH = os.path.join('../models', MODEL_ARCH,  'lut.pkl')

model = models.__dict__[MODEL_ARCH]()
for i in range(4):
    layer = getattr(model.features, str(i*2))
    layer.weight.data = torch.zeros_like(layer.weight.data)
model = model.cuda()

network_utils = networkUtils.helloworld(model, INPUT_DATA_SHAPE)

class TestNetworkUtils_helloworld(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNetworkUtils_helloworld, self).__init__(*args, **kwargs)
        
        
    def check_network_def(self, network_def, input_channels, output_channels):
        for layer_idx in range(len(network_def)):
            self.assertEqual(network_def[layer_idx][0], input_channels[layer_idx], "Number of input channels in network_def not match with that of the model")
            self.assertEqual(network_def[layer_idx][1], output_channels[layer_idx], "Number of output channels in network_def not match with that of the model")
            self.assertEqual(len(network_def), 4, "Number of layers in network_def is not equal to that in the original model")
    
    
    def test_num_sumplifiable_blocks(self):
        self.assertEqual(network_utils.get_num_simplifiable_blocks(), 3, "Number of simplifiabe blocks error")
    
    
    def test_network_def(self):
        network_def = network_utils.get_network_def_from_model(model)
        output_channels = [16, 32, 64, 10]
        input_channels = [3, 16, 32, 64]
        for layer_idx in range(len(network_def)):
            self.assertEqual(network_def[layer_idx][0], input_channels[layer_idx],  "Number of input channels in network_def not match with that of the model")
            self.assertEqual(network_def[layer_idx][1], output_channels[layer_idx], "Number of output channels in network_def not match with that of the model")
        self.assertEqual(len(network_def), 4, "Number of layers in network_def is not equal to that in the original model")
        
        
    def test_compute_weights(self):
        network_def = network_utils.get_network_def_from_model(model)
        num_w = network_utils.compute_resource(network_def, 'WEIGHTS')
        num_w_gt = 3*3*(3*16 + 16*32 + 32*64 + 64*10)
        self.assertEqual(num_w, num_w_gt, "Number of weights error")
        
        
    def test_compute_flops(self):
        network_def = network_utils.get_network_def_from_model(model)
        network_utils.build_lookup_table(network_def, resource_type='FLOPS', lookup_table_path=LOOKUP_TABLE_PATH)
        with open(LOOKUP_TABLE_PATH, 'rb') as file_id:
            lookup_table = pickle.load(file_id)
            self.assertEqual(len(lookup_table), 4, "Lookup table has wrong number of layers")
            for layer_idx in range(4):
                feature_flops_dict = lookup_table[layer_idx]
                for feature_points in feature_flops_dict.keys():
                    entry = feature_flops_dict[feature_points]
                    self.assertEqual(32*32*9*feature_points[0]*feature_points[1], entry, "Lookup table entry error")
        flops = network_utils.compute_resource(network_def, 'FLOPS', lookup_table_path=LOOKUP_TABLE_PATH)
        flops_gt = 3*3*(3*16 + 16*32 + 32*64 + 64*10)*32*32
        self.assertEqual(flops, flops_gt, "FLOPS estimation error")
    
    
    def test_extra_history_info(self):
        network_def = network_utils.get_network_def_from_model(model)
        network_def[0] = (3, 8)
        network_def[1] = (8, 32)
        num_filters_str = network_utils.extra_history_info(network_def)
        self.assertEqual(num_filters_str, "8 32 64 10", "Network architecture in extra_history_info is wrong")
        
        
    def test_simplift_network_def(self):
        network_def = network_utils.get_network_def_from_model(model)
        
        constraint_1 = 29232 - 3*3*(8*3 + 8*32)
        simp_network_def_1, simp_resource_1 = network_utils.simplify_network_def_based_on_constraint(network_def, 0, constraint_1, "WEIGHTS")
        self.check_network_def(simp_network_def_1, [3, 8, 32, 64], [8, 32, 64, 10])
        
        constraint_2 = 29232 - 3*3*(16*16 + 16*64)
        simp_network_def_2, simp_resource_2 = network_utils.simplify_network_def_based_on_constraint(network_def, 1, constraint_2, "WEIGHTS")
        self.check_network_def(simp_network_def_2, [3, 16, 16, 64], [16, 16, 64, 10])
        
        constraint_3 = 29232 - 3*3*(16*32 + 16*10)
        simp_network_def_3, simp_resource_3 = network_utils.simplify_network_def_based_on_constraint(network_def, 2, constraint_3, "WEIGHTS")
        self.check_network_def(simp_network_def_3, [3, 16, 32, 48], [16, 32, 48, 10])
    
    
    def test_simplift_network_def_constraint_too_tight(self):
        network_def = network_utils.get_network_def_from_model(model)
        
        constraint_1 = -1
        simp_network_def_1, simp_resource_1 = network_utils.simplify_network_def_based_on_constraint(network_def, 0, constraint_1, "WEIGHTS")
        self.check_network_def(simp_network_def_1, [3, 1, 32, 64], [1, 32, 64, 10])
        
        constraint_2 = 0
        simp_network_def_2, simp_resource_2 = network_utils.simplify_network_def_based_on_constraint(network_def, 1, constraint_2, "WEIGHTS")
        self.check_network_def(simp_network_def_2, [3, 16, 1, 64], [16, 1, 64, 10])
        
        constraint_3 = 1
        simp_network_def_3, simp_resource_3 = network_utils.simplify_network_def_based_on_constraint(network_def, 2, constraint_3, "WEIGHTS")
        self.check_network_def(simp_network_def_3, [3, 16, 32, 1], [16, 32, 1, 10])
        
    
    def test_simplify_model(self):    
        network_def = network_utils.get_network_def_from_model(model)
        
        constraint_1 = 29232 - 3*3*(9*3 + 9*32)
        simp_network_def_1, simp_resource_1 = network_utils.simplify_network_def_based_on_constraint(network_def, 0, constraint_1, "WEIGHTS")
        self.check_network_def(simp_network_def_1, [3, 7, 32, 64], [7, 32, 64, 10])
        simp_model_1 = network_utils.simplify_model_based_on_network_def(simp_network_def_1, model)
        update_network_def_1 = network_utils.get_network_def_from_model(simp_model_1)
        self.check_network_def(update_network_def_1, [3, 7, 32, 64], [7, 32, 64, 10])
        
        constraint_2 = 29232 - 3*3*(19*16 + 19*64)
        simp_network_def_2, simp_resource_2 = network_utils.simplify_network_def_based_on_constraint(network_def, 1, constraint_2, "WEIGHTS")
        self.check_network_def(simp_network_def_2, [3, 16, 13, 64], [16, 13, 64, 10])
        simp_model_2 = network_utils.simplify_model_based_on_network_def(simp_network_def_2, model)
        update_network_def_2 = network_utils.get_network_def_from_model(simp_model_2)
        self.check_network_def(update_network_def_2, [3, 16, 13, 64], [16, 13, 64, 10])
        
        constraint_3 = 29232 - 3*3*(15*32 + 15*10)
        simp_network_def_3, simp_resource_3 = network_utils.simplify_network_def_based_on_constraint(network_def, 2, constraint_3, "WEIGHTS")
        self.check_network_def(simp_network_def_3, [3, 16, 32, 49], [16, 32, 49, 10])
        simp_model_3 = network_utils.simplify_model_based_on_network_def(simp_network_def_3, model)
        update_network_def_3 = network_utils.get_network_def_from_model(simp_model_3)
        self.check_network_def(update_network_def_3, [3, 16, 32, 49], [16, 32, 49, 10])
        
        
        acc_1 = network_utils.evaluate(simp_model_1)
        self.assertEqual(acc_1, 1, "Evaluation function error")
        
        acc_2 = network_utils.evaluate(simp_model_2)
        self.assertEqual(acc_2, 5, "Evaluation function error")
        
        acc_3 = network_utils.evaluate(simp_model_3)
        self.assertEqual(acc_3, 80, "Evaluation function error")
        
        input = torch.randn((4, 3, 32, 32)).cuda()
        _ = simp_model_1(input)
        _ = simp_model_2(input)
        _ = simp_model_3(input)
        
    
    def test_finetune(self):
        model_0 = copy.deepcopy(model)

        for layer_idx in range(4):
            layer = getattr(model_0.features, str(layer_idx*2))
            layer.weight.data = torch.zeros_like(layer.weight.data)
            
        model_finetune = network_utils.fine_tune(model_0, iterations=1)
        for layer_idx in range(4):
            layer = getattr(model_finetune.features, str(layer_idx*2))
            temp = (layer.weight.data == (torch.zeros_like(layer.weight.data).cuda() + layer_idx))
            temp = torch.min(temp)
            temp = temp.item()
            self.assertTrue(temp, "Fintune function error when iteration = 1")
        
        model_finetune_0 = network_utils.fine_tune(model_0, iterations=0)
        for layer_idx in range(4):
            layer = getattr(model_finetune_0.features, str(layer_idx*2))
            temp = (layer.weight.data == (torch.zeros_like(layer.weight.data).cuda()))
            temp = torch.min(temp)
            temp = temp.item()
            self.assertTrue(temp, "Finetune function error when iteration = 0")
        

if __name__ == '__main__':
    unittest.main()
