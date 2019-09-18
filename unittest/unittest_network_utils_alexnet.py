import torch
import os
import sys
sys.path.append(os.path.abspath('../'))
import pickle
import network_utils as networkUtils
import unittest
import nets as models
from constants import *
import copy

MODEL_ARCH = 'alexnet'
INPUT_DATA_SHAPE = (3, 224, 224)
LOOKUP_TABLE_PATH = os.path.join('../models', MODEL_ARCH, 'lut.pkl')
DATASET_PATH = '../data/'

model = models.__dict__[MODEL_ARCH](num_classes=10)
network_utils = networkUtils.__dict__[MODEL_ARCH](model, INPUT_DATA_SHAPE, DATASET_PATH)

class TestNetworkUtils_alexnet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNetworkUtils_alexnet, self).__init__(*args, **kwargs)
    
    
    def check_network_def(self, network_def, input_channels, output_channels, only_num_channels=False):
        self.assertEqual(len(network_def), 8, "network_def length error")
        layer_idx = 0
        
        kernel_size_gt = [(11, 11), (5, 5), (3, 3), (3, 3), (3, 3), (1, 1), (1, 1), (1, 1)]
        stride_gt = [(4, 4), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        padding_gt = [(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)]
        for layer_name, layer_properties in network_def.items():
            self.assertEqual(layer_properties[KEY_NUM_IN_CHANNELS], input_channels[layer_idx], "network_def num of input channels error")
            self.assertEqual(layer_properties[KEY_NUM_OUT_CHANNELS], output_channels[layer_idx], "network_def num of output channels error")
            self.assertFalse(layer_properties[KEY_IS_DEPTHWISE], "network_def is_depthwise error")
            self.assertEqual(layer_properties[KEY_GROUPS], 1, "network_def group error")
            self.assertEqual(layer_properties[KEY_KERNEL_SIZE], kernel_size_gt[layer_idx], "network_def kernel size error")
            self.assertEqual(layer_properties[KEY_PADDING], padding_gt[layer_idx], "network_def padding error")
            self.assertEqual(layer_properties[KEY_STRIDE], stride_gt[layer_idx], "network_def stride error")
  
            if layer_idx < 5:
                self.assertEqual(layer_properties[KEY_LAYER_TYPE_STR], 'Conv2d', "network_def layer type string error")
            else:
                self.assertEqual(layer_properties[KEY_LAYER_TYPE_STR], 'Linear', "network_def layer type string error")
            
            input_feature_map_spatial_size = [224, 27, 13, 13, 13, 1, 1, 1]
            output_feature_map_spatial_size = [55, 27, 13, 13, 13, 1, 1, 1]
            if not only_num_channels:
                self.assertEqual(layer_properties[KEY_INPUT_FEATURE_MAP_SIZE], [1, input_channels[layer_idx], 
                    input_feature_map_spatial_size[layer_idx], input_feature_map_spatial_size[layer_idx]],
                    "network_def input feature map size error")
                self.assertEqual(layer_properties[KEY_OUTPUT_FEATURE_MAP_SIZE], [1, output_channels[layer_idx], 
                    output_feature_map_spatial_size[layer_idx], output_feature_map_spatial_size[layer_idx]],
                    "network_def output feature map size error")
            #print(layer_idx)
            layer_idx += 1
       
        
    def gen_layer_weight(self, tensor):
        gen_tensor = torch.zeros_like(tensor)
        for i in range(gen_tensor.shape[0]):
            gen_tensor[i, ::] += i
        return gen_tensor
    
            
    def test_network_def(self):
        network_def = network_utils.get_network_def_from_model(model)
        #print(network_def)      
        
        input_channels = [3, 64, 192, 384, 256, 9216, 4096, 4096]
        output_channels = [64, 192, 384, 256, 256, 4096, 4096, 10]
        self.check_network_def(network_def, input_channels, output_channels)
        self.assertEqual(network_utils.get_num_simplifiable_blocks(), 7, "Num of simplifiable blocks error")
        
        
    def test_compute_resource(self):
        network_def = network_utils.get_network_def_from_model(model)
        num_w = network_utils.compute_resource(network_def, 'WEIGHTS')
        num_mac = network_utils.compute_resource(network_def, 'FLOPS')
        
        self.assertEqual(num_w, 57035456, "Num of weights error")
        self.assertEqual(num_mac, 710133440, "Num of MACs error")
        
        
    def test_extra_history_info(self):
        network_def = network_utils.get_network_def_from_model(model)
        output_feature_info = network_utils.extra_history_info(network_def)
        output_channels = [64, 192, 384, 256, 256, 4096, 4096, 10]
        output_channels_str = [str(x) for x in output_channels]
        output_feature_info_gt = ' '.join(output_channels_str)
        self.assertEqual(output_feature_info, output_feature_info_gt, "extra_history_info error")
        
        
    def delta_to_layer_num_channels(self, delta, simp_block_idx):
        input_channels_gt = [3, 64, 192, 384, 256, 9216, 4096, 4096]
        output_channels_gt = [64, 192, 384, 256, 256, 4096, 4096, 10]
        
        output_channels_gt[simp_block_idx]    = output_channels_gt[simp_block_idx] - delta
        if simp_block_idx == 4:
            input_channels_gt[simp_block_idx+1]   = input_channels_gt[simp_block_idx+1] - delta*36
        else:
            input_channels_gt[simp_block_idx+1]   = input_channels_gt[simp_block_idx+1] - delta

        return input_channels_gt, output_channels_gt
        
    
    def run_simplify_network_def_and_check_for_one_resource_type(self, constraint, resource_type, simp_block_indices, delta, res_gt):
        network_def = network_utils.get_network_def_from_model(model)
        
        for i in range(len(simp_block_indices)):
            simp_block_idx = simp_block_indices[i]
            simp_network_def, simp_resource = network_utils.simplify_network_def_based_on_constraint(network_def, simp_block_idx, constraint, resource_type)
            self.assertEqual(simp_resource, res_gt[i], "Simplified network resource {} error ({})".format(resource_type, simp_block_idx))
            input_channels_gt, output_channels_gt = self.delta_to_layer_num_channels(delta[i], simp_block_idx)
            self.check_network_def(simp_network_def, input_channels_gt, output_channels_gt, only_num_channels=True)
            print(i)
    
    def test_simplify_network_def_based_on_constraint(self):
        total_num_w   = 57035456 
        total_num_mac = 710133440 
        constraint_num_w = total_num_w*0.975
        constraint_num_mac = total_num_mac*0.975

        simp_block_indices = [0, 1, 4, 6]
        delta_w   = [56, 184, 16, 352]
        delta_mac = [8, 16, 40, 4088]
        
        num_w_gt   = [56746328, 56105152, 54639296, 55590144]
        num_mac_gt = [673355240, 682126016, 688660160, 693348112]
        
        self.run_simplify_network_def_and_check_for_one_resource_type(constraint=constraint_num_w, 
            resource_type="WEIGHTS", simp_block_indices=simp_block_indices, 
            delta=delta_w, res_gt=num_w_gt)
        self.run_simplify_network_def_and_check_for_one_resource_type(constraint=constraint_num_mac, 
            resource_type="FLOPS", simp_block_indices=simp_block_indices, 
            delta=delta_mac, res_gt=num_mac_gt)
        
    
    def test_simplify_model_based_on_network_def(self):
        network_def = network_utils.get_network_def_from_model(model)
        total_num_w   = 57035456
        constraint_num_w = total_num_w*0.975
        simp_block_indices = [0, 1, 4, 6]
        delta_w   = [56, 184, 16, 352]
        topk_w    = [8, 8, 240, 3744]
        
        conv_idx = [0, 3, 6, 8, 10]
        fc_idx   = [1, 4, 6]

        for i in range(len(simp_block_indices)):
            simp_block_idx = simp_block_indices[i]
            simp_network_def, _ = network_utils.simplify_network_def_based_on_constraint(network_def, 
                simp_block_idx, constraint_num_w, "WEIGHTS")
            simp_model = network_utils.simplify_model_based_on_network_def(simp_network_def, model)
            updated_network_def = network_utils.get_network_def_from_model(simp_model)
            input_channels_gt, output_channels_gt = self.delta_to_layer_num_channels(delta_w[i], simp_block_idx)
            self.check_network_def(updated_network_def, input_channels_gt, output_channels_gt)
            
            for block_idx in range(7):
                if block_idx < 5: # conv
                    layer = getattr(model, 'features')
                    layer = getattr(layer, str(conv_idx[block_idx]))
                    simp_layer = getattr(simp_model, 'features')
                    simp_layer = getattr(simp_layer, str(conv_idx[block_idx]))
                else:
                    layer = getattr(model, 'classifier')
                    layer = getattr(layer, str(fc_idx[block_idx - 5]))
                    simp_layer = getattr(simp_model, 'classifier')
                    simp_layer = getattr(simp_layer, str(fc_idx[block_idx - 5]))

                if block_idx != simp_block_idx and block_idx != simp_block_idx + 1:
                    equal_weight = (layer.weight.data == simp_layer.weight.data)
                    equal_bias   = (layer.bias.data == simp_layer.bias.data)
                    self.assertTrue(equal_weight.min(), "simplify_model_based_on_network_def modify unrelated layers (weights)")
                    self.assertTrue(equal_bias.min(), "simplify_model_based_on_network_def modify unrelated layers (biases)")
                elif block_idx == simp_block_idx:
                    layer_weight = layer.weight.data
                    layer_weight = layer_weight.view(layer_weight.shape[0], -1)
                    layer_weight_norm = layer_weight*layer_weight
                    layer_weight_norm = layer_weight_norm.sum(1)
                    _, kept_filter_idx = torch.topk(layer_weight_norm, topk_w[i], sorted=False)
                    kept_filter_idx, _ = torch.sort(kept_filter_idx)
                    
                    equal_prune_weights = (layer.weight.data[kept_filter_idx, ::] == simp_layer.weight.data)
                    self.assertTrue(equal_prune_weights.min(), "Output channels of the pruned layer error")
                    
                    equal_prune_biases = (layer.bias.data[kept_filter_idx] == simp_layer.bias.data)
                    self.assertTrue(equal_prune_biases.min(), "Output channels of the pruned layer error")
                    
                    # check the input features of the next layer
                    if (block_idx + 1) < 5: # conv
                        next_layer = getattr(model, 'features')
                        next_layer = getattr(next_layer, str(conv_idx[block_idx + 1]))
                        simp_next_layer = getattr(simp_model, 'features')
                        simp_next_layer = getattr(simp_next_layer, str(conv_idx[block_idx + 1]))
                    else:
                        next_layer = getattr(model, 'classifier')
                        next_layer = getattr(next_layer, str(fc_idx[(block_idx + 1) - 5]))
                        simp_next_layer = getattr(simp_model, 'classifier')
                        simp_next_layer = getattr(simp_next_layer, str(fc_idx[(block_idx + 1) - 5]))
                    
                    if block_idx != 4:   
                        if block_idx < 5:
                            equal_weights = (next_layer.weight.data[:, kept_filter_idx, ::] == simp_next_layer.weight.data)
                        else:
                            equal_weights = (next_layer.weight.data[:, kept_filter_idx] == simp_next_layer.weight.data)
                        self.assertTrue(equal_weights.min(), "Input channels of the layer after the pruned layer error")
                    else: # conv -> FC
                        kept_filter_idx_fc = []
                        for filter_idx in kept_filter_idx:
                            for i in range(36):
                                kept_filter_idx_fc.append(filter_idx*36 + i)
                        equal_weights = (next_layer.weight.data[:, kept_filter_idx_fc] == simp_next_layer.weight.data)
                        self.assertTrue(equal_weights.min(), "Input channels of the FC layer after the pruned conv layer error")
           
            
        
    def test_build_latency_lookup_table(self):
        network_def = network_utils.get_network_def_from_model(model)
        lookup_table_path = './unittest_lookup_table.plk'
        min_conv_feature_size = 32
        min_fc_feature_size   = 1024
        measure_latency_batch_size = 1
        measure_latency_sample_times = 1
        
        network_utils.build_lookup_table(network_def, 'LATENCY', lookup_table_path, min_conv_feature_size,
                                         min_fc_feature_size, measure_latency_batch_size, measure_latency_sample_times)
        
        with open(lookup_table_path, 'rb') as file_id:
            lookup_table = pickle.load(file_id)
        self.assertEqual(len(lookup_table), 8, "Lookup table length error")
        
        input_channels_gt = [3, 64, 192, 384, 256, 9216, 4096, 4096]
        output_channels_gt = [64, 192, 384, 256, 256, 4096, 4096, 10]
        layer_idx = 0
        
        for layer_name, layer_properties in lookup_table.items():
            self.assertEqual(layer_properties[KEY_IS_DEPTHWISE], network_def[layer_name][KEY_IS_DEPTHWISE], "lookup table layer properties error (is_depthwise)")
            self.assertEqual(layer_properties[KEY_NUM_IN_CHANNELS], network_def[layer_name][KEY_NUM_IN_CHANNELS], "lookup table layer properties error (num_in_channels)")
            self.assertEqual(layer_properties[KEY_NUM_OUT_CHANNELS], network_def[layer_name][KEY_NUM_OUT_CHANNELS], "lookup table layer properties error (num_out_channels)")
            self.assertEqual(layer_properties[KEY_KERNEL_SIZE], network_def[layer_name][KEY_KERNEL_SIZE], "lookup table layer properties error (kernel_size)")
            self.assertEqual(layer_properties[KEY_STRIDE], network_def[layer_name][KEY_STRIDE], "lookup table layer properties error (stride)")
            self.assertEqual(layer_properties[KEY_PADDING], network_def[layer_name][KEY_PADDING], "lookup table layer properties error (padding)")
            self.assertEqual(layer_properties[KEY_GROUPS], network_def[layer_name][KEY_GROUPS], "lookup table layer properties error (groups)")
            self.assertEqual(layer_properties[KEY_LAYER_TYPE_STR], network_def[layer_name][KEY_LAYER_TYPE_STR], "lookup table layer properties error (layer_type_str)")
            self.assertEqual(layer_properties[KEY_INPUT_FEATURE_MAP_SIZE], network_def[layer_name][KEY_INPUT_FEATURE_MAP_SIZE], "lookup table layer properties error (input_feature_size)")
        
            layer_latency_table = layer_properties[KEY_LATENCY]
            
            num_in_samples = input_channels_gt[layer_idx]
            num_output_samples = output_channels_gt[layer_idx]
            if layer_idx < 5:
                if num_in_samples < min_conv_feature_size:
                    num_in_samples = 1
                else: 
                    num_in_samples = num_in_samples/min_conv_feature_size
                num_output_samples = num_output_samples/min_conv_feature_size
            else:
                num_in_samples = num_in_samples/min_fc_feature_size
                if num_output_samples < min_fc_feature_size:
                    num_output_samples = 1
                else:
                    num_output_samples = num_output_samples/min_fc_feature_size
                    
            self.assertEqual(len(layer_latency_table), num_in_samples*num_output_samples, "Layerwise latency dict length error (layer index: {})".format(layer_idx))
            layer_idx += 1
            
        os.remove(lookup_table_path)
        
if __name__ == '__main__':
    unittest.main()
