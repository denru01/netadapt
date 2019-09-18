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

MODEL_ARCH = 'mobilenet'
INPUT_DATA_SHAPE = (3, 224, 224)
LOOKUP_TABLE_PATH = os.path.join('../models', MODEL_ARCH, 'lut.pkl')
DATASET_PATH = '../data/'

model = models.__dict__[MODEL_ARCH](num_classes=10)
network_utils = networkUtils.__dict__[MODEL_ARCH](model, INPUT_DATA_SHAPE, DATASET_PATH)

class TestNetworkUtils_mobilenet(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNetworkUtils_mobilenet, self).__init__(*args, **kwargs)
    
    
    def check_network_def(self, network_def, input_channels, output_channels, only_num_channels=False):
        self.assertEqual(len(network_def), 28, "network_def length error")
        layer_idx = 0
        for layer_name, layer_properties in network_def.items():
            self.assertEqual(layer_properties[KEY_NUM_IN_CHANNELS], input_channels[layer_idx], "network_def num of input channels error")
            self.assertEqual(layer_properties[KEY_NUM_OUT_CHANNELS], output_channels[layer_idx], "network_def num of output channels error")
            
            if layer_idx % 2 == 1 and layer_idx != 27:
                self.assertTrue(layer_properties[KEY_IS_DEPTHWISE], "network_def is_depthwise error")
                self.assertEqual(layer_properties[KEY_GROUPS], layer_properties[KEY_NUM_IN_CHANNELS], "network_def group error")
            else:
                self.assertFalse(layer_properties[KEY_IS_DEPTHWISE], "network_def is_depthwise error")
                self.assertEqual(layer_properties[KEY_GROUPS], 1, "network_def group error")
            if layer_idx == 27 or (layer_idx % 2 == 0 and layer_idx != 0):
                self.assertEqual(layer_properties[KEY_KERNEL_SIZE], (1, 1), "network_def kernel size error")
                self.assertEqual(layer_properties[KEY_PADDING], (0, 0), "network_def padding error")
            else:
                self.assertEqual(layer_properties[KEY_KERNEL_SIZE], (3, 3), "network_def kernel size error")
                self.assertEqual(layer_properties[KEY_PADDING], (1, 1), "network_def padding error")
            if layer_idx != 27:
                self.assertEqual(layer_properties[KEY_LAYER_TYPE_STR], 'Conv2d', "network_def layer type string error")
            else:
                self.assertEqual(layer_properties[KEY_LAYER_TYPE_STR], 'Linear', "network_def layer type string error")
            if layer_idx in [0, 3, 7, 11, 23]:
                self.assertEqual(layer_properties[KEY_STRIDE], (2, 2),  "network_def stride error")
            else:
                self.assertEqual(layer_properties[KEY_STRIDE], (1, 1),  "network_def stride error")
            input_feature_map_spatial_size = [224, 112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 
                                              14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 1]
            output_feature_map_spatial_size = [112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 
                                              14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 1]
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
        #print(len(network_def))
        input_channels = [3, 32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
                          512, 512, 512, 512, 512, 512, 1024, 1024, 1024]
        output_channels = [32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
                          512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 10]
        self.check_network_def(network_def, input_channels, output_channels)
        self.assertEqual(network_utils.get_num_simplifiable_blocks(), 14, "Num of simplifiable blocks error")
        
        
    def test_compute_resource(self):
        network_def = network_utils.get_network_def_from_model(model)
        num_w = network_utils.compute_resource(network_def, 'WEIGHTS')
        num_mac = network_utils.compute_resource(network_def, 'FLOPS')
        self.assertEqual(num_w, 3195328, "Num of weights error")
        self.assertEqual(num_mac, 567726592, "Num of MACs error")
        
        
    def test_extra_history_info(self):
        network_def = network_utils.get_network_def_from_model(model)
        output_feature_info = network_utils.extra_history_info(network_def)
        output_channels = [32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
                          512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 10]
        output_channels_str = [str(x) for x in output_channels]
        output_feature_info_gt = ' '.join(output_channels_str)
        self.assertEqual(output_feature_info, output_feature_info_gt, "extra_history_info error")
        
    
    def delta_to_layer_num_channels(self, delta, simp_block_idx):
        input_channels_gt = [3, 32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
            512, 512, 512, 512, 512, 512, 1024, 1024, 1024]
        output_channels_gt = [32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
            512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 10]

        if simp_block_idx == 0:
            input_channels_gt[simp_block_idx + 1] = input_channels_gt[simp_block_idx + 1] - delta
            input_channels_gt[simp_block_idx + 2] = input_channels_gt[simp_block_idx + 2] - delta
            output_channels_gt[simp_block_idx]    = output_channels_gt[simp_block_idx] - delta
            output_channels_gt[simp_block_idx+1]  = output_channels_gt[simp_block_idx+1] - delta
        elif simp_block_idx != 13:
            print(input_channels_gt)
            print(output_channels_gt)
            input_channels_gt[2*simp_block_idx+1] = input_channels_gt[2*simp_block_idx+1] - delta
            input_channels_gt[2*simp_block_idx+2] = input_channels_gt[2*simp_block_idx+2] - delta
            output_channels_gt[2*simp_block_idx]  = output_channels_gt[2*simp_block_idx] - delta
            output_channels_gt[2*simp_block_idx+1]= output_channels_gt[2*simp_block_idx+1] - delta
        else:
            output_channels_gt[2*simp_block_idx] = output_channels_gt[2*simp_block_idx] - delta
            input_channels_gt[2*simp_block_idx+1] = input_channels_gt[2*simp_block_idx+1] - delta
        return input_channels_gt, output_channels_gt
    
    
    def run_simplify_network_def_and_check_for_one_resource_type(self, constraint, resource_type, simp_block_indices, delta, res_gt):
        network_def = network_utils.get_network_def_from_model(model)
        
        for i in range(len(simp_block_indices)):
            simp_block_idx = simp_block_indices[i]
            simp_network_def, simp_resource = network_utils.simplify_network_def_based_on_constraint(network_def, simp_block_idx, constraint, resource_type)
            self.assertEqual(simp_resource, res_gt[i], "Simplified network resource {} error".format(resource_type))
            input_channels_gt, output_channels_gt = self.delta_to_layer_num_channels(delta[i], simp_block_idx)
            self.check_network_def(simp_network_def, input_channels_gt, output_channels_gt, only_num_channels=True)
     
    
    def test_simplify_network_def_based_on_constraint(self):
        total_num_w   = 3195328
        total_num_mac = 567726592
        constraint_num_w = total_num_w*0.975
        constraint_num_mac = total_num_mac*0.975

        simp_block_indices = [0, 1, 5, 7, 9, 11, 13]
        delta_w   = [24, 56, 104, 80, 80, 56, 80]
        delta_mac = [16, 24, 48, 72, 72, 96, 288]
        
        num_w_gt   = [3192928, 3185864, 3114520, 3112688, 3112688, 3108808, 3112608]
        num_mac_gt = [547656192, 547781632, 553191232, 553148896, 553148896, 553233568, 553273024]
        
        self.run_simplify_network_def_and_check_for_one_resource_type(constraint=constraint_num_w, 
            resource_type="WEIGHTS", simp_block_indices=simp_block_indices, 
            delta=delta_w, res_gt=num_w_gt)
        self.run_simplify_network_def_and_check_for_one_resource_type(constraint=constraint_num_mac, 
            resource_type="FLOPS", simp_block_indices=simp_block_indices, 
            delta=delta_mac, res_gt=num_mac_gt)
        
    
    def test_simplify_model_based_on_network_def(self):
        network_def = network_utils.get_network_def_from_model(model)
        total_num_w   = 3195328
        constraint_num_w = total_num_w*0.975
        simp_block_indices = [0, 1, 5, 7, 9, 11, 13]
        delta_w   = [24, 56, 104, 80, 80, 56, 80]
        topk_w    = [8, 8, 152, 432, 432, 456, 944]

        for i in range(len(simp_block_indices)):
            simp_block_idx = simp_block_indices[i]
            simp_network_def, _ = network_utils.simplify_network_def_based_on_constraint(network_def, 
                simp_block_idx, constraint_num_w, "WEIGHTS")
            simp_model = network_utils.simplify_model_based_on_network_def(simp_network_def, model)
            updated_network_def = network_utils.get_network_def_from_model(simp_model)
            input_channels_gt, output_channels_gt = self.delta_to_layer_num_channels(delta_w[i], simp_block_idx)
            self.check_network_def(updated_network_def, input_channels_gt, output_channels_gt)
            
            conv_layers = getattr(model, 'model')
            simp_conv_layers = getattr(simp_model, 'model')
            for block_idx in range(14):
                module = getattr(conv_layers, str(block_idx))
                simp_module = getattr(simp_conv_layers, str(block_idx))
                if block_idx != simp_block_idx and block_idx != simp_block_idx + 1:
                    if block_idx != 0:
                        for layer_idx in ['0', '1', '3', '4']:
                            layer = getattr(module, layer_idx)
                            simp_layer = getattr(simp_module, layer_idx)
                            if layer_idx in ['0', '3']:
                                equal = (simp_layer.weight.data == layer.weight.data)
                                self.assertTrue(equal.min(), "simplify_model_based_on_network_def modify unrelated conv layers")
                            else:
                                equal_weight = (simp_layer.weight.data == layer.weight.data)
                                equal_bias   = (simp_layer.bias.data == layer.bias.data)
                                equal_num_features = (simp_layer.num_features == layer.num_features)
                                self.assertTrue(equal_weight.min(), "simplify_model_based_on_network_def modify unrelated batchnorm layers (weight)")
                                self.assertTrue(equal_bias.min(), "simplify_model_based_on_network_def modify unrelated batchnorm layers (bias)")
                                self.assertTrue(equal_num_features, "simplify_model_based_on_network_def modify unrelated batchnorm layers (num_features)")
                    else:
                        layer = getattr(module, '0')
                        simp_layer = getattr(simp_module, '0')
                        equal = (simp_layer.weight.data == layer.weight.data)
                        self.assertTrue(equal.min(), "simplify_model_based_on_network_def modify unrelated conv layers")
                        
                        layer = getattr(module, '1')
                        simp_layer = getattr(simp_module, '1')
                        equal_weight = (simp_layer.weight.data == layer.weight.data)
                        equal_bias   = (simp_layer.bias.data == layer.bias.data)
                        equal_num_features = (simp_layer.num_features == layer.num_features)
                        self.assertTrue(equal_weight.min(), "simplify_model_based_on_network_def modify unrelated batchnorm layers (weight)")
                        self.assertTrue(equal_bias.min(), "simplify_model_based_on_network_def modify unrelated batchnorm layers (bias)")
                        self.assertTrue(equal_num_features, "simplify_model_based_on_network_def modify unrelated batchnorm layers (num_features)")
                        
                elif block_idx == simp_block_idx:
                    # check (regular/pointwise layer output channels and input channels of the next depthwise layer)
                    # or check (pointwise layer output channels and nput features of the next FC layer)
                    if block_idx == 0:
                        layer = getattr(module, '0')
                        simp_layer = getattr(simp_module, '0')
                    else: # pointwise
                        # first check depthwise layer within the same block
                        layer = getattr(module, '0')
                        simp_layer = getattr(module, '0')
                        equal_dep = (layer.weight.data == simp_layer.weight.data)
                        self.assertTrue(equal_dep.min(), "Depthwise layer within the target block error")
                        
                        layer = getattr(module, '3')
                        simp_layer = getattr(simp_module, '3')
                        
                    layer_weight = layer.weight.data
                    weight_vector = layer_weight.view(layer_weight.shape[0], -1)
                    weight_norm = weight_vector*weight_vector
                    weight_norm = torch.sum(weight_norm, dim=1)
                    _, kept_filter_idx = torch.topk(weight_norm, topk_w[i], sorted=False)
                    kept_filter_idx, _ = kept_filter_idx.sort()
                    
                    weight_gt = layer_weight[kept_filter_idx, :, :, :]
                    weight_simp = simp_layer.weight.data
                    equal_weight = (weight_gt == weight_simp)
                    
                    self.assertTrue(equal_weight.min(), "Output channels of the pruned layer error")
                    
                    # modify input channels of the next few layers
                    if block_idx != 13: # depthwise -> batchnorm -> pointwise
                        next_module = getattr(conv_layers, str(block_idx+1))
                        simp_next_module = getattr(simp_conv_layers, str(block_idx+1))
                        
                        dep_layer = getattr(next_module, '0')
                        simp_dep_layer = getattr(simp_next_module, '0')
                        dep_layer_weight = dep_layer.weight.data[kept_filter_idx, :, :, :]
                        equal_dep_weights = (dep_layer_weight == simp_dep_layer.weight.data)
                        self.assertTrue(equal_dep_weights.min(), "Input channels of the depthwise layer after pruned layers error")
                        
                        batchnorm_layer = getattr(next_module, '1')
                        simp_batchnorm_layer = getattr(simp_next_module, '1')
                        batchnorm_layer_weight = batchnorm_layer.weight.data[kept_filter_idx]
                        equal_batchnorm_weights = (batchnorm_layer_weight == simp_batchnorm_layer.weight.data)
                        self.assertTrue(equal_batchnorm_weights.min(), "Weights of the batchnorm layer after pruned layers error")
                        
                        batchnorm_layer_bias = batchnorm_layer.bias.data[kept_filter_idx]
                        equal_batchnorm_bias = (batchnorm_layer_bias == simp_batchnorm_layer.bias.data)
                        self.assertTrue(equal_batchnorm_bias.min(), "Biases of the batchnorm layer after pruned layers error")
                        
                        equal_batchnorm_num_features = (len(kept_filter_idx) == simp_batchnorm_layer.num_features)
                        self.assertTrue(equal_batchnorm_num_features, "Number of features of the batchnorm layer after pruned layers error")
                        
                        pt_layer = getattr(next_module, '3')
                        simp_pt_layer = getattr(simp_next_module, '3')
                        pt_layer_weight = pt_layer.weight.data[:, kept_filter_idx, :, :]
                        equal_pt_weights = (pt_layer_weight == simp_pt_layer.weight.data)
                        self.assertTrue(equal_pt_weights.min(), "Input channels of the pointwise layer after pruned layers error")
                        
                    else: # FC
                        fc_layer = getattr(model, 'fc')
                        simp_fc_layer = getattr(simp_model, 'fc')
                        fc_layer_weight = fc_layer.weight.data
                        fc_layer_weight = fc_layer_weight[:, kept_filter_idx]
                        equal_fc_weights = (fc_layer_weight == simp_fc_layer.weight.data)
                        self.assertTrue(equal_fc_weights.min(), "Input features of FC layer error")
       
    def test_simplify_model_based_on_network_def_check_weights(self):
        # make sure we prune the correct filters by checking the weights of a pruned model
        # the weights of the original model are initialized to certain values
       
        total_num_w   = 3195328
        constraint_num_w = total_num_w*0.975
        simp_block_indices = [0, 1, 5, 7, 9, 11, 13]
        delta_w   = [24, 56, 104, 80, 80, 56, 80]
        topk_w    = [8, 8, 152, 432, 432, 456, 944]
        
        # initialze model weights
        model_init = copy.deepcopy(model)
        conv_layers = getattr(model_init, 'model')
        for block_idx in range(14):
            module = getattr(conv_layers, str(block_idx))
            
            # regular/depthwise
            layer = getattr(module, '0')
            layer.weight.data = self.gen_layer_weight(layer.weight.data)
            
            if block_idx != 0: # pointwise    
                layer = getattr(module, '3')
                layer.weight.data = self.gen_layer_weight(layer.weight.data)
        model_init.fc.weight.data = self.gen_layer_weight(model_init.fc.weight.data)        
        
        
        network_def = network_utils.get_network_def_from_model(model_init)

        for i in range(len(simp_block_indices)):
            simp_block_idx = simp_block_indices[i]
            simp_network_def, _ = network_utils.simplify_network_def_based_on_constraint(network_def, 
                simp_block_idx, constraint_num_w, "WEIGHTS")
            simp_model = network_utils.simplify_model_based_on_network_def(simp_network_def, model_init)
            updated_network_def = network_utils.get_network_def_from_model(simp_model)
            input_channels_gt, output_channels_gt = self.delta_to_layer_num_channels(delta_w[i], simp_block_idx)
            self.check_network_def(updated_network_def, input_channels_gt, output_channels_gt)

            simp_conv_layers = getattr(simp_model, 'model')
            for block_idx in range(14):
                if block_idx == simp_block_idx:
                    simp_module = getattr(simp_conv_layers, str(block_idx))

                    if block_idx == 0:
                        simp_layer = getattr(simp_module, '0')
                    else: # pointwise
                        simp_layer = getattr(simp_module, '3')
                    
                    for weight_idx in range(topk_w[i]):
                        equal_weights = (simp_layer.weight.data[weight_idx, ::] == delta_w[i] + weight_idx)
                        self.assertTrue(equal_weights.min(), "Weights of the pruned layers error")
                        
                    if simp_block_idx != 13:
                        # check the next depthwise layer
                        simp_module = getattr(simp_conv_layers, str(block_idx+1))
                        simp_layer = getattr(simp_module, '0')
                        for weight_idx in range(topk_w[i]):
                            equal_weights = (simp_layer.weight.data[weight_idx, ::] == delta_w[i] + weight_idx)
                            self.assertTrue(equal_weights.min(), "Weights of the pruned layers error")
        
            
    def test_build_latency_lookup_table(self):
        network_def = network_utils.get_network_def_from_model(model)
        lookup_table_path = './unittest_lookup_table.plk'
        min_conv_feature_size = 32
        min_fc_feature_size   = 128
        measure_latency_batch_size = 1
        measure_latency_sample_times = 1
        
        network_utils.build_lookup_table(network_def, 'LATENCY', lookup_table_path, min_conv_feature_size,
                                         min_fc_feature_size, measure_latency_batch_size, measure_latency_sample_times)
        
        with open(lookup_table_path, 'rb') as file_id:
            lookup_table = pickle.load(file_id)
        self.assertEqual(len(lookup_table), 28, "Lookup table length error")
        
        input_channels_gt = [3, 32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
            512, 512, 512, 512, 512, 512, 1024, 1024, 1024]
        output_channels_gt = [32, 32, 64, 64, 128, 128, 128, 128,  256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 
            512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 10]
        layer_idx = 0
        
        dep_layer_latency_dict_list = []
        pt_layer_latency_dict_list  = []
        
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
            if layer_idx != 27:
                if num_in_samples < min_conv_feature_size:
                    num_in_samples = 1
                else: 
                    num_in_samples = num_in_samples/min_conv_feature_size
                num_output_samples = num_output_samples/min_conv_feature_size
            else:
                num_in_samples = num_in_samples/min_fc_feature_size
                if num_output_samples < min_fc_feature_size:
                    num_output_samples = 1
            if layer_idx != 27 and layer_idx % 2 == 1: 
                self.assertEqual(len(layer_latency_table), num_in_samples, "Layerwise latency dict length error (layer index: {})".format(layer_idx))
            else:                 
                self.assertEqual(len(layer_latency_table), num_in_samples*num_output_samples, "Layerwise latency dict length error (layer index: {})".format(layer_idx))
        
            if layer_idx >= 13 and layer_idx <= 22:
                if layer_idx % 2 == 0: # pointwise layer
                    pt_layer_latency_dict_list.append(layer_latency_table)
                else: # depthwise layer
                    dep_layer_latency_dict_list.append(layer_latency_table)
            
            layer_idx += 1
            
        # check whether same layers have the same results
        for i in range(1, len(dep_layer_latency_dict_list)):
            latency_dict_gt = dep_layer_latency_dict_list[0]
            latency_dict    = dep_layer_latency_dict_list[i]
            for key, value in latency_dict_gt.items():
                self.assertEqual(latency_dict_gt[key], latency_dict[key], "Lookup talbe of same depthwise layers ({}) error".format(i))
        
        for i in range(1, len(pt_layer_latency_dict_list)):
            latency_dict_gt = pt_layer_latency_dict_list[0]
            latency_dict    = pt_layer_latency_dict_list[i]
            for key, value in latency_dict_gt.items():
                self.assertEqual(latency_dict_gt[key], latency_dict[key], "Lookup talbe of same pointwise layers ({}) error".format(i))
        
        os.remove(lookup_table_path)
    
if __name__ == '__main__':
    unittest.main()
