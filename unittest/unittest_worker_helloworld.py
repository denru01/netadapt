import torch
import os
import sys
sys.path.append(os.path.abspath('../'))

import network_utils as networkUtils
import unittest
from constants import *
import subprocess
import time
import common
import sys
import nets as models
import shutil


MODEL_ARCH = 'helloworld'
INPUT_DATA_SHAPE = (3, 32, 32)

FLOPS_LOOKUP_TABLE_PATH = os.path.join('../models', MODEL_ARCH, 'lut.pkl')

WORKER_FOLDER = os.path.join('../models', MODEL_ARCH, 'unittest_worker')
if not os.path.exists(WORKER_FOLDER):
    os.mkdir(WORKER_FOLDER)
    print('Create directory', WORKER_FOLDER)
_WORKER_PY_FILENAME = '../worker.py'

MODEL_PATH = os.path.join('../models', MODEL_ARCH, 'model_0.pth.tar')

model = models.__dict__[MODEL_ARCH]()
for i in range(4):
    layer = getattr(model.features, str(i*2))
    layer.weight.data = torch.zeros_like(layer.weight.data)
torch.save(model, MODEL_PATH)

DATASET_PATH = './'
network_utils = networkUtils.__dict__[MODEL_ARCH](model, INPUT_DATA_SHAPE, DATASET_PATH)
            

class TestWorker_helloworld(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWorker_helloworld, self).__init__(*args, **kwargs)
        
        
    def run_worker(self, constraint, netadapt_iteration, block, resource_type, short_term_fine_tune_iteration, finetune_lr=0.001, lookup_table_path=''):
         gpu = 0
         with open(os.path.join(WORKER_FOLDER, common.WORKER_LOG_FILENAME_TEMPLATE.format(netadapt_iteration, block)), 'w') as file_id:
             command_list = [sys.executable, _WORKER_PY_FILENAME, WORKER_FOLDER, \
                             MODEL_PATH, str(block), resource_type, str(constraint), \
                             str(netadapt_iteration), str(short_term_fine_tune_iteration), str(gpu), \
                             lookup_table_path, DATASET_PATH] + [str(e) for e in INPUT_DATA_SHAPE] + [MODEL_ARCH] + [str(finetune_lr)]
             print(command_list)
             return subprocess.call(command_list, stdout=file_id, stderr=file_id)
             #return os.system(' '.join(command_list))
       
        
    def check_worker_simplify_and_finetune(self, constraint, netadapt_iteration, block, resource_type, 
                                           short_term_fine_tune_iteration, resource_gt, acc_gt, network_def_gt,
                                           finetune_lr=0.001, lookup_table_path=''):
        t = time.time()
        returncode = self.run_worker(constraint, netadapt_iteration, block, 
                                         resource_type, short_term_fine_tune_iteration, 
                                         finetune_lr=0.001, lookup_table_path=lookup_table_path)
        print('Worker finish time: {}s'.format(time.time() - t))
        
        # Check return code
        self.assertEqual(returncode, 0, "Normal worker return value error")
        saved_model = torch.load(os.path.join(WORKER_FOLDER,
                            common.WORKER_MODEL_FILENAME_TEMPLATE.format(netadapt_iteration, block)))
        acc = network_utils.evaluate(saved_model)
        network_def_saved_model = network_utils.get_network_def_from_model(saved_model)
        res = network_utils.compute_resource(network_def_saved_model, resource_type=resource_type, lookup_table_path=lookup_table_path)
            
        with open(os.path.join(WORKER_FOLDER, common.WORKER_ACCURACY_FILENAME_TEMPLATE.format(netadapt_iteration, block)),
                  'r') as file_id:
            saved_acc = float(file_id.read())
        with open(os.path.join(WORKER_FOLDER, common.WORKER_RESOURCE_FILENAME_TEMPLATE.format(netadapt_iteration, block)),
                  'r') as file_id:
            saved_res = float(file_id.read())
            
        self.assertEqual(acc, acc_gt, "Evaluation of simplified model error")
        self.assertEqual(acc, saved_acc,   "The value in accuracy file is not equal to accuracy of somplified model")
        self.assertEqual(res, resource_gt, "Resource of simplified model error")
        self.assertEqual(res, saved_res,   "The value in resource file is not equal to resource of somplified model")
        
        for idx in range(4):
            layer = getattr(saved_model.features, str(idx*2))
            temp = (layer.weight.data == (torch.zeros_like(layer.weight.data) + idx*short_term_fine_tune_iteration))
            temp = torch.min(temp)
            temp = temp.item()
            self.assertTrue(temp, "Model weights after short-term fine-tune are incorrect")
            
            self.assertEqual(network_def_saved_model[idx], network_def_gt[idx], "network_def of simplified model is incorrect")
        return
        
        
    def test_worker_simplify_and_finetune_weights(self):
        '''
            Check simplifying and finetuning block 0~2
        '''

        netadapt_iteration = 2
        all_resource_weights = 29232
        
        constraint_weights = [all_resource_weights -  3*3*(8*(3 + 32)), 
                              all_resource_weights -  3*3*(7*(16 + 64)),
                              all_resource_weights -  3*3*(31*(32 + 10))]
        resource_weights_gt = [all_resource_weights - 3*3*(8*(3 + 32)), 
                              all_resource_weights -  3*3*(7*(16 + 64)),
                              all_resource_weights -  3*3*(31*(32 + 10))]
        eval_acc_gt = [1, 5, 80]

        network_def_gt = [
                [(3, 8), (8, 32), (32, 64), (64, 10)],
                [(3, 16), (16, 25), (25, 64), (64, 10)],
                [(3, 16), (16, 32), (32, 33), (33, 10)]
                ]

        for block in range(3):
            self.check_worker_simplify_and_finetune(constraint=constraint_weights[block], 
                netadapt_iteration=netadapt_iteration, block=block, 
                resource_type="WEIGHTS", short_term_fine_tune_iteration=block, finetune_lr=0.001, 
                lookup_table_path='', resource_gt=resource_weights_gt[block], acc_gt=eval_acc_gt[block], 
                network_def_gt=network_def_gt[block])
        return
        
        
    def test_worker_simplify_and_finetune_flops(self):
        '''
            Check simplifying and finetuning block 0~2
        '''

        netadapt_iteration = -1
        all_resource_flops = 29232*32*32
        
        constraint_flops = [all_resource_flops -  3*3*(8*(3 + 32))*32*32, 
                              all_resource_flops -  3*3*(7*(16 + 64))*32*32,
                              all_resource_flops -  3*3*(31*(32 + 10))*32*32]
        resource_flops_gt = [all_resource_flops - 3*3*(8*(3 + 32))*32*32, 
                              all_resource_flops -  3*3*(7*(16 + 64))*32*32,
                              all_resource_flops -  3*3*(31*(32 + 10))*32*32]
        
        
        eval_acc_gt = [1, 5, 80]
        
        network_def_gt = [
                [(3, 8), (8, 32), (32, 64), (64, 10)],
                [(3, 16), (16, 25), (25, 64), (64, 10)],
                [(3, 16), (16, 32), (32, 33), (33, 10)]
                ]

        for block in range(3):
            self.check_worker_simplify_and_finetune(constraint=constraint_flops[block], 
                netadapt_iteration=netadapt_iteration, block=block, 
                resource_type="FLOPS", short_term_fine_tune_iteration=block, finetune_lr=0.001, 
                lookup_table_path=FLOPS_LOOKUP_TABLE_PATH, resource_gt=resource_flops_gt[block], acc_gt=eval_acc_gt[block], 
                network_def_gt=network_def_gt[block])
        return
    
    
    def test_worker_weights_tight_constraint(self):
        '''
            Check simplifying and finetuning block 0~2
        '''

        netadapt_iteration = 2
        all_resource_weights = 29232
        
        constraint_weights = [0, 1, -1]
        resource_weights_gt = [all_resource_weights - 3*3*(15*(3 + 32)), 
                              all_resource_weights -  3*3*(31*(16 + 64)),
                              all_resource_weights -  3*3*(63*(32 + 10))]
        eval_acc_gt = [1, 5, 80]

        network_def_gt = [
                [(3, 1), (1, 32), (32, 64), (64, 10)],
                [(3, 16), (16, 1), (1, 64), (64, 10)],
                [(3, 16), (16, 32), (32, 1), (1, 10)]
                ]
       
        for block in range(3):
            self.check_worker_simplify_and_finetune(constraint=constraint_weights[block], 
                netadapt_iteration=netadapt_iteration, block=block, 
                resource_type="WEIGHTS", short_term_fine_tune_iteration=block, finetune_lr=0.001, 
                lookup_table_path='', resource_gt=resource_weights_gt[block], acc_gt=eval_acc_gt[block], 
                network_def_gt=network_def_gt[block])
        return
    
    
    def test_worker_block_out_of_bound(self):
        netadapt_iteration = -5
        all_resource_flops = 29232*32*32
        returncode = self.run_worker(constraint=all_resource_flops, 
                        netadapt_iteration=netadapt_iteration, block=5, 
                        resource_type="FLOPS", short_term_fine_tune_iteration=5, finetune_lr=0.001)

        self.assertEqual(returncode, 1, "Abnormal worker not detected")
        
       
if __name__ == '__main__':
    unittest.main()
    shutil.rmtree(WORKER_FOLDER)