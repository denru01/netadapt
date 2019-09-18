import torch
import os
import network_utils as networkUtils
import nets as models
import unittest
import pickle
from constants import *
import subprocess
import common
import sys
import shutil


MODEL_ARCH = 'helloworld'
INPUT_DATA_SHAPE = (3, 32, 32)

FLOPS_LOOKUP_TABLE_PATH = os.path.join('models', MODEL_ARCH, 'lut.pkl')

MODEL_PATH = os.path.join('models', MODEL_ARCH, 'model_0.pth.tar')

model = models.__dict__[MODEL_ARCH]()
for i in range(4):
    layer = getattr(model.features, str(i*2))
    layer.weight.data = torch.zeros_like(layer.weight.data)
torch.save(model, MODEL_PATH)

DATASET_PATH = './'
network_utils = networkUtils.__dict__[MODEL_ARCH](model, INPUT_DATA_SHAPE, DATASET_PATH)
            
SHORT_TERM_FINE_TUNE_ITERATION = 5
MAX_ITERS = 3
BUDGET_RATIO = 0.8
INIT_REDUCTION_RATIO = 0.025
REDUCTION_DECAY = 1.0
FINETUNE_LR = 0.001
SAVE_INTERVAL = 1


def run_master(working_folder, resource_type='WEIGHT',
               budget_ratio=BUDGET_RATIO,
               budget=None,
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               init_reduction=None,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               max_iters=MAX_ITERS,
               lookup_table_path=None,
               resume=False):
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
        print('Create directory', working_folder)
    with open(os.path.join(working_folder, 'master_log.txt'), 'w') as file_id:
        command_list = [sys.executable, 'master.py', working_folder, str(3), str(32), str(32), 
                        '-im', MODEL_PATH, 
                        '-gp', str(0), str(1), str(2), 
                        '-mi', str(max_iters), 
                        '-bur', str(budget_ratio),
                        '-rt', resource_type, 
                        '-irr', str(init_reduction_ratio), 
                        '-rd', str(reduction_decay), 
                        '-lr', str(finetune_lr),
                        '-st', str(short_term_fine_tune_iteration),
                        '-dp', DATASET_PATH,
                        '--arch', MODEL_ARCH,
                        '-si', str(SAVE_INTERVAL)]
        if lookup_table_path != None:
            command_list = command_list + ['-lt', lookup_table_path]
        if resume:
            command_list = command_list + ['--resume']
        if init_reduction != None:
            command_list = command_list + ['-ir', str(init_reduction)]
        if budget != None:
            command_list = command_list + ['-bu', str(budget)]
        print(command_list)
        return subprocess.call(command_list, stdout=file_id, stderr=file_id)


class NormalUsage(unittest.TestCase):
    '''
        No ValueError
    '''
    def __init__(self, *args, **kwargs):
        super(NormalUsage, self).__init__(*args, **kwargs)
        
        
    def check_master_results(self, working_folder, acc_gt, res_gt, output_feature_gt, resource_type,  
                             short_term_fine_tune_iteration, max_iters, lookup_table_path):
        history_path = os.path.join(working_folder, 'master', 'history.txt')
        with open(history_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]    
        self.assertEqual(len(content)-1, max_iters+1, "master/history.txt length error")
        
        for i in range(1, len(content)):
            print('Check iteration {}'.format(i-1))
            tokens = content[i].split(',')
            print(tokens)
            
            # check accuracy
            load_model_path = tokens[4]
            saved_model = torch.load(load_model_path)  
            acc = network_utils.evaluate(saved_model)
            saved_acc = float(tokens[1])
            self.assertEqual(acc, saved_acc, "The accuracy of saved model is not equal to that in history.txt")
            self.assertEqual(acc_gt[i-1], acc, "The accuracy of saved model is incorrect")
            
            # check resource
            saved_network_def = network_utils.get_network_def_from_model(saved_model)
            resource = network_utils.compute_resource(saved_network_def, resource_type, lookup_table_path)
            saved_resource = float(tokens[2])
            self.assertEqual(resource, saved_resource, "The resource of saved model is not equal to that in history txt")
            self.assertEqual(res_gt[i-1], resource, "The resource of saved model is incorrect.")
            
            # check simplified block idx
            if i != 1:
                tokens_pre = content[i-1].split(',')
                output_features = tokens[5].split(' ')
                output_features_pre = tokens_pre[5].split(' ')
                find_simplified_block = False
                for output_idx in range(len(output_features)):
                    if output_features[output_idx] != output_features_pre[output_idx]:
                        if not find_simplified_block:
                            self.assertEqual(output_idx, int(tokens[3]), "Not simplify the block as described in master/history.txt")
                            self.assertEqual(output_features[output_idx], str(output_feature_gt[i-1][output_idx]), "Simplified block has incorrect # of output channels")
                            find_simplified_block = True
                        else:
                            self.assertEqual(1, 0, "Simplified block index error")    
                    else:
                        self.assertTrue(output_idx != int(tokens[3]), "Simplify the incorrect block")
                        
            # check network_def
            for idx in range(4):
                if idx == 0:
                    self.assertEqual(saved_network_def[idx], (3, output_feature_gt[i-1][idx]), "network_def of simplified model error")
                else:
                    self.assertEqual(saved_network_def[idx], (output_feature_gt[i-1][idx-1], output_feature_gt[i-1][idx]), "network_def of simplified model error")
            
            # check model weights
            for idx in range(4):
                layer = getattr(saved_model.features, str(idx*2))
                temp = (layer.weight.data == (torch.zeros_like(layer.weight.data) + idx*short_term_fine_tune_iteration*(i-1)))
                temp = torch.min(temp)
                temp = temp.item()
                self.assertTrue(temp, "Model weights after short-term fine-tune are incorrect")
                
            
    def test_master_weights(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_weights')
        res_gt = [29232, 28476, 27531, 26181]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               max_iters=MAX_ITERS,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=None)
        shutil.rmtree(working_folder)
        
    
    def test_master_flops_with_built_lookup_table(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_flops_with_built_lookup_table')
        lookup_table_path = os.path.join('models', MODEL_ARCH, 'lut.pkl')
        res_gt = [29232*32*32, 28476*32*32, 27531*32*32, 26181*32*32]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='FLOPS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               max_iters=MAX_ITERS,
               lookup_table_path=lookup_table_path,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='FLOPS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=lookup_table_path)
        shutil.rmtree(working_folder)
        
        
    def test_master_flops_without_built_lookup_table(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_flops_without_built_lookup_table')
        lookup_table_path = os.path.join('models', MODEL_ARCH, 'not_built_flops_lut.pkl')
        res_gt = [29232*32*32, 28476*32*32, 27531*32*32, 26181*32*32]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='FLOPS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=lookup_table_path,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='FLOPS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=lookup_table_path)
        os.remove(lookup_table_path)
        shutil.rmtree(working_folder)
        
        
    def master_flops_without_built_lookup_table_resume(self, working_folder, max_iters_before_resume):
        # run `max_iters_before_resume` iteration 
        # modify history 
        # resume
        lookup_table_path = os.path.join('models', MODEL_ARCH, 'not_built_flops_lut.pkl')
        res_gt = [29232*32*32, 28476*32*32, 27531*32*32, 26181*32*32]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='FLOPS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               max_iters=max_iters_before_resume, 
               lookup_table_path=lookup_table_path,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='FLOPS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=max_iters_before_resume, lookup_table_path=lookup_table_path)
        
        with open(os.path.join(working_folder, 'master', 'history.pickle'), 'rb') as file_id:
            history_pkl = pickle.load(file_id)
        his_args = history_pkl['master_args']
        his_args.max_iters = MAX_ITERS
        history_pkl['master_args'] = his_args
        with open(os.path.join(working_folder, 'master', 'history.pickle'), 'wb') as file_id:
            pickle.dump(history_pkl, file_id)
            
        run_master(working_folder, resource_type='FLOPS',
               budget_ratio=0,  
               init_reduction_ratio=0,
               reduction_decay=0, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               max_iters=max_iters_before_resume, 
               lookup_table_path=' ',
               resume=True)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='FLOPS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=lookup_table_path)
        
        os.remove(lookup_table_path)
        
        
    def test_master_weights_budget_met(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_weights_budget_met')
        res_gt = [29232, 28476]
        acc_gt = [95, 80]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0.9999,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=1, lookup_table_path=None)
        history_path = os.path.join(working_folder, 'master', 'history.txt')
        with open(history_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]    
        self.assertEqual(len(content)-1, 2, "Master does not terminate when budget is met") 
        shutil.rmtree(working_folder)
        
     
    def test_master_weights_use_init_reduction_not_ratio(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_weights_use_init_reduction_not_ratio')
        res_gt = [29232, 28476, 27531, 26181]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=0,
               init_reduction=INIT_REDUCTION_RATIO*29232,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=None) 
        shutil.rmtree(working_folder)
        
    
    def test_master_weights_use_budget_not_ratio(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_weights_use_budget_not_ratio')
        res_gt = [29232, 28476, 27531, 26181]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0,
               budget=29232*BUDGET_RATIO,
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=None)  
        shutil.rmtree(working_folder)
        
    
    def test_master_delete_previous_files_and_not_resume(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_delete_previous_files_and_not_resume')
        res_gt = [29232, 28476, 27531, 26181]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 62, 10],
                             [13, 32, 62, 10],
                             [13, 30, 62, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        for file in os.listdir(os.path.join(working_folder, 'master')):
            os.remove(os.path.join(working_folder, 'master', file))
        for file in os.listdir(os.path.join(working_folder, 'worker')):
            os.remove(os.path.join(working_folder, 'worker', file))
        returncode = run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=None)
        self.assertEqual(returncode, 0, "Master function error when all previous files are deleted.")
        shutil.rmtree(working_folder)
        
        
    def test_master_resume(self):
        max_iters_before_resume = 1
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_flops_without_built_lookup_table-resume_' + str(max_iters_before_resume))
        self.master_flops_without_built_lookup_table_resume(working_folder, max_iters_before_resume)
        shutil.rmtree(working_folder)
        
        max_iters_before_resume = 2
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_flops_without_built_lookup_table-resume_' + str(max_iters_before_resume))
        self.master_flops_without_built_lookup_table_resume(working_folder, max_iters_before_resume)
        shutil.rmtree(working_folder)
        
    def test_constraint_tight(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_constraint_tight')
        res_gt = [29232, 5418, 693]
        acc_gt = [95, 80, 85]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 1, 10],
                             [1, 32, 1, 10]
                             ]
        run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0,  
               init_reduction_ratio=1,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               max_iters=MAX_ITERS-1,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS-1, lookup_table_path=None)
        
        master_log_path = os.path.join(working_folder, 'master_log.txt')
        with open(master_log_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]    
        
        warning_counter = 0
        for line in content:
            if 'UserWarning' in line:
                warning_counter += 1
        self.assertEqual(warning_counter, 2, "Target resource warning by master error")        
        shutil.rmtree(working_folder)
        
    
    # normal usage
    # however, if constraint is too tight or not achievable, 
    # it will raise ValueError
    def test_constraint_not_achievable(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_constraint_not_achievable')
        res_gt = [29232, 5418, 693, 135]
        acc_gt = [95, 80, 85, 90]
        output_feature_gt = [[16, 32, 64, 10],
                             [16, 32, 1, 10],
                             [1, 32, 1, 10],
                             [1, 1, 1, 10]
                             ]
        returncode = run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0,  
               init_reduction_ratio=1,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               max_iters=MAX_ITERS+1,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.check_master_results(working_folder, acc_gt, res_gt, output_feature_gt, resource_type='WEIGHTS',
                             short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION, 
                             max_iters=MAX_ITERS, lookup_table_path=None)
        self.assertEqual(returncode, 1, "Master resource constraint not achievable error")
        
        master_log_path = os.path.join(working_folder, 'master_log.txt')
        with open(master_log_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]    
        
        warning_counter = 0
        for line in content:
            if 'UserWarning' in line:
                warning_counter += 1
        self.assertEqual(warning_counter, 3, "Target resource warning by master error")        
        shutil.rmtree(working_folder)
        
        
class ValueErrCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ValueErrCase, self).__init__(*args, **kwargs)
    
    
    def test_not_resume_and_previous_files_exist(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_not_resume')
        returncode = run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0,  
               init_reduction_ratio=1,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               max_iters=1,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.assertEqual(returncode, 0, "Normal master execution error")
        returncode = run_master(working_folder, resource_type='WEIGHTS',
               budget_ratio=0,  
               init_reduction_ratio=1,
               reduction_decay=REDUCTION_DECAY, 
               finetune_lr=FINETUNE_LR,
               max_iters=1,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=None,
               resume=False)
        self.assertEqual(returncode, 1, "Master does not detect the error incurred when previous files exist and `--resume` is not specified")
        shutil.rmtree(working_folder)
        
    
    def test_resume_no_lookup_table(self):
        working_folder = os.path.join('models', MODEL_ARCH, 'unittest_master_resume_no_lookuptable')
        lookup_table_path = os.path.join('models', MODEL_ARCH, 'not_built_flops_lut.pkl')
        run_master(working_folder, resource_type='FLOPS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               max_iters=1, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=lookup_table_path,
               resume=False)
        os.remove(lookup_table_path)
        with open(os.path.join(working_folder, 'master', 'history.pickle'), 'rb') as file_id:
            history_pkl = pickle.load(file_id)
        his_args = history_pkl['master_args']
        his_args.max_iters = MAX_ITERS
        history_pkl['master_args'] = his_args
        with open(os.path.join(working_folder, 'master', 'history.pickle'), 'wb') as file_id:
            pickle.dump(history_pkl, file_id)
        returncode = run_master(working_folder, resource_type='FLOPS',
               budget_ratio=BUDGET_RATIO,  
               init_reduction_ratio=INIT_REDUCTION_RATIO,
               reduction_decay=REDUCTION_DECAY, 
               max_iters=1, 
               finetune_lr=FINETUNE_LR,
               short_term_fine_tune_iteration=SHORT_TERM_FINE_TUNE_ITERATION,
               lookup_table_path=lookup_table_path,
               resume=True)
        self.assertEqual(returncode, 1, "Master does not detect the error incurred when resuming from previous iterations but lookup table not found")
        shutil.rmtree(working_folder)
        
        
        
if __name__ == '__main__':

    unittest.main()
