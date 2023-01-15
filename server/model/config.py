import socket
import copy
from shutil import copytree
import os
import time


dset_root = {}
dset_root['cub'] = '../data/simplifed_cub'

test_code = False 

if 'node' in socket.gethostname() or test_code:
    nfs_dset = copy.deepcopy(dset_root)
    if test_code:
        local_path = os.path.join(os.getenv("HOME"), 'my_local_test')
    else:
        local_path = '/local/image_datasets'

    if not os.path.isdir(local_path):
        os.makedirs(local_path)
    for x in dset_root.items():
        folder_name = os.path.basename(x[1])
        dset_root[x[0]] = os.path.join(local_path, folder_name)
        
def wait_dataset_copy_finish(dataset):
    flag_file = os.path.join(dset_root[dataset] + '_flag',
                            'flag_ready.txt')
    while True:
        with open(flag_file, 'r') as f:
            status = f.readline()
        if status == 'True':
            break
        time.sleep(600)


def setup_dataset(dataset):
    # my_tmp = os.path.join(os.getenv("HOME"), 'tmp')
    my_tmp = os.path.join(os.getcwd(), 'tmp')
    
    if not os.path.isdir(my_tmp):
        os.makedirs(my_tmp)
        os.environ["TMPDIR"] = my_tmp
    if 'node' in socket.gethostname():
        if not os.path.isdir(dset_root[dataset]):
            if os.path.isdir(os.path.join(dset_root[dataset] + '_flag')):
                wait_dataset_copy_finish(dataset)
            else:
                gypsum_copy_data_to_local(dataset)
        else:
            wait_dataset_copy_finish(dataset)


def gypsum_copy_data_to_local(dataset):
    flag_file = os.path.join(dset_root[dataset] + '_flag', 'flag_ready.txt')

    os.makedirs(dset_root[dataset] + '_flag')
    with open(flag_file, 'w') as f:
        f.write('False')
    if test_code:
        import pdb
        pdb.set_trace()
        pass
    copytree(nfs_dset[dataset], dset_root[dataset])

    if test_code:
        pdb.set_trace()
        pass
    with open(flag_file, 'w') as f:
        f.write('True')

