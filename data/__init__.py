from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj
from data.dataset import Sen2_MTC_New_Multi, Sen2_MTC_New_Single, SEN12MS, SEN12OPTMS, SyntheticSEN12MS, \
    SyntheticSARSEN12MS, SyntheticSEN12MS_v2, SyntheticSEN12MS_v3_mask


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    #phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    phase_dataset = SyntheticSEN12MS_v3_mask(path = opt['datasets'][opt['phase']]['which_dataset']['args']['data_root'],
                                       mode = dataset_opt['args']['mode'], rand_use = dataset_opt['args']['rand_use'], use_mask = dataset_opt['args']['use_mask'])
    #val_dataset = None
    val_dataset_opt = opt['datasets']['val']['which_dataset']
    #val_dataset = init_obj(val_dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = SyntheticSEN12MS_v3_mask(opt['datasets']['val']['which_dataset']['args']['data_root'],
                                       'val', use_mask = dataset_opt['args']['use_mask'])
    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    # dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    # valid_split = dataloder_opt.get('validation_split', 0)    
    
    # ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
    # if valid_split > 0.0 or 'debug' in opt['name']: 
    #     if isinstance(valid_split, int):
    #         assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
    #         valid_len = valid_split
    #     else:
    #         valid_len = int(data_len * valid_split)
    #     data_len -= valid_len
    #     phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets
