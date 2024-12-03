from core.praser import init_obj
from models.network_x0_dpm_solver import Network
from models.model import Palette
def create_model(**cfg_model):
    """ create_model """
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    #model = init_obj(model_opt, logger, default_file_name='models.model', init_type='Model')
    print(cfg_model['networks'])
    print(model_opt['args'].keys())
    print(model_opt['args']['sample_num'])
    print(model_opt['args']['task'])
    print(model_opt['args']['optimizers'])
    print(model_opt['args']['ema_scheduler'])
    #print(cfg_model)
    #print(model_opt)
    model = Palette(model_opt['args']['networks'], model_opt['args']['losses'], model_opt['args']['sample_num'], model_opt['args']['task'], model_opt['args']['optimizers'], model_opt['args']['ema_scheduler'], phase_loader = model_opt['args']['phase_loader'], writer = model_opt['args']['writer'], logger = model_opt['args']['logger'], val_loader = model_opt['args']['val_loader'], metrics = model_opt['args']['metrics'], opt = opt)
    return model

def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    #net = init_obj(network_opt, logger, default_file_name='network_x0_dpm_solver', init_type='Network')
    net = Network(opt['model']['which_networks'][0]['args']['unet'],opt['model']['which_networks'][0]['args']['beta_schedule'], opt['model']['which_networks'][0]['args']['module_name'])
    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__, network_opt['args'].get('init_type', 'default')))
        net.init_weights()
    return net


def define_loss(logger, loss_opt):
    return init_obj(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

def define_metric(logger, metric_opt):
    return init_obj(metric_opt, logger, default_file_name='models.metric', init_type='Metric')

