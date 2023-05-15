import logging
logger = logging.getLogger('base')


def create_model1D(opt):
    from .model import DDPM1D as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m