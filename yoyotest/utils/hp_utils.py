import logging

from mlflow import log_param

logger = logging.getLogger(__name__)


def check_and_log_hp(names, hps, allow_extra=True):  # pragma: no cover
    """Check and log hyper-parameters.

    Args:
        names (list): names of all expected hyper parameters
        hps (dict): all hyper-parameters from the config file
        allow_extra (bool): Can have more hyper-parameters than explicitly stated
    """
    check_hp(names, hps, allow_extra=allow_extra)
    log_hp(names, hps)


def check_hp(names, hps, allow_extra=True):
    """Check if required hyper-parameters are all present.

    Args:
        names (list): names of all expected hyper parameters
        hps (dict): all hyper-parameters from the config file
        allow_extra (bool): Can have more hyper-parameters than explicitly stated
    """
    missing = set()
    for name in names:
        if name not in hps:
            missing.add(name)
    extra = hps.keys() - names

    if len(missing) > 0:
        logger.error('please add the missing hyper-parameters: {}'.format(missing))
    if len(extra) > 0 and not allow_extra:
        logger.error('please remove the extra hyper-parameters: {}'.format(extra))
    if len(missing) > 0 or (len(extra) > 0 and not allow_extra):
        raise ValueError('fix according to the error message above')


def log_hp(names, hps):  # pragma: no cover
    """Log the hyper-parameters.

    Args:
        names (list): list with names of hyper parameters to log
        hps (dict): all hyper-parameters from the config file
    """
    for name in sorted(names):
        log_param(name, hps[name])
        logger.info('\thp "{}" => "{}"'.format(name, hps[name]))
