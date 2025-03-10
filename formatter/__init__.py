import logging

from .Basic import BasicFormatter
from .cv.BasicResnetFormatter import BasicResnetFormatter
from .cv.Cifar10Formatter import Cifar10Formatter


logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "BasicResnet": BasicResnetFormatter,
    "Cifar10": Cifar10Formatter
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
