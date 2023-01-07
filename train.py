import argparse
import numpy as np
import data_loaders as module_data
import models.metrics as module_metric
import blocks.configs as module_config
import trainers as module_trainer
from models import *
from configs.config_parser import ConfigParser
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    # setup data_loaders instances
    data_loader = config.init_obj('data_loader', module_data)
    # data_loader.set_dataset()
    valid_data_loader = data_loader.split_validation()

    # build blocks blocks, then print to console
    model_name = config["model"]
    # print(f"the block size: {data_loader.get_block_size()}")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # print(device_ids)

    model_config = config.init_obj('config', module_config)
    print(model_config)

    model = eval(f"{model_name}(model_config)")
    logger.info(model)
    print(f"the devices: {device_ids}")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        # blocks = torch.nn.parallel.DistributedDataParallel(blocks, device_ids=device_ids)
    model = model.to(device)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # model_name attribute was added because when there is GPU blocks is of DataParallel
    trainer = config.init_obj('trainers', module_trainer)
    trainer.setup(
        model=model,
        train_data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        metric_ftns=metrics,
        config=config
    )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Transformer for simulating PacBio AAV sequencing reads')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--blocks', '--modelName'], type=str, target='modelName')
    # ]
    # config = ConfigParser.from_args(args, options)
    config = ConfigParser.from_args(args)
    main(config)
