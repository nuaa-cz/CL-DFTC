import warnings
warnings.filterwarnings("ignore")

import sys
import logging
import argparse

from config import Config
from utils import tool_funcs
from task.airtrajcluster import AirTrajCluster
from model.airtrajcl import AirTrajCL

def parse_args():
    # don't set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description = "CL-DFTC/joint_train.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    enc_name = Config.trajcluster_encoder_name
    metrics = tool_funcs.Metrics()

    airtrajcl = AirTrajCL()
    airtrajcl.load_checkpoint()
    airtrajcl.to(Config.device)
    task = AirTrajCluster(airtrajcl, Config.trajcl_aug1, Config.trajcl_aug2)
    metrics.add(task.train())

    logging.info('[EXPFlag]model={},dataset={},{}'.format(enc_name, Config.dataset_prefix, str(metrics)))
    return


# nohup python joint_train.py --dataset pvg &> result &
if __name__ == '__main__':
    Config.dataset = 'pvg'
    Config.update(parse_args())

    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    main()
    