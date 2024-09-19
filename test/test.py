from exp_logger.logger import Logger
import exp_logger
import os

if __name__ == '__main__':
    logger = Logger(os.path.dirname(exp_logger.__file__),
                    args = {'algo': 'test',
                            'env_id': 'test-v1'})
