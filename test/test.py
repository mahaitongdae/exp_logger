from src.logger import Logger
import src
import os

if __name__ == '__main__':
    logger = Logger(os.path.dirname(src.__file__),
                    args = {'algo': 'test',
                            'env_id': 'test-v1'})
