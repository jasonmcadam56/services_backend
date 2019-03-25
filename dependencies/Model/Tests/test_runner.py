from unittest import mock
import unittest
from EyeTrack import runner 

class TestRunner(unittest.TestCase):
    """
        This is a class used to test Model Runner.
    """

    @mock.patch('EyeTrack.runner.np.load')
    @mock.patch('EyeTrack.runner.ArgumentParser.parse_args')
    def test_main_test(self, parse_args, patch_load):
        parse_args.return_value = mock.Mock(verbose=True, data_arch='mit')
        patch_load.return_value = {'train_eye_left': [0], 'train_right_eye': [0]}
        runner.main()



if __name__ == '__main__':
    unittest.main()