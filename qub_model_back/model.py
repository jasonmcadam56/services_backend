from EyeTrack import runner


def run(args):
    """
    :param args (list):
        --type (str)    : model type ie cnn or grid         eg... '--type=cnn'
        --test (flag)   : pass in to only do a test run     eg... '--test'
        --modelLoc (str): path to the model to be used      eg... '--modelLoc=<path>'
        --data(str)     : data set to pass to the model     eg... '--data=<path>'

        eg...   args = ['--type=cnn', 'test', ...]
    """

    runner.main(args)
