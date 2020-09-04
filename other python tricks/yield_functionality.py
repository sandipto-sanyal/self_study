def generator(seed: int = 0):
    '''
    This is a sample generator which yields natural numbers one by one
    This denotes python\'s lazy generation concept.

    Parameters
    ----------
    seed : int, optional
        The initial number from which to start the generation.
        The default is 0.

    Yields
    ------
    int
        The next natural number.

    '''
    yield seed # this will give the n upon invocation
    yield from generator(seed+1) # this will define what to yield next upon invocation
    
if __name__ == '__main__':
    s = generator(seed=10)
    print('Data type of generator: {}'.format(type(s))) # output is <class 'generator'>
    print('Next number upon generation: {}'.format(next(s))) # output is 10
    print('Next number upon generation: {}'.format(next(s))) # output is 11
    print('Next number upon generation: {}'.format(next(s))) # output is 12