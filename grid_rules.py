import numpy as np
from math import pi, sin, exp, tanh

class Activator(object):
    '''abstract template for activation functions for neurons'''
    def __init__(self):
        raise NotImplementedError, 'abstract class, use specific function'

    def __cal__(self, grid):
        pass


class ThreshholdActivator(Activator):
    '''Simple, deterministic activator'''
    def __init__(self, reactivation_barrier, threshhold=.5):
        '''
        Parameters:
            reactivation_barrier (float): value that determines how difficult residual activation
        in the neuron makes reactivation
            threshhold (float): how readily the neuron activates. 0: never activates, 1: always active
        '''
        self.reactivation_barrier = reactivation_barrier
        self.threshhold = threshhold
 
    def __call__(self, grid):
        #sigmoid on activation potential in neuron grid
        grid.activation =  np.ceil(
            grid.activation - (self.reactivation_barrier * grid.neurons) - self.threshhold)
        #where activation was greater than threshhold, neurons are activated (set to 1)
        grid.neurons = np.where(grid.activation > grid.neurons, grid.activation, grid.neurons)


class Grid(object):
    '''
    A 2D grid of neurons. Neurons are connected only to their four (up, down, left, right)
    neighbors. The first and last row are the input and output layers, and the horizontal direction
    is periodic (cylindrical grid). An example network:

     I0     I1      I2    I3
    H00  H01  H02  H03
    H10  H11  H12  H13
     O0    O1    O2    O3

    Neuron H00 has neighbors I0, H01, H10, and H03.
    '''
    def __init__(self, shape, activator, activation_decay=1.0, reward_bias=0):
        '''
        Parameters:
            shape (tuple of two int): the height and width of the network
            activator (Activator): function that determines when neurons turn on
            activation_decay (0 <= float <= 1): governs the rate that neuron activation decays
        '''
        self.shape = shape
        self.activator = activator
        self.activation_decay = activation_decay
        self.reward_bias = reward_bias
        self.neurons = np.zeros(shape)
        self.symmetric_neurons = np.zeros(shape)
        self.activation = np.zeros(shape)
        self.biases = np.zeros(shape)
        #synapse sequence (donor to receiver): top, bottom, left, right:
        self.synapses = [np.random.uniform(0, 1, shape) for i in range(4)]
        #break connection between input and output layers:
        self.synapses[1][-1][:] = 0
        self.reward = 1.0 #leaves synapses unchanged. Positive reinforces synapses, negative degrades them

    def clear(self):
        self.neurons[:] = 0

    def synapse_clear(self):
        for synapse in self.synapses:
            synapse[:] = 0

    def set_activation(self):
        self.activation[:] = self.biases[:]
        for axis in 0, 1: #shifting rows or columns
            for direction in 0, 1: #increasing or decreasing index
                self.activation += self.synapses[2*axis + direction] * np.roll(
                    self.neurons, 2*direction-1, axis)

    def update_neurons(self, input_array):
        '''input_array (np_array): what values to set the input layer to'''
        self.neurons *= self.activation_decay
        self.activator(self)
        self.neurons[0][:] = input_array[:]
        self.symmetric_neurons = self.neurons - .5

    def update_reward(self, desired_output):
        change = .5 + self.reward_bias - np.mean(np.square(desired_output  - self.neurons[-1]))
        #self.reward += change*(1-abs(self.reward))

    def update_synapses(self):
        for axis in 0, 1:
            for direction in 0, 1:
                self.synapses[2*axis + direction] += self.reward * self.neurons * np.roll(
                    self.neurons, 2*direction-1, axis)
        self.synapses[1][-1][:] = 0

    def update(self, input_array, desired_output):
        self.update_neurons(input_array)
        self.set_activation()
        self.update_reward(desired_output)

def activation_test_suite():
    '''
    Checks that Grid correctly calculates neighbor contributions to activation
    Neurons are set to pattern
    1  0  1
    1  1  1
    0  0  1
    The active synapses per neuron are
    .1  .2  .3
    .4  .5  .6
    .7  .8  .9
    '''
    activator = ThreshholdActivator(.5, .4)
    grid = Grid((3,3), activator, 1.0)
    active_synapse = np.arange(.1, 1.0, .1).reshape((3, 3))
    neurons = np.array( [[1, 0, 1], [1, 1, 1], [0, 0, 1]] )
    grid.biases[:] = .1
    grid.neurons[:] = neurons

    def test_activation(expected, synapse):
        print 'testing activation function'
        grid.synapse_clear()
        grid.synapses[synapse][:] = active_synapse[:]
        grid.set_activation()
        assert np.all(np.round(grid.activation, 1) == expected
                      ), 'Error: unexpected output \n %r\n expected\n %r' %(
            grid.activation, expected)
    
    #test 1:
    expected = np.array( [[.1, .1, .4],
                                       [.5, .1, .7],
                                       [.8, .9, 1.0]] )
    test_activation(expected, 1)
    print 'Passed test of top synapse connections'
    
    #test 2
    grid.synapses[1][-1][:] = 0
    expected = np.array( [[.1, .1, .4],
                                       [.5, .1, .7],
                                       [.1, .1, .1]] )
    grid.set_activation()
    assert np.all(np.round(grid.activation, 1) == expected
                  ), 'Error: unexpected output \n %r\n expected\n %r'%(
            grid.activation, expected)
    print 'Passed test that breaking connection between input and output layers works'
    
    #test 3:
    expected = np.array( [[.2, .3, .4],
                                       [.1, .1, .7],
                                       [.8, .1, 1.0]] )
    test_activation(expected, 0)
    print 'Passed test of bottom synapse connections'
    
    #test 4
    expected = np.array( [[.2, .3, .1],
                                       [.5, .6, .7],
                                       [.8, .1, .1]] )
    test_activation(expected, 3)
    print 'Passed test of left synapse connections'
    
    #test 5
    expected = np.array( [[.1, .3, .4],
                                       [.5, .6, .7],
                                       [.1, .9, .1]] )
    test_activation(expected, 2)
    print 'Passed test of right synapse connections'
    grid.neurons[:] = .5
    grid.update_neurons(np.array([-.5, -.5, -.5]))
    expected = np.array( [[-.5, -.5, -.5],
                                       [.5, .5, 1],
                                      [.5, 1, .5]] )
    assert np.all(grid.neurons == expected
                  ), 'Error in activator. Unexpected:\n%r\nExpected:\n%r\n' %(grid.neurons, expected)
    print 'Passed test of activator function'
    print 'All tests passed'


class Drive(object):
    '''abstract class for input functions for neural grids'''
    def __init__(self, length):
        self.clock = 0
        self.array = np.zeros(length)

    def __call__(self, dt):
        self.clock += dt
        return self.array


class PulseDrive(Drive):
    def __init__(self, length, period):
        self.multiplier = 2*pi/period
        Drive.__init__(self, length)

    def __call__(self, dt):
        self.clock += dt
        self.array[:] = sin(self.multiplier*self.clock)
        return self.array


if __name__ == '__main__':
    activation_test_suite()
