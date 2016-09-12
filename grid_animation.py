import numpy as np
import pyglet
from pyglet.gl import *
from grid_rules import Grid, ThreshholdActivator, PulseDrive
from math import sqrt

def generate_quads(width, height, length):
    '''create information necessary for OpenGL to render quads, formatted for passing to the GPU
    Parameters:
        width (int): the width of the grid in neurons
        height (int): the height of the grid in neurons
        length (int): length of grid edges in pixels
    Returns:
        vertices, normals, colors, indices (lists of OpenGL types)
    '''
    #set up the coordinates of all corners. Note: neurons are CORNERS of the squares
    vertices = []
    colors = []
    normals = []
    for j in range(height):
        for i in range(width):
            vertices.extend([i*length, j*length, 1.0])
            colors.extend([i*1.0/width, j*1.0/height, (i+j)%2, 1.0])
            normals.extend([0, 0, 1])
    vertices = (GLfloat * len(vertices))(*vertices)
    colors = (GLfloat * len(colors))(*colors)
    normals = (GLfloat * len(normals))(*normals)
    
    #create list of quad indices
    indices = []
    for x in range(width-1):
        for y in range(height-1):
            index0 = y*width + x
            index1 = index0 + 1
            index2 = index1 + width
            index3 = index2 - 1
            if (x+y)%2:
                indices.extend([index0, index1, index2])
                indices.extend([index2, index3, index0])
            else:
                indices.extend([index0, index1, index3])
                indices.extend([index2, index3, index1])
    indices = (GLuint * len(indices))(*indices)
    return vertices, normals, colors, indices 


class GridDisplay(object):
    def __init__(self, grid, length):
        '''
        Parameters:
            grid (grid_rules.Grid)
            length (int): the size of square edges in the grid (in pixels)
        '''

        self.grid = grid
        self.height, self.width = grid.shape        
        self.window = pyglet.window.Window((self.width-1)*length, (self.height-1)*length)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.list = glGenLists(1) #creates a link to the GPU list location
        self.vertices, self.normals, self.colors, self.indices = generate_quads(
                                                                        self.width, self.height, length)     
        def draw():
            glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, self.vertices)
            glNormalPointer(GL_FLOAT, 0, self.normals)
            glColorPointer(4, GL_FLOAT, 0, self.colors)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)
            glPopClientAttrib()
        @self.window.event
        def on_draw():
            self.window.clear()
            draw()

    def set_colors(self):
        self.colors = []
        for el in np.nditer(self.grid.neurons):
            self.colors.extend([.9*sqrt(el), .6*el, el**2, 1.0])
        self.colors = (GLfloat*len(self.colors))(*self.colors)

    def update(self, input_line, desired_output):
        self.grid.update(input_line, desired_output)
        self.set_colors()

if __name__ == '__main__':
    threshhold = .6
    reactivation_barrier = 2.0
    width = 100
    height = 100
    length = 5
    activation_decay = .95
    reward_bias = 0
    
    activator = ThreshholdActivator(reactivation_barrier, threshhold)
    grid = Grid((height, width), activator, activation_decay, reward_bias)
    display = GridDisplay( grid, length )

    drive = PulseDrive(width, 4.0)
    def update(dt):
        display.update(np.ones(width), np.zeros(width))
    pyglet.clock.schedule_interval(update, 1.0/60)
    pyglet.app.run()
