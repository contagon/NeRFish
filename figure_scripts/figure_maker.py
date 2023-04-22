import matplotlib.pyplot as plt

class FigureMaker(object):
    def __init__(self):
        self.font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 22}

        plt.rc('font', **self.font)

        # Set the plot style to be beautiful.
        plt.style.use('seaborn')

        # And the colors to be pastel.
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Pastel1.colors)
        plt.rcParams['axes.facecolor'] = 'white'

        # And the grid to be white.
        plt.rcParams['axes.facecolor'] = 'white'