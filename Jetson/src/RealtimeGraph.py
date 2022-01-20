from matplotlib import pyplot as plt

class RealtimeGraph:
    def __init__(self, length):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.figure.canvas.mpl_connect('close_event', lambda: exit(0))
        self.graph_line, = self.ax.plot(range(length), [0 for _ in range(length)])

    def setScale(self, min_y, max_y):
        plt.ylim([min_y, max_y])

    def show(self, y):
        self.graph_line.set_ydata(y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()