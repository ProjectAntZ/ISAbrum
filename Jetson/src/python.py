import serial
import keyboard
from matplotlib import pyplot as plt
import re


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


SENSORS_NUMBER = 3

ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
graph = RealtimeGraph(SENSORS_NUMBER)
graph.setScale(0, 200)
y = [0 for _ in range(SENSORS_NUMBER)]

while True:
    msg = ser.readline().decode('ascii')
    if len(msg) > 0:
        if 'Sensors: ' in msg:
            s = int(re.search('Sensors: (\d+)', msg).group(1))
            d = int(re.search('Distance: (\d+)', msg).group(1))
            y[s] = d
            graph.show(y)
        else:
            print(msg)
    if keyboard.is_pressed('s'):
        ser.write(bytes('shoot\n', 'ascii'))

    if keyboard.is_pressed('r'):
        ser.write(bytes('reload\n', 'ascii'))

    if keyboard.is_pressed('q'):
        break

ser.close()
