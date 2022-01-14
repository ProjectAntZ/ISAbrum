import serial
import keyboard

ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
while True:
    msg = ser.readline().decode('ascii')
    if len(msg)>0:
        print(msg)
    if keyboard.is_pressed('s'):
        ser.write(bytes('shoot\n', 'ascii')) 

    if keyboard.is_pressed('r'):
        ser.write(bytes('reload\n', 'ascii')) 
        
    if keyboard.is_pressed('q'):
        break

ser.close()
