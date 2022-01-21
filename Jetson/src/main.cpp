#include <Arduino.h>
#include "variables.h"
#include "functions.hpp"
#include "NERFTrigger.hpp"
#include <NewPing.h>
#include <Encoder.h>
#include <Axle.hpp>
#include "Sonars.hpp"


// // Moduł przekaźników Iduino 2 kanały z optoizolacją - styki 10A/250VAC - cewka 5V
// // https://botland.com.pl/moduly-przekaznikow/14266-modul-przekaznikow-iduino-2-kanaly-z-optoizolacja-styki-10a250vac-cewka-5v-5903351242332.html
// // GND - IN1 - IN2 - VCC
// // Activated on LOW state !! - contrary to documentation
// // OR
// // // Moduł przekaźników 2 kanały - styki 10A/250VAC - cewka 5V
// // // https://botland.com.pl/przekazniki-przekazniki-arduino/2043-modul-przekaznikow-2-kanaly-styki-10a-250vac-cewka-5v-5904422302429.html
// // VCC - IN1 - IN2 - GND
// // Activated on LOW state 
// // Linear actuator - blue to IN1, green to IN2

float YawCalibrationCenter = 71.0f;
float PitchCalibrationCenter = 61.0f;
const int relayIN1_pin = 6; //LEFT on module
const int relayIN2_pin = 7; //RIGHT on module
int defaultNotEnergized = HIGH;

int lSpeed = 0;
int rSpeed = 0;

void Brake()
{
    lSpeed = 0;
    rSpeed = 0;
  MotorL_Brake();
  MotorR_Brake();
}

void turn(int speed) {
    if(lSpeed != speed && rSpeed != -speed)
    {
        Brake();
        delay(500);
        MotorL_Move(speed);
        MotorR_Move(-speed);
        lSpeed = speed;
        rSpeed = -speed;
    }
}

void forward(int speed) {
    if(lSpeed != speed && rSpeed != speed)
    {
        Brake();
        delay(500);
        MotorL_Move(speed);
        MotorR_Move(speed);
        lSpeed = speed;
        rSpeed = speed;
    }
}

class NERF {
private:
    NERFTrigger nerfTrigger;
    int darts;
public:
    NERF() {
        nerfTrigger = NERFTrigger(relayIN1_pin, relayIN2_pin, std::chrono::milliseconds(3000));
        nerfTrigger.initialize();
        reload();
    }

    void shoot() {
        if(!nerfTrigger.getCanShoot()) {
            Serial.println("Not ready");
            return;
        }
        
        if(darts>0) {
            nerfTrigger.shoot();
            darts-=1;
            Serial.println("Pew Pew Darts left: " + String(darts));
        } else Serial.println("No darts");
    }

    void reload() {
        darts = 6;
        Serial.println("Reloaded");
    }
};

NERF trigger;
Sonars sonars;

void setup()
{
    // Use to check for any errors in using timers
    // TeensyTimerTool::attachErrFunc(TeensyTimerTool::ErrorHandler(Serial));

    initSerial(115200);
    delay(3000);

    initMotors();
    Brake();
     // Each platform has to do this independently, checked manually
    calibrateServo(ServoSelector::Yaw, (int)YawCalibrationCenter);
    calibrateServo(ServoSelector::Pitch, (int)PitchCalibrationCenter);

    initServos();
    centerServos();
    sonars.update();
    delay(500);
}

String msg;
bool targetFound = true;
void loop()
{
    if(Serial.available()!=0) {
        msg = Serial.readStringUntil('\n');
        Serial.print(msg);
        if(msg == "shoot")
        {
            Brake();
            trigger.shoot();
        }
        else if(msg == "reload")
        {
            trigger.reload();
        }
        else if(msg=="left")
        {
            targetFound = true;
            turn(-80);
        }
        else if(msg=="right")
        {
            targetFound = true;
            turn(80);
        }
        else if(msg=="eliminated")
        {
            targetFound = false;
        }
        else if(msg=="brake")
        {
            Brake();
        }
        else if(msg=="forward")
        {
            forward(80);
        }
        else if(msg=="backward")
        {
            forward(-80);
        }

        // Serial.flush();
    }

    if(!targetFound)
    {
        if (sonars.getShortestDistance() < 30) turn(90);
        else if (sonars.getShortestDistance() > 50) forward(60);
    }
    sonars.update();
}
