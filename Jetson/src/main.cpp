#include <Arduino.h>
#include "variables.h"
#include "functions.hpp"
#include "NERFTrigger.hpp"
#include <NewPing.h>
#include <Encoder.h>
#include "ISAMobile.h"
#include <Axle.hpp>


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

float YawCalibrationCenter = 80.0f;
float PitchCalibrationCenter = 58.0f;
const int relayIN1_pin = 6; //LEFT on module
const int relayIN2_pin = 7; //RIGHT on module
int defaultNotEnergized = HIGH;
bool canShoot = false;
#define SONAR_NUM      3 // Number of sensors.
#define MAX_DISTANCE 200 // Maximum distance (in cm) to ping.
#define PING_INTERVAL 30 // Milliseconds between sensor pings (29ms is about the min to avoid cross-sensor echo).

unsigned long pingTimer[SONAR_NUM]; // Holds the times when the next ping should happen for each sensor.
unsigned int distances[SONAR_NUM];         // Where the ping distances are stored.
uint8_t currentSensor = 0;          // Keeps track of which sensor is active.
bool isObstacle[SONAR_NUM] = {false};



NewPing sonar[SONAR_NUM] = {   // Sensor object array.
  NewPing(ultrasound_trigger_pin[(int)UltraSoundSensor::Left], 
                     ultrasound_echo_pin[(int)UltraSoundSensor::Left], 
                     MAX_DISTANCE), // Each sensor's trigger pin, echo pin, and max distance to ping.
  NewPing(ultrasound_trigger_pin[(int)UltraSoundSensor::Front], 
                     ultrasound_echo_pin[(int)UltraSoundSensor::Front], 
                     MAX_DISTANCE),
  NewPing(ultrasound_trigger_pin[(int)UltraSoundSensor::Right], 
                     ultrasound_echo_pin[(int)UltraSoundSensor::Right], 
                     MAX_DISTANCE)
};
bool isTurn = false;
void turn(int speed) {
    if(!isTurn)
    {
        Brake();
        delay(500);
    isTurn = true;
    MotorL_Move(speed);
    MotorR_Move(-speed);
    }
}

void forward(int speed) {
    if(isTurn)
    {
        Brake();
        delay(500);
    
    isTurn = false;
    MotorL_Move(speed);
    MotorR_Move(speed);
    }
}

void processPingResult(uint8_t sensor, int distanceInCm) {
  // The following code would be replaced with your code that does something with the ping result.
  if(distanceInCm < 40 && distanceInCm != 0)
  {
     isObstacle[sensor] = true;
     distances[sensor] = distanceInCm;
     Serial.println("Sensor: " + String(sensor) + "; Distance: " + String(distanceInCm));
  }
  else if(distanceInCm > 50)
  {
      isObstacle[sensor] = false;
  }  
  
}

void echoCheck() {
  if (sonar[currentSensor].check_timer())
    processPingResult(currentSensor, sonar[currentSensor].ping_result / US_ROUNDTRIP_CM);
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
        if(!nerfTrigger.getCanShoot()) return;
        
        if(darts>0) {
            Serial.println("Shooting");
            nerfTrigger.shoot();
            darts-=1;
        } else Serial.println("No darts");
    }

    void reload() {
        darts = 6;
    }
};

NERF trigger;


void setup()
{
    // Use to check for any errors in using timers
    // TeensyTimerTool::attachErrFunc(TeensyTimerTool::ErrorHandler(Serial));

    initSerial(115200);
    delay(3000);
    Serial.println(" | Serial Done");

      pingTimer[0] = millis() + 75;

    for (uint8_t i = 1; i < SONAR_NUM; i++)
        pingTimer[i] = pingTimer[i - 1] + PING_INTERVAL;

    initMotors();
    Brake();
     // Each platform has to do this independently, checked manually
    calibrateServo(ServoSelector::Yaw, (int)YawCalibrationCenter);
    calibrateServo(ServoSelector::Pitch, (int)PitchCalibrationCenter);

    initServos();
    centerServos();

    delay(500);

}

String msg;

void loop()
{
    // trigger.shoot();
    // delay(50);

    if(Serial.available()!=0) {
        msg = Serial.readString();

        if(msg.indexOf("shoot")!=-1) {
            trigger.shoot();
        }
        if(msg.indexOf("reload")!=-1) {
            trigger.reload();
        }
        Serial.flush();
    }

    for (uint8_t i = 0; i < SONAR_NUM; i++) {
    if (millis() >= pingTimer[i]) {
      pingTimer[i] += PING_INTERVAL * SONAR_NUM;
      sonar[currentSensor].timer_stop();
      currentSensor = i;
      sonar[currentSensor].ping_timer(echoCheck);
    }  
  }
  bool isObstacleBool = false;
   for (uint8_t i = 0; i < SONAR_NUM; i++) {
       if(isObstacle[i])
        isObstacleBool = true;
   }
   isObstacleBool?turn(100) : forward(70);
   
}






