#include <NewPing.h>

#define SONAR_NUM      3 // Number of sensors.
#define MAX_DISTANCE 200 // Maximum distance (in cm) to ping.
#define PING_INTERVAL 30 // Milliseconds between sensor pings (29ms is about the min to avoid cross-sensor echo).

class Sonars {
private:
  unsigned long pingTimer[SONAR_NUM]; // Holds the times when the next ping should happen for each sensor.
  unsigned int distances[SONAR_NUM];  // Where the ping distances are stored.
  unsigned int shortestDistance = !((unsigned int)0);
  uint8_t currentSensor = 0;          // Keeps track of which sensor is active.

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

  void processPingResult(uint8_t sensor, int distanceInCm) {
  // The following code would be replaced with your code that does something with the ping result.
  if(distanceInCm != 0)
  {
     if (shortestDistance > distanceInCm) shortestDistance = distanceInCm;
     distances[sensor] = distanceInCm;
     Serial.println("Sensor: " + String(sensor) + "; Distance: " + String(distanceInCm));
  }
  
}

  void echoCheck() {
    if (sonar[currentSensor].check_timer())
      processPingResult(currentSensor, sonar[currentSensor].ping_result / US_ROUNDTRIP_CM);
  }
public:
  Sonars() {
    pingTimer[0] = millis() + 75;

    for (uint8_t i = 1; i < SONAR_NUM; i++)
        pingTimer[i] = pingTimer[i - 1] + PING_INTERVAL;
  }

  void update() {
    for (uint8_t i = 0; i < SONAR_NUM; i++) {
      if (millis() >= pingTimer[i]) {
        pingTimer[i] += PING_INTERVAL * SONAR_NUM;
        sonar[currentSensor].timer_stop();
        currentSensor = i;
        sonar[currentSensor].ping_timer(echoCheck);
      }  
    }
  }

  unsigned int getShortestDistance() {
    return shortestDistance;
  }
}
