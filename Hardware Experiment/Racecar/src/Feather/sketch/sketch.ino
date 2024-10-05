#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Servo
#define SERVOMIN  120 // this is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  470// this is the 'maximum' pulse length count (out of 4096)
#define SERVO_FREQ 50

// Initialize the PCA9685
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

// Keep track of the time for the lost connection behavior
unsigned long previousMillis = 0;

// Function to clamp the angle of the servo
float clamp(float vmin, float val, float vmax){
    return max(min(val, vmax), vmin);
}

void setup() {
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(SERVO_FREQ); // Analog servos run at ~50 Hz updates

  Serial.begin(115200);
  while (!Serial) {
    delay(1); // will pause until serial monitor opens
  }

  // Serial.println("Listening...");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // If bytes are available to read from the serial port
  if(Serial.available() > 0) {
    // Serial.println("i GOT");
    String data = Serial.readStringUntil('\n');
    float servo_angle = clamp(30, data.toFloat(), 140);

    // map angle of 0 to 180 to Servo min and max
    uint16_t pulselen = map(servo_angle, 0, 180, SERVOMIN, SERVOMAX);
    
    pca.setPWM(4, 0, pulselen);
    previousMillis = currentMillis;
  }

  // In case of connection loss
  if (currentMillis - previousMillis > 500) {
    uint16_t pulselen = map(85, 0, 180, SERVOMIN, SERVOMAX);
    pca.setPWM(4, 0, pulselen);
  }
}