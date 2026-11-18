/*
 * AI Car Simple V2 - USB serial goal-navigation test firmware
 * Commands and encoder telemetry use USB Serial at 115200 baud.
 * Commands:
 *   V,servo,pwm      - Forward (pwm > 0)
 *   B,servo,pwm      - Backward (pwm > 0, motor ngược chiều)
 *   S                - Stop
 *   P:pulse          - Servo only
 */

#include <Servo.h>

// Motor pins
const uint8_t M1_RPWM = 5;
const uint8_t M1_LPWM = 6;
const uint8_t M1_REN = 22;
const uint8_t M1_LEN = 23;
const uint8_t M2_RPWM = 7;
const uint8_t M2_LPWM = 8;
const uint8_t M2_REN = 24;
const uint8_t M2_LEN = 25;

// Servo
const uint8_t SERVO_PIN = 9;

// Encoder pins
const uint8_t LEFT_ENCODER_A = 2;
const uint8_t LEFT_ENCODER_B = 3;
const uint8_t RIGHT_ENCODER_A = 10;
const uint8_t RIGHT_ENCODER_B = 11;

// Giới hạn
const int SERVO_MIN_US = 1500;
const int SERVO_MAX_US = 2600;
const int SERVO_CENTER_US = 2300;
const int MAX_PWM = 50;
const unsigned long MOTOR_COMMAND_TIMEOUT_MS = 500;

// Cài đặt motor
const bool INVERT_MOTOR_1 = false;
const bool INVERT_MOTOR_2 = false;

// Encoder tracking
const bool INVERT_LEFT_ENCODER = false;
const bool INVERT_RIGHT_ENCODER = false;
volatile long leftEncoderTicks = 0;
volatile long rightEncoderTicks = 0;
volatile bool previousRightA = false;

// Serial buffer
const size_t LINE_BUFFER_SIZE = 64;
char lineBuffer[LINE_BUFFER_SIZE];
size_t lineLength = 0;

// State
Servo steering;
int currentServo = SERVO_CENTER_US;
int currentPWM = 0;
bool isBackward = false;  // Track direction
unsigned long lastEncoderReportMs = 0;
const unsigned long ENCODER_REPORT_INTERVAL = 50;
unsigned long lastDriveCommandMs = 0;

// ============================================================
// MOTOR CONTROL
// ============================================================

void setMotor(uint8_t rpwm, uint8_t lpwm, int speedValue, bool invert) {
  speedValue = constrain(speedValue, -255, 255);
  if (invert) {
    speedValue = -speedValue;
  }

  if (speedValue > 0) {
    analogWrite(lpwm, 0);
    analogWrite(rpwm, speedValue);
  } else if (speedValue < 0) {
    analogWrite(rpwm, 0);
    analogWrite(lpwm, -speedValue);
  } else {
    analogWrite(rpwm, 0);
    analogWrite(lpwm, 0);
  }
}

void stopMotors() {
  currentPWM = 0;
  isBackward = false;
  setMotor(M1_RPWM, M1_LPWM, 0, INVERT_MOTOR_1);
  setMotor(M2_RPWM, M2_LPWM, 0, INVERT_MOTOR_2);
}

void drive(int servo_us, int pwm, bool backward) {
  // Giới hạn giá trị
  currentServo = constrain(servo_us, SERVO_MIN_US, SERVO_MAX_US);
  currentPWM = constrain(pwm, 0, MAX_PWM);
  isBackward = backward;
  lastDriveCommandMs = millis();
  
  // Điều khiển servo
  steering.writeMicroseconds(currentServo);
  
  // Điều khiển motor (âm nếu backward)
  int motorSpeed = backward ? -currentPWM : currentPWM;
  setMotor(M1_RPWM, M1_LPWM, motorSpeed, INVERT_MOTOR_1);
  setMotor(M2_RPWM, M2_LPWM, motorSpeed, INVERT_MOTOR_2);
}

// ============================================================
// ENCODER ISR
// ============================================================

void leftEncoderIsr() {
  if (digitalRead(LEFT_ENCODER_A) == HIGH) {
    const bool forward = (digitalRead(LEFT_ENCODER_B) == HIGH) ^ INVERT_LEFT_ENCODER;
    leftEncoderTicks += forward ? 1 : -1;
  }
}

ISR(PCINT0_vect) {
  const bool currentA = digitalRead(RIGHT_ENCODER_A);
  if (currentA && !previousRightA) {
    const bool forward = (digitalRead(RIGHT_ENCODER_B) == LOW) ^ INVERT_RIGHT_ENCODER;
    rightEncoderTicks += forward ? 1 : -1;
  }
  previousRightA = currentA;
}

// ============================================================
// COMMAND PARSING
// ============================================================

void handleLine(char *line) {
  // Command: V,servo,pwm (Forward)
  if (strncmp(line, "V,", 2) == 0) {
    char *servoToken = line + 2;
    char *comma = strchr(servoToken, ',');
    
    if (comma == NULL) {
      Serial.println("ERROR: Invalid V command format");
      return;
    }
    
    *comma = '\0';
    int servo_us = atoi(servoToken);
    int pwm = atoi(comma + 1);
    
    if (servo_us < SERVO_MIN_US || servo_us > SERVO_MAX_US) {
      Serial.print("ERROR: Servo out of range (");
      Serial.print(SERVO_MIN_US);
      Serial.print("-");
      Serial.print(SERVO_MAX_US);
      Serial.println(")");
      return;
    }
    
    if (pwm < 0 || pwm > MAX_PWM) {
      Serial.print("ERROR: PWM out of range (0-");
      Serial.print(MAX_PWM);
      Serial.println(")");
      return;
    }
    
    drive(servo_us, pwm, false);  // Forward
    return;
  }
  
  // Command: B,servo,pwm (Backward)
  if (strncmp(line, "B,", 2) == 0) {
    char *servoToken = line + 2;
    char *comma = strchr(servoToken, ',');
    
    if (comma == NULL) {
      Serial.println("ERROR: Invalid B command format");
      return;
    }
    
    *comma = '\0';
    int servo_us = atoi(servoToken);
    int pwm = atoi(comma + 1);
    
    if (servo_us < SERVO_MIN_US || servo_us > SERVO_MAX_US) {
      Serial.print("ERROR: Servo out of range (");
      Serial.print(SERVO_MIN_US);
      Serial.print("-");
      Serial.print(SERVO_MAX_US);
      Serial.println(")");
      return;
    }
    
    if (pwm < 0 || pwm > MAX_PWM) {
      Serial.print("ERROR: PWM out of range (0-");
      Serial.print(MAX_PWM);
      Serial.println(")");
      return;
    }
    
    drive(servo_us, pwm, true);  // Backward
    return;
  }
  
  // Command: S (Stop)
  if (strcmp(line, "S") == 0) {
    stopMotors();
    steering.writeMicroseconds(SERVO_CENTER_US);
    currentServo = SERVO_CENTER_US;
    Serial.println("STOPPED");
    return;
  }
  
  // Command: P:pulse (Servo only - legacy)
  if (strncmp(line, "P:", 2) == 0) {
    int pulse = atoi(line + 2);
    pulse = constrain(pulse, SERVO_MIN_US, SERVO_MAX_US);
    steering.writeMicroseconds(pulse);
    currentServo = pulse;
    Serial.print("SERVO:");
    Serial.println(pulse);
    return;
  }
  
  // Unknown command
  Serial.print("ERROR: Unknown command: ");
  Serial.println(line);
}

void receiveCommands() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    
    if (ch == '\r') {
      continue;
    }
    
    if (ch == '\n') {
      if (lineLength > 0) {
        lineBuffer[lineLength] = '\0';
        handleLine(lineBuffer);
      }
      lineLength = 0;
      continue;
    }
    
    if (lineLength < LINE_BUFFER_SIZE - 1) {
      lineBuffer[lineLength++] = ch;
    } else {
      lineLength = 0;
      Serial.println("ERROR: Line too long");
    }
  }
}

// ============================================================
// ENCODER TELEMETRY
// ============================================================

void reportEncoders() {
  unsigned long now = millis();
  if (now - lastEncoderReportMs < ENCODER_REPORT_INTERVAL) {
    return;
  }
  lastEncoderReportMs = now;
  
  long leftTicks, rightTicks;
  noInterrupts();
  leftTicks = leftEncoderTicks;
  rightTicks = rightEncoderTicks;
  interrupts();
  
  // Format: E,left,right,servo
  Serial.print("E,");
  Serial.print(leftTicks);
  Serial.print(',');
  Serial.print(rightTicks);
  Serial.print(',');
  Serial.println(currentServo);
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  
  pinMode(M1_RPWM, OUTPUT);
  pinMode(M1_LPWM, OUTPUT);
  pinMode(M1_REN, OUTPUT);
  pinMode(M1_LEN, OUTPUT);
  pinMode(M2_RPWM, OUTPUT);
  pinMode(M2_LPWM, OUTPUT);
  pinMode(M2_REN, OUTPUT);
  pinMode(M2_LEN, OUTPUT);
  
  digitalWrite(M1_REN, HIGH);
  digitalWrite(M1_LEN, HIGH);
  digitalWrite(M2_REN, HIGH);
  digitalWrite(M2_LEN, HIGH);
  
  stopMotors();
  
  pinMode(LEFT_ENCODER_A, INPUT_PULLUP);
  pinMode(LEFT_ENCODER_B, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_A, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_B, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_A), leftEncoderIsr, CHANGE);
  previousRightA = digitalRead(RIGHT_ENCODER_A);
  PCICR |= _BV(PCIE0);
  PCMSK0 |= _BV(PCINT4);
  
  steering.attach(SERVO_PIN);
  steering.writeMicroseconds(SERVO_CENTER_US);
  
  Serial.println(F("AI Car Simple V2 USB - Ready"));
  Serial.println(F("Commands:"));
  Serial.println(F("  V,servo,pwm - Forward"));
  Serial.println(F("  B,servo,pwm - Backward"));
  Serial.println(F("  S - Stop"));
  Serial.println(F("  P:pulse - Servo only"));
  
  Serial.println("READY");
}

// ============================================================
// LOOP
// ============================================================

void loop() {
  receiveCommands();
  if (currentPWM > 0 &&
      millis() - lastDriveCommandMs > MOTOR_COMMAND_TIMEOUT_MS) {
    stopMotors();
    steering.writeMicroseconds(SERVO_CENTER_US);
    currentServo = SERVO_CENTER_US;
    Serial.println("WATCHDOG_STOP");
  }
  reportEncoders();
}
