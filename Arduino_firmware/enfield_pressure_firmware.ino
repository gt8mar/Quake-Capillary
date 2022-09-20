// Filename: pressureRegulator.ino
// Hardware: teensy 4.1
// ------------------------------------------------------------------------
// This file controls an Enfield pressure regulator with air input of 2psi.
// The pressure range is therefore 0 to 2 psi. 


const int pwm = 2 ;         //initializing pin 2 as ‘pwm’ variable
const int buttonPin1 = 4;    // initializing pin 4 as a button variable
const int buttonPin2 = 5;   //initializing pin 5 as a button variable
const int LEDPin = 13;
const int thorlabsLEDPin = 7; 

int buttonState1 = 0;
int buttonState2 = 0;

void setup()
{
    pinMode(pwm,OUTPUT) ;     //Set pin 2 as output
    pinMode(buttonPin1, INPUT);  //Set pin 3 as input
    pinMode(buttonPin2, INPUT);  //Set pin 4 as input
    pinMode(LEDPin, OUTPUT); 
    pinMode(thorlabsLEDPin, OUTPUT);
    
}



void loop()
{
  // read the state of the button    
    buttonState1 = digitalRead(buttonPin1);
    buttonState2 = digitalRead(buttonPin2);

    if (buttonState1 == HIGH && buttonState2 == LOW){
      digitalWrite(LEDPin, HIGH);
      
      delay(1000);
      
      analogWrite(pwm,127) ;         //setting pwm to increase PSI from 1 to 1.6, incrementing by 0.1 
      digitalWrite(thorlabsLEDPin, LOW);
      delay(100);
            digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ;             //delay of 1s
  
      analogWrite(pwm,153) ;    // 1.2 psi
      digitalWrite(thorlabsLEDPin, LOW);
      delay(100);
                  digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ; 
     
      analogWrite(pwm,178) ;     // 1.4 psi
      digitalWrite(thorlabsLEDPin, LOW);
      delay(100);
                  digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ; 
  
      analogWrite(pwm,203) ;     // 1.6 psi
            digitalWrite(thorlabsLEDPin, LOW);
            delay(100);
                        digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ;
  
      analogWrite(pwm,178) ;      //1.4 psi
            digitalWrite(thorlabsLEDPin, LOW);
            delay(100);
                        digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ; 
      
      analogWrite(pwm,152) ;       //1.2 psi
            digitalWrite(thorlabsLEDPin, LOW);
            delay(100);
                        digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ;
  
      analogWrite(pwm,127) ;     // 1 psi
            digitalWrite(thorlabsLEDPin, LOW);
            delay(100);
                        digitalWrite(thorlabsLEDPin, HIGH);
      delay(1000) ; 
      
      }
    else if(buttonState2 == HIGH){
      digitalWrite(LEDPin, LOW);
      delay(500);

      analogWrite(pwm, 127);
      delay(10000);
      }
    else if(buttonState1 == LOW && buttonState2 == LOW){
      digitalWrite(LEDPin, LOW);
      analogWrite(pwm, 0);
      }

      
}
