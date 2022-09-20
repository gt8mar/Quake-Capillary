
const int pwm = 2 ;         //initializing pin 2 as ‘pwm’ variable
const int buttonPin1 = 3;    // initializing pin 3 as a button variable
const int buttonPin2 = 4;   //initializing pin 4 as a button variable


void setup()
{
    pinMode(pwm,OUTPUT) ;     //Set pin 2 as output
    pinMode(buttonPin1, INPUT);  //Set pin 3 as input
    pinMode(buttonPin2, INPUT);  //Set pin 4 as input
}



void loop()
{
  // read the state of the button
    buttonState1 = digitalRead(buttonPin1);
    buttonState2 = digitalRead(buttonPin2);

    if (buttonState1 == HIGH && buttonState2 == LOW){
    
      delay(1000);
      
      analogWrite(pwm,6) ;         //setting pwm to increase PSI from 0 to 1, incrementing by 0.1 
      delay(1000) ;             //delay of 5s
  
      analogWrite(pwm,19) ; 
      delay(1000) ; 
     
      analogWrite(pwm,32) ;     // 0.5 psi
      delay(1000) ; 
  
      analogWrite(pwm,45) ; 
      delay(1000) ;
  
      analogWrite(pwm,57) ; 
      delay(1000) ; 
      
      analogWrite(pwm,70) ;       //1.1 psi
      delay(1000) ;
  
      analogWrite(pwm,57) ;     
      delay(1000) ; 
      
      analogWrite(pwm,45) ; 
      delay(1000) ;
       
      analogWrite(pwm,32) ;       // 0.5 psi
      delay(1000) ;
       
      analogWrite(pwm,19) ; 
      delay(1000) ; 
  
      analogWrite(pwm,6) ;         // 0.1 psi
      delay(1000) ;
      }
    else if(buttonState1 == HIGH && buttonState2 == HIGH){
      delay(500);

      analogWrite(pwm, 64);
      delay(2000);
      }
    //else if(buttonState1 == LOW){    }

      
}
