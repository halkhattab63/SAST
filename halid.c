PORTA EQU 0OH
PORTB EQU 02H
PORTC EQU 04H
PORT_CON EQU 06H

DATA ENDS

CODE SEGMENT

ORG 0000H

MOV DX, PORT_CON
MOV AL, 83H         ; Portların giriş/çıkış ayarlarını yapıyoruz (örneğin: 0x83)
OUT DX, AL

MOV SI, 0

XX:
MOV DX, PORTA       ; PORTA'yı okuyarak BASLA anahtarını kontrol ediyoruz
IN AL, DX
AND AL, 01H         ; Sadece BASLA anahtarının bitini kontrol ediyoruz
CMP AL, 01H
JZ XX               ; BASLA anahtarı kapalıysa tekrar kontrol et

MOV AL, 03H
MOV DX, PORTC
OUT DX, AL          ; Motoru başlat

JMP START

START:
MOV AL, S1[SI]      ; Mod verisini bellekten al
MOV DX, PORTC
OUT DX, AL          ; Motorun modunu ayarla

CMP SI, 07H         ; Bellek dizisinin sonuna gelindi mi kontrol et
JZ RESET            ; Evetse RESET noktasına git

MOV DX, PORTA
IN AL, DX
AND AL, 02H         ; MODE anahtarını kontrol et
CMP AL, 02H
JZ MODE             ; MODE anahtarı kapalıysa Half Mod'a geç

MOV AL, 02H
MOV DX, PORTB
OUT DX, AL          ; Full Mod'da çalışmak için PORTB'ye veri gönder

JMP FULL

MODE:
ADD SI, 1           ; Half Mod'da çalışmak için bellek dizisini ilerlet
MOV AL, S1[SI]
MOV DX, PORTB
OUT DX, AL          ; Half Mod'da çalışmak için PORTB'ye veri gönder

CALL BEKLE          ; Bekleme fonksiyonunu çağır

JMP XX

FULL:
MOV BX, 0           ; Full Mod için başlangıç ayarları
JMP START

RESET:
MOV SI, 0           ; Bellek dizisini başa al
JMP XX

BEKLE:
MOV CX, 0AFD0H      ; Bekleme döngüsü
LI: LOOP LI
RET

ORG 1000H

S1 DB 0BH, 0CH, 0DH, 0EH, 0FH, 0AH, 0BH, 0CH

CODE ENDS
END





/************************************/

#define _XTAL_FREQ 10000000

#define RS RD2
#define EN RD3
#define D4 RD4
#define D5 RD5
#define D6 RD6
#define D7 RD7

#define IN1 RB0
#define IN2 RB1
#define START RB3
#define STOP RB4
#define ARTTIR RB5
#define AZALT RB6

#include <xc.h>
#include <stdio.h>
#include "Im016.h"  // Assuming this is a custom LCD library

// Configuration bits for PIC16F877A
#pragma config FOSC = HS
#pragma config WDTE = OFF
#pragma config PWRTE = OFF
#pragma config BOREN = ON
#pragma config LVP = OFF
#pragma config CPD = OFF
#pragma config WRT = OFF
#pragma config CP = OFF

int bekle(int saniye) {
    while(saniye > 0) {
        __delay_ms(1000);
        saniye--;
    }
    return 0;
}

void main(void) {
    int ref, speed, t = 0;
    char s[16];

    TRISB = 0xF8;  // RB0, RB1, RB3, RB4, RB5, RB6 are inputs, others are outputs
    TRISC = 0x00;  // All PORTC pins are outputs
    TRISD = 0x00;  // All PORTD pins are outputs
    ADCON1 = 0x04; // Configure AN0 as analog input, rest as digital

    Led_Init();

    CCP1CON = 0x0F;  // Configure CCP1 for PWM mode
    PR2 = 0xFF;      // Load the period register
    T2CON = 0b00000101;  // Enable Timer2 with prescaler 1:4

    while(1) {
        Led_Clear();
        ADCON0 = 0b00000001;  // Select AN0 and turn on ADC
        __delay_ms(2);        // Acquisition time
        ADCON0bits.GO = 1;    // Start conversion
        while(ADCON0bits.GO); // Wait for conversion to complete

        ref = ((ADRESH << 2) + (ADRESL >> 6));
        speed = (unsigned int)(ref * 3000.0 / 1023.0);

        if(speed >= 3000) { speed = 3000; }

        sprintf(s, "Speed: %d rpm", speed);
        Led_Set_Cursor(1, 1);
        Led_Write_String(s);

        if(ARTTIR == 0) { t++; }
        if(AZALT == 0) { t--; }
        if(t < 0) { t = 0; }

        sprintf(s, "Time: %d s", t);
        Led_Set_Cursor(2, 1);
        Led_Write_String(s);

        if(START == 0) {
            CCPR1L = speed / 4;  // Set the duty cycle
            IN1 = 1; IN2 = 0;    // Start the motor
            bekle(t);
            t = 0;
            CCPR1L = 0;
            IN1 = 0; IN2 = 0;    // Stop the motor
            Led_Set_Cursor(2, 11);
            Led_Write_String("Stopped");
            __delay_ms(500);
        }
    }
}




****************

#include <xc.h>
#include <pic16f877a.h>

#define _XTAL_FREQ 10000000

#define START RB0
#define STOP RB1
#define ARTTIR RB2
#define AZALT RB3
#define YON RB4

#define IN1 RB5
#define IN2 RB6
#define ENA RB7
#define LED RB7

int display(int c, int b) {
    int d[12];
    d[0] = 0xC0;  // 0
    d[1] = 0xF9;  // 1
    d[2] = 0xA4;  // 2
    d[3] = 0xB0;  // 3
    d[4] = 0x99;  // 4
    d[5] = 0x92;  // 5
    d[6] = 0x82;  // 6
    d[7] = 0xF8;  // 7
    d[8] = 0x80;  // 8
    d[9] = 0x90;  // 9
    d[10] = 0x86; // e
    d[11] = 0x8E; // d
    
    PORTC = 0x80;
    PORTD = d[c]; 
    __delay_ms(2);
    
    PORTC = 0x40; 
    PORTD = d[b]; 
    __delay_ms(2);

    return 0;
}

void main(void) {
    int b = 1; 
    int c = 11; 
    int set = 0; 

    TRISB = 0b00011111;  // RB0-RB4 inputs, others outputs
    TRISC = 0b00000000;  // All outputs
    TRISD = 0b00000000;  // All outputs

    PR2 = 0xFF; 
    T2CON = 0b00000101;

    PORTB = 0;
    PORTC = 0;
    PORTD = 0;

    while (1) {
        if (ARTTIR == 0) { 
            b++; 
            __delay_ms(100); 
        }
        if (AZALT == 0) { 
            b--; 
            __delay_ms(100); 
        }
        if (b > 9) { b = 9; }
        if (b < 1) { b = 1; }

        if (START == 0) {
            set = 1; 
        }
        if (STOP == 0) { 
            b = 1;
            set = 0; 
        }

        if (set == 1) {
            CCPR1L = (b * 255) / 9; // PWM duty cycle ayarı
            IN1 = YON; 
            IN2 = !YON; 
            LED = 1; 
            c = 10; // 'e' harfi
        } else {
            CCPR1L = 0; 
            IN1 = 0;
            IN2 = 0;
            LED = 0; 
            c = 11; // 'd' harfi
        }

        display(c, b);
    }

    return;
}




/******************************* */




DATA SEGMENT

PORTA EQU 00H
PORTB EQU 02H
PORTC EQU 04H
PORT_CON EQU 06H

DATA ENDS

CODE SEGMENT

ORG 0000H

MOV DX, PORT_CON
MOV AL, 88H  ; Set all PORTB pins as inputs except the motor control pins
OUT DX, AL

START:

MOV AL, 8EH  ; Set initial display value to 'd'
MOV DX, PORTB
OUT DX, AL

MOV DX, PORTA
IN AL, DX

CMP AL, 01H  ; Check if ARTTIR button is pressed
JZ ARTTIR

CMP AL, 02H  ; Check if AZALT button is pressed
JZ AZALT

CMP AL, 04H  ; Check if BASLA button is pressed
JZ BASLA

JMP XX

XX:
MOV AL, BL
MOV DX, PORTC
OUT DX, AL

CALL BEKLE

JMP START

ARTTIR:
INC BL
CMP BL, 09H
JG XX
JMP START

AZALT:
DEC BL
CMP BL, 00H
JL XX
JMP START

RESET:
MOV BL, 00H
JMP START

BASLA:
CMP BL, 00H
JZ HATA

MOV AL, 8BH  ; Set display to 'b'
MOV DX, PORTB
OUT DX, AL

ADD BL, 00H  ; No operation, just to keep the structure
MOV AL, BL
MOV DX, PORTC
OUT DX, AL

CALL BEKLE

SUB BL, 01H
MOV AL, BL
CMP BL, 00H
JNE START

MOV AL, 8EH  ; Set display to 'd'
MOV DX, PORTB
OUT DX, AL

JMP START

HATA:
MOV AL, 89H  ; Set display to 'h'
MOV DX, PORTB
OUT DX, AL

JMP XX

BEKLE:
MOV CX, 08FD0H
L1: LOOP L1
RET

CODE ENDS
END
