# CudaBouncingBall

## Background

Simple program to draw a bouncing ball into a MP4 file.

## Structure

Shape.cu -- Data structure for various shapes. In this case I use an enumerated type because the C++ version I was using didn't
appear to support inheritance properly... guessing that is a limit of the architecture, as running different functions with the same 
call is tantamount to a large set of conditional branches.  
  I.E.  (Dog) MyObject.GetBreed() is tantamount to:
            if (typeof(MyObject) == typeof(Dog))  Dog::GetBreed
            elif (typeof(MyObject) == typeof(Cat))  Cat::GetBreed
            elif (typeof(MyObject) == typeof(Hamster))  Hamster::GetBreed()
 which would be a pain in a SIMD architecture.            
    
BouncingBall.cu -- Main Program.
Build.sh -- Build script

##
