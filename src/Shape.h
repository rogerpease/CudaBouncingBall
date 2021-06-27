#include <fstream> 

#include <unistd.h> 
#include <iostream>
#include <cstdlib>
#include <vector> 
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std; 

//
//  Simple Shape Class. Keeping in a separate file so I can easily run CICD tests on it. 
// 

//
// In a larger project I might name these Color_Blue or put them in some sort of enumerated type or their own namespace. 
// 
typedef struct { char red; char green; char blue; } Color_t;
const Color_t Blue  {.red=0,.green=0,.blue=0xFF};
const Color_t White {.red=0xFF,.green=0xFF,.blue=0xFF};
const Color_t Red   {.red=0xFF,.green=0x0,.blue=0x0};

//
// Another option is to put this in Shape as a return class. 
// 
class PixelColorResult
{
   public:
     __device__ PixelColorResult(bool result, Color_t color) {  IsInPixel=result; Color=color; } 
     __device__ PixelColorResult() {  IsInPixel = false; } 
     bool IsInPixel;  // Is our pixel in our shape. 
     Color_t Color;  //  What Color? 

};

class Shape  
{
   public: 

    typedef enum { Circle, Rectangle } ShapeEnum; 	    
    Shape (ShapeEnum shape, int centerx,int centery,int height,int width,Color_t color); 
    Shape (ShapeEnum shape, int centerx,int centery,int radius,Color_t color); 
    __device__ PixelColorResult IsInShape(int pixelx, int pixely);

    //
    // Got this off a CUDA link but it's a good way to allocate for managed memory. 
    // 
    void *operator new(size_t len);
    __host__ float GetLeft(); 
    __host__ float GetRight();
    __host__ float GetTop();
    __host__ float GetCenterX();
    __host__ float GetCenterY();
    __host__ float GetTravelAngle();
    __host__ float GetBottom();
    __host__ void SetTravelAngle(float degrees); 
    __host__ void Move(float magnitude,int frameHeight, int FrameWidth);


   private: 
    ShapeEnum shape; 
    // One of the big advantages of objects is that they force you (if you use private variables) to separate the way you store your data from
    // the way you retrieve your data. I started with these as integers but realized that if I wanted to 
    // improve the movement accuracy and keep the data within the object I should use floats. 
    float travelAngleDegrees; 
    float centerX; 
    float centerY;
    float radius; 
    float radiusSquared; // Keep the square of the radius to keep from taking a square root. 
    float heightDiv2; 
    float widthDiv2; 
    Color_t shapeColor;
    bool debug; 
};



