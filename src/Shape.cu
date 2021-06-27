#include <unistd.h> 
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "Shape.h"
using namespace cv;
using namespace std; 

// X is row, Y is Col. 

//
// Main Shape constructor. 
// In a normal uniprocessor I would do this with a base Shape Class and parent classes for the specific shapes, but
//  GPUs have some restrictions on that. My guess is they are running some sort of SIMT architecture and virtual functions  
//  set you up to call different functions which have different instructions. 
//


Shape::Shape (ShapeEnum shape, int centerx,int centery,int height,int width,Color_t color): 
	    centerX(centerx),centerY(centery),heightDiv2(height/2),widthDiv2(width/2),shapeColor(color) 
{ 
  if (shape == ShapeEnum::Rectangle) 
  { 
     this->shape = shape;  
  }
  else
  {
     printf("Constructor with three elements is only valid for: Circle Rectangle"); 
     exit(0); 
  }
  debug = false; 
}

Shape::Shape (ShapeEnum shape, int centerx,int centery,int radius,Color_t color): 
	    centerX(centerx),centerY(centery),radius(radius),radiusSquared(radius*radius),shapeColor(color)
{ 
  if (shape == ShapeEnum::Circle) 
  { 
    this->shape = shape;  
  }
  else
  {
    printf("Constructor with three elements is only valid for: Circle"); 
    exit(0); 
  }
  debug = false; 
}


__host__ void Shape::SetTravelAngle(float degrees)  { this->travelAngleDegrees = degrees; } 
__host__ float Shape::GetTravelAngle()  { return this->travelAngleDegrees; } 

__host__ void Shape::Move(float magnitude,int frameHeight, int frameWidth)
{
   if (debug) printf("Old X: %f Y:%f mag %f Dir %f\n",centerX,centerY,magnitude, travelAngleDegrees);
   float XOffset = -1*magnitude*cos(this->travelAngleDegrees*3.14/180); 
   float YOffset =    magnitude*sin(this->travelAngleDegrees*3.14/180); 

   float centerToLREdge = (shape == ShapeEnum::Circle) ? radius : widthDiv2;
   float centerToTBEdge = (shape == ShapeEnum::Circle) ? radius : heightDiv2;
   // collision with top. If coming up at 15 degrees we want to flip to down at 180-15 degrees. Use .0 to indicate float
   // so the result doesn't get cast to an integer.  
   bool topCollision   = ((centerX-centerToTBEdge+XOffset) < 0);
   bool botCollision   = ((centerX+centerToTBEdge+XOffset) > frameHeight);
   bool leftCollision  = ((centerY-centerToLREdge+YOffset) < 0);
   bool rightCollision = ((centerY+centerToLREdge+YOffset) > frameWidth);

   bool doubleCollision = false; 
//   bool singleCollision = topCollision || botCollision || leftCollision || rightCollision;

   if (((topCollision) && (leftCollision)) || ((topCollision) && (rightCollision))   ||
       ((botCollision) && (leftCollision)) || ((botCollision) && (rightCollision)))  
   { 
      travelAngleDegrees += 180; 
      doubleCollision = true; 
   } 
   else if (topCollision)  { this->travelAngleDegrees = (180.0 - travelAngleDegrees); } 
   // collision with bottom. If coming down at 179 degrees we want to flip to down at 360-179 = 181 degrees.
   else if (botCollision) { this->travelAngleDegrees = (180-travelAngleDegrees);            } 
   // collision with left. If coming left at 260 degrees we want to flip up to 90- (270-260) = 80 degrees.
   else if (leftCollision)          { this->travelAngleDegrees = 90+(270.0-travelAngleDegrees);      } 
   // collision with right. If coming right at 89 degrees we want to flip up to 270 - (90 -89) = 270 degrees.
   else if (rightCollision)  { this->travelAngleDegrees = (270+(90.0-travelAngleDegrees));      } 
   while (travelAngleDegrees < 0) travelAngleDegrees += 360; 
   while (travelAngleDegrees > 360) travelAngleDegrees -= 360; 
   // Recompute 
   XOffset = -1*magnitude*cos(this->travelAngleDegrees*3.14/180); 
   YOffset =    magnitude*sin(this->travelAngleDegrees*3.14/180); 
   this->centerX += XOffset; 
   this->centerY += YOffset; 
   if (debug)
   {
     printf("New X: %f Y:%f %f \n",centerX,centerY,travelAngleDegrees);
     if (topCollision)   { printf("TopCollision\n"); } 
     if (botCollision)   { printf("BotCollision\n"); } 
     if (leftCollision)  { printf("LeftCollision\n"); } 
     if (rightCollision) { printf("RightCollision\n"); } 
   }
}

__device__ PixelColorResult Shape::IsInShape(int pixelx, int pixely)
{ 
  if (shape == ShapeEnum::Rectangle) 
  {	      
    if ((pixelx >= centerX-heightDiv2) &&  (pixelx <= centerX+heightDiv2) &&
        (pixely >= centerY-widthDiv2)  &&  (pixely <= centerY+widthDiv2))
    { 
       return PixelColorResult(true,this->shapeColor); 
    }
  }
  if (shape == ShapeEnum::Circle) 
  {	      
    int px = pixelx - centerX;
    int py = pixely - centerY;
    if (((px*px) + (py*py)) <= radiusSquared) 
      return PixelColorResult(true,this->shapeColor); 
  }	
  return PixelColorResult(); 
} 
__host__ float Shape::GetCenterX()  { return centerX; }
__host__ float Shape::GetCenterY()  { return centerY; }

__host__ float Shape::GetLeft() 
{ 
  if (shape == ShapeEnum::Rectangle) { return centerY-widthDiv2; } 	      
  if (shape == ShapeEnum::Circle)    { return centerY-radius; } 	      
  assert(false); // We should never enter this condition. If we do the best thing to do (usually) is 
                 // log the failure and reset the system. 
               
  return -1;  // To avoid compiler warning. 
}	
__host__ float Shape::GetRight() 
{ 
  if (shape == ShapeEnum::Rectangle) { return centerY+widthDiv2; } 	      
  if (shape == ShapeEnum::Circle)    { return centerY+radius; } 	      
  assert(false);
  return -1;  // To avoid compiler warning. 
}	
__host__ float Shape::GetTop() 
{ 
  if (shape == ShapeEnum::Rectangle) { return centerX-heightDiv2; } 	      
  if (shape == ShapeEnum::Circle)    { return centerX-radius; } 	      
  assert(false); 
  return -1;  // To avoid compiler warning. 
                 
}	
__host__ float Shape::GetBottom() 
{ 
  if (shape == ShapeEnum::Rectangle) { return centerX+heightDiv2; } 	      
  if (shape == ShapeEnum::Circle)    { return centerX+radius; } 	      
  assert(false); 
  return -1;  // To avoid compiler warning. 
}	

// Got this off a CUDA link but it's a good way to allocate for managed memory. 
void * Shape::operator new(size_t len) {
  void *ptr;
  cudaMallocManaged(&ptr, len);
  cudaDeviceSynchronize();
  return ptr;
}


//
// Run CICD for Shape. 
//   THis could be put into a make Test target if I were using Makefiles. 
//

#ifdef TEST 

__global__ void GPUCheckRectangle(Shape * shape,int top, int bottom, int left, int right,int *result) 
{
    *result = 0; 

    assert(shape->IsInShape(top-1,left).IsInPixel == 0); 	
    assert(shape->IsInShape(top-1,left-1).IsInPixel == 0); 	
    assert(shape->IsInShape(top,left).IsInPixel == 1); 	
    assert(shape->IsInShape(bottom+1,left).IsInPixel == 0); 	
    assert(shape->IsInShape(bottom,left-1).IsInPixel == 0); 	
    assert(shape->IsInShape(bottom,left).IsInPixel == 1); 	
    *result = 27; 
}
__global__ void GPUCheckCircle(Shape * shape,int centerX, int centerY, int radius,int *result) 
{
    *result = 0; 
    assert(shape->IsInShape(centerX-radius-1,centerY).IsInPixel == 0); 	
    assert(shape->IsInShape(centerX-radius,centerY).IsInPixel == 1); 	
    *result = 100; 
}

int main() 
{
    int *deviceResult; 
    int *hostResult = (int *) malloc(sizeof(int)); 
    assert(hostResult != NULL); 

    assert(cudaMalloc(&deviceResult,sizeof(int)) == cudaSuccess); 
    cudaMemset(deviceResult,0,sizeof(int)); 
    printf("Shape GPU Functions\n"); 
    Shape * redRectangle = new Shape(Shape::ShapeEnum::Rectangle,600,600,100,60,Red);
    GPUCheckRectangle<<<1,1>>>(redRectangle,550,650,570,630,deviceResult); 
    cudaDeviceSynchronize();

    printf("Host Functions\n"); 
    assert(redRectangle->GetLeft()    == 570); 
    assert(redRectangle->GetRight()   == 630); 
    assert(redRectangle->GetTop()     == 550); 
    assert(redRectangle->GetBottom()  == 650); 

    cudaMemcpy(hostResult,deviceResult,sizeof(int),cudaMemcpyDeviceToHost); 
    assert(*hostResult == 27); 

    cudaMemset(deviceResult,0,sizeof(int)); 
    Shape * blueCircle = new Shape(Shape::ShapeEnum::Circle,100,100,10,Red);
    GPUCheckCircle<<<1,1>>>(blueCircle,100,100,10,deviceResult); 
    cudaMemcpy(hostResult,deviceResult,sizeof(int),cudaMemcpyDeviceToHost); 
    assert(*hostResult == 100); 


    // Test Movement. 

    redRectangle = new Shape(Shape::ShapeEnum::Rectangle,10,80,16,50,Red);
    redRectangle->SetTravelAngle(5); 
    redRectangle->Move(10.0,1080,1920); 
    printf("RR Angle %f X%f Y%f \n",redRectangle->GetTravelAngle(),redRectangle->GetCenterX(),redRectangle->GetCenterY()); 
   
    printf("PASS!\n"); return 0; 
}


#endif 
