#define NUMTHREADS 16

#include <fstream> 

#include <unistd.h> 
#include <iostream>
#include <cstdlib>
 
using namespace std; 
//
// RDP: Simple CUDA code for drawing a bouncing ball.  //

// To keep this simple I am using RGBA which aligns all my pixels on word boundaries. 
// It'd get more complicated if I were doing 10-bit color depths or non-planar formats. 

typedef unsigned int Color_t;


class PixelColorResult
{
   public:
     bool IsInPixel;  // Is our pixel in our shape. 
     Color_t Color;  //  What Color? 

};

class Shape  
{
    virtual PixelColorResult ShapePresence(); 
    Color_t color; 
    Color_t PixelColor(int X, int Y, Color_t 
}

class Ball : Shape  
{ 
  PixelColorResult ShapePresence
  public:
    int centerX; 
    int centerY;
    int radiusSquared; // Keep the square of the radius to keep from taking a square root. 
};

class Rectangle : Shape  
{ 
  public:
    int centerX; 
    int centerY;
    int radiusSquared; // Keep the square of the radius to keep from taking a square root. 

}



typedef struct 
{
  int width;
  int height;
  int BytesPerPixel;
  Color_t BackgroundColor; 
} FrameConfig_t;

//
// Return 1 if row,col is within ballConfig
//  

__global__ void DrawFrame(unsigned int * fPtr, FrameConfig_t frameConfig,std::vector<Shape> Objects )
{

  printf("%d\n",ballConfig.area()); 
  int colStart  = (blockIdx.y    ) * frameConfig.width/gridDim.y; 
  int colEnd    = (blockIdx.y + 1) * frameConfig.width/gridDim.y; 

  int rowStart  = (blockIdx.x    ) * frameConfig.height/gridDim.x; 
  int rowEnd    = (blockIdx.x + 1) * frameConfig.height/gridDim.x; 
  for (int row = rowStart; row < rowEnd; row++) 
    for (int col = colStart; col < colEnd; col++) 
    {
       unsigned int *p = fPtr+row*frameConfig.width + col;
       //
       // You need to be careful with pointers in CUDA. I had a bug in my frame computation and it manifested as the kernel prematurely aborting.  
       // It turned out I had indexed past the pointer's allocated space. 
       //
       if (((row-ballConfig.centerX)*(row-ballConfig.centerX) + (col-ballConfig.centerY)*(col-ballConfig.centerY)) < (float) (ballConfig.radiusSquared)) 
       {
         *p = ballConfig.color;
       } 
       else 
       { 
         *p = frameConfig.BackgroundColor;
       } 
    }  
}

 

int main()
{

    FrameConfig_t frameConfig = { .width = 1920, .height = 1080, .BytesPerPixel=4,.BackgroundColor = 0xFFFFFFFF }; 

    Ball blueball = { .centerX = 100, .centerY = 100, .radiusSquared = 2500,.color = 0xFFFF0000 }; 
    dim3 cpus(4,4);  

    unsigned int * framePtr; 

    int frameLen = frameConfig.height*frameConfig.width*frameConfig.BytesPerPixel;
    printf("FrameLen: %d\n",frameLen); 
    printf("sizeof(unsigned int): %ld\n",sizeof(unsigned int)); 

    cudaError_t res = cudaMalloc(&framePtr, frameLen);
    if (res != 0) { printf("CUDA Result %d\n", res);  exit(0); } 
    else {printf("Allocated %p to %p\n",framePtr,framePtr+(frameLen/4));} 
  
    printf("Drawing Frame\n"); 
    DrawFrame<<<cpus,1>>>(framePtr, frameConfig,blueball);
    cudaGetErrorString(cudaPeekAtLastError() );
    printf("Drew Frame\n"); 
    sleep(1); 
    
    unsigned int * hostPtr;
    hostPtr = (unsigned int *) malloc(frameLen); 
    cudaMemcpy(hostPtr,framePtr,frameLen,cudaMemcpyDeviceToHost); 

    std::ofstream ofs("frame.raw", ios::binary);
    ofs.write((const char *) hostPtr, frameLen);
    ofs.close();

}


