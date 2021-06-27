#define NUMTHREADS 16

#include <fstream> 
#include <unistd.h> 
#include <iostream>
#include <cstdlib>
#include <vector> 
#include <opencv2/opencv.hpp>
#include "Shape.h" 

using namespace cv;
using namespace std; 


typedef struct 
{
  int width;
  int height;
  int BytesPerPixel;
  Color_t BackgroundColor; 
} FrameConfig_t;


__global__ void DrawFrame(unsigned char * fPtr, FrameConfig_t frameConfig,Shape ** shapes)
{

  int colStart  = (blockIdx.y    ) * frameConfig.width/gridDim.y; 
  int colEnd    = (blockIdx.y + 1) * frameConfig.width/gridDim.y; 

  int rowStart  = (blockIdx.x    ) * frameConfig.height/gridDim.x; 
  int rowEnd    = (blockIdx.x + 1) * frameConfig.height/gridDim.x; 

  for (int row = rowStart; row < rowEnd; row++) 
    for (int col = colStart; col < colEnd; col++) 
    {
       unsigned char *p = fPtr+ (row*frameConfig.width + col)*3;
       //
       // You need to be careful with pointers in CUDA. I had a bug in my frame computation and it manifested as the kernel prematurely aborting.  
       // It turned out I had indexed past the pointer's allocated space. 
       //
       int shapeNum = 0;
       Shape * shapePtr; 
       *p = frameConfig.BackgroundColor.blue; 
       *(p+1) = frameConfig.BackgroundColor.green; 
       *(p+2) = frameConfig.BackgroundColor.red; 
       while ((shapePtr = (*(shapes+shapeNum++))) != nullptr)
       {
         auto shapePixel = shapePtr->IsInShape(row, col);
         if (shapePixel.IsInPixel)
         {
            *p = shapePixel.Color.blue; 
            *(p+1) = shapePixel.Color.green; 
            *(p+2) = shapePixel.Color.red; 
	 }
       }
    }  
}



int main()
{

    FrameConfig_t frameConfig = { .width = 640 , .height = 480, .BytesPerPixel=3,.BackgroundColor = White}; 

    Shape ** myshapes;
    cudaError_t res = cudaMallocManaged(&myshapes, sizeof(Shape *)*5);

    // Allocate each shape 

    Shape * blueBall     = new Shape(Shape::ShapeEnum::Circle,100,100,60,Blue); 
    blueBall->SetTravelAngle(120);
    Shape * redRectangle = new Shape(Shape::ShapeEnum::Rectangle,300,300,100,60,Red); 
    redRectangle->SetTravelAngle(260);
    *(myshapes + 0) = blueBall; 
    *(myshapes + 1) = redRectangle; 
    *(myshapes + 2) = nullptr; 

    dim3 cpus(4,4);  

    unsigned char * framePtr; 

    int frameLen = frameConfig.height*frameConfig.width*frameConfig.BytesPerPixel;
    printf("FrameLen: %d\n",frameLen); 

    res = cudaMalloc(&framePtr, frameLen);
    if (res != 0) { printf("CUDA Result %d\n", res);  exit(0); } 
    else {printf("Allocated %p to %p\n",framePtr,framePtr+(frameLen/4));} 

    unsigned int * hostPtr;
    hostPtr = (unsigned int *) malloc(frameLen); 
  
    Mat image(480,640,CV_8UC3);
    image.data = (unsigned char *) hostPtr; 

    VideoWriter video("BouncingShapes.avi", cv::VideoWriter::fourcc('M','J','P','G'), 60, Size(640,480));
    for (int i = 0; i < 1200;i++)
    {
      // GPUs in desktop systems usually have a separate memory architecture with a separate set of memory.
      // This means that they must be synced by PCI transactions, and either you use cudaMemcpy or you use
      // cudaManagedMemory with an OS smart enough to figure out what memory is where. 
      // From what I've read (which may or may not be accurate), the Jetson nano, using an older architecture, 
      //  uses the same physical memory as the ARM CPU but doesn't support paging, so 
      //  you can write from the host and device will pick up the proper memory, but when you read back on the
      //  host side you need to do a cudaMemcpy. This may be wrong or might be improved
      //  upon but it's working for now. 
      //    
      Shape * shapePtr;
      int shapeNum = 0; 
      while ((shapePtr = (*(myshapes+shapeNum++))) != nullptr)
         shapePtr->Move(5.0,480,640);
      DrawFrame<<<cpus,1>>>(framePtr, frameConfig,myshapes);
      cudaGetErrorString(cudaPeekAtLastError());
      cudaMemcpy(hostPtr,framePtr,frameLen,cudaMemcpyDeviceToHost); 
      video.write(image); 
    }

    video.release(); 
    destroyAllWindows(); // destroys the window showing image
    return 0;
}


