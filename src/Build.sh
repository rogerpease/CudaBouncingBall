export TEGRA_ARMABI=aarch64-linux-gnu
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig/ 
set -e 

/usr/local/cuda-10.2/bin/nvcc Shape.cu  `pkg-config --cflags opencv4` -D TEST -o ShapeTest
./ShapeTest 

/usr/local/cuda-10.2/bin/nvcc Shape.cu -dc `pkg-config --cflags opencv4` -c -o Shape.o
/usr/local/cuda-10.2/bin/nvcc BouncingBall.cu  -dc `pkg-config --cflags opencv4` -c -o BouncingBall.o

/usr/local/cuda-10.2/bin/nvcc BouncingBall.o Shape.o -o BouncingBall  `pkg-config --libs opencv4` \
        -lpthread -lv4l2 -lEGL -lGLESv2 -lX11 \
        -lcuda -lcudart 

