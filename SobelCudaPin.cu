#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define THREADS 16

char* filename;
const char* seq_out = "gradientSEQ.png";
const char* cuda_out = "gradientCUDA.png";

void sobelSeq(unsigned char *image, unsigned char *output, int width, int height) {
    float gx = 0;
    float gy = 0;
    float s = 0;
    for (int i = 1; i < height-1; ++i) {
        for (int j = 1; j < width-1; ++j) {
            gx = image[(i-1)*width+j-1]+2*image[i*width+j-1]+image[(i+1)*width+j-1]
            -image[(i-1)*width+j+1]-2*image[i*width+j+1]-image[(i+1)*width+j+1];
            gy = image[(i-1)*width+j-1]+2*image[(i-1)*width+j]+image[(i-1)*width+j+1]
            -image[(i+1)*width+j-1]-2*image[(i+1)*width+j]-image[(i+1)*width+j+1];
            s = (int) sqrt(gx*gx+gy*gy);
            s = s > 255 ? 255:s;
            output[i*width+j] = s;
        }
    }
}

__global__ void KernelSobelElement (unsigned char *image, unsigned char *output, int width, int height) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  float gx = 0;
  float gy = 0;
  float s = 0;
  if(i > 0 && j > 0 && i < (height-1) && j < (width-1)) {
      gx = image[(i-1)*width+j-1]+2*image[i*width+j-1]+image[(i+1)*width+j-1]
      -image[(i-1)*width+j+1]-2*image[i*width+j+1]-image[(i+1)*width+j+1];
      gy = image[(i-1)*width+j-1]+2*image[(i-1)*width+j]+image[(i-1)*width+j+1]
      -image[(i+1)*width+j-1]-2*image[(i+1)*width+j]-image[(i+1)*width+j+1];
      s = (int) sqrtf(gx*gx+gy*gy);
      s = s > 255 ? 255:s;
      output[i*width+j] = s;
  }
}

void CheckCudaError(char sms[], int line);
float GetTime(void);

int main(int argc, char** argv) {
    CheckCudaError((char *) "First line", __LINE__);
    int width,height, pixelWidth; //meta info de la imagen
    unsigned char *image; //imagen
    unsigned int nThreads=16;
    if (argc == 1){filename = "lenna.png";}
    else if (argc == 2) {filename = argv[1];}
    else if (argc == 3) {filename = argv[1]; nThreads = atoi(argv[2]);}
    else {printf("Usage: ./cudacode.exe filename\n"); exit(0); }

    printf("Reading image...\n");
    image = stbi_load(filename, &width, &height, &pixelWidth, 1);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return (-1);
    }
    printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

    //Lectura feta
    cudaEvent_t E1, E2;
    unsigned int numBytes = width*height*sizeof(char);
    cudaEventCreate(&E1); cudaEventCreate(&E2);

    
    unsigned char *imX, *imZ;
    unsigned char *image_d;
    unsigned char *image_o;
    float TiempoEle, TiempoSEQ, t1, t2;

    image_d = (unsigned char*) malloc(numBytes);

    // Ejecucion Secuencial, se ejecuta varias veces para evitar problemas de precision con el clock
    t1=GetTime();
    for (int t = 0; t<10; t++)
        sobelSeq(image, image_d, width, height);
    t2=GetTime();
    TiempoSEQ = (t2 - t1) / 10.0;

    stbi_write_png(seq_out,width,height,1,image_d,0);
    printf("Sequential image written with time %4.6f\n ms", TiempoSEQ);


  
// Pinned
    cudaMallocHost(&image_o, numBytes);
    cudaMallocHost(&image_d, numBytes);

    image_o=image;
    
    // Obtener Memoria en el device
    cudaMalloc((void**)&imX, numBytes);
    cudaMalloc((void**)&imZ, numBytes);
    CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

    // Copiar datos desde el host en el device
    cudaMemcpy(imX, image_o, numBytes, cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

    int nBlocksFil = (height+nThreads-1)/nThreads;
    int nBlocksCol = (width+nThreads-1)/nThreads;

    //dim3 dimGridE(Ncol/nThreads, Nfil/nThreads, 1);
    dim3 dimGridE(nBlocksCol, nBlocksFil, 1);
    dim3 dimBlockE(nThreads, nThreads, 1);

    printf("\n");
    printf("Kernel Elemento a Elemento MEMPIN\n");
    printf("Dimension problema: %d filas x %d columnas\n", height, width);
    printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockE.x, dimBlockE.y, dimBlockE.z, dimBlockE.x * dimBlockE.y * dimBlockE.z);
    printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridE.x, dimGridE.y, dimGridE.z, dimGridE.x * dimGridE.y * dimGridE.z);

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);

    KernelSobelElement<<<dimGridE, dimBlockE>>>(imX, imZ, width, height);

    CheckCudaError((char *) "Invocar Kernel", __LINE__);

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    // Obtener el resultado desde el host
    cudaMemcpy(image_d, imZ, numBytes, cudaMemcpyDeviceToHost);
    CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

    cudaFree(imX); cudaFree(imZ);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&TiempoEle, E1, E2);

    cudaEventDestroy(E1); cudaEventDestroy(E2);

    stbi_write_png(cuda_out,width,height,1,image_d,0);
    printf("Image Written with CUDA execution with time: %4.6f\n ms", TiempoEle);
}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;

  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }
  //else printf("(OK) %s \n", sms);
}

float GetTime(void)        {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}
