#include <stdio.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"


const char *filename = "lenna.png";
const char* outputname = "gradient.png";
const int maskx[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
const int masky[3][3] = {{1, 2, -1}, {0, 0, -0}, {-1, -2, -1}};
unsigned char *image; //imagen
int width,height, pixelWidth; //meta info de la imagen

int main() {
    printf("Reading image...\n");
    image = stbi_load(filename, &width, &height, &pixelWidth, 1);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return (-1);
    }
    printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
    unsigned char image_d[height][width];
    int gx = 0;
    int gy = 0;
    int s = 0;
    // Sequential sobel
    for (int i = 1; i < height-1; ++i) {
        for (int j = 1; j < width-1; ++j) {
            gx = image[(i-1)*width+j-1]+2*image[i*width+j-1]+image[(i+1)*width+j-1]
            -image[(i-1)*width+j+1]-2*image[i*width+j+1]-image[(i+1)*width+j+1];
            gy = image[(i-1)*width+j-1]+2*image[(i-1)*width+j]+image[(i-1)*width+j+1]
            -image[(i+1)*width+j-1]-2*image[(i+1)*width+j]-image[(i+1)*width+j+1];
            s = (int) sqrt(gx*gx+gy*gy);
            s = s > 255 ? 255:s;
            image_d[i][j] = s;
        }
    }
    stbi_write_png(outputname,width,height,1,image_d,0);
    printf("Image Written");
}
