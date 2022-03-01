#include <math.h>
#include <string.h>
#include <stdio.h>


#define EPSILON 1e-3

const char *image_filename = "1.bmp";
unsigned int width, height;
unsigned int *hImage = nullptr;

float gaussian[50];

struct float4{
    float x;
    float y;
    float z;
    float w;

    float4(){};
    float4(float value) {x=y=z=w=value;}
};

typedef struct {unsigned char x,y,z,w;} uchar4;

extern "C" void loadImageData(int argc, char **argv);
extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width, unsigned int *height, const char *name);



////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void updateGaussianGold(float delta, int radius);
extern "C" void bilateralFilterGold(unsigned int *pSrc, unsigned int *pDest,
                                    float e_d, int w, int h, int r);


void updateGaussianGold(float delta, int radius) {
  for (int i = 0; i < 2 * radius + 1; i++) {
    int x = i - radius;
    gaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
}

float heuclideanLen(float4 a, float4 b, float d) {
  float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
              (b.z - a.z) * (b.z - a.z) + (b.w - a.w) * (b.w - a.w);

  return expf(-mod / (2 * d * d));
}

unsigned int hrgbaFloatToInt(float4 rgba) {
  unsigned int w = (((unsigned int)(fabs(rgba.w) * 255.0f)) & 0xff) << 24;
  unsigned int z = (((unsigned int)(fabs(rgba.z) * 255.0f)) & 0xff) << 16;
  unsigned int y = (((unsigned int)(fabs(rgba.y) * 255.0f)) & 0xff) << 8;
  unsigned int x = ((unsigned int)(fabs(rgba.x) * 255.0f)) & 0xff;

  return (w | z | y | x);
}

float4 hrgbaIntToFloat(unsigned int c) {
  float4 rgba;
  rgba.x = (c & 0xff) * 0.003921568627f;          //  /255.0f;
  rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;   //  /255.0f;
  rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;  //  /255.0f;
  rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;  //  /255.0f;
  return rgba;
}

float4 mul(float a, float4 b) {
  float4 ans;
  ans.x = a * b.x;
  ans.y = a * b.y;
  ans.z = a * b.z;
  ans.w = a * b.w;

  return ans;
}

float4 add4(float4 a, float4 b) {
  float4 ans;
  ans.x = a.x + b.x;
  ans.y = a.y + b.y;
  ans.z = a.z + b.z;
  ans.w = a.w + b.w;

  return ans;
}

void bilateralFilterGold(unsigned int *pSrc, unsigned int *pDest, float e_d,
                         int w, int h, int r) {
  float4 *hImage = new float4[w * h];
  float domainDist, colorDist, factor;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      hImage[y * w + x] = hrgbaIntToFloat(pSrc[y * w + x]);
    }
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float4 t(0.0f);
      float sum = 0.0f;

      for (int i = -r; i <= r; i++) {
        int neighborY = y + i;

        // clamp the neighbor pixel, prevent overflow
        if (neighborY < 0) {
          neighborY = 0;
        } else if (neighborY >= h) {
          neighborY = h - 1;
        }

        for (int j = -r; j <= r; j++) {
          domainDist = gaussian[r + i] * gaussian[r + j];

          // clamp the neighbor pixel, prevent overflow
          int neighborX = x + j;

          if (neighborX < 0) {
            neighborX = 0;
          } else if (neighborX >= w) {
            neighborX = w - 1;
          }

          colorDist = heuclideanLen(hImage[neighborY * w + neighborX],
                                    hImage[y * w + x], e_d);
          factor = domainDist * colorDist;
          sum += factor;
          t = add4(t, mul(factor, hImage[neighborY * w + neighborX]));
        }
      }

      pDest[y * w + x] = hrgbaFloatToInt(mul(1 / sum, t));
    }
  }

  delete[] hImage;
}





void loadImageData() {


  // LoadBMPFile((uchar4 **)&hImage, &width, &height, image_path);
    LoadBMPFile((uchar4 **)&hImage, &width, &height, "./1.bmp");

    if (!hImage) {
    fprintf(stderr, "Error opening file\n");
    // exit(EXIT_FAILURE);
  }

  printf("Loaded , %d x %d pixels\n\n", width, height);
}





int main()
{
  loadImageData();
  bilateralFilterGold(hImage,hImage,0,width,height,1);
}