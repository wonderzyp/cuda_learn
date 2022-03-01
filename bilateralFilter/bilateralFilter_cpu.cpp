#include <math.h>
#include <string.h>


#define EPSILON 1e-3

float gaussian[50];

struct float4{
    float x;
    float y;
    float z;
    float w;

    float4(){};
    float4(float value) {x=y=z=w=value;}
};


//2022-3-1 以下两个函数未理解
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

void bilateralFilterGold(unsigned int *pSrc, unsigned int *pDest, float e_d,
                        int w, int h, int r){
    float4 *hImage = new float4[w*h];
    float domainDist, colorDist, factor;

    for (int y=0; y<h; ++y){
      for (int x=0; x<w; ++x) {
        hImage[y*w+x] = hrgbaFloatToInt(pSrc[y*w+x]);
      }
    }


}