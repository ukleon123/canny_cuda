#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. 함수는 Colab 환경에서 동작해야 합니다.
// 2. 자유롭게 구현하셔도 되지만 모든 함수에서 GPU를 활용해야 합니다.
// 3. CPU_Func.cu에 있는 Image_Check함수에서 True가 Return되어야 하며, CPU코드에 비해 속도가 빨라야 합니다.
/////////////////////////////////////////////////////////////////////////

void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {}
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {}
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle){}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max){}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max) {}