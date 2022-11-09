#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. �Լ��� Colab ȯ�濡�� �����ؾ� �մϴ�.
// 2. �����Ӱ� �����ϼŵ� ������ ��� �Լ����� GPU�� Ȱ���ؾ� �մϴ�.
// 3. CPU_Func.cu�� �ִ� Image_Check�Լ����� True�� Return�Ǿ�� �ϸ�, CPU�ڵ忡 ���� �ӵ��� ����� �մϴ�.
/////////////////////////////////////////////////////////////////////////

void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {}
void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {}
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle){}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t *angle,uint8_t *sobel, uint8_t *suppression_pixel, uint8_t& min, uint8_t& max){}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max) {}