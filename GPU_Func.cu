#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. 함수는 Colab 환경에서 동작해야 합니다.
// 2. 자유롭게 구현하셔도 되지만 모든 함수에서 GPU를 활용해야 합니다.
// 3. CPU_Func.cu에 있는 Image_Check함수에서 True가 Return되어야 하며, CPU코드에 비해 속도가 빨라야 합니다.
/////////////////////////////////////////////////////////////////////////

__constant__ gaussian_filter[25];

__device__ void conv2d_5x5_device(float* filter, uint8_t* pixel, int x, int y, int width, int &v)
{
    v += pixel[(y + threadIdx.x) * width + x + threadIdx.y] * filter [threadIdx.x * 5 + threadIdx.y];
    __syncthreads();
}

__device__ void conv2d_3x3_device(int* filter_y, int* filter_x, uint_8* pixel, int x, int y, int width, int &gx, int &gy)
{
    gx += (int)pixel[(y + threadIdx.x) * width + x + threadIdx.y] * filter_y[i * 3 + threadIdx.y];
    gy += (int)pixel[(y + threadIdx.x) * width + x + threadIdx.y] * filter_x[i * 3 + threadIdx.y];
    __syncthreads();
}

__global__ set_gaussian_filter()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    __shared__ float filter[25];
    gaussian_filter[x * 5 + y] = (1 / (2 * 3.14)) * exp(-(x - 2) * (x - 2) + (y - 2) * (y - 2) / 2);
    __syncthreads();
    
    cudaMemcpyToSymbol(gaussian_filter, &filter, sizeof(float) * 25);
}

__global__ zero_padding(int width, int height, int filter_size, uint8_t* gray, uint8_t* padded)
{
    int x = blockIdx.x + 2;
    int y = blockIdx.y + 2;
    int offset = (int)(filter_size / 2);
    padded[x * (width + filter_size - 1) + y] = gray[((x - (int)(filter / 2)) * width + (y - (int)(filter / 2))) * 3];
}

__global__ void grayscale_kernel(uint8_t* buf, uint8_t* gray, int start_add)
{
    int idx = start_add + blockIdx.x * 3;
    gray[idx + threadIdx.x] = (buf[idx] * 0.114 + buf[idx + 1] * 0.587 + buf[i + 2] * 0.299);
}

__global__ void noise_reduction_kernel(int height, int width, uint8_t* padded uint8_t* gaussian)
{
    __shared__ uint8_t v;
    conv2d_5x5_device(float* filter, uint8_t* pixel, int x, int y, int width, int &v);
    if (threadIdx.x < 3)
        gaussian[(blockIdx.x * width + blockIdx.y) * 3 + threadIdx.x] = v;
}

__global__ void intensity_gradient_kernel(uint8_t* gaussian, int width, uint8_t* sobel, uint8_t* angle)
{
    __shared__ int filter_x[9] = {  -1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1};
    __shared__ int filter_x[9] = {  1, 2, 1,
                                    0, 0, 0,
                                    -1, -2, -1};
    __shared__ int gx = 0;
    __shared__ int gy = 0;
    __shared__ int t = 0;
    conv2d_3x3_device(filter_y, filter_x, gaussian, blockIdx.x, blockIdx.y, width, gx, gy);
    if(!blockIdx.x && !blockIdx.y)
    {
        t = sqrt(gx * gx + gy * gy);
        uint8_t v = 0;
        if (t > 255)
            v = 255;
        else
            v = t;
        sobel[(blockIdx.x * width + blockIdx.y) * 3] = v;
        sobel[(blockIdx.y * width + blockIdx.y) * 3 + 1] = v;
        sobel[(blockIdx.y * width + blockIdx.y) * 3 + 2] = v;

        float t_angle = 0;
        if(gy != 0 || gx != 0)
            t_angle = (float)atan2(gy, gx) * 180 / 3.14;
        if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
            angle[blockIdx.x * width + blockIdx.y] = 0;
        else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
            angle[blockIdx.x * width + blockIdx.y] = 45;
        else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
            angle[blockIdx.x * width + blockIdx.y] = 90;
        else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
            angle[blockIdx.x * width + blockIdx.y] = 135;
    }
}

__global__ void non_maximum_suppression_kernel(int width, uint8_t* sobel, uint8_t* angle, uint8_t* suppression_pixel, uint8_t& max, uint8_t& min)
{
    uint8_t p1 = 0;
    uint8_t p2 = 0;
    if (angle[blockIdx.y * width + blockIdx.x] == 0) {
        p1 = sobel[((blockIdx.y + 1) * width + blockIdx.x) * 3];
        p2 = sobel[((blockIdx.y - 1) * width + blockIdx.x) * 3];
    }
    else if (angle[blockIdx.y * width + blockIdx.x] == 45) {
        p1 = sobel[((blockIdx.y + 1) * width + blockIdx.x - 1) * 3];
        p2 = sobel[((bloxkIdx.y - 1) * width + blockIdx.x + 1) * 3];
    }
    else if (angle[blockIdx.y * width + blockIdx.x] == 90) {
        p1 = sobel[((bloxkIdx.y) * width + blockIdx.x + 1) * 3];
        p2 = sobel[((blockIdx.y) * width + blockIdx.x - 1) * 3];
    }
    else {
        p1 = sobel[((blockIdx.y + 1) * width + blockIdx.x + 1) * 3];
        p2 = sobel[((blockIdx.y - 1) * width + blockIdx.x - 1) * 3];
    }
    uint8_t v = sobel[(blockIdx.y * width + blockIdx.x) * 3];
    if(min > v)
        min = v;
    if(max < v)
        max = v;
    if ((v >= p1) && (v >= p2)) {
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3] = v;
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3 + 1] = v;
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3 + 2] = v;
    }
    else {
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3] = 0;
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3 + 1] = 0;
        suppression_pixel[(blockIdx.y * width + blockIdx.x) * 3 + 2] = 0;
    }

}

__global__ void hysterisis_thresholding()
{

}

void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len)
{
    uint8_t* dev_buf;
    uint8_t* dev_ gray;
    dim3 dimGrid(len - start_add / 3);
    dim3 dimBlock(3);
    cudaMalloc((void**) dev_buf, sizeof(uint_8) * len);
    cudaMalloc((void**) dev_gray, sizeof(uint_8) * len);
    cudaMemcpy(dev_buf, buf, sizeof(uint_8) * len, cudaMemcpyHostToDevice);
    grayscale_kernel<<<dimGrid, dimBlock>>>();
    cudaMemcpy(gray, dev_gray, sizeof(uint_8) * len, cudaMemcpyDeviceToHost);
    cudaFree(dev_buf);
    cudaFree(dev_gray);
}
void GPU_Noise_Reduction(int width, int height, uint8_t* gray, uint8_t* gaussian)
{
    uint8_t* dev_gray;
    uint8_t* dev_padded;
    uint8_t* dev_gaussian;
    cudaMalloc((void**) dev_gray, sizeof(uint8_t) * width * height);
    cudaMalloc((void**) dev_padded, sizeof(uint8_t) * (width + 4) * (height + 4));
    cudaMalloc((void**) dev_gaussian, sizeof(uint_8) * width * height);
    dim3 dimGrid(1);
    dim3 dimBlock(5, 5);
    set_gaussian_filter<<<dimGrid, dimBlock>>>();
    
    dim3 dimGrid(height, width);
    dim3 dimBlock(1);
    cudaMemset(dev_padded, 0, sizeof(uint8_t) * (width + 4) * (height + 4);
    cudaMemcpy(dev_gray, gray, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice);
    zero_padding<<<dimGrid, dimBlock>>>(width, height, 5, dev_gray, dev_padded);
    
    dim3 dimBlock(5, 5);
    noise_reduction_kernel<<<dimGrid, dimBlock>>>();
    cudaMemcpy(gaussian, dev_gaussian, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);
    cudaFree(dev_gray);
    cudaFree(dev_padded);
    cudaFree(dev_gaussian);

}
void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t* angle)
{
    uint8_t* padded;
    uint8_t* dev_padded;
    uint8_t* dev_gaussian;

    uint8_t* dev_sobel;
    uint8_t* dev_angle;

    dim3 dimBlock(1);
    dim3 dimGrid(height, width);
    
    padded = (uint8_t*)malloc((width + 2) * (height + 2));
    dev_padded = (void**)cudaMalloc(sizeof(uint8_t) * (width + 2) * (height + 2));
    dev_sobel = (void**)cudaMalloc(sizeof(uint8_t) * (width) * (height));
    dev_angle = (void**)cudaMalloc(sizeof(uint8_t) * (width) * (height));

    memset(tmp, (uint8_t)0, (width + 2) * (height + 2));
    cudaMemcpy(dev_tmp, tmp, sizeof(uint8_t) * (width + 2) * (height + 2), cudaMemcpyHostToDevice);
    zero_padding<<<dimGrid, dimBlock>>>(width, height, 3, dev_gaussian, dev_padded);
    
    dim3 dimBlock(3, 3);
    intensity_gradient_kernel <<<dimGrid, dimBlock>>>(dev_padded, width, dev_sobel, dev_angle);
    cudaMemcpy(sobel, dev_sobel, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(angle, dev_angle, sizeof(uint8_t) * windth * height,cudaMemcpyDeviceToHost);

    cudaFree(dev_gaussian);
    cudaFree(dev_padded);
    cudeFree(dev_sobel);
    cudaFree(dev_angle);
}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t* angle, uint8_t *sobel, uint8_t* suppression_pixel, uint8_t& min, uint8_t& max)
{
    uint8_t* dev_angle;
    uint8_t* dev_sobel;
    uint8_t* dev_suppression_pixel;
    uint8_t* dev_max;
    uint8_t* dev_min;

    dev_angle =
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel,uint8_t *hysteresis, uint8_t min, uint8_t max)
{

}
