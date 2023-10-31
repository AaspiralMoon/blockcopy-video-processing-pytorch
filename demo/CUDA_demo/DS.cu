#include <cuda_runtime.h>

__global__ void DS(float *curr_img, float *ref_block, float *prev_bbox, int y, int x) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;  // 当前线程处理的k值
    
    LDSP = [0, -2, -1, -1, 1, -1, -2, 0, 0, 0, 2, 0, -1, 1, 1, 1, 0, 2]
    SDSP = [0, -1, -1, 0, 0, 0, 1, 0, 0, 1]

    // 检查是否超出了LDSP的范围
    if (k >= 9 || k == 4) {
        return;
    }

    // 计算diamond坐标
    int yDiamond = y + LDSP[2 * k + 1];
    int xDiamond = x + LDSP[2 * k];

    // 检查边界条件
    if (yDiamond < 0 || yDiamond + blockH >= h || xDiamond < 0 || xDiamond + blockW >= w) {
        return;
    }

    // 计算MAD
    float MAD = 0.0;
    for (int dy = 0; dy < blockH; dy++) {
        for (int dx = 0; dx < blockW; dx++) {
            int imgIndex = (yDiamond + dy) * w + xDiamond + dx;
            int refIndex = dy * blockW + dx;
            mad_value += fabsf(curr_img[imgIndex] - ref_block[refIndex]);
        }
    }
    MAD /= (blockW * blockH);

    costs[k] = MAD;

    if (cost < *min_cost) {
        atomicExch(min_cost, cost);  // 如果当前成本更低，则更新最小成本
        atomicExch(point, k);        // 同时更新索引
    }
}

