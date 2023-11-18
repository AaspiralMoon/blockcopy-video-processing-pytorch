#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

torch::Tensor _costMAD(const torch::Tensor& block1, const torch::Tensor& block2) {
    auto block1_float = block1.toType(torch::kFloat32);
    auto block2_float = block2.toType(torch::kFloat32);
    return torch::mean(torch::abs(block1_float - block2_float));
}

bool _checkBounded(int xval, int yval, int w, int h, int blockW, int blockH) {
    return !(yval < 0 || yval + blockH >= h || xval < 0 || xval + blockW >= w);
}

std::vector<int> OBDS(const torch::Tensor& imgCurr, 
                                        const torch::Tensor& ref_block, 
                                        const std::vector<int>& prev_bbox) {
    int h = imgCurr.size(0);
    int w = imgCurr.size(1);
    
    int x1 = prev_bbox[0], y1 = prev_bbox[1], x2 = prev_bbox[2], y2 = prev_bbox[3];
    int blockW = x2 - x1;
    int blockH = y2 - y1;

    std::vector<float> costs(9, 65537.0f);
    int computations = 0;
    std::vector<int> bboxCurr;

    std::vector<std::vector<int>> LDSP = {{0, -2}, {-1, -1}, {1, -1}, {-2, 0}, {0, 0}, {2, 0}, {-1, 1}, {1, 1}, {0, 2}};
    std::vector<std::vector<int>> SDSP = {{0, -1}, {-1, 0}, {0, 0}, {1, 0}, {0, 1}};

    int x = x1; // (x, y) large diamond center point
    int y = y1;

    // Start search
    costs[4] = _costMAD(imgCurr.slice(0, y1, y2).slice(1, x1, x2), ref_block).item<float>();
    float cost = 0.0f;
    int point = 4;
    if (costs[4] != 0.0f) {
        computations += 1;
        for (int k = 0; k < 9; ++k) {
            int yDiamond = y + LDSP[k][1];
            int xDiamond = x + LDSP[k][0];
            if (!_checkBounded(xDiamond, yDiamond, w, h, blockW, blockH)) {
                continue;
            }
            if (k == 4) {
                continue;
            }
            costs[k] = _costMAD(imgCurr.slice(0, yDiamond, yDiamond + blockH).slice(1, xDiamond, xDiamond + blockW), ref_block).item<float>();
            computations += 1;
        }

        point = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
        cost = costs[point];
    }

    int SDSPFlag = 1; // SDSPFlag = 1, trigger SDSP
    int cornerFlag = 0; // cornerFlag = 0 by default
    int xLast;
    int yLast;
    if (point != 4) {
        SDSPFlag = 0;
        cornerFlag = (std::abs(LDSP[point][0]) == std::abs(LDSP[point][1])) ? 0 : 1;
        xLast = x;
        yLast = y;
        x += LDSP[point][0];
        y += LDSP[point][1];
        std::fill(costs.begin(), costs.end(), 65537.0f);
        costs[4] = cost;
    }

    // SDSP Search Loop
    while (SDSPFlag == 0) {
        if (cornerFlag == 1) { // If the next MBD point is at the corner
            for (int k = 0; k < 9; ++k) {
                int yDiamond = y + LDSP[k][1];
                int xDiamond = x + LDSP[k][0];
                if (!_checkBounded(xDiamond, yDiamond, w, h, blockW, blockH)) {
                    continue;
                }
                if (k == 4 || 
                    (xDiamond >= xLast - 1 && xDiamond <= xLast + 1 && 
                    yDiamond >= yLast - 1 && yDiamond <= yLast + 1)) {
                    continue;
                }
                costs[k] = _costMAD(imgCurr.slice(0, yDiamond, yDiamond + blockH).slice(1, xDiamond, xDiamond + blockW), ref_block).item<float>();
                computations += 1;
            }
        } else { // If the next MBD point is at the edge
            std::vector<int> lst;
            if (point == 1) lst = {0, 1, 3};
            else if (point == 2) lst = {0, 2, 5};
            else if (point == 6) lst = {3, 6, 8};
            else if (point == 7) lst = {5, 7, 8};

            for (int idx : lst) {
                int yDiamond = y + LDSP[idx][1];
                int xDiamond = x + LDSP[idx][0];
                if (!_checkBounded(xDiamond, yDiamond, w, h, blockW, blockH)) {
                    continue;
                }
                costs[idx] = _costMAD(imgCurr.slice(0, yDiamond, yDiamond + blockH).slice(1, xDiamond, xDiamond + blockW), ref_block).item<float>();
                computations += 1;
            }
        }

        point = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
        cost = costs[point];

        SDSPFlag = (point == 4) ? 1 : 0;
        if (!SDSPFlag) {
            cornerFlag = (std::abs(LDSP[point][0]) == std::abs(LDSP[point][1])) ? 0 : 1;
            xLast = x;
            yLast = y;
            x += LDSP[point][0];
            y += LDSP[point][1];
            std::fill(costs.begin(), costs.end(), 65537.0f);
            costs[4] = cost;
        }
    }

    // Final SDSP Search
    std::fill(costs.begin(), costs.end(), 65537.0f);
    costs[2] = cost;
    for (int k = 0; k < 5; ++k) {
        int yDiamond = y + SDSP[k][1];
        int xDiamond = x + SDSP[k][0];
        if (!_checkBounded(xDiamond, yDiamond, w, h, blockW, blockH)) {
            continue;
        }
        if (k == 2) {
            continue;
        }
        costs[k] = _costMAD(imgCurr.slice(0, yDiamond, yDiamond + blockH).slice(1, xDiamond, xDiamond + blockW), ref_block).item<float>();
        computations += 1;
    }

    point = 2;
    cost = 0.0f;
    if (costs[2] != 0.0f) {
        point = std::distance(costs.begin(), std::min_element(costs.begin(), costs.end()));
        cost = costs[point];
    }

    x += SDSP[point][0];
    y += SDSP[point][1];

    bboxCurr = {x, y, x + blockW, y + blockH};

    return bboxCurr;
}

PYBIND11_MODULE(obds_torch_extension, m) {
    m.def("OBDS", &OBDS, "Object-based Diamond Search");
}