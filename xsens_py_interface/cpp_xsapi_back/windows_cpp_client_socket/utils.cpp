//
// Created by teamat on 2020-01-25.
//

#include "utils.h"

using namespace std;

double delta_t(std::chrono::high_resolution_clock::time_point p_t1, std::chrono::high_resolution_clock::time_point p_t2)
{
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(p_t2 - p_t1);
    return time_span.count();
}
