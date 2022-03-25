#pragma once
#include <iostream>
#include <thread>

#include <CL/sycl.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>


#define MAX_ITER 100

void PlotSet_DPCPP(std::string FileName, uint64_t rez, double RE_Start, double RE_End, double IM_Start, double IM_End);

void PlotSet_CPU(std::string FileName, uint64_t rez, double RE_Start, double RE_End, double IM_Start, double IM_End);