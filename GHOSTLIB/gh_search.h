#pragma once

#include <opencv2\opencv.hpp>
#include <AssimpCV.h>

unsigned int find_best_frame(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps);