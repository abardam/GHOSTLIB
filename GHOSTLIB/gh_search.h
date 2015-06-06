#pragma once

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include "gh_common.h"

typedef std::vector<std::vector<std::vector<int>>> BodypartFrameCluster;

unsigned int find_best_frame(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<std::vector<int>>& frame_clusters);
std::vector<unsigned int> sort_best_frames(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<std::vector<int>>& frame_clusters = std::vector<std::vector<int>>());

BodypartFrameCluster cluster_frames(unsigned int K, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed, unsigned int max_iterations=50);

//opencv serialization
void write(cv::FileStorage& fs, const std::string& s, const BodypartFrameCluster& n);
void read(const cv::FileNode& node, BodypartFrameCluster& n, const BodypartFrameCluster& default_value = BodypartFrameCluster());