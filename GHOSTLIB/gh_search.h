#pragma once

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include "gh_common.h"

#define TEXTURE_VALID_PIXEL_THRESHOLD 10

typedef std::vector<std::vector<std::vector<int>>> BodypartFrameCluster;

unsigned int find_best_frame(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<std::vector<int>>& frame_clusters);
std::vector<unsigned int> sort_best_frames(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<cv::Vec3f> precalculated_vecs, const std::vector<std::vector<int>>& frame_clusters = std::vector<std::vector<int>>());

BodypartFrameCluster cluster_frames(unsigned int K, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed, unsigned int max_iterations=50);

BodypartFrameCluster cluster_frames_keyframes(int frame_gap, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed);
std::vector<cv::Vec3f> precalculate_vecs(const BodyPartDefinition& bpd, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed);

//opencv serialization
void write(cv::FileStorage& fs, const std::string& s, const BodypartFrameCluster& n);
void read(const cv::FileNode& node, BodypartFrameCluster& n, const BodypartFrameCluster& default_value = BodypartFrameCluster());