#pragma once
#include <opencv2\opencv.hpp>

void inverse_point_mapping(const cv::Mat& neutral_pts,
	const std::vector<cv::Point2i> _2d_points,
	const cv::Mat& target_cameramatrix, const cv::Mat& target_camerapose,
	const cv::Mat& target_img, cv::Mat& output_img,
	cv::Mat& neutral_pts_occluded, std::vector<cv::Point2i>& _2d_points_occluded,
	bool debug = false);