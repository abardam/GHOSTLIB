#pragma once
#include <opencv2\opencv.hpp>

void inverse_point_mapping(const cv::Mat& _2d_pointmat_multiplied_by_depth_but_not_yet_inv_camera_matrix,
	std::vector<cv::Point2i> _2d_points,
	const cv::Mat& source_cameramatrix, const cv::Mat& target_cameramatrix, const cv::Mat& source_camerapose, const cv::Mat& target_camerapose,
	const cv::Mat& target_img, cv::Mat& output_img);