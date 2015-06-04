#pragma once

#include <opencv2\opencv.hpp>
#include <AssimpCV.h>
#include <recons_common.h>

void process_and_save_occlusions(const cv::Mat& render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const std::string& video_directory);