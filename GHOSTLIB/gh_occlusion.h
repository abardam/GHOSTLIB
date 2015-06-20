#pragma once

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include <recons_common.h>

void process_and_save_occlusions(const cv::Mat& render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const std::string& video_directory);

//same as the other one but each body part has its own individual render_pretexture
//will intersect with the original render_pretexture
void process_and_save_occlusions_expanded(const cv::Mat& render_pretexture,
	const std::vector<cv::Mat>& bodypart_render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const std::string& video_directory,
	const std::vector<bool>& validity = std::vector<bool>());