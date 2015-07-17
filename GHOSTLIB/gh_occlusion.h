#pragma once

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include <recons_common.h>
#include <recons_voxel.h>
#include <recons_voxel_body.h>

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

void process_occlusions_texturemap_cylinder(const cv::Mat& render_pretexture,
	const std::vector<cv::Mat>& bodypart_render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeHardMap& snhmap,
	const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const cv::Mat& camera_matrix,
	const std::vector<VoxelMatrix>& bodypart_voxels,
	float voxel_size,
	std::vector<cv::Mat>& bodypart_textures,
	std::vector<cv::Mat>& bodypart_textureweights,
	const std::vector<bool>& validity = std::vector<bool>());

void process_occlusions_texturemap_triangles(
	std::vector<std::vector<unsigned int>> bodypart_triangle_indices,
	std::vector<std::vector<float>> bodypart_triangle_vertices,
	std::vector<std::vector<std::vector<unsigned int>>> bodypart_triangle_UV,
	const cv::Mat& render_depth,
	int anim_frame,
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeHardMap& snhmap,
	const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const cv::Mat& camera_matrix,
	const std::vector<VoxelMatrix>& bodypart_voxels,
	float voxel_size,
	std::vector<cv::Mat>& bodypart_textures,
	std::vector<cv::Mat>& bodypart_textureweights,
	const std::vector<bool>& validity = std::vector<bool>());