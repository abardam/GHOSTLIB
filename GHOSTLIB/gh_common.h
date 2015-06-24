#pragma once

#include <cv_draw_common.h>
#include <cv_skeleton.h>
#include <recons_voxel.h>
#include <recons_cylinder.h>

typedef std::vector<std::vector<std::vector<int>>> BodypartFrameCluster;

//represents a single, processed frame
struct FrameDataProcessed{
	std::vector<CroppedMat> mBodyPartImages;

	std::vector<bool> mValidity; //basically whether we can use the body part images or not

	cv::Mat mCameraMatrix;
	cv::Mat mCameraPose;
	SkeletonNodeHard mRoot;

	int mnFacing;

	unsigned int mWidth, mHeight;

	cv::Mat mBackgroundImage;

	FrameDataProcessed(unsigned int num_bodyparts, unsigned int width, unsigned int height, const cv::Mat& camera_matrix, const cv::Mat& camera_pose, const SkeletonNodeHard& root):
		mBodyPartImages(num_bodyparts),
		mValidity(num_bodyparts),
		mWidth(width),
		mHeight(height),
		mCameraMatrix(camera_matrix),
		mCameraPose(camera_pose),
		mRoot(root){}
};

void load_processed_frames(const std::vector<std::string>& filepaths, const std::string& extension, unsigned int num_bodyparts, std::vector<FrameDataProcessed>& frameDataProcesseds, bool load_bg);

void load_packaged_file(std::string filename,
	BodyPartDefinitionVector& bpdv,
	std::vector<FrameDataProcessed>& frame_datas,
	BodypartFrameCluster& bodypart_frame_cluster,
	std::vector<std::vector<float>>& triangle_vertices,
	std::vector<std::vector<unsigned int>>& triangle_indices,
	std::vector<VoxelMatrix>& voxels, float& voxel_size,
	std::vector<Cylinder>& cylinders);