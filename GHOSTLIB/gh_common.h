#pragma once

#include <recons_common.h>
#include <cv_draw_common.h>

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

void load_processed_frames(const std::vector<std::string>& filepaths, unsigned int num_bodyparts, std::vector<FrameDataProcessed>& frameDataProcesseds, bool load_bg);