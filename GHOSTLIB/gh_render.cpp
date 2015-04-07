#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include "gh_render.h"

void inverse_point_mapping(const cv::Mat& _2d_pointmat_multiplied_by_depth_but_not_yet_inv_camera_matrix,
	std::vector<cv::Point2i> _2d_points,
	const cv::Mat& source_cameramatrix, const cv::Mat& target_cameramatrix, const cv::Mat& source_camerapose, const cv::Mat& target_camerapose,
	const cv::Mat& target_img, cv::Mat& output_img){

	cv::Mat neutral_transform = source_camerapose.inv() * source_cameramatrix.inv();

	cv::Mat reprojected_pts = target_cameramatrix * target_camerapose * neutral_transform * _2d_pointmat_multiplied_by_depth_but_not_yet_inv_camera_matrix;
	divide_pointmat_by_z(reprojected_pts);

	for (int j = 0; j < reprojected_pts.cols; ++j){
		int orig_x = _2d_points[j].x;
		int orig_y = _2d_points[j].y;
		int repro_x = reprojected_pts.ptr<float>(0)[j];
		int repro_y = reprojected_pts.ptr<float>(1)[j];

		if (CLAMP(repro_x, repro_y, output_img.cols, output_img.rows) && CLAMP(orig_x, orig_y, target_img.cols, target_img.rows)){
			output_img.ptr < cv::Vec3b >(orig_y)[orig_x] = target_img.ptr<cv::Vec3b>(repro_y)[repro_x];
		}
	}
}

