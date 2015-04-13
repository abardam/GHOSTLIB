#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include "gh_render.h"

void inverse_point_mapping(const cv::Mat& neutral_pts,
	const std::vector<cv::Point2i> _2d_points,
	const cv::Mat& target_cameramatrix, const cv::Mat& target_camerapose,
	const cv::Mat& target_img, cv::Mat& output_img,
	cv::Mat& neutral_pts_occluded, std::vector<cv::Point2i>& _2d_points_occluded,
	bool debug){

	//cv::Mat neutral_transform = source_camerapose.inv() * source_cameramatrix.inv();

	cv::Mat reprojected_pts = target_cameramatrix * target_camerapose * neutral_pts; //neutral_transform * _2d_pointmat_multiplied_by_depth_but_not_yet_inv_camera_matrix;
	divide_pointmat_by_z(reprojected_pts);

	cv::Mat debug_img;
	if (debug){
		debug_img = target_img.clone();
	}

	std::vector<cv::Vec4f> neutral_pts_occluded_v;
	neutral_pts_occluded_v.reserve(reprojected_pts.cols);

	for (int j = 0; j < reprojected_pts.cols; ++j){
		int orig_x = _2d_points[j].x;
		int orig_y = _2d_points[j].y;
		int repro_x = reprojected_pts.ptr<float>(0)[j];
		int repro_y = reprojected_pts.ptr<float>(1)[j];

		if (CLAMP(repro_x, repro_y, output_img.cols, output_img.rows) && CLAMP(orig_x, orig_y, target_img.cols, target_img.rows)){
			cv::Vec3b color = target_img.ptr<cv::Vec3b>(repro_y)[repro_x];

			if (debug){
				debug_img.ptr<cv::Vec3b>(repro_y)[repro_x] = cv::Vec3b(0xff, 0xff, 0);
				cv::imshow("debug image", debug_img);
				cv::imshow("output", output_img);
				cv::waitKey(1);
			}

			if (color == cv::Vec3b(0xff, 0, 0)){
				cv::Vec4f pt = neutral_pts.col(j);
				neutral_pts_occluded_v.push_back(pt);
				_2d_points_occluded.push_back(_2d_points[j]);
			}
			else{
				output_img.ptr < cv::Vec3b >(orig_y)[orig_x] = color;
			}
		}
	}

	neutral_pts_occluded = pointvec_to_pointmat(neutral_pts_occluded_v);
}

