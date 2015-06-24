#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include "gh_render.h"

void inverse_point_mapping(const cv::Mat& neutral_pts,
	const std::vector<cv::Point2i> _2d_points,
	const cv::Mat& target_cameramatrix, const cv::Mat& target_camerapose,
	const cv::Mat& target_img,
	cv::Point2i offset, cv::Mat& output_img,
	cv::Mat& neutral_pts_occluded, std::vector<cv::Point2i>& _2d_points_occluded,
	bool try_render_white,
	bool debug){

	bool alpha_channel = output_img.channels() == 4;

	//cv::Mat neutral_transform = source_camerapose.inv() * source_cameramatrix.inv();

	cv::Mat reprojected_pts = target_cameramatrix * target_camerapose * neutral_pts; //neutral_transform * _2d_pointmat_multiplied_by_depth_but_not_yet_inv_camera_matrix;
	divide_pointmat_by_z(reprojected_pts);

	cv::Mat debug_img;
	if (debug){
		debug_img = target_img.clone();
	}

	std::vector<cv::Vec4f> neutral_pts_occluded_v;
	neutral_pts_occluded_v.reserve(reprojected_pts.cols);


	if (try_render_white){
		for (int j = 0; j < reprojected_pts.cols; ++j){
			int orig_x = _2d_points[j].x;
			int orig_y = _2d_points[j].y;
			int repro_x = reprojected_pts.ptr<float>(0)[j] - offset.x;
			int repro_y = reprojected_pts.ptr<float>(1)[j] - offset.y;

			if (CLAMP(repro_x, repro_y, target_img.cols, target_img.rows) && CLAMP(orig_x, orig_y, output_img.cols, output_img.rows)){
				cv::Vec3b color = target_img.ptr<cv::Vec3b>(repro_y)[repro_x];

				if (debug){
					debug_img.ptr<cv::Vec3b>(repro_y)[repro_x] = cv::Vec3b(0xff, 0xff, 0);
					cv::imshow("debug image", debug_img);
					cv::imshow("output", output_img);
					char q = cv::waitKey(1);
					if (q == 'q') debug = false;
				}

				if (color == cv::Vec3b(0xff, 0, 0) || color == cv::Vec3b(0xff,0xff,0xff)){
					neutral_pts_occluded_v.push_back(neutral_pts.col(j));
					_2d_points_occluded.push_back(_2d_points[j]);
				}
				else if (color == cv::Vec3b(0xff, 0xff, 0xff)){}
				else{
					if (alpha_channel){
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](0) = color(0);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](1) = color(1);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](2) = color(2);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](3) = 0xff;
					}
					else{
						output_img.ptr<cv::Vec3b>(orig_y)[orig_x] = color;
					}
				}
			}
		}
	}
	else{
		for (int j = 0; j < reprojected_pts.cols; ++j){
			int orig_x = _2d_points[j].x;
			int orig_y = _2d_points[j].y;
			int repro_x = reprojected_pts.ptr<float>(0)[j] - offset.x;
			int repro_y = reprojected_pts.ptr<float>(1)[j] - offset.y;

			if (CLAMP(repro_x, repro_y, target_img.cols, target_img.rows) && CLAMP(orig_x, orig_y, output_img.cols, output_img.rows)){
				cv::Vec3b color = target_img.ptr<cv::Vec3b>(repro_y)[repro_x];

				if (debug){
					debug_img.ptr<cv::Vec3b>(repro_y)[repro_x] = cv::Vec3b(0xff, 0xff, 0);
					cv::imshow("debug image", debug_img);
					cv::imshow("output", output_img);
					char q = cv::waitKey(1);
					if (q == 'q') debug = false;
				}

				if (color == cv::Vec3b(0xff, 0, 0)){
					neutral_pts_occluded_v.push_back(neutral_pts.col(j));
					_2d_points_occluded.push_back(_2d_points[j]);
				}
				else if (color == cv::Vec3b(0xff, 0xff, 0xff)){}
				else{
					if (alpha_channel){
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](0) = color(0);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](1) = color(1);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](2) = color(2);
						output_img.ptr<cv::Vec4b>(orig_y)[orig_x](3) = 0xff;
					}
					else{
						output_img.ptr<cv::Vec3b>(orig_y)[orig_x] = color;
					}
				}
			}
		}
	}

	neutral_pts_occluded = pointvec_to_pointmat(neutral_pts_occluded_v);
}

