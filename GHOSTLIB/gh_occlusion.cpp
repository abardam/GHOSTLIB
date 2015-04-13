#include "gh_occlusion.h"
#include <cv_draw_common.h>


void process_and_save_occlusions(const cv::Mat& render_pretexture, 
	const cv::Mat& render_depth, int anim_frame, 
	const BodyPartDefinitionVector& bpdv, 
	const std::vector<FrameData>& frame_datas, 
	const std::string& video_directory){

	unsigned int win_width = render_pretexture.cols;
	unsigned int win_height = render_pretexture.rows;

	std::vector<cv::Mat> bodypart_pts(bpdv.size());
	std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
	std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());

	std::vector<std::vector<cv::Point2i>> nonbodypart_pts_2d_v(bpdv.size());

	for (int y = 0; y < win_height; ++y){
		for (int x = 0; x < win_width; ++x){

			unsigned int bodypart_id;
			bool is_bodypart = false;

			for (int i = 0; i < bpdv.size(); ++i){
				cv::Vec3b orig_color = render_pretexture.ptr<cv::Vec3b>(y)[x];
				cv::Vec3b bp_color(bpdv[i].mColor[2] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[0] * 0xff);

				if (orig_color == bp_color
					){
					float depth = render_depth.ptr<float>(y)[x];
					bodypart_pts_2d_withdepth_v[i].push_back(cv::Vec4f(depth*x, depth*y,
						depth, 1));
					bodypart_pts_2d_v[i].push_back(cv::Point2i(x, y));
					is_bodypart = true;
					bodypart_id = i;
					break;
				}
			}

			if (is_bodypart){
				for (int i = 0; i < bpdv.size(); ++i){
					if (i == bodypart_id) continue;

					nonbodypart_pts_2d_v[i].push_back(cv::Point2i(x, y));
				}
			}
		}
	}

	//now apply to the RGB images

	std::stringstream filename_ss;

	std::vector<cv::Vec3b> crop_colors;
	crop_colors.push_back(cv::Vec3b(0xff, 0xff, 0xff));
	//crop_colors.push_back(cv::Vec3b(0xff, 0, 0));

	for (int i = 0; i < bpdv.size(); ++i){
		cv::Mat bodypart_image(win_height, win_width, CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));

		for (int j = 0; j < nonbodypart_pts_2d_v[i].size(); ++j){
			int x = nonbodypart_pts_2d_v[i][j].x;
			int y = nonbodypart_pts_2d_v[i][j].y;

			bodypart_image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0xff, 0, 0);
		}
		for (int j = 0; j < bodypart_pts_2d_v[i].size(); ++j){
			int x = bodypart_pts_2d_v[i][j].x;
			int y = bodypart_pts_2d_v[i][j].y;

			bodypart_image.ptr<cv::Vec3b>(y)[x] = frame_datas[anim_frame].mmColor.ptr<cv::Vec3b>(y)[x];
		}

		filename_ss.str("");
		filename_ss << video_directory << "\\bodypart" << i << "frame" << anim_frame << ".xml.gz";

		cv::FileStorage fs;
		fs.open(filename_ss.str(), cv::FileStorage::WRITE);

		fs << "bodypart" << i << "frame" << anim_frame << "cropped_mat" << crop_mat(bodypart_image, crop_colors);

		fs.release();
	}
}

