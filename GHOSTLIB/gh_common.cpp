#include "gh_common.h"


void load_processed_frames(const std::vector<std::string>& filepaths, unsigned int num_bodyparts, std::vector<FrameDataProcessed>& frameDataProcesseds){

	cv::FileStorage fs;

	for (auto it = filepaths.begin(); it != filepaths.end(); ++it){
		fs.open(*it, cv::FileStorage::READ);

		if (!fs.isOpened()) continue;

		std::cout << "loading processed " << *it << std::endl;

		float win_width, win_height, fovy;
		cv::Mat camera_extrinsic, camera_intrinsic;

		if (fs["camera_intrinsic"].empty()){
			fs["camera_intrinsic_mat"] >> camera_intrinsic;
			cv::Mat color_tmp;
			fs["color"] >> color_tmp;
			win_width = color_tmp.cols;
			win_height = color_tmp.rows;
		}
		else{
			fs["camera_intrinsic"]["width"] >> win_width;
			fs["camera_intrinsic"]["height"] >> win_height;
			fs["camera_intrinsic"]["fovy"] >> fovy;
			camera_intrinsic = generate_camera_intrinsic(win_width, win_height, fovy);
		}

		fs["camera_extrinsic"] >> camera_extrinsic;

		SkeletonNodeHard root;
		fs["skeleton"] >> root;

		
		FrameDataProcessed frameData(num_bodyparts, win_width, win_height, camera_intrinsic, camera_extrinsic, root);


		fs.release();

		//find the frame number
		//usually it is in the filename
		unsigned int frame;
		std::string path;
		{
			unsigned int startpos = it->find_last_of("/\\")+1;
			unsigned int endpos = it->find_first_of(".", startpos);

			std::string frame_str = it->substr(startpos, endpos-startpos);
			frame = atoi(frame_str.c_str());

			path = it->substr(0, startpos);
		}

		std::stringstream ss;

		for (int bp = 0; bp < num_bodyparts; ++bp){
			ss.str("");
			ss << path << "\\bodypart" << bp << "frame" << frame << ".xml.gz";

			fs.open(ss.str(), cv::FileStorage::READ);

			if (!fs.isOpened()) continue;

			CroppedMat cropped_mat;

			fs["cropped_mat"] >> cropped_mat;

			frameData.mBodyPartImages[bp] = cropped_mat;

			if (cropped_mat.mMat.empty()){
				frameData.mValidity[bp] = false;
			}
			else{
				frameData.mValidity[bp] = true;
			}
		}

		frameDataProcesseds.push_back(frameData);
	}
}