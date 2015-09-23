#include "gh_common.h"
#include <recons_common.h>


void load_processed_frames(const std::vector<std::string>& filepaths, const std::string& extension, unsigned int num_bodyparts, std::vector<FrameDataProcessed>& frameDataProcesseds, bool load_bg){

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

		int facing;
		fs["facing"] >> facing;
		
		FrameDataProcessed frameData(num_bodyparts, win_width, win_height, camera_intrinsic, camera_extrinsic, root);
		frameData.mnFacing = facing;

		cv::Mat full_body_image;
		fs["color"] >> full_body_image;
		std::vector<cv::Vec3b> crop_colors;
		crop_colors.push_back(cv::Vec3b(0xff, 0xff, 0xff));
		
		frameData.mBodyImage = crop_mat(full_body_image, crop_colors);

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
			ss << path << "\\bodypart" << bp << "frame" << frame << extension;

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

		if(load_bg){
			ss.str("");
			ss << path << "\\background_" << "frame" << frame << extension;

			fs.open(ss.str(), cv::FileStorage::READ);

			if (!fs.isOpened()) continue;

			cv::Mat bg_mat;

			fs["mat"] >> bg_mat;

			frameData.mBackgroundImage = bg_mat;

		}

		frameDataProcesseds.push_back(frameData);
	}
}


void load_packaged_file(std::string filename,
	BodyPartDefinitionVector& bpdv,
	std::vector<FrameDataProcessed>& frame_datas,
	BodypartFrameCluster& bodypart_frame_cluster,
	std::vector<std::vector<float>>& triangle_vertices,
	std::vector<std::vector<unsigned int>>& triangle_indices,
	std::vector<VoxelMatrix>& voxels, float& voxel_size,
	std::vector<Cylinder>& cylinders){


	std::string path;
	{
		unsigned int startpos = filename.find_last_of("/\\") + 1;
		path = filename.substr(0, startpos);
	}

	int win_width, win_height;

	cv::FileStorage savefile;
	savefile.open(filename, cv::FileStorage::READ);

	savefile["width"] >> win_width;
	savefile["height"] >> win_height;

	cv::FileNode bpdNode = savefile["bodypartdefinitions"];
	bpdv.clear();
	for (auto it = bpdNode.begin(); it != bpdNode.end(); ++it)
	{
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}

	cv::FileNode frameNode = savefile["frame_datas"];
	frame_datas.clear();
	for (auto it = frameNode.begin(); it != frameNode.end(); ++it){
		cv::Mat camera_pose, camera_matrix;
		SkeletonNodeHard root;
		int facing;
		(*it)["camera_extrinsic"] >> camera_pose;
		(*it)["camera_intrinsic_mat"] >> camera_matrix;
		(*it)["skeleton"] >> root;
		(*it)["facing"] >> facing;
		FrameDataProcessed frame_data(bpdv.size(), 0, 0, camera_matrix, camera_pose, root);
		frame_data.mnFacing = facing;

		std::string body_image_path;
		(*it)["body_image_path"] >> body_image_path;
		frame_data.mBodyImage.mMat = cv::imread(path + "/" + body_image_path);
		(*it)["body_image_offset"] >> frame_data.mBodyImage.mOffset;
		(*it)["body_image_size"] >> frame_data.mBodyImage.mSize;


		frame_datas.push_back(frame_data);
	}

	cv::FileNode clusterNode = savefile["bodypart_frame_cluster"];
	bodypart_frame_cluster.clear();
	bodypart_frame_cluster.resize(bpdv.size());
	for (auto it = clusterNode.begin(); it != clusterNode.end(); ++it){
		int bodypart;
		(*it)["bodypart"] >> bodypart;
		cv::FileNode clusterClusterNode = (*it)["clusters"];
		for (auto it2 = clusterClusterNode.begin(); it2 != clusterClusterNode.end(); ++it2){
			int main_frame;
			(*it2)["main_frame"] >> main_frame;

			CroppedMat image;
			if ((*it2)["image"].empty()){
				std::string image_path;
				(*it2)["image_path"] >> image_path;
				image.mMat = cv::imread(path + "/" + image_path);

				if (image.mMat.empty()) continue;
				(*it2)["image_offset"] >> image.mOffset;
				(*it2)["image_size"] >> image.mSize;

			}
			else{
				(*it2)["image"] >> image;
			}

			win_width = image.mSize.width;
			win_height = image.mSize.height;

			frame_datas[main_frame].mBodyPartImages.resize(bpdv.size());
			frame_datas[main_frame].mBodyPartImages[bodypart] = image;
			std::vector<int> cluster;
			cluster.push_back(main_frame);
			bodypart_frame_cluster[bodypart].push_back(cluster);
		}
	}

	cv::FileNode vertNode = savefile["triangle_vertices"];
	triangle_vertices.clear();
	for (auto it = vertNode.begin(); it != vertNode.end(); ++it){
		triangle_vertices.push_back(std::vector<float>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			float vert;
			(*it2) >> vert;
			triangle_vertices.back().push_back(vert);
		}
	}


	cv::FileNode indNode = savefile["triangle_indices"];
	triangle_indices.clear();
	for (auto it = indNode.begin(); it != indNode.end(); ++it){
		triangle_indices.push_back(std::vector<unsigned int>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			int ind;
			(*it2) >> ind;
			triangle_indices.back().push_back(ind);
		}
	}

	cv::FileNode voxNode = savefile["voxels"];
	voxels.clear();
	for (auto it = voxNode.begin(); it != voxNode.end(); ++it){
		int width, height, depth;
		(*it)["width"] >> width;
		(*it)["height"] >> height;
		(*it)["depth"] >> depth;
		voxels.push_back(VoxelMatrix(width, height, depth));
	}

	savefile["voxel_size"] >> voxel_size;

	cv::FileNode cylNode = savefile["cylinders"];
	cylinders.clear();
	for (auto it = cylNode.begin(); it != cylNode.end(); ++it){
		float width, height;
		(*it)["width"] >> width;
		(*it)["height"] >> height;
		cylinders.push_back(Cylinder(width, height));
	}

	savefile.release();

	frame_datas[0].mWidth = win_width;
	frame_datas[0].mHeight = win_height;

}

