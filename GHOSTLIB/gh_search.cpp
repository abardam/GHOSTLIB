#include <queue>

#include "gh_search.h"

float calculate_score(cv::Vec3f a_x, cv::Vec3f a_y, cv::Vec3f a_z, cv::Vec3f b_x, cv::Vec3f b_y, cv::Vec3f b_z){
	return cv::norm(a_x - b_x) + cv::norm(a_y - b_y) + cv::norm(a_z - b_z);
}

void generate_score_vectors(const cv::Mat& mat, cv::Vec3f& x, cv::Vec3f& y, cv::Vec3f& z){
	x = (mat(cv::Range(0, 3), cv::Range(0, 1)));
	x = cv::normalize(x);
	y = (mat(cv::Range(0, 3), cv::Range(1, 2)));
	y = cv::normalize(x);
	z = (mat(cv::Range(0, 3), cv::Range(2, 3)));
	z = cv::normalize(x);
}
#if 0
unsigned int find_best_frame_2(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<std::vector<int>>& frame_clusters){

	bool clustered = !frame_clusters.empty();

	const cv::Mat& cmp_rot_only = cmp_camerapose(cv::Range(0, 3), cv::Range(0, 3));

	cv::Vec3f cmp_x;
	cv::Vec3f cmp_y;
	cv::Vec3f cmp_z;

	generate_score_vectors(cmp_camerapose, cmp_x, cmp_y, cmp_z);

	float best_score;
	unsigned int best_frame = snhmaps.size();;

	int num_frames = clustered ? frame_clusters.size() : snhmaps.size();

	for (int i = 0; i < num_frames; ++i){

		unsigned int frame = clustered&&!frame_clusters[i].empty() ? frame_clusters[i][0] : i;

		const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[frame], XXX)(cv::Range(0, 3), cv::Range(0, 3));

		cv::Vec3f cand_x;
		cv::Vec3f cand_y;
		cv::Vec3f cand_z;

		generate_score_vectors(cand_rot_only, cand_x, cand_y, cand_z);

		//float score = cv::norm(cmp_x - cand_x) + cv::norm(cmp_y - cand_y) + cv::norm(cmp_z - cand_z);
		float score = calculate_score(cmp_x, cmp_y, cmp_z, cand_x, cand_y, cand_z);

		if (best_frame == snhmaps.size() || score < best_score){
			best_frame = frame;
			best_score = score;
		}
	}

	return best_frame;
}
#endif

unsigned int find_best_frame(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<std::vector<int>>& frame_clusters){
	
	bool clustered = !frame_clusters.empty();

	cv::Mat zero(cv::Vec4f(0, 0, 0, 1));
	cv::Mat cmp_pt = cmp_camerapose.inv() * zero;
	cv::Vec4f cmp_pt_v = cmp_pt;
	cmp_pt_v(2) = 0;

	float best_score;
	unsigned int best_frame = snhmaps.size();;

	int num_frames = clustered ? frame_clusters.size() : snhmaps.size();

	for (int i = 0; i < num_frames; ++i){

		unsigned int frame = clustered&&!frame_clusters[i].empty()?frame_clusters[i][0]:i;

		cv::Mat cand_pt = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose).inv()* zero;
		cv::Vec4f cand_pt_v = cand_pt;
		cand_pt_v(2) = 0;

		float score = cv::norm(cmp_pt_v - cand_pt_v);

		if (best_frame == snhmaps.size() || score < best_score){
			best_frame = frame;
			best_score = score;
		}
	}

	return best_frame;
}

struct CmpFrameScore{
	bool operator()(std::pair<unsigned int, float> a, std::pair<unsigned int, float> b){
		return a.second > b.second;
	}
};

//this is what we are using currently
std::vector<unsigned int> sort_best_frames(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed, const std::vector<cv::Vec3f> precalculated_vecs, const std::vector<std::vector<int>>& frame_clusters){

	bool clustered = !frame_clusters.empty();

	const cv::Mat& cmp_rot_only = cmp_camerapose(cv::Range(0, 3), cv::Range(0, 3));


	cv::Vec3f cmp_rot_vec;

	cv::Rodrigues(cmp_rot_only, cmp_rot_vec);

	std::priority_queue<std::pair<unsigned int, float>, std::vector<std::pair<unsigned int, float>>, CmpFrameScore> frame_pqueue;


	int num_frames = clustered ? frame_clusters.size() : snhmaps.size();

	for (int i = 0; i < num_frames; ++i){
		unsigned int frame = clustered&&!frame_clusters[i].empty() ? frame_clusters[i][0] : i;

		if (framedatas_processed[frame].mnFacing == FACING_SIDE) continue;

		//const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose)(cv::Range(0, 3), cv::Range(0, 3));

		cv::Vec3f cand_rot_vec;
		//cv::Rodrigues(cand_rot_only, cand_rot_vec); //precalculate
		cand_rot_vec = precalculated_vecs[frame];

		float score = cv::normL2Sqr<float,float>((&cmp_rot_vec[0]), (&cand_rot_vec[0]), 3);

		frame_pqueue.push(std::pair<unsigned int, float>(frame, score));
	}

	std::vector<unsigned int> frame_v(frame_pqueue.size());

	for (int i = 0; i<frame_v.size(); ++i){
		frame_v[i] = frame_pqueue.top().first;
		frame_pqueue.pop();
	}

	return frame_v;
}

std::vector<cv::Vec3f> precalculate_vecs(const BodyPartDefinition& bpd, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed){
	std::vector<cv::Vec3f> precalculated_vecs(snhmaps.size());
	for (int frame = 0; frame < snhmaps.size(); ++frame){
		const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose)(cv::Range(0, 3), cv::Range(0, 3));
		cv::Vec3f cand_rot_vec;
		cv::Rodrigues(cand_rot_only, cand_rot_vec);
		precalculated_vecs[frame] = cand_rot_vec;
	}
	return precalculated_vecs;
}

BodypartFrameCluster cluster_frames_2(unsigned int K, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed, unsigned int max_iterations){
	if (K > snhmaps.size()) K = snhmaps.size();
	BodypartFrameCluster bodypart_cluster_ownership(bpdv.size());

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<cv::Mat> camera_pose_clusters(K);
		for (int j = 0; j < K; ++j){
			camera_pose_clusters[j] = get_bodypart_transform(bpdv[i], snhmaps[j], frame_data_processed[j].mCameraPose);
		}

		std::vector<std::vector<int>> cluster_ownership(K);

		for (int k = 0; k < max_iterations; ++k){
			cluster_ownership.clear();
			cluster_ownership.resize(K);
			std::vector<cv::Mat> cluster_center(K);

			for (int k = 0; k < K; ++k){
				cluster_center[k] = cv::Mat::zeros(4, 4, CV_32F);
			}

			//assign frames to clusters
			for (int frame = 0; frame < snhmaps.size(); ++frame){

				if (!frame_data_processed[frame].mValidity[i]) continue;

				unsigned int best_cluster = K;
				float best_score;

				cv::Vec3f cand_x;
				cv::Vec3f cand_y;
				cv::Vec3f cand_z;

				cv::Mat cand_cam_pose = get_bodypart_transform(bpdv[i], snhmaps[frame], frame_data_processed[frame].mCameraPose);

				generate_score_vectors(cand_cam_pose, cand_x, cand_y, cand_z);

				for (int cluster = 0; cluster < K; ++cluster){

					cv::Vec3f cmp_x;
					cv::Vec3f cmp_y;
					cv::Vec3f cmp_z;

					generate_score_vectors(camera_pose_clusters[cluster], cmp_x, cmp_y, cmp_z);

					float score = calculate_score(cand_x, cand_y, cand_z, cmp_x, cmp_y, cmp_z);

					if (best_cluster == K || score < best_score){
						best_cluster = cluster;
						best_score = score;
					}
				}

				cluster_ownership[best_cluster].push_back(frame);

				cv::Mat(cand_x).copyTo(cand_cam_pose(cv::Range(0, 3), cv::Range(0, 1)));
				cv::Mat(cand_y).copyTo(cand_cam_pose(cv::Range(0, 3), cv::Range(1, 2)));
				cv::Mat(cand_z).copyTo(cand_cam_pose(cv::Range(0, 3), cv::Range(2, 3)));

				cv::add(cluster_center[best_cluster], cand_cam_pose, cluster_center[best_cluster]);
			}

			for (int cluster = 0; cluster < K; ++cluster){
				//average each cluster center

				cluster_center[cluster] /= cluster_ownership[cluster].size();

				//then assign
				camera_pose_clusters[cluster] = cluster_center[cluster];
			}
		}

		bodypart_cluster_ownership[i] = cluster_ownership;
	}

	return bodypart_cluster_ownership;
}

bool check_validity(const FrameDataProcessed& frame_data, int bodypart){

	const cv::Mat& tex = frame_data.mBodyPartImages[bodypart].mMat;
	int num_pixels = 0;
	for (int y = 0; y<tex.rows; ++y){
		for (int x = 0; x<tex.cols; ++x){
			if (tex.ptr<cv::Vec3b>(y)[x](0) != 0xff) ++num_pixels;
		}
	}

	return ((num_pixels + 0.0f) > TEXTURE_VALID_PIXEL_THRESHOLD);
}

BodypartFrameCluster cluster_frames(unsigned int K, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed, unsigned int max_iterations){
	if (K > snhmaps.size()) K = snhmaps.size();
	BodypartFrameCluster bodypart_cluster_ownership(bpdv.size());

	cv::Range rotation_range[2];
	rotation_range[0] = rotation_range[1] = cv::Range(0, 3);

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<cv::Vec3f> camera_pose_clusters(K);
		for (int j = 0; j < K; ++j){
			cv::Vec3f temp;
			cv::Rodrigues(get_bodypart_transform(bpdv[i], snhmaps[j], frame_data_processed[j].mCameraPose)(rotation_range), temp);
			camera_pose_clusters[j] = temp;
		}

		std::vector<std::vector<int>> cluster_ownership(K);
		std::vector<bool> validity(snhmaps.size());

		std::vector<cv::Vec3f> frame_sphere_pts(snhmaps.size());
		for (int frame = 0; frame < snhmaps.size(); ++frame){
			cv::Vec3f temp;
			cv::Rodrigues(get_bodypart_transform(bpdv[i], snhmaps[frame], frame_data_processed[frame].mCameraPose)(rotation_range), temp);
			frame_sphere_pts[frame] = temp;
			validity[frame] = check_validity(frame_data_processed[frame], i);
		}


		for (int k = 0; k < max_iterations; ++k){

			std::cout << "bp: " << i << "iteration: " << k << " ";

			float total_cluster_movement = 0;

			cluster_ownership.clear();
			cluster_ownership.resize(K);
			std::vector<cv::Vec3f> cluster_center(K);

			for (int k = 0; k < K; ++k){
				cluster_center[k] = cv::Vec3f(0,0,0);
			}

			//assign frames to clusters
			for (int frame = 0; frame < snhmaps.size(); ++frame){

				if (!frame_data_processed[frame].mValidity[i] || !validity[frame]) continue;

				unsigned int best_cluster = K;
				float best_score;

				for (int cluster = 0; cluster < K; ++cluster){

					float score = cv::norm(camera_pose_clusters[cluster] - frame_sphere_pts[frame]);

					if (best_cluster == K || score < best_score){
						best_cluster = cluster;
						best_score = score;
					}
				}

				cluster_ownership[best_cluster].push_back(frame);

				cluster_center[best_cluster] += frame_sphere_pts[frame];
			}

			for (int cluster = 0; cluster < K; ++cluster){

				double num_frames_in_cluster = cluster_ownership[cluster].size();
				//average each cluster center
				cluster_center[cluster] = cluster_center[cluster] / (num_frames_in_cluster!=0?num_frames_in_cluster:1);

				//take the difference
				total_cluster_movement += cv::norm(cluster_center[cluster] - camera_pose_clusters[cluster]);

				//then assign
				camera_pose_clusters[cluster] = cluster_center[cluster];
			}

			std::cout << "total cluster movement: " << total_cluster_movement << std::endl;
			if (total_cluster_movement < 0.1) break;
			else if (total_cluster_movement != total_cluster_movement){
				std::cout << "something went wrong\n";
			}

		}

		bodypart_cluster_ownership[i] = cluster_ownership;
	}

	return bodypart_cluster_ownership;
}

BodypartFrameCluster cluster_frames_keyframes(int frame_gap, const BodyPartDefinitionVector& bpdv, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& frame_data_processed){

	BodypartFrameCluster bodypart_cluster_ownership(bpdv.size());


	for (int i = 0; i < bpdv.size(); ++i){
		int frame_counter = 0;
		for (int j = 0; j < snhmaps.size(); ++j){
			if (frame_counter >= frame_gap || 
				(j>0 && frame_data_processed[j].mnFacing != frame_data_processed[j-1].mnFacing)){

				if (frame_data_processed[j].mValidity[i]){

					bodypart_cluster_ownership[i].push_back(std::vector<int>());
					bodypart_cluster_ownership[i].back().push_back(j);
					frame_counter = 0;
				}
				else{
					frame_counter = frame_gap;
				}

			}
			++frame_counter;
		}
	}

	return bodypart_cluster_ownership;
}


//opencv serialization
void write(cv::FileStorage& fs, const std::string& s, const BodypartFrameCluster& n){

	fs << s << "{" << "bodyparts" << "[";

	//body parts
	for (auto it = n.begin(); it != n.end(); ++it){
		fs << "{" << "clusters" << "[";

		//clusters
		for (auto it2 = it->begin(); it2 != it->end(); ++it2){

			fs << "{" << "frames" << *it2 << "}";
		}

		fs << "]" << "}";
	}

	fs << "]" << "}";

}
void read(const cv::FileNode& node, BodypartFrameCluster& n, const BodypartFrameCluster& default_value){
	if (node.empty()){
		n = default_value;
	}
	else{
		n.clear();

		cv::FileNode bodyparts = node["bodyparts"];

		for (auto it = bodyparts.begin(); it != bodyparts.end(); ++it){
			n.push_back(std::vector<std::vector<int>>());

			cv::FileNode clusters = (*it)["clusters"];

			for (auto it2 = clusters.begin(); it2 != clusters.end(); ++it2){
				
				cv::FileNode frames = (*it2)["frames"];

				std::vector<int> frames_vector;

				frames >> frames_vector;

				n.back().push_back(frames_vector);
			}
		}
	}
}