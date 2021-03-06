#include <queue>

#include "gh_search.h"

#define DIST_ROTATION 1
#define DIST_ZCMP 2

#define TEXTURE_DIST DIST_ROTATION

#define ROTATION_MAGNITUDE_THRESHOLD 0.01

float rotation_magnitude(const cv::Mat& rot){
	cv::Vec3f rod;
	cv::Rodrigues(rot, rod);
	return rod(0)*rod(0) + rod(1)*rod(1) + rod(2)*rod(2);
}

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

typedef std::pair<std::string, int> BodyPartFrame;
typedef std::map<BodyPartFrame, cv::Mat> BodyPartFrameTransformMap;
typedef std::pair<BodyPartFrame, cv::Mat> BodyPartFrameTransformEntry;

BodyPartFrameTransformMap bodypart_frame_transform_map;

cv::Mat get_bodypart_transform_saveframe(const BodyPartDefinition& bpd, int frame, const std::vector<SkeletonNodeHardMap>& snhmaps, const std::vector<FrameDataProcessed>& framedatas_processed){
	BodyPartFrame bpf(bpd.mBodyPartName, frame);

	BodyPartFrameTransformMap::iterator it = bodypart_frame_transform_map.find(bpf);

	if (it == bodypart_frame_transform_map.end()){
		cv::Mat transform = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose).inv();
		bodypart_frame_transform_map.insert(BodyPartFrameTransformEntry( bpf, transform));
		return transform;
	}
	else{
		return it->second;
	}
}

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

		//cv::Mat cand_pt = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose).inv()* zero;
		cv::Mat cand_pt = get_bodypart_transform_saveframe(bpd, frame, snhmaps, framedatas_processed).inv()* zero;
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
	cv::Mat front_vec(cv::Vec3f(0, 0, 1));

	//cv::Vec3f cmp_rot_vec;
	//cv::Rodrigues(cmp_rot_only, cmp_rot_vec);

	cv::Mat cmp_rot_front_mat = cmp_rot_only * front_vec;
	cv::Vec3f cmp_rot_front = cmp_rot_front_mat;
	cmp_rot_front = cv::normalize(cmp_rot_front);

	std::priority_queue<std::pair<unsigned int, float>, std::vector<std::pair<unsigned int, float>>, CmpFrameScore> frame_pqueue;


	int num_frames = clustered ? frame_clusters.size() : snhmaps.size();

	for (int i = 0; i < num_frames; ++i){
		unsigned int frame = clustered&&!frame_clusters[i].empty() ? frame_clusters[i][0] : i;

		if (framedatas_processed[frame].mnFacing == FACING_SIDE) continue;

		const cv::Mat& cand_rot_only = get_bodypart_transform_saveframe(bpd, frame, snhmaps, framedatas_processed)(cv::Range(0, 3), cv::Range(0, 3));
		//const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose)(cv::Range(0, 3), cv::Range(0, 3));

#if TEXTURE_DIST == DIST_ROTATION
		cv::Mat cmp_cand = cmp_rot_only * cand_rot_only.t();
		float score = rotation_magnitude(cmp_cand);
#elif TEXTURE_DIST == DIST_ZCMP
		cv::Vec3f cand_rot_vec;
		cand_rot_vec = precalculated_vecs[frame];

		float score = acos(cmp_rot_front.dot(cand_rot_vec));
#endif
		//float score = cv::normL2Sqr<float,float>((&cmp_rot_vec[0]), (&cand_rot_vec[0]), 3);

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

	cv::Mat front_vec(cv::Vec3f(0, 0, 1));

	for (int frame = 0; frame < snhmaps.size(); ++frame){
		const cv::Mat& cand_rot_only = get_bodypart_transform_saveframe(bpd, frame, snhmaps, framedatas_processed)(cv::Range(0, 3), cv::Range(0, 3));
		//const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[frame], framedatas_processed[frame].mCameraPose)(cv::Range(0, 3), cv::Range(0, 3));
		//cv::Vec3f cand_rot_vec;
		//cv::Rodrigues(cand_rot_only, cand_rot_vec);

		cv::Mat cand_rot_mul = cand_rot_only * front_vec;
		precalculated_vecs[frame] = cand_rot_mul;
		precalculated_vecs[frame] = cv::normalize(precalculated_vecs[frame]);
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
	rotation_range[0] = cv::Range(0,3);
	rotation_range[1] = cv::Range(0,3);

#if TEXTURE_DIST == DIST_ZCMP

	cv::Mat front_vec(cv::Vec3f(0, 0, 1));

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<cv::Vec3f> camera_pose_clusters(K);

		for (int j = 0; j < K; ++j){
			//cv::Vec3f temp;
			//cv::Rodrigues(get_bodypart_transform(bpdv[i], snhmaps[j], frame_data_processed[j].mCameraPose)(rotation_range), temp);

			cv::Mat temp = get_bodypart_transform(bpdv[i], snhmaps[j], frame_data_processed[j].mCameraPose)(rotation_range)* front_vec;
			camera_pose_clusters[j] = temp;
			camera_pose_clusters[j] = cv::normalize(camera_pose_clusters[j]);
		}

		std::vector<std::vector<int>> cluster_ownership(K);
		std::vector<bool> validity(snhmaps.size());

		std::vector<cv::Vec3f> frame_sphere_pts(snhmaps.size());

		for (int frame = 0; frame < snhmaps.size(); ++frame){
			//cv::Vec3f temp;
			//cv::Rodrigues(get_bodypart_transform(bpdv[i], snhmaps[frame], frame_data_processed[frame].mCameraPose)(rotation_range), temp);

			cv::Mat temp = get_bodypart_transform(bpdv[i], snhmaps[frame], frame_data_processed[frame].mCameraPose)(rotation_range)* front_vec;
			frame_sphere_pts[frame] = temp;
			frame_sphere_pts[frame] = cv::normalize(frame_sphere_pts[frame]);
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

					//float score = cv::norm(camera_pose_clusters[cluster] - frame_sphere_pts[frame]);
					float score = acos(camera_pose_clusters[cluster].dot(frame_sphere_pts[frame]));
					//float score = rotation_magnitude(camera_pose_clusters[cluster] * frame_sphere_pts[frame].t());

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

		//set 0 to the one closest to the center
		for (int j = 0; j < cluster_ownership.size(); ++j){

			cv::Vec3f center(0, 0, 0);
			for (int k = 0; k < cluster_ownership[j].size(); ++k){
				center += frame_sphere_pts[cluster_ownership[j][k]];
			}
			center /= (cluster_ownership[j].size() + 0.0f);
			int closest_frame_index=-1;
			float closest_distance;
			for (int k = 0; k < cluster_ownership[j].size(); ++k){
				float distance = cv::norm(frame_sphere_pts[cluster_ownership[j][k]] - center);
				if (closest_frame_index == -1 || closest_distance > distance){
					closest_distance = distance;
					closest_frame_index = k;
				}
			}
			if (closest_frame_index != -1){
				int temp_frame = cluster_ownership[j][0];
				cluster_ownership[j][0] = cluster_ownership[j][closest_frame_index];
				cluster_ownership[j][closest_frame_index] = temp_frame;
			}

		}

		bodypart_cluster_ownership[i] = cluster_ownership;
	}
#elif TEXTURE_DIST == DIST_ROTATION
	for (int bp = 0; bp < bpdv.size(); ++bp){
		for (int frame = 0; frame < snhmaps.size(); ++frame){
			if (bodypart_cluster_ownership[bp].empty()){
				//i.e. if this is the first frame

				//we'll make a new cluster
				bodypart_cluster_ownership[bp].push_back(std::vector<int>());
				
				//and put this frame as the base
				bodypart_cluster_ownership[bp].back().push_back(frame);
			}
			else{
				//otherwise, lets check the bases and get the lowest theta

				int closest_cluster = -1;
				float lowest_difference;

				const cv::Mat& transform = get_bodypart_transform(bpdv[bp], snhmaps[frame], cv::Mat::eye(4, 4, CV_32F))(rotation_range);

				for (int cluster = 0; cluster < bodypart_cluster_ownership[bp].size(); ++cluster){
					//check the difference in magnitude of this base from the guy

					int cand_frame = bodypart_cluster_ownership[bp][cluster][0];

					const cv::Mat& cand_transform = get_bodypart_transform(bpdv[bp], snhmaps[cand_frame], cv::Mat::eye(4, 4, CV_32F))(rotation_range);

					float difference = rotation_magnitude(transform * cand_transform.t());

					if (closest_cluster == -1 || lowest_difference > difference){
						lowest_difference = difference;
						closest_cluster = cluster;
					}
				}

				//now lets check if it's below the threshold
				if (lowest_difference < ROTATION_MAGNITUDE_THRESHOLD){
					//if it is, add it to the cluster

					bodypart_cluster_ownership[bp][closest_cluster].push_back(frame);

					//now recalculate the base

					int cluster_base = -1;
					float ssd;
					for (int frame_in_cluster = 0; frame_in_cluster < bodypart_cluster_ownership[bp][closest_cluster].size(); ++frame_in_cluster){
						//calculate the difference in magnitude of this frame from all the others

						const cv::Mat& transform = get_bodypart_transform(bpdv[bp], snhmaps[bodypart_cluster_ownership[bp][closest_cluster][frame_in_cluster]], cv::Mat::eye(4, 4, CV_32F))(rotation_range);

						float ssd_cand = 0;
						for (int frame_in_cluster2 = 0; frame_in_cluster2 < bodypart_cluster_ownership[bp][closest_cluster].size(); ++frame_in_cluster2){

							const cv::Mat& transform2 = get_bodypart_transform(bpdv[bp], snhmaps[bodypart_cluster_ownership[bp][closest_cluster][frame_in_cluster2]], cv::Mat::eye(4, 4, CV_32F))(rotation_range);
							float diff = rotation_magnitude(transform * transform2.t());
							ssd_cand += diff*diff;
						}

						if (cluster_base == -1 || ssd > ssd_cand){
							cluster_base = frame_in_cluster;
							ssd = ssd_cand;
						}
					}

					//swap base positions

					int old_base_frame = bodypart_cluster_ownership[bp][closest_cluster][0];
					bodypart_cluster_ownership[bp][closest_cluster][0] = bodypart_cluster_ownership[bp][closest_cluster][cluster_base];
					bodypart_cluster_ownership[bp][closest_cluster][cluster_base] = old_base_frame;
				}
				else{
					//if it's not, create a new cluster

					//we'll make a new cluster
					bodypart_cluster_ownership[bp].push_back(std::vector<int>());

					//and put this frame as the base
					bodypart_cluster_ownership[bp].back().push_back(frame);
				}
			}
		}

		std::cout << "bodypart " << bp << ": " << bodypart_cluster_ownership[bp].size() << " clusters created\n";

	}
#endif


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