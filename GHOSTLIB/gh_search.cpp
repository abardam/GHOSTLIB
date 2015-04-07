#include "gh_search.h"

unsigned int find_best_frame(const BodyPartDefinition& bpd, const cv::Mat& cmp_camerapose, const std::vector<SkeletonNodeHardMap>& snhmaps){
	
	const cv::Mat& cmp_rot_only = cmp_camerapose(cv::Range(0, 3), cv::Range(0, 3));

	cv::Vec3f cmp_x = cmp_rot_only(cv::Range(0,3),cv::Range(0,1));
	cv::Vec3f cmp_y = cmp_rot_only(cv::Range(0,3),cv::Range(1,2));
	cv::Vec3f cmp_z = cmp_rot_only(cv::Range(0,3),cv::Range(2,3));

	cmp_x = cv::normalize(cmp_x);
	cmp_y = cv::normalize(cmp_y);
	cmp_z = cv::normalize(cmp_z);

	float best_score;
	unsigned int best_frame = snhmaps.size();;

	for (int i = 0; i < snhmaps.size(); ++i){

		const cv::Mat& cand_rot_only = get_bodypart_transform(bpd, snhmaps[i])(cv::Range(0, 3), cv::Range(0, 3));

		cv::Vec3f cand_x = cand_rot_only(cv::Range(0, 3), cv::Range(0, 1));
		cv::Vec3f cand_y = cand_rot_only(cv::Range(0, 3), cv::Range(1, 2));
		cv::Vec3f cand_z = cand_rot_only(cv::Range(0, 3), cv::Range(2, 3));

		cand_x = cv::normalize(cand_x);
		cand_y = cv::normalize(cand_y);
		cand_z = cv::normalize(cand_z);

		float score = cv::norm(cmp_x - cand_x) + cv::norm(cmp_y - cand_y) + cv::norm(cmp_z - cand_z);

		if (best_frame == snhmaps.size() || score < best_score){
			best_frame = i;
			best_score = score;
		}
	}

	return best_frame;
}