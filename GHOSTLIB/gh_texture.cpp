#include "gh_texture.h"


void cartesian_to_cylinder(const cv::Vec3f& pt, float& distance, float& azimuth, float& height){
	distance = sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
	azimuth = atan2(pt[1], pt[0]);
	height = pt[2];
}
void cartesian_to_cylinder(const cv::Vec4f& pt, float& distance, float& azimuth, float& height){
	distance = sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
	azimuth = atan2(pt[1], pt[0]);
	height = pt[2];
}
void cartesian_to_cylinder_nodist(const cv::Vec4f& pt, float& azimuth, float& height){
	azimuth = atan2(pt[1], pt[0]);
	height = pt[2];
}

void cylinder_to_uv(float cyl_azimuth, float cyl_height, float vox_height, float im_width, float im_height, int& U, int& V){
	U = cyl_azimuth * ((im_width - 1) / (2*CV_PI)) + (im_width - 1) / 2;
	V = cyl_height / vox_height * im_height;
}

std::vector<std::vector<unsigned int>> generate_triangle_UV(std::vector<unsigned int> triangle_indices, std::vector<float> triangle_vertices, unsigned int width, unsigned int height){

	std::vector<std::vector<unsigned int>> triangle_UV;

	unsigned int num_rows, num_cols, num_tri,
		tri_width, tri_height;

	num_tri = triangle_indices.size() / 3;
	triangle_UV.resize(num_tri);

	unsigned int upper_bound_sqrt = ceil(sqrt(num_tri));
	num_cols = upper_bound_sqrt;
	num_rows = upper_bound_sqrt;
	tri_width = floor((width) / (ceil(num_cols / 2)) - 4);
	tri_height = floor((height - 2 * num_rows + 2) / num_rows);

	for (int y = 0; y < num_rows; ++y){
		for (int x = 0; x < num_cols; x+=2){

			if (y*num_cols + x >= num_tri) break;

			unsigned int u1, v1, u2, v2, u3, v3;
			u1 = (x / 2) * (tri_width + 4);
			v1 = y * (tri_height + 2);
			u2 = u1;
			v2 = v1 + tri_height;
			u3 = u1 + tri_width;
			v3 = v2;

			triangle_UV[y*num_cols + x].push_back(u1);
			triangle_UV[y*num_cols + x].push_back(v1);
			triangle_UV[y*num_cols + x].push_back(u2);
			triangle_UV[y*num_cols + x].push_back(v2);
			triangle_UV[y*num_cols + x].push_back(u3);
			triangle_UV[y*num_cols + x].push_back(v3);


			if (y*num_cols + x + 1 >= num_tri) break;

			u1 = u1 + 2;
			v1 = v1;
			u2 = u1 + tri_width;
			v2 = v1;
			u3 = u2;
			v3 = v2 + tri_height;

			triangle_UV[y*num_cols + x+1].push_back(u1);
			triangle_UV[y*num_cols + x+1].push_back(v1);
			triangle_UV[y*num_cols + x+1].push_back(u2);
			triangle_UV[y*num_cols + x+1].push_back(v2);
			triangle_UV[y*num_cols + x+1].push_back(u3);
			triangle_UV[y*num_cols + x+1].push_back(v3);
		}
	}

	return triangle_UV;
}