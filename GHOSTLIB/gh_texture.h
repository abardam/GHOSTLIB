#include <opencv2\opencv.hpp>

void cartesian_to_cylinder(const cv::Vec3f& pt, float& distance, float& azimuth, float& height);
void cartesian_to_cylinder(const cv::Vec4f& pt, float& distance, float& azimuth, float& height);
void cartesian_to_cylinder_nodist(const cv::Vec4f& pt, float& azimuth, float& height);
void cylinder_to_uv(float cyl_azimuth, float cyl_height, float vox_height, float im_width, float im_height, int& U, int& V);

std::vector<std::vector<unsigned int>> generate_triangle_UV(std::vector<unsigned int> triangle_indices, std::vector<float> triangle_vertices, unsigned int width, unsigned int height);