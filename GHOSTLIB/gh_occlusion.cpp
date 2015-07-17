#include "gh_occlusion.h"
#include <cv_draw_common.h>
#include "gh_texture.h"
#include <cv_pointmat_common.h>

void process_and_save_occlusions(const cv::Mat& render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const std::string& video_directory){

	unsigned int win_width = render_pretexture.cols;
	unsigned int win_height = render_pretexture.rows;

	std::vector<cv::Mat> bodypart_pts(bpdv.size());
	std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
	std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());

	std::vector<std::vector<cv::Point2i>> nonbodypart_pts_2d_v(bpdv.size());
	std::vector<cv::Point2i> bg_pts_2d_v(bpdv.size());

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
			else{
				bg_pts_2d_v.push_back(cv::Point2i(x, y));
			}
		}
	}

	//now apply to the RGB images

	std::stringstream filename_ss;

	std::vector<cv::Vec3b> crop_colors;
	crop_colors.push_back(cv::Vec3b(0xff, 0xff, 0xff));
	//crop_colors.push_back(cv::Vec3b(0xff, 0, 0));

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
		cv::Size(3,3),
		cv::Point(1, 1));


	for (int i = 0; i < bpdv.size(); ++i){
		cv::Mat bodypart_image(win_height, win_width, CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));

		cv::Mat white_mask(win_height, win_width, CV_8U, cv::Scalar(0xff));

		for (int j = 0; j < nonbodypart_pts_2d_v[i].size(); ++j){
			int x = nonbodypart_pts_2d_v[i][j].x;
			int y = nonbodypart_pts_2d_v[i][j].y;

			bodypart_image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0xff, 0, 0);
			white_mask.ptr<unsigned char>(y)[x] = 0;
		}
		for (int j = 0; j < bodypart_pts_2d_v[i].size(); ++j){
			int x = bodypart_pts_2d_v[i][j].x;
			int y = bodypart_pts_2d_v[i][j].y;

			bodypart_image.ptr<cv::Vec3b>(y)[x] = frame_color.ptr<cv::Vec3b>(y)[x];
			white_mask.ptr<unsigned char>(y)[x] = 0;
		}

		cv::Mat whiter_mask;
		cv::dilate(white_mask, whiter_mask, element);
		for (int j = 0; j < bodypart_image.rows * bodypart_image.cols; ++j){
			if (whiter_mask.ptr<unsigned char>()[j] == 0xff){
				if (bodypart_image.ptr<cv::Vec3b>()[j] != cv::Vec3b(0xff, 0, 0)){
					bodypart_image.ptr<cv::Vec3b>()[j] = cv::Vec3f(0xff, 0xff, 0xff);
				}
			}
		}

		filename_ss.str("");
		filename_ss << video_directory << "\\bodypart" << i << "frame" << anim_frame << ".xml.gz";

		cv::FileStorage fs;
		fs.open(filename_ss.str(), cv::FileStorage::WRITE);

		fs << "bodypart" << i << "frame" << anim_frame << "cropped_mat" << crop_mat(bodypart_image, crop_colors);
		fs << "facing" << frame_facing;

		fs.release();
	}

	cv::Mat bg_image(win_height, win_width, CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));

	if (!frame_fullcolor.empty()){
		for (int i = 0; i < bg_pts_2d_v.size(); ++i){
			int x = bg_pts_2d_v[i].x;
			int y = bg_pts_2d_v[i].y;

			bg_image.ptr<cv::Vec3b>(y)[x] = frame_fullcolor.ptr<cv::Vec3b>(y)[x];

		}
	}


	filename_ss.str("");
	filename_ss << video_directory << "\\background_" << "frame" << anim_frame << ".xml.gz";

	cv::FileStorage fs;
	fs.open(filename_ss.str(), cv::FileStorage::WRITE);

	fs << "frame" << anim_frame << "mat" << bg_image;
	fs << "facing" << frame_facing;

	fs.release();
}



void process_and_save_occlusions_expanded(const cv::Mat& render_pretexture,
	const std::vector<cv::Mat>& bodypart_render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const std::string& video_directory,
	const std::vector<bool>& validity){

	if (frame_color.empty()){
		std::cout << "frame " << anim_frame << " empty" << std::endl;
		return;
	}

	unsigned int win_width = render_pretexture.cols;
	unsigned int win_height = render_pretexture.rows;

	std::vector<cv::Mat> bodypart_pts(bpdv.size());
	std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
	std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());

	std::vector<std::vector<cv::Point2i>> nonbodypart_pts_2d_v(bpdv.size());
	std::vector<cv::Point2i> bg_pts_2d_v(bpdv.size());

	for (int y = 0; y < win_height; ++y){
		for (int x = 0; x < win_width; ++x){

			unsigned int bodypart_id;
			bool is_bodypart = false;

			for (int i = 0; i < bpdv.size(); ++i){
				cv::Vec3b orig_color = render_pretexture.ptr<cv::Vec3b>(y)[x];
				cv::Vec3b bp_orig_color = bodypart_render_pretexture[i].ptr<cv::Vec3b>(y)[x];
				cv::Vec3b bp_color(bpdv[i].mColor[2] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[0] * 0xff);

				if ((bp_orig_color == bp_color
					&& orig_color != bg_color) 
					|| orig_color == bp_color
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
			else{
				bg_pts_2d_v.push_back(cv::Point2i(x, y));
			}
		}
	}

	//now apply to the RGB images

	std::stringstream filename_ss;

	std::vector<cv::Vec3b> crop_colors;
	crop_colors.push_back(cv::Vec3b(0xff, 0xff, 0xff));
	//crop_colors.push_back(cv::Vec3b(0xff, 0, 0));

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
		cv::Size(5, 5),
		cv::Point(2, 2));

	for (int i = 0; i < bpdv.size(); ++i){
		cv::Mat bodypart_image(win_height, win_width, CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));
		cv::Mat white_mask(win_height, win_width, CV_8U, cv::Scalar(0xff));

		for (int j = 0; j < nonbodypart_pts_2d_v[i].size(); ++j){
			int x = nonbodypart_pts_2d_v[i][j].x;
			int y = nonbodypart_pts_2d_v[i][j].y;

			bodypart_image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0xff, 0, 0);
			white_mask.ptr<unsigned char>(y)[x] = 0;
		}
		for (int j = 0; j < bodypart_pts_2d_v[i].size(); ++j){
			int x = bodypart_pts_2d_v[i][j].x;
			int y = bodypart_pts_2d_v[i][j].y;


			if (validity.empty() || validity[i]){
				bodypart_image.ptr<cv::Vec3b>(y)[x] = frame_color.ptr<cv::Vec3b>(y)[x];
			}
			else{
				bodypart_image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0xff, 0, 0);
			}
			white_mask.ptr<unsigned char>(y)[x] = 0;
		}


		cv::Mat whiter_mask;
		cv::dilate(white_mask, whiter_mask, element);
		for (int y = 0; y < bodypart_image.rows; ++y){
			for (int x = 0; x < bodypart_image.cols; ++x){
				if (whiter_mask.ptr<unsigned char>(y)[x] == 0xff){
					if (bodypart_image.ptr<cv::Vec3b>(y)[x](0) != 0xff){
						//bodypart_image.ptr<cv::Vec3b>()[j] = cv::Vec3f(0xff, 0xff, 0xff);
						//lets take the average, not including blue and white
						int r = 0, b = 0, g = 0;
						int ncol = 0;
						for (int _y = -2; _y <= 2; ++_y){
							for (int _x = -2; _x <= 2; ++_x){
								if (CLAMP(x+_x, y+_y, bodypart_image.cols, bodypart_image.rows)){
									const cv::Vec3b& color = bodypart_image.ptr<cv::Vec3b>(y + _y)[x + _x];

									if (color(0) != 0xff){
										r += color(2);
										g += color(1);
										b += color(0);
										++ncol;
									}
								}
							}
						}

						bodypart_image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(b / ncol, g / ncol, r / ncol);
					}
				}
			}
		}

		filename_ss.str("");
		filename_ss << video_directory << "\\bodypart" << i << "frame" << anim_frame << ".xml.gz";

		cv::FileStorage fs;
		fs.open(filename_ss.str(), cv::FileStorage::WRITE);

		fs << "bodypart" << i << "frame" << anim_frame << "cropped_mat" << crop_mat(bodypart_image, crop_colors);
		fs << "facing" << frame_facing;

		fs.release();
	}

	cv::Mat bg_image(win_height, win_width, CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));

	if (!frame_fullcolor.empty()){
		for (int i = 0; i < bg_pts_2d_v.size(); ++i){
			int x = bg_pts_2d_v[i].x;
			int y = bg_pts_2d_v[i].y;

			bg_image.ptr<cv::Vec3b>(y)[x] = frame_fullcolor.ptr<cv::Vec3b>(y)[x];

		}
	}


	filename_ss.str("");
	filename_ss << video_directory << "\\background_" << "frame" << anim_frame << ".xml.gz";

	cv::FileStorage fs;
	fs.open(filename_ss.str(), cv::FileStorage::WRITE);

	fs << "frame" << anim_frame << "mat" << bg_image;
	fs << "facing" << frame_facing;

	fs.release();
}


void process_occlusions_texturemap_cylinder(const cv::Mat& render_pretexture,
	const std::vector<cv::Mat>& bodypart_render_pretexture,
	const cv::Mat& render_depth, int anim_frame,
	const BodyPartDefinitionVector& bpdv, 
	const SkeletonNodeHardMap& snhmap,
	const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const cv::Mat& camera_matrix,
	const std::vector<VoxelMatrix>& bodypart_voxels,
	float voxel_size,
	std::vector<cv::Mat>& bodypart_textures,
	std::vector<cv::Mat>& bodypart_textureweights,
	const std::vector<bool>& validity){

	cv::Mat camera_matrix_inv = camera_matrix.inv();

	if (frame_color.empty()){
		std::cout << "frame " << anim_frame << " empty" << std::endl;
		return;
	}

	unsigned int win_width = render_pretexture.cols;
	unsigned int win_height = render_pretexture.rows;

	std::vector<cv::Mat> bodypart_pts(bpdv.size());
	std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
	std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());
	std::vector<std::vector<cv::Vec3f>> bodypart_pts_norm(bpdv.size());

	std::vector<std::vector<cv::Point2i>> nonbodypart_pts_2d_v(bpdv.size());
	std::vector<cv::Point2i> bg_pts_2d_v(bpdv.size());

	for (int y = 0; y < win_height-1; ++y){
		for (int x = 0; x < win_width-1; ++x){

			unsigned int bodypart_id;
			bool is_bodypart = false;

			for (int i = 0; i < bpdv.size(); ++i){
				cv::Vec3b orig_color = render_pretexture.ptr<cv::Vec3b>(y)[x];
				cv::Vec3b bp_orig_color = bodypart_render_pretexture[i].ptr<cv::Vec3b>(y)[x];
				cv::Vec3b bp_color(bpdv[i].mColor[2] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[0] * 0xff);

				if ((bp_orig_color == bp_color
					&& orig_color != bg_color)
					|| orig_color == bp_color
					){
					float depth = render_depth.ptr<float>(y)[x];
					bodypart_pts_2d_withdepth_v[i].push_back(cv::Vec4f(depth*x, depth*y,
						depth, 1));
					bodypart_pts_2d_v[i].push_back(cv::Point2i(x, y));

					//calculate normal
					{
						float pts_implane[] = {
							depth*x, render_depth.ptr<float>(y)[x + 1] * (x + 1), render_depth.ptr<float>(y + 1)[x] * (x),
							depth*y, render_depth.ptr<float>(y)[x + 1] * (y), render_depth.ptr<float>(y + 1)[x] * (y + 1),
							depth, render_depth.ptr<float>(y)[x + 1], render_depth.ptr<float>(y + 1)[x],
							1, 1, 1

						};

						cv::Mat pts_m_implane(4, 3, CV_32F, pts_implane);
						cv::Mat pts_m = camera_matrix_inv * pts_m_implane;

						cv::Vec3f pt = pts_m(cv::Range(0,3),cv::Range(0,1));
						cv::Vec3f pt1x = pts_m(cv::Range(0,3),cv::Range(1,2));
						cv::Vec3f pt1y = pts_m(cv::Range(0,3),cv::Range(2,3));

						cv::Vec3f norm = cv::normalize((pt1x - pt).cross(pt1y - pt));

						bodypart_pts_norm[i].push_back(norm);
					}

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
			else{
				bg_pts_2d_v.push_back(cv::Point2i(x, y));
			}
		}
	}

	//now apply to the RGB images

	for (int i = 0; i < bpdv.size(); ++i){

		for (int j = 0; j < nonbodypart_pts_2d_v[i].size(); ++j){
			int x = nonbodypart_pts_2d_v[i][j].x;
			int y = nonbodypart_pts_2d_v[i][j].y;

			//occluded, ignore
		}
		for (int j = 0; j < bodypart_pts_2d_v[i].size(); ++j){
			int x = bodypart_pts_2d_v[i][j].x;
			int y = bodypart_pts_2d_v[i][j].y;


			if (validity.empty() || validity[i]){
				//add to bodypart textures
				//first calculate UV coordinates

				cv::Mat voxel_coord_m = get_voxel_transform(bodypart_voxels[i].width, bodypart_voxels[i].height, bodypart_voxels[i].depth, voxel_size).inv() * get_bodypart_transform(bpdv[i], snhmap, cv::Mat::eye(4, 4, CV_32F)).inv() * camera_matrix_inv * cv::Mat(bodypart_pts_2d_withdepth_v[i][j]);
				cv::Vec4f voxel_coord = voxel_coord_m;

				float distance, azimuth, height;
				cartesian_to_cylinder_nodist(voxel_coord, azimuth, height);

				//map azimuth and height to U and V of bodypart texture
				
				int U, V;
				cylinder_to_uv(azimuth, height, bodypart_voxels[i].height, bodypart_textures[i].cols, bodypart_textures[i].rows, U, V);

				cv::Mat ptloc_m = camera_matrix_inv * cv::Mat(bodypart_pts_2d_withdepth_v[i][j]);
				cv::Vec3f ptloc(ptloc_m.ptr<float>(0)[0],
					ptloc_m.ptr<float>(1)[0],
					ptloc_m.ptr<float>(2)[0]);

				float addweight = bodypart_pts_norm[i][j].dot(cv::normalize(-ptloc));
				addweight = addweight * addweight * addweight * addweight;

				if (addweight > 0 && CLAMP(U, V, bodypart_textures[i].cols, bodypart_textures[i].rows)){
					cv::Vec3b color = bodypart_textures[i].ptr<cv::Vec3b>(V)[U];
					cv::Vec3b add_color = frame_color.ptr<cv::Vec3b>(y)[x];
					if (add_color != cv::Vec3b(0xff, 0xff, 0xff)){
						float wt = bodypart_textureweights[i].ptr<float>(V)[U];
						int b = (color[0] * wt + add_color[0] * addweight) / (wt + addweight);
						int g = (color[1] * wt + add_color[1] * addweight) / (wt + addweight);
						int r = (color[2] * wt + add_color[2] * addweight) / (wt + addweight);
						bodypart_textures[i].ptr<cv::Vec3b>(V)[U] = cv::Vec3b(b,g,r);
						bodypart_textureweights[i].ptr<float>(V)[U] += addweight;
					}
				}
			}
			else{
				//invalid, ignore
			}
		}


	}

}

#define TRIANGLE_TEXTURE_MAX_DEPTH_DIFF 0.005

void process_occlusions_texturemap_triangles(
	std::vector<std::vector<unsigned int>> bodypart_triangle_indices,
	std::vector<std::vector<float>> bodypart_triangle_vertices,
	std::vector<std::vector<std::vector<unsigned int>>> bodypart_triangle_UV,
	const cv::Mat& render_depth, 
	int anim_frame,
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeHardMap& snhmap,
	const cv::Vec3b& bg_color,
	const cv::Mat& frame_color, const cv::Mat& frame_fullcolor,
	const int& frame_facing,
	const cv::Mat& camera_matrix,
	const std::vector<VoxelMatrix>& bodypart_voxels,
	float voxel_size,
	std::vector<cv::Mat>& bodypart_textures,
	std::vector<cv::Mat>& bodypart_textureweights,
	const std::vector<bool>& validity){

	cv::Mat camera_matrix_inv = camera_matrix.inv();

	if (frame_color.empty()){
		std::cout << "frame " << anim_frame << " empty" << std::endl;
		return;
	}

	unsigned int win_width = render_depth.cols;
	unsigned int win_height = render_depth.rows;

	for (int i = 0; i < bodypart_triangle_vertices.size(); ++i){
		//step 1: loop thru the triangles of each body part to determine which are on the surface of the rendered shape
		unsigned int num_vertices = bodypart_triangle_vertices[i].size() / 3;
		std::vector<cv::Point2f> vertex_loc(num_vertices);
		std::vector<cv::Vec3f> vertex_pt(num_vertices);
		std::vector<bool> vertex_visible(num_vertices);

		cv::Mat vertices_local_t(num_vertices, 3, CV_32F, bodypart_triangle_vertices[i].data());
		cv::Mat vertices_local_3 = vertices_local_t.t();
		cv::Mat vertices_local = cv::Mat::ones(4, num_vertices, CV_32F);
		vertices_local_3.copyTo(vertices_local(cv::Range(0, 3), cv::Range(0, num_vertices)));
		cv::Mat vertices_transformed = get_bodypart_transform(bpdv[i], snhmap, cv::Mat::eye(4, 4, CV_32F)) * get_voxel_transform(bodypart_voxels[i].width, bodypart_voxels[i].height, bodypart_voxels[i].depth, voxel_size)
			* vertices_local;
		cv::Mat vertices_projected = camera_matrix * vertices_transformed;
		divide_pointmat_by_z(vertices_projected);

		for (int j = 0; j < num_vertices; ++j){
			int x = vertices_projected.ptr<float>(0)[j];
			int y = vertices_projected.ptr<float>(1)[j];

			if (CLAMP(x, y, win_width, win_height)){
				float depth = render_depth.ptr<float>(y)[x];
				if (abs(depth - vertices_transformed.ptr<float>(2)[j]) < TRIANGLE_TEXTURE_MAX_DEPTH_DIFF){
					vertex_visible[j] = true;
					vertex_loc[j] = cv::Point2f(x, y);

					float wx = vertices_transformed.ptr<float>(0)[j];
					float wy = vertices_transformed.ptr<float>(1)[j];
					float wz = vertices_transformed.ptr<float>(2)[j];

					vertex_pt[j] = cv::Vec3f(wx, wy, wz);
				}
				else{
					vertex_visible[j] = false;
				}
			}
			else{
				vertex_visible[j] = false;
			}
		}

		//step 2: check validity of each triangle based on the validity of each of the vertices in it
		
		unsigned int num_triangles = bodypart_triangle_indices[i].size() / 3;
		for (int j = 0; j < num_triangles; ++j){
			unsigned int vert1, vert2, vert3;
			vert1 = bodypart_triangle_indices[i][j * 3];
			vert2 = bodypart_triangle_indices[i][j * 3 + 1];
			vert3 = bodypart_triangle_indices[i][j * 3 + 2];

			if (vertex_visible[vert1] &&
				vertex_visible[vert2] &&
				vertex_visible[vert3]){

				//calculate affine transform between img and texture
				cv::Point2f src[3];
				cv::Point2f dst[3];

				src[0] = vertex_loc[vert1];
				src[1] = vertex_loc[vert2];
				src[2] = vertex_loc[vert3];

				dst[0] = cv::Point2f(bodypart_triangle_UV[i][j][0], bodypart_triangle_UV[i][j][1]);
				dst[1] = cv::Point2f(bodypart_triangle_UV[i][j][2], bodypart_triangle_UV[i][j][3]);
				dst[2] = cv::Point2f(bodypart_triangle_UV[i][j][4], bodypart_triangle_UV[i][j][5]);

				cv::Mat affine_transform = cv::getAffineTransform(src, dst);

				cv::Mat warped_img(bodypart_textures[i].size(), CV_8UC3, cv::Scalar(0xff, 0xff, 0xff));
				cv::warpAffine(frame_color, warped_img, affine_transform, warped_img.size());

				cv::Mat mask_img(warped_img.size(), CV_8U, cv::Scalar(0));

				std::vector<cv::Point2i> pts;
				pts.push_back(dst[0]);
				pts.push_back(dst[1]);
				pts.push_back(dst[2]);

				cv::fillConvexPoly(mask_img, pts, cv::Scalar(0xff));

				//copy over to the texture, based on weight
				cv::Vec3f tri_norm = cv::normalize(vertex_pt[vert2] - vertex_pt[vert1]).cross(vertex_pt[vert3] - vertex_pt[vert1]);
				cv::Vec3f tri_center = cv::normalize((vertex_pt[vert1] + vertex_pt[vert2] + vertex_pt[vert3]) / 3);
				float add_weight = (tri_norm).dot(tri_center);

				if (add_weight > 0){
					for (int p = 0; p < bodypart_textures[i].cols * bodypart_textures[i].rows; ++p){
						if (mask_img.ptr<unsigned char>()[p] == 0xff){
							cv::Vec3b color = bodypart_textures[i].ptr<cv::Vec3b>()[p];
							cv::Vec3b add_color = warped_img.ptr<cv::Vec3b>()[p];
							if (add_color != cv::Vec3b(0xff, 0xff, 0xff)){
								float wt = bodypart_textureweights[i].ptr<float>()[p];
								int b = (color[0] * wt + add_color[0] * add_weight) / (wt + add_weight);
								int g = (color[1] * wt + add_color[1] * add_weight) / (wt + add_weight);
								int r = (color[2] * wt + add_color[2] * add_weight) / (wt + add_weight);
								bodypart_textures[i].ptr<cv::Vec3b>()[p] = cv::Vec3b(b, g, r);
								bodypart_textureweights[i].ptr<float>()[p] += add_weight;
							}
						}
					}
				}


			}
		}


	}


}