#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT1344 DescriptorType;

std::string scene_filename_;

//Algorithm params
bool show_keypoints_(false);
bool show_correspondences_(false);
bool use_cloud_resolution_(false);
bool use_hough_(true);
float model_ss_(0.01f);
float scene_ss_(0.01f);
float rf_rad_(0.015f);
float descr_rad_(0.02f);
float cg_size_(0.01f);
float cg_thresh_(5.0f);
int a = 0;
int i = 0;
float cont_ban, cont_man, cont_nar, cont_per = 0;
int cont_banins, cont_manins, cont_narins, cont_perins = 0;

void
showHelp(char* filename)
{
	std::cout << std::endl;
	std::cout << "***************************************************************************" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "***************************************************************************" << std::endl << std::endl;
	std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "     -h:                     Show this help." << std::endl;
	std::cout << "     -k:                     Show used keypoints." << std::endl;
	std::cout << "     -c:                     Show used correspondences." << std::endl;
	std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
	std::cout << "                             each radius given by that value." << std::endl;
	std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
	std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
	std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
	std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
	std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
	std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
	std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine(int argc, char* argv[])
{
	//Show help
	if (pcl::console::find_switch(argc, argv, "-h"))
	{
		showHelp(argv[0]);
		exit(0);
	}

	//Model & scene filenames
	std::vector<int> filenames;
	filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
	if (filenames.size() != 1)
	{
		std::cout << "Filenames missing.\n";
		showHelp(argv[0]);
		exit(-1);
	}

	scene_filename_ = argv[filenames[0]];

	//Program behavior
	if (pcl::console::find_switch(argc, argv, "-k"))
	{
		show_keypoints_ = true;
	}
	if (pcl::console::find_switch(argc, argv, "-c"))
	{
		show_correspondences_ = true;
	}
	if (pcl::console::find_switch(argc, argv, "-r"))
	{
		use_cloud_resolution_ = true;
	}

	std::string used_algorithm;
	if (pcl::console::parse_argument(argc, argv, "--algorithm", used_algorithm) != -1)
	{
		if (used_algorithm.compare("Hough") == 0)
		{
			use_hough_ = true;
		}
		else if (used_algorithm.compare("GC") == 0)
		{
			use_hough_ = false;
		}
		else
		{
			std::cout << "Wrong algorithm name.\n";
			showHelp(argv[0]);
			exit(-1);
		}
	}

	//General parameters
	pcl::console::parse_argument(argc, argv, "--model_ss", model_ss_);
	pcl::console::parse_argument(argc, argv, "--scene_ss", scene_ss_);
	pcl::console::parse_argument(argc, argv, "--rf_rad", rf_rad_);
	pcl::console::parse_argument(argc, argv, "--descr_rad", descr_rad_);
	pcl::console::parse_argument(argc, argv, "--cg_size", cg_size_);
	pcl::console::parse_argument(argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr& cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

int main(int argc, char* argv[])
{
	parseCommandLine(argc, argv);

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType>());
	pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>());
	pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>());
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>());
	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZRGBA>);


	if (pcl::io::loadPCDFile(scene_filename_, *cloud) < 0)
	{
		std::cout << "Error loading scene cloud." << std::endl;
		showHelp(argv[0]);
		return (-1);
	}


	// Create the filtering object z
	pcl::PassThrough<pcl::PointXYZRGBA> pass_z;
	pass_z.setInputCloud(cloud);
	pass_z.setFilterFieldName("z");
	pass_z.setFilterLimits(0.0, 0.58);
	pass_z.filter(*cloud_filtered_z);




	pcl::PassThrough<pcl::PointXYZRGBA> pass_x;
	pass_x.setInputCloud(cloud_filtered_z);
	pass_x.setFilterFieldName("y");
	pass_x.setFilterLimits(0, 0.16);
	pass_x.filter(*cloud_filtered_x);




	pcl::PCDWriter writer;
	writer.write("scene.pcd", *cloud_filtered_x);

	pcl::io::loadPCDFile("scene.pcd", *scene);

	if (use_cloud_resolution_)
	{
		float resolution = static_cast<float> (computeCloudResolution(scene));
		if (resolution != 0.0f)
		{
			model_ss_ *= resolution;
			scene_ss_ *= resolution;
			rf_rad_ *= resolution;
			descr_rad_ *= resolution;
			cg_size_ *= resolution;
		}

		std::cout << "Model resolution:       " << resolution << std::endl;
		std::cout << "Model sampling size:    " << model_ss_ << std::endl;
		std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
		std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
		std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
		std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
	}

	//
	//  Compute Normals
	//
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch(10);
	norm_est.setInputCloud(scene);
	norm_est.compute(*scene_normals);

	//
	//  Downsample Clouds to Extract keypoints
	//

	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(scene);
	uniform_sampling.setRadiusSearch(scene_ss_);
	uniform_sampling.filter(*scene_keypoints);


	pcl::SHOTColorEstimation<PointType, NormalType, pcl::SHOT1344> descr_est;
	descr_est.setSearchMethod(pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>));


	descr_est.setInputCloud(scene_keypoints);
	descr_est.setInputNormals(scene_normals);
	descr_est.setRadiusSearch(descr_rad_);
	descr_est.setSearchSurface(scene);
	descr_est.compute(*scene_descriptors);

	std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;


	std::cout << "Scene total points: " << scene->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;


	std::cout << "Se termino calculos de scene" << std::endl;

	std::string dir_des_ban;
	std::string dir_key_ban;
	std::string dir_nor_ban;





	while (i < 36) {


		if (a == 0) {
			dir_des_ban = "basedatos/descriptores_banana/";
			dir_key_ban = "basedatos/keypoints_banana/";
			dir_nor_ban = "basedatos/normales_banana/";
			pcl::io::loadPCDFile("nubes_puntos/filteredbanana/" + std::to_string(i) + ".pcd", *model);
			std::cout << "Nube de puntos de banana: " << i << std::endl;
		}

		if (a == 1) {
			dir_des_ban = "basedatos/descriptores_manzana/";
			dir_key_ban = "basedatos/keypoints_manzana/";
			dir_nor_ban = "basedatos/normales_manzana/";
			pcl::io::loadPCDFile("nubes_puntos/filteredmanzana/" + std::to_string(i) + ".pcd", *model);
			std::cout << "Nube de puntos de manzana: " << i << std::endl;
		}

		if (a == 2) {
			dir_des_ban = "basedatos/descriptores_naranja/";
			dir_key_ban = "basedatos/keypoints_naranja/";
			dir_nor_ban = "basedatos/normales_naranja/";
			pcl::io::loadPCDFile("nubes_puntos/filterednaranja/" + std::to_string(i) + ".pcd", *model);
			std::cout << "Nube de puntos de naranja: " << i << std::endl;
		}

		if (a == 3) {
			dir_des_ban = "basedatos/descriptores_pera/";
			dir_key_ban = "basedatos/keypoints_pera/";
			dir_nor_ban = "basedatos/normales_pera/";
			pcl::io::loadPCDFile("nubes_puntos/filteredpera/" + std::to_string(i) + ".pcd", *model);
			std::cout << "Nube de puntos de pera: " << i << std::endl;
		}

		pcl::io::loadPCDFile(dir_nor_ban + std::to_string(i) + ".pcd", *model_normals);
		pcl::io::loadPCDFile(dir_des_ban + std::to_string(i) + ".pcd", *model_descriptors);
		pcl::io::loadPCDFile(dir_key_ban + std::to_string(i) + ".pcd", *model_keypoints);

		std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;




		//
		//  Find Model-Scene Correspondences with KdTree
		//

		pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

		pcl::KdTreeFLANN<DescriptorType> match_search;
		match_search.setInputCloud(model_descriptors);

		//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
		for (std::size_t i = 0; i < scene_descriptors->size(); ++i)
		{
			std::vector<int> neigh_indices(1);
			std::vector<float> neigh_sqr_dists(1);
			if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) //skipping NaNs
			{
				continue;
			}
			int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
			if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
			{
				pcl::Correspondence corr(neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
				model_scene_corrs->push_back(corr);
			}
		}
		std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;


		//
		//  Actual Clustering
		//
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
		std::vector<pcl::Correspondences> clustered_corrs;


		//  Using Hough3D
		if (use_hough_)
		{
			//
			//  Compute (Keypoints) Reference Frames only for Hough
			//
			pcl::PointCloud<RFType>::Ptr model_rf(new pcl::PointCloud<RFType>());
			pcl::PointCloud<RFType>::Ptr scene_rf(new pcl::PointCloud<RFType>());

			pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
			rf_est.setFindHoles(true);
			rf_est.setRadiusSearch(rf_rad_);

			rf_est.setInputCloud(model_keypoints);
			rf_est.setInputNormals(model_normals);
			rf_est.setSearchSurface(model);
			rf_est.compute(*model_rf);

			rf_est.setInputCloud(scene_keypoints);
			rf_est.setInputNormals(scene_normals);
			rf_est.setSearchSurface(scene);
			rf_est.compute(*scene_rf);

			//  Clustering
			pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
			clusterer.setHoughBinSize(cg_size_);
			clusterer.setHoughThreshold(cg_thresh_);
			clusterer.setUseInterpolation(true);
			clusterer.setUseDistanceWeight(false);

			clusterer.setInputCloud(model_keypoints);
			clusterer.setInputRf(model_rf);
			clusterer.setSceneCloud(scene_keypoints);
			clusterer.setSceneRf(scene_rf);
			clusterer.setModelSceneCorrespondences(model_scene_corrs);

			//clusterer.cluster (clustered_corrs);
			clusterer.recognize(rototranslations, clustered_corrs);
		}
		else // Using GeometricConsistency
		{
			pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
			gc_clusterer.setGCSize(cg_size_);
			gc_clusterer.setGCThreshold(cg_thresh_);

			gc_clusterer.setInputCloud(model_keypoints);
			gc_clusterer.setSceneCloud(scene_keypoints);
			gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

			//gc_clusterer.cluster (clustered_corrs);
			gc_clusterer.recognize(rototranslations, clustered_corrs);
		}

		//
		//  Output results
		//

		std::cout << "Model instances found: " << rototranslations.size() << std::endl << std::endl;
		for (std::size_t i = 0; i < rototranslations.size(); ++i)
		{
			std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
			std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

			if (a == 0) {
				cont_ban += clustered_corrs[i].size();
				cont_banins++;
			}
			if (a == 1) {
				cont_man += clustered_corrs[i].size();
				cont_manins++;
			}
			if (a == 2) {
				cont_nar += clustered_corrs[i].size();
				cont_narins++;
			}
			if (a == 3) {
				cont_per += clustered_corrs[i].size();
				cont_perins++;
			}

		}



		if (a == 3) {
			a = 0;
			i++;
		}
		else {
			a++;
		}

	} //End for bananna
	std::cout << "El valor de bananas final es: " << cont_ban << std::endl;
	std::cout << "El valor de manzanas final es: " << cont_man << std::endl;
	std::cout << "El valor de naranjas final es: " << cont_nar << std::endl;
	std::cout << "El valor de peras final es: " << cont_per << std::endl << std::endl;

	std::cout << "La cantidad de instancias de bananos son " << cont_banins << std::endl;
	std::cout << "La cantidad de instancias de manzanas son " << cont_manins << std::endl;
	std::cout << "La cantidad de instancias de naranjas son " << cont_narins << std::endl;
	std::cout << "La cantidad de instancias de peras son " << cont_perins << std::endl;


}