#pragma once
#ifndef OBJECT_DESCRIPTION_H_
#define OBJECT_DESCRIPTION_H_


#include <opencv2/core/core.hpp>
#include "BoW.h"//需要词袋的头文件



/// Forward declarations to avoid unnecessary includes
namespace ntk {
	class Pose3D;
}

namespace pcl {
	template <typename PointType> 
	class PointCloud;
	struct PointXYZRGB;
}


/**
 * \brief Object Classfication namespace
 */
namespace ocl {

//各个描述子的大小长度
/// Size of an FPFH descriptor
const int FPFH_LEN = 33;

/// Size of a SIFT descriptor
const int SIFT_LEN = 128;

/// Size of custom HoG descriptor
const int HOG_LEN = 36;

/// Number of blocks which make up a HoG descriptor
const int NUM_HOG_BLOCKS = 105;
	

/**
 * \brief Describes an object using FPFH, SIFT, HOG, and BoW descriptors
 */
class ObjectDescription
{
private:
	cv::Mat fpfhDescriptors;  //描述子
	cv::Mat siftDescriptors;
	cv::Mat hogDescriptors;
	cv::Mat fpfhBOW;          //词袋中的单词
	cv::Mat siftBOW;
	cv::Mat hogBOW;
	std::string fpfhVocabPath;  //特征路径
	std::string siftVocabPath;
	std::string hogVocabPath;
	BoW fpfhAssigner;
	BoW siftAssigner;
	BoW hogAssigner;

	/// Sets up the respective feature type vocabularies  设置各自的特征类型词汇表
	void setupBOWAssignment();

	/// Assign the Bag of Words representation for a set of FPFH descriptors  用词袋代替FPFH描述子
	void assignFPFHBOW();

	/// Assign the Bag of Words representation for a set of SIFT descriptors
	void assignSIFTBOW();

	/// Assign the Bag of Words representation for a set of HOG descriptors
	void assignHOGBOW();

	/// Get the Fast Point Feature Histogram representation for the given object point cloud
	void extractFPFHDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);   //提取所给点云的FPFH描述子

	/// Get the Scale Invariant Feature Transform representation for the given object point cloud
	void extractSIFTDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);//提取所给点云的SIFT描述子

	/// Get the Histogram of Oriented Gradients representation for the given object point cloud
	void extractHOGDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);  //提取所给点云的HOG描述子


public:
	ObjectDescription(const std::string& pathToVocabs = ".");
	virtual ~ObjectDescription();

	/// Sets the path of the FPFH visual vocabulary   设置FPFH词袋的路径
	inline void setFPFHVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->fpfhVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}
	
	/// Sets the path of the SIFT visual vocabulary
	inline void setSIFTVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->siftVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}

	/// Sets the path of the HOG visual vocabulary
	inline void setHOGVocabPath(const std::string& vocabPath) {
		if (!vocabPath.empty()) {
			this->hogVocabPath = vocabPath;
		} else {
			//TODO add error handling here
		}		
	}

	/// Get this object's FPFH BoW representation   活得物体FPFH词袋的数据
	inline cv::Mat getFPFHBow() const {
		return fpfhBOW;
	}

	/// Get this object's SIFT BoW representation
	inline cv::Mat getSIFTBow() const {
		return siftBOW;
	}

	/// Get this object's HOG BoW representation
	inline cv::Mat getHOGBow() const {
		return hogBOW;
	}

	/// Get a vector of all BoWs   获得三个特征的词袋
	inline std::vector<cv::Mat> getAllBows() {
		std::vector<cv::Mat> allBows;

		allBows.push_back(getFPFHBow());
		allBows.push_back(getSIFTBow());
		allBows.push_back(getHOGBow());

		return allBows;
	}

	/// Get the multi-model feature descriptions for this object  活得点云多个特征模型
	void extractFeatureDescriptors(const pcl::PointCloud<pcl::PointXYZRGB>& cloud);

	/// Assigns the BoWs representations to describe this object  用词袋来代替这个物体
	void assignBOWs();


	


};

} /* ocl */

#endif /* OBJECT_DESCRIPTION_H_ */

