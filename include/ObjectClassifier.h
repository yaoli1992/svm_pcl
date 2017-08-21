#pragma once
#ifndef OBJECT_CLASSIFIER_H_
#define OBJECT_CLASSIFIER_H_

#include <cstdlib>
#include <string>
#include "SVMUtils.hpp"

// Forward declaration
namespace cv {
	class Mat;
}

namespace ocl {


/// Stores all information relevant to an object category
struct ObjectCategory {
	double categoryLabel;           //分类物体的标签申明
	std::string categoryName;       //分类物体的名称
	std::vector<double> classConfidence;  //分类物体的可信度

	ObjectCategory() : categoryLabel(-1.0), categoryName("Unknown") {}   //物体分类函数的初始化
};


class ObjectClassifier
{
private:
	std::string fpfhModelPath;   //fpfh模型的路径
	std::string siftModelPath;   //sift模型的路径
	std::string hogModelPath;    //hog模型的路径
	std::string combinedModelPath;  //结合所有特征的模型路径
	SVMModelPtr fpfhModel;       //SVM的模型
	SVMModelPtr siftModel;
	SVMModelPtr hogModel;
	SVMModelPtr combModel;

	/// Converts a BoW vector into a format for use by LibSVM   把词袋向量转为SVM数据库格式
	void convertBowIntoSVMData(const cv::Mat& bowVector, SVMData& svmData);

public:
	/// Constructor
	ObjectClassifier(const std::string& modelsDirPath);

	/// Destructor
	virtual ~ObjectClassifier();

	/**
	 * \brief Sets the file path for FPFH SVM classifier model.
	 * This function assumes that FPFH.model is present in the working directory
	 */
	void setFPFHModelPath(const std::string& modelPath = "./FPFH.model") {
		if (!modelPath.empty())
			fpfhModelPath = modelPath;
		else
			fpfhModelPath = "./FPFH.model";
	}

	/**
	 * \brief Sets the file path for SIFT SVM classifier model.
	 * This function assumes that SIFT.model is present in the working directory
	 */
	void setSIFTModelPath(const std::string& modelPath = "./SIFT.model") {
		if (!modelPath.empty())
			siftModelPath = modelPath;
		else
			siftModelPath = "./SIFT.model";
	}

	/**
	 * \brief Sets the file path for HOG SVM classifier model.
	 * This function assumes that HOG.model is present in the working directory
	 */
	void setHOGModelPath(const std::string& modelPath = "./HOG.model") {
		if (!modelPath.empty())
			hogModelPath = modelPath;
		else
			hogModelPath = "./HOG.model";
	}

	/**
	 * \brief Sets the file path for Combined Ensemble SVM classifier model.
	 * This function assumes that Combined.model is present in the working directory
	 */
	void setCombinedModelPath(const std::string& modelPath = "./Combined.model") {
		if (!modelPath.empty())
			combinedModelPath = modelPath;
		else
			combinedModelPath = "./Combined.model";
	}
         //用特征来表示是什么物体
	/// Determines the category to which an object belongs using its feature representation
	void classify(const cv::Mat& fpfhBow, const cv::Mat& siftBow, const cv::Mat& hogBow);

	/// Measures the confidence of the classification result using the Renyi entropy  用熵来估计分类的可信度
	double measureConfidence() const;


public:
	ObjectCategory objectCategory;

};

} // ocl

#endif /* OBJECT_CLASSIFIER_H_ */

