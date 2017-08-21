#include <opencv2/core/core.hpp>
#include "ObjectClassifier.h"

#define myLog2(x) std::log(x) / std::log(2.0)
//分类函数
namespace ocl {


#ifndef OCL_DEBUG
#define OCL_DEBUG 1
#endif


ObjectClassifier::ObjectClassifier(const std::string& modelsDirPath) {
	setFPFHModelPath(modelsDirPath + "/FPFH.model");
	setSIFTModelPath(modelsDirPath + "/SIFT.model");
	setHOGModelPath(modelsDirPath + "/HOG.model");
	setCombinedModelPath(modelsDirPath + "/Combined.model");

	fpfhModel = SVMUtils::loadModel(fpfhModelPath.c_str());
	siftModel = SVMUtils::loadModel(siftModelPath.c_str());
	hogModel = SVMUtils::loadModel(hogModelPath.c_str());
	combModel = SVMUtils::loadModel(combinedModelPath.c_str());

	if (fpfhModel == NULL || siftModel == NULL || hogModel == NULL || combModel == NULL) {
		fprintf(stderr, "Error: SVM model loading error. Ensure that the file paths are correct!\n");
	}

}


ObjectClassifier::~ObjectClassifier() {
	if (fpfhModel) free(fpfhModel);
	if (siftModel) free(siftModel);
	if (hogModel)  free(hogModel);
	if (combModel) free(combModel);
}

//把词袋向量转为SVM数据库格式
void ObjectClassifier::convertBowIntoSVMData(const cv::Mat& bowVector, SVMData& svmData) {
	svmData.classNum = -1;

	for (int j = 0; j < bowVector.cols; j++) {
		SVMNode node;
		node.index = j + 1;
		node.value = static_cast<double>(bowVector.at<float>(0, j));

		svmData.nodes.push_back(node);
	}
}

//分类物体的名称
std::string getClassName(const size_t & idx) {
	std::string categories[11] = {"Bottle", "Bowl", "Box", "Can", "Carton", "Cup", "Mug", "Spray-Can", "Tin", "Tube", "Tub"};
	std::string name;

	if (idx >= 0 && idx < 11) {   //如果在这个范围就是
		name = categories[idx];
	} else {
		name = "Unknown";   //超过范围就说明不知道是什么物体
	}

	return name;
}

//分类的函数用到三个特征都是mat型数据fpfh  sift  hog都是BOW （bag of word）词袋
void ObjectClassifier::classify(const cv::Mat& fpfhBow, const cv::Mat& siftBow, const cv::Mat& hogBow) {
	SVMData fpfhData, siftData, hogData, combData;   //申明SVMdata
	SVMNodePtr fpfhDataArr, siftDataArr, hogDataArr, combDataArr;
	double *fpfhProbEstimates, *siftProbEstimates, *hogProbEstimates, *combProbEstimates;
	
	// TODO: ensure that matrices are not empty or invalid before proceeding把词袋转化为SVMdata
	convertBowIntoSVMData(fpfhBow, fpfhData);
	convertBowIntoSVMData(siftBow, siftData);
	convertBowIntoSVMData(hogBow, hogData);

	// TODO: ensure that model exists and is loaded   确认模型存在并载入了
	int numClasses = svm_get_nr_class(fpfhModel);

	// TODO: ensure that numClasses > 0
	fpfhProbEstimates = (double*) malloc (numClasses * sizeof(double));
	siftProbEstimates = (double*) malloc (numClasses * sizeof(double));
	hogProbEstimates  = (double*) malloc (numClasses * sizeof(double));
	combProbEstimates = (double*) malloc (numClasses * sizeof(double));

	fpfhDataArr = SVMUtils::convertNodeVectorToArray(fpfhData.nodes);  //转向量位矩阵
	siftDataArr = SVMUtils::convertNodeVectorToArray(siftData.nodes);
	hogDataArr = SVMUtils::convertNodeVectorToArray(hogData.nodes);

	int attrNum = 0; //< Combined data attribute number
         //向量机预测可能性
	double fpfhClass = svm_predict_probability(fpfhModel, fpfhDataArr, fpfhProbEstimates);
	for (int i = 0; i < numClasses; i++) {
		SVMNode fpfhNode;
		fpfhNode.index = ++attrNum;
		fpfhNode.value = fpfhProbEstimates[i];
		combData.nodes.push_back(fpfhNode);
	}
	
	if (OCL_DEBUG == 1)
		printf("FPFH class: %s\n", getClassName(static_cast<int>(fpfhClass) - 1).c_str());

	double siftClass = svm_predict_probability(siftModel, siftDataArr, siftProbEstimates);
	for (int i = 0; i < numClasses; i++) {
		SVMNode siftNode;
		siftNode.index = ++attrNum;
		siftNode.value = siftProbEstimates[i];
		combData.nodes.push_back(siftNode);
	}
	
	if (OCL_DEBUG == 1)
		printf("SIFT class: %s\n", getClassName(static_cast<int>(siftClass) - 1).c_str());

	double hogClass = svm_predict_probability(hogModel, hogDataArr, hogProbEstimates);
	for (int i = 0; i < numClasses; i++) {
		SVMNode hogNode;
		hogNode.index = ++attrNum;
		hogNode.value = hogProbEstimates[i];
		combData.nodes.push_back(hogNode);
	}

	if (OCL_DEBUG == 1)
		printf("HOG class: %s\n", getClassName(static_cast<int>(hogClass) - 1).c_str());
			
	double sum = 0.0;
	for (size_t j = 0; j < combData.nodes.size(); ++j) {
		sum += combData.nodes[j].value;
	}

	for (size_t j = 0; j < combData.nodes.size(); ++j) {
		combData.nodes[j].value /= sum;
	}

	combDataArr = SVMUtils::convertNodeVectorToArray(combData.nodes);//节点向量转位矩阵数据
	objectCategory.categoryLabel = svm_predict_probability(combModel, combDataArr, combProbEstimates);
	for (int k = 0; k < numClasses; k++) {
		objectCategory.classConfidence.push_back(combProbEstimates[k]);
	}

	free(fpfhDataArr);
	free(siftDataArr);
	free(hogDataArr);
	free(combDataArr);
	free(fpfhProbEstimates);
	free(siftProbEstimates);
	free(hogProbEstimates);
	free(combProbEstimates);	
}

//估计分类结果的可信度
double ObjectClassifier::measureConfidence() const {
	double sum = 0.;
	double renyiEntropy = 0.;   //熵值  是一评价估计结果的方法
	double confidence = 0.;     //可信度
	size_t length = objectCategory.classConfidence.size();
	
	for (size_t i = 0; i < length; ++i) {
		sum += (objectCategory.classConfidence[i] * objectCategory.classConfidence[i]);  
	}
	renyiEntropy = -1 * myLog2(sum);    //详情https://en.wikipedia.org/wiki/Rényi_entropy   log运算计算熵
	renyiEntropy /= myLog2(static_cast<double>(length));
	confidence = 1 - renyiEntropy;

	return confidence;
}


} /* ocl */
