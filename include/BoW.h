#pragma once
#ifndef BOW_ASSIGNER_H_
#define BOW_ASSIGNER_H_

#include <opencv2/features2d/features2d.hpp>

namespace ocl {

class BoW
{
public:
  BoW();

  virtual ~BoW();

	void setVocabulary(const cv::Mat& vocab);    //设置词袋是mat型数据

	const cv::Mat& getVocabulary() const;    //得到词袋

	int getVocabularySize() const;         //得到词袋的大小

	void compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor); //对于给定的描述子计算词袋
	

private:
	cv::Mat vocabulary;    //申明单词数据类型

	cv::Ptr<cv::DescriptorMatcher> matcher;  //描述子的配准
};


} /* ocl */

#endif
