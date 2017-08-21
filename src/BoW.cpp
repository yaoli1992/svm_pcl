#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "BoW.h"

namespace ocl {

BoW::BoW() {
  // create descriptor matcher using BruteForce
	matcher  = cv::DescriptorMatcher::create("BruteForce");   //配准的方法还是常用的暴力匹配方法
}


BoW::~BoW() {
}

//设置词袋的单词
void BoW::setVocabulary(const cv::Mat& vocab) {
	matcher->clear();   //首先把matcher清除
	vocabulary = vocab;
	matcher->add( std::vector<cv::Mat>(1, vocab) );
}


const cv::Mat& BoW::getVocabulary() const {
	return vocabulary;
}

//单词如果不为空  那么单词的大小就位单词的rows行
int BoW::getVocabularySize() const {
	return vocabulary.empty() ? 0 : vocabulary.rows;
}

//计算描述子的单词描述
void BoW::compute(const cv::Mat& queryDesc, cv::Mat& bowDescriptor) {
	bowDescriptor.release();  //释放词袋
	std::vector<cv::DMatch> matches;
	matcher->match(queryDesc, matches);   //对给定的描述子进行配准

	bowDescriptor = cv::Mat(1, getVocabularySize(), CV_32F, cv::Scalar::all(0.0));  //这个词袋的描述子为1行 size大小的float类型的mat型数据
	float *descPtr = (float*) bowDescriptor.data;   //然后把得到的mat型的词袋的描述子赋值给descPtr

	for (size_t i = 0; i < matches.size(); i++) {
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx; // cluster index  //分类的索引
		CV_Assert( queryIdx == static_cast<int>(i));

		descPtr[trainIdx] = descPtr[trainIdx] + 1.0f;
	}
  // normalize BoW descriptor   均值化 BOW描述子
	bowDescriptor /= queryDesc.rows;
}


} /* ocl */
