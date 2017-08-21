#include <cstdlib>
#include <cstdio>
#include <boost/timer.hpp>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>  
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include "ObjectDescription.h"
#include "ObjectClassifier.h"

//在这里只是使用到了pcl库是为了打开载入点云数据，SVM还是以opencv的库
using namespace std;

static string getClassName(const size_t & idx) {
	string categories[11] = {"Bottle", "Bowl", "Box", "Can", "Carton", "Cup", "Mug", "Spray-Can", "Tin", "Tube", "Tub"};
	
	if (idx >= 0 && idx < 11) {
		return categories[idx];
  }
  else {
		return "Unknown";
	}
}


/** print usage
  */
void printUsage(const char* prog_name) {
    cout<< "\n\nUsage: " << prog_name << " [options]\n\n"
        << "Options:\n"
        << "-------------------------------------------\n"
        << "-h          this help\n"
        << "-data       target rgbd point cloud, with .ply type\n"
        << "-vocab      vocabulary data directory\n"
        << "-svm        svm model directory\n";
}


int main(int argc, char *argv[]) {
  if (pcl::console::find_argument (argc, argv, "-h") != -1) {
    printUsage(argv[0]);
    return 0;
  }
  string cloud_name, vocab_dir, svm_dir;
  pcl::console::parse_argument (argc, argv, "-data", cloud_name);
  pcl::console::parse_argument (argc, argv, "-vocab", vocab_dir);
  pcl::console::parse_argument (argc, argv, "-svm", svm_dir);
  cout << "detect " << cloud_name << " using vocabulary: " << vocab_dir << " and svm " << svm_dir << endl;
  pcl::PointCloud<pcl::PointXYZRGB> loadedCloud;
  pcl::io::loadPCDFile<pcl::PointXYZRGB>(cloud_name, loadedCloud); //载入我们需要判断的物体

  ocl::ObjectDescription desc(vocab_dir);//载入所有点云物体的描述

  ocl::ObjectClassifier classifier(svm_dir);  //根据特征使用SVM 的数据进行份类
  {
    pcl::console::TicToc tt; tt.tic();
    desc.extractFeatureDescriptors(loadedCloud);  //提取载入点云的描述子
    desc.assignBOWs();
    classifier.classify(desc.getFPFHBow(), desc.getSIFTBow(), desc.getHOGBow()); //利用三个特征来判断这个载入点云物体
    cout << "Time: " << tt.toc() << endl;
  }

  cout << "Object class: " << getClassName(classifier.objectCategory.categoryLabel - 1) << endl;   //输出判断结果就是人工的标记的物体
  cout << "With confidence value: " << classifier.measureConfidence() << endl;    //输出可信度
  cout << "scores:\n";
  for (size_t j = 0; j < classifier.objectCategory.classConfidence.size(); j++)
    cout << "\t" << j+1 << " -> " << classifier.objectCategory.classConfidence[j] << endl;  //输出所有用于分类的物体对该点云的可信度
	return EXIT_SUCCESS;
}
