#include "SPextractor.h"
#include "SPmatcher.h"
#include <string>

using namespace cv;
using namespace ORB_SLAM2;
std::string txt_path;

int main()
{
    const std::string path1 = "/media/lyc/linyicheng15827404778/dataset/1-outdoor-five_speed/sequence01/image/00200.png";
    const std::string path2 = "/media/lyc/linyicheng15827404778/dataset/1-outdoor-five_speed/sequence01/image/00202.png";

    cv::Mat img1 = cv::imread(path1, CV_8UC1);
    cv::Mat img2 = cv::imread(path2, CV_8UC1);

    cv::resize(img1, img1, cv::Size(640, 480));
    cv::resize(img2, img2, cv::Size(640, 480));

    std::vector<cv::KeyPoint> key1, key2;
    cv::Mat des1, des2;

    auto extractor = new SPextractor(1000,1.2,4,0.015,0.007);
    for (int i = 0; i < 1000;i ++){
        extractor->operator()(img1, cv::Mat(), key1, des1);
        extractor->operator()(img2, cv::Mat(), key2, des2);
    }

    cv::Mat show1, show2;
    drawKeypoints(img1, key1, show1  ,Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img2, key2, show2  ,Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    imshow("Display Image 1", show1);
    imshow("Display Image 2", show2);
    waitKey(0);
}

