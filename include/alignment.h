
#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include<opencv2/opencv.hpp>
#include<face_object.h>
#include "face_object.h"

// #include<tuple>

class Alignment
{
public:
    float v1[5][2] = {{192.98138, 239.94708},
                      {318.90277, 240.1936},
                      {256.63416, 314.01935},
                      {201.26117, 371.41043},
                      {313.08905, 371.15118}};



    cv::Mat syntonym_align_matrix = cv::Mat(5,2,CV_32FC1, v1);

    // syntonym_align_matrix *= 4;

    // cv::Mat syntonym_align_matrix = 4 * align_matrix;



    cv::Mat similarTransform(cv::Mat src,cv::Mat dst);
    int MatrixRank(cv::Mat M);
    cv::Mat varAxis0(const cv::Mat &src);
    cv::Mat meanAxis0(const cv::Mat &src);
    cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B);
    int align(cv::Mat& img, FaceObject& face );
    void reverse_align(cv::Mat& img, std::vector<FaceObject> &faceobjects);


};
#endif // ALIGNMENT_H
