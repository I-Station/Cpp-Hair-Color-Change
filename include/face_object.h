

#ifndef FACEOBJECT_H
#define FACEOBJECT_H

#include <opencv2/core/core.hpp>
#include <torch/torch.h>


struct FaceObject
{
    int image_width, image_height;
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    cv::Mat  m;
    cv::Mat  m_inverse;
    cv::Mat output;
    cv::Mat changed ;
    cv::Mat aligned;
    torch::Tensor tensor_img;
    float prob;
};


#endif