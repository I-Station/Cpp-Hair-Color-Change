#ifndef CV_UTILS_H
#define CV_UTILS_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <face_object.h>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

class Cv_utils 
{
public:

    torch::Tensor std = torch::tensor({0.485, 0.456, 0.406});
    torch::Tensor mean = torch::tensor({0.229, 0.224, 0.225});

    // torch::Tensor std = torch::tensor({0.5, 0.5, 0.5}).unsqueeze(1).unsqueeze(2);
    // torch::Tensor mean = torch::tensor({0.5, 0.5, 0.5}).unsqueeze(1).unsqueeze(2);

    cv::Mat tensor2im(torch::Tensor tensor_image, int channel_size) ;
    
    cv::Mat tensormask2im(torch::Tensor tensor_image);

    torch::Tensor img2tensor(cv::Mat img);

    // void save_tensor(torch::Tensor, std::string &path);

    void change_color(torch::Tensor output_network , std::vector<FaceObject> &faceobjects , std::vector<int> color );

    void mask_smooting(cv::Mat& mask);

    cv::Mat reverse_image(cv::Mat& reverse_face,cv::Mat& rgb);

    void rgb_normalize(cv::Mat rgb, cv::Mat& output);
  
    void read_image(std::string path, cv::Mat& img);

    

private:
        std::vector<double> norm_mean = {0.485, 0.456, 0.406};
        std::vector<double> norm_std = {0.229, 0.224, 0.225};
        cv::Mat hsv_color;
        cv::Mat hsv_aligned;
        cv::Mat one_image= cv::Mat(512, 512, CV_8UC3, cv::Scalar(1.0,1.0,1.0));


};
#endif //CV_UTILS_H