#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <fstream>
#include <iostream>
#include <memory>
#include <algorithm> 


// #include <opencv2/opencv.hpp>
#include <face_object.h>
// #include <faceParser.h>
#include <faceParserTorch.h>
#include <cv_utils.h>
#include "alignment.h"


#include "FaceDetector.h"


#define LOGPRINT(x) std::cout << x << std::endl;

int main() {

auto cuda_available = torch::cuda::is_available();
torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
std::cout << (cuda_available ? "CUDA available. Working on GPU." : "Working on CPU.") << '\n';


bool use_gpu = false;
std::string path_img ="/Users/yasinmac/Downloads/test.jpeg" ; //"/media/syn1/HDD/Face-Detector-1MB-with-landmark/Face_Detector_ncnn/sample.jpg";


cv::Mat img_real;
cv::Mat img_scale;
std::vector<bbox> boxes;
float scale;
const int max_side = 320;

cv::Mat output_image;
cv::Mat out_mask;
torch::Tensor tensor_img;
torch::Tensor output_network ;

std::vector<torch::Tensor> aligned_images;

std::string param = "/Users/yasinmac/Desktop/yasin_/github/c-/models/face.param";
std::string bin = "/Users/yasinmac/Desktop/yasin_/github/c-/models/face.bin";

std::string path_model = "/Users/yasinmac/Desktop/yasin_/github/c-/models/traced_faceParser.pt";

faceParser face_parser;
Cv_utils cv_utils;
Alignment alignment;
std::vector<FaceObject> faceobjects;


// slim or RFB
Detector detector(param, bin, false);




face_parser.load(use_gpu,path_model);


// read aligned image 
cv_utils.read_image(path_img, img_real);

output_image = img_real.clone();
// img_real.convertTo(output_image,CV_32F);

// detector

float long_side = std::max(img_real.cols, img_real.rows);
scale = max_side/long_side;
cv::resize(img_real, img_scale, cv::Size(img_real.cols*scale, img_real.rows*scale));
detector.Detect(img_scale, boxes);



detector.box2faceobject( boxes, faceobjects,  scale);



for (int i = 0; i < boxes.size(); ++i) {
    
    
    alignment.align(img_real, faceobjects[i]);
    faceobjects[i].tensor_img = cv_utils.img2tensor(faceobjects[i].aligned);
    aligned_images.push_back(faceobjects[i].tensor_img);
    
    }

torch::Tensor tensor_image = torch::cat({aligned_images}, 0);


face_parser.inference(tensor_image, output_network);

std::vector<int> color =  {255, 255, 255};

cv_utils.change_color(output_network, faceobjects, color );

alignment.reverse_align(output_image, faceobjects);

cv::imwrite("/Users/yasinmac/Desktop/yasin_/github/c-/test_out.png", output_image);




// for (int j = 0; j < boxes.size(); ++j) {
    
    
    
//     alignment.align(img_real, faceobjects[j]);

//     cv::imwrite("/Users/yasinmac/Desktop/yasin_/github/c-/aligned.png", faceobjects[j].syn_aligned);
    


//     cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
//     cv::rectangle(img_real, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
//     char test[80];
//     sprintf(test, "%f", boxes[j].s);

//     cv::putText(img_real, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
//     cv::circle(img_real, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
//     cv::circle(img_real, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
//     cv::circle(img_real, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
//     cv::circle(img_real, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
//     cv::circle(img_real, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
// }

// cv::imwrite("/Users/yasinmac/Desktop/yasin_/github/c-/out.png", img_real);




std::cout << "Done!" << std::endl;
return 0;



}




