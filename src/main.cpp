#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <fstream>
#include <iostream>
#include <memory>
#include <algorithm> 
#include <face_object.h>
#include <faceParserTorch.h>
#include <cv_utils.h>
#include "alignment.h"

#include "FaceDetector.h"


#define LOGPRINT(x) std::cout << x << std::endl;

int main() {



std::cout << "file path " << GetCurrentWorkingDir() << std::endl;

auto cuda_available = torch::cuda::is_available();
torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
std::cout << (cuda_available ? "CUDA available. Working on GPU." : "Working on CPU.") << '\n';


bool use_gpu = false;

std::string input_path ="/test_data/test.png" ; 
std::string output_path =  "/test_data/test_out.png";

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

std::string param = "/models/face.param";
std::string bin = "/models/face.bin";
std::string path_model = "/models/traced_faceParser.pt";

faceParser face_parser;
Cv_utils cv_utils;
Alignment alignment;
std::vector<FaceObject> faceobjects;



Detector detector(param, bin, false);
face_parser.load(use_gpu,path_model);


// read aligned image 
cv_utils.read_image(input_path, img_real);

output_image = img_real.clone();

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

std::vector<int> color =  {255, 51, 255};

cv_utils.change_color(output_network, faceobjects, color );

alignment.reverse_align(output_image, faceobjects);

cv::imwrite(output_path, output_image);


std::cout << "Done!" << std::endl;
return 0;



}




