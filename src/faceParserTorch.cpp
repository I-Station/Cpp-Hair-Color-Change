

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <faceParserTorch.h>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.



int faceParser::load( bool use_gpu, std::string model_path)
{
   
    if(use_gpu)
    {   
        network_face_parser = torch::jit::load(model_path,torch::kCUDA);
       
    }
    else
    {   
        std::cout << model_path << std::endl;
        network_face_parser = torch::jit::load(model_path,torch::kCPU);
    }

    network_face_parser.eval();

    return 0;   
}



void faceParser::inference(torch::Tensor &input ,torch::Tensor &output )
{   

    output = network_face_parser.forward( {input}).toTensor();

    
}



