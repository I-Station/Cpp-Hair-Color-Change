
#ifndef FACEPARSER_H
#define FACEPARSER_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include "net.h"
#include <iostream>
#include <face_object.h>



#include <string>
#include <vector>

#include <torch/torch.h>
#include "face_object.h"

class faceParser
{
public:

    int load(bool use_gpu, std::string model_path);

    void inference(torch::Tensor& input , torch::Tensor &generated);
    
private:
    torch::jit::script::Module network_face_parser;
};




#endif