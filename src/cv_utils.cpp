#include "cv_utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <face_object.h>



// Cv_utils cv_utils;

cv::Mat Cv_utils::tensor2im(torch::Tensor tensor_image, int channel_size) {
    tensor_image = tensor_image.cpu();
    tensor_image = tensor_image.mul_(std).add_(mean);
    tensor_image = tensor_image.mul_(255).clamp(0, 255).toType(torch::kU8).permute({1,2,0}).detach();
    tensor_image = tensor_image.reshape({ 512 * 512 * 1});
    
    
    cv::Mat img_out = cv::Mat(512, 512, CV_8UC1, tensor_image.data_ptr()).clone();
        
    return img_out;
}


cv::Mat Cv_utils::tensormask2im(torch::Tensor tensor_image) {
    tensor_image = tensor_image.cpu();
    tensor_image = tensor_image.unsqueeze(0).mul_(1).clamp(0, 1).toType(torch::kU8).permute({1,2,0}).detach();
    tensor_image = tensor_image.reshape({ 512 * 512 * 1});
    
    cv::Mat img_out = cv::Mat(512, 512, CV_8UC1, tensor_image.data_ptr()).clone();
    
    return img_out;
}

torch::Tensor Cv_utils::img2tensor(cv::Mat img) {


    // auto tensor_image = torch::from_blob(img.data, {  img.rows,img.cols, 3 }, torch::kU8);
    torch::Tensor tensor_image = torch::from_blob(img.data, {  img.rows,img.cols, 3 }, torch::kU8);
    tensor_image = tensor_image.div(255).unsqueeze(0);
    tensor_image = tensor_image.permute({ 0, 3, 1, 2 });


    tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor_image.toType(torch::kFloat));

// torch::data::transforms::Normalize<>(norm_mean, norm_std)(input_tensor);
    // tensor_image = tensor_image.toType(torch::kFloat).sub_(norm_mean).div_(norm_std);
    return tensor_image;


}
// void Cv_utils::save_tensor(torch::Tensor tensor, std::string &path){
//     // torch::Tensor bytes_1 = torch::jit::pickle_save(tensor);
//     std::byte bytes_1 = torch::jit::pickle_save(tensor);
//     std::ofstream fout1(path, std::ios::out | std::ios::binary);
//     fout1.write(bytes_1.data(), bytes_1.size());
//     fout1.close();
// }

void mask_smooting(cv::Mat& mask)
{
    int value = 10;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2 * value + 1,
                         2 * value + 1),
        cv::Point(value, value));

    cv::erode(mask,mask, kernel,cv::Point(-1, -1), 1);
    cv::blur(mask,mask,cv::Size(50,50),cv::Point(-1,-1));
}


cv::Mat Cv_utils::reverse_image(cv::Mat& reverse_face,cv::Mat& rgb)
{
    cv::Mat total_output;
    cv::cvtColor( reverse_face, total_output,cv::COLOR_BGR2GRAY);
    cv::Mat idx;
    cv::findNonZero(total_output, idx);
    std::cout << idx.size() << std::endl;
    return total_output;

}
void Cv_utils::rgb_normalize(cv::Mat rgb, cv::Mat& output)
{
    cv::Mat rgb_tmp;
    cv::Mat output_src;
    rgb.convertTo(rgb_tmp, CV_32F);
    output.convertTo(output, CV_32F);
    std::vector<cv::Mat> rgb_channels;
    std::vector<cv::Mat> output_channels;


    cv::split(rgb,rgb_channels);
    cv::split(output,output_channels);

    for(int i=0; i<3; i++)
    {
        cv::Mat tmp_src, tmp_target;
        cv::Mat target_mean, target_std,source_mean, source_std;
        cv::meanStdDev(rgb_channels[i], target_mean, target_std);
        cv::meanStdDev(output_channels[i], source_mean, source_std);
        tmp_src = (output_channels[i]-source_mean)/source_std;
        tmp_target = tmp_src.mul(target_std)+target_mean;
        output_channels[i] = tmp_target;
    }
    cv::merge(output_channels, output_src);
    output = output_src;

}

void Cv_utils::mask_smooting(cv::Mat& mask)
{
    int value = 10;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2 * value + 1,
                        2 * value + 1),
        cv::Point(value, value));

    cv::erode(mask,mask, kernel,cv::Point(-1, -1), 1);
    cv::blur(mask,mask,cv::Size(10,10),cv::Point(-1,-1));
}

void Cv_utils::read_image(std::string path, cv::Mat &img)
{

    img = cv::imread(path);
    // cvtColor( img, img, CV_BGR2RGB );
    // cv::cvtColor( img, img, cv::COLOR_BGR2RGB);
    // cv::resize(img, img, cv::Size(128, 128));

}




void Cv_utils::change_color(torch::Tensor output_network , std::vector<FaceObject> &faceobjects, std::vector<int> color )
{     

    auto mask = torch::where(output_network == 17, torch::ones_like(output_network), torch::zeros_like(output_network));

    for(size_t i = 0; i < faceobjects.size(); i++)
    {   

        cv::Mat out_mask = tensormask2im(mask[i]);
        cv::cvtColor(out_mask, out_mask, cv::COLOR_GRAY2BGR);


        cv::Mat bgr_color(out_mask.rows, out_mask.cols, CV_8UC3, cv::Scalar(color[0],color[1],color[2]));
        cv::Mat hsv_color;
        cv::cvtColor(bgr_color, hsv_color, cv::COLOR_BGR2HSV);

        cv::Mat hsv_aligned;
        cv::cvtColor(faceobjects[i].aligned, hsv_aligned, cv::COLOR_BGR2HSV);

        cv::Mat hsv_aligned_channel[3];
        cv::split(hsv_aligned, hsv_aligned_channel);

        cv::Mat hsv_color_channel[3];
        cv::split(hsv_color, hsv_color_channel);

        hsv_aligned_channel[0] = hsv_color_channel[0];

        cv::merge(hsv_aligned_channel, 3, hsv_aligned);

        cv::cvtColor(hsv_aligned, hsv_aligned, cv::COLOR_HSV2BGR);
        
        cv::Mat one_image(out_mask.rows, out_mask.cols, CV_8UC3, cv::Scalar(1.0,1.0,1.0));

        faceobjects[i].changed = hsv_aligned.mul(out_mask) +  faceobjects[i].aligned.mul(one_image-out_mask) ;
        // cv::imwrite("/Users/yasinmac/Desktop/yasin_/github/c-/aligned_changed.png", finall);



    }}
    



// -----------------------CALCULATE FPS-----------------------
// double temp = 0;
// for ( int i = 0; i<100 ; i++)
// {
//     int64 start = cv::getTickCount();
//     generated = generator.forward( {tensor_image, identity}).toTensor();
//     double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
//     temp += fps;
    
// }

// std::cout << "FPS Generator : " << temp/(double)100 << std::endl;
