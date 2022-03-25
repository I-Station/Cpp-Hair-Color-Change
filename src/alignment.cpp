#include "alignment.h"
#include<opencv2/opencv.hpp>
#include<face_object.h>
#include "cv_utils.h"
#define LOGPRT(x) std::cout << x << std::endl;
Cv_utils cv_utils;


cv::Mat Alignment::similarTransform(cv::Mat src,cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S,U, V);

    // the SVD function in opencv differ from scipy .


    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    float scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat  temp2 = src_mean.t(); //src_mean.T
    cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;

    return T;
}



int Alignment::MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}

cv::Mat Alignment::varAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return meanAxis0(temp_);

}

cv::Mat Alignment::meanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}

cv::Mat Alignment::elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}

// std::tuple <cv::Mat ,cv::Mat, cv::Mat> alignment(cv::Mat& img, float tmp_landmark[5][2])
// {


//     cv::Mat landmark = cv::Mat(5,2,CV_32FC1, tmp_landmark);
//     cv::Mat similarity_matrix = similarTransform(landmark,syntonym_align_matrix);
    
    
//     cv::Mat m = similarity_matrix.rowRange(0,2).clone();
    
//     similarity_matrix = similarity_matrix.inv();
    
//     cv::Mat m_inverse = similarity_matrix.rowRange(0,2);
//     cv::Mat aligned_face;
//     cv::warpAffine(img,aligned_face,m, cv::Size(128, 128));
    
//     return std::make_tuple(aligned_face, m, m_inverse) ;
    
// }


int Alignment::align(cv::Mat& img, FaceObject& face )
{

    face.image_width = img.cols;
    face.image_height = img.rows;
    float tmp_landmark[5][2] = {{face.landmark[0].x, face.landmark[0].y},
                        {face.landmark[1].x, face.landmark[1].y},
                        {face.landmark[2].x, face.landmark[2].y},
                        {face.landmark[3].x, face.landmark[3].y},
                        {face.landmark[4].x, face.landmark[4].y}};



    cv::Mat landmark = cv::Mat(5,2,CV_32FC1, tmp_landmark);
    cv::Mat similarity_matrix = similarTransform(landmark,syntonym_align_matrix);
    
    
    face.m = similarity_matrix.rowRange(0,2).clone();
    
    similarity_matrix = similarity_matrix.inv();
    
    face.m_inverse = similarity_matrix.rowRange(0,2);
    cv::Mat aligned_face;
    
    cv::warpAffine(img,aligned_face,face.m, cv::Size(512, 512));
    face.aligned = aligned_face;

    return 0 ;

}


void Alignment::reverse_align(cv::Mat& img, std::vector<FaceObject> &faceobjects)
{



    for(size_t i = 0; i < faceobjects.size(); i++)
    {   



    cv::Mat out_mask(faceobjects[i].aligned.rows, faceobjects[i].aligned.cols,CV_8UC3, CV_RGB(1.0,1.0,1.0));
    cv::Mat reverse_aligned ,reverse_mask ;
    
    cv::warpAffine(out_mask, reverse_mask, faceobjects[i].m_inverse, cv::Size(faceobjects[i].image_width, faceobjects[i].image_height));
    cv::warpAffine(faceobjects[i].changed, reverse_aligned, faceobjects[i].m_inverse, cv::Size(faceobjects[i].image_width, faceobjects[i].image_height));
 
    cv::Mat one_image(reverse_mask.rows,reverse_mask.cols,CV_8UC3,CV_RGB(1.0,1.0,1.0));


    img = img.mul(one_image-reverse_mask) + reverse_aligned.mul(reverse_mask);
    
        }



}

