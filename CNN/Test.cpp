#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "CNN.cpp"

using namespace std;
using namespace cv;

int main()
{
    Mat src;
    src = imread("T.jpg");    //图像加载
    if (src.empty()) {
        cout << "could not load image..." << endl;
        return -1;
    }
    else cout << "load succsessful." << endl;
    imshow("input", src);
    Mat rst;
    float xradio = (float)128 / (float)src.rows;
    float yradio = (float)128 / (float)src.cols;
    resize(src, rst, Size(), xradio, yradio);
    imshow("sss",rst);
    cout << rst.rows << " " << rst.cols;
    int height = rst.rows;
    int width = rst.cols;
    float* blue = new float[(height+2) * (width+2)]();
    float* green = new float[(height+2) * (width+2)]();
    float* red = new float[(height+2) * (width+2)]();

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float	b = src.at<Vec3b>(row, col)[0];  //读取通道值
            float	g = src.at<Vec3b>(row, col)[1];
            float	r = src.at<Vec3b>(row, col)[2];
            blue[(row+1) * (width+2) + col + 1] = b / 256.0f;
            green[(row+1) * (width+2) + col + 1] = g / 256.0f;
            red[(row+1) * (width+2) + col + 1] = r / 256.0f;
        }
    }
//    cout << "blue" << endl;
//    for(int i = 0; i < 130; i++){
//        for(int j = 0; j < 130; j++){
//            cout << blue[i * 130 + j] << " ";
//        }
//        cout << endl;
//    }

//    cout << "green" << endl;
//    for(int i = 0; i < 130; i++){
//        for(int j = 0; j < 130; j++){
//            cout << green[i * 130 + j] << " ";
//        }
//        cout << endl;
//    }

//    cout << "red" << endl;
//    for(int i = 0; i < 130; i++){
//        for(int j = 0; j < 130; j++){
//            cout << red[i * 130 + j] << " ";
//        }
//        cout << endl;
//    }


    Start_CNN(blue, green, red, 128);
    delete[]blue;
    delete[]green;
    delete[]red;
    waitKey(0);
    return 0;
}
