#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <sys/time.h>
#include "Fast_CNN.cpp"

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
//    imshow("sss",rst);
//    cout << rst.rows << " " << rst.cols;
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
            blue[(row+1) * (width+2) + col + 1] = b / 255.0f;
            green[(row+1) * (width+2) + col + 1] = g / 255.0f;
            red[(row+1) * (width+2) + col + 1] = r / 255.0f;
        }
    }
    auto start = std::chrono::steady_clock::now();
//start
    Start_CNN(blue, green, red, 128);
//end
    auto end = std::chrono::steady_clock::now();
    cout << "cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;
    delete[]blue;
    delete[]green;
    delete[]red;
    waitKey(0);
    return 0;
}
