#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("./T.jpg");
    imshow("wudi", image);
    waitKey(0);

    cout << "Hello World!" << endl;
    return 0;
}
