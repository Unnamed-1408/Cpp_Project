#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "face_binary_cls.cpp"

using namespace std;

struct T{
    int size;
    int WNL;
    float **List;
};
float *ConvBNReLU(int layer, int fliter_num, T *In_channel);

void Start_CNN(float *B, float* G, float *R, int size){
    T *start = new T;
    start->size = 3;
    start->List = new float*[3];
    start->List[0] = B;
    start->List[1] = G;
    start->List[2] = R;
    start->WNL = 128;

    T *first = new T;
    T *second = new T;
    T *third = new T;
    first->size = 16;
    first->List = new float*[16];
    float *a = ConvBNReLU(0, 0, start);

    for(int i = 0; i < 130; i++){
        for(int j = 0; j < 130; j++){
//            if(a[i *130 + j] > 0)
                cout << a[i * 130 + j] << " ";
        }
        cout << endl;
    }
}

float *ConvBNReLU(int layer, int fliter_num, T *In_channel){
    int x_ptr = 1;
    int y_ptr = 1;
    float * Storage = new float[(In_channel->WNL + 2) * (In_channel->WNL + 2)]();
    float* fliter = conv_params[layer].p_weight + fliter_num * 27;
    for(;y_ptr <= In_channel->WNL; y_ptr++){
        for(x_ptr = 1; x_ptr <= In_channel->WNL; x_ptr++){
            fliter = conv_params[layer].p_weight + fliter_num * 27;
//            cout << "wuhu" << endl;
            float tmp = 0;
            for(int channel_ptr = 0; channel_ptr < In_channel->size; channel_ptr++){
//                fliter = conv_params[layer].p_weight + fliter_num * 27;
//                float tmp = 0;
//                cout << "I'm in" << endl;
//                for(int i = 0; i < 3; i++){
                    tmp += fliter[0] * In_channel->List[channel_ptr][(y_ptr-1) * (In_channel->WNL+2) + x_ptr - 1];
//                    cout << tmp << endl;
                    tmp += fliter[1] * In_channel->List[channel_ptr][(y_ptr-1) * (In_channel->WNL+2) + x_ptr];
//                    cout << tmp << endl;
                    tmp += fliter[2] * In_channel->List[channel_ptr][(y_ptr-1) * (In_channel->WNL+2) + x_ptr + 1];
//                    cout << tmp << endl;
                    tmp += fliter[3] * In_channel->List[channel_ptr][(y_ptr) * (In_channel->WNL+2) + x_ptr - 1];
//                    cout << tmp << endl;
                    tmp += fliter[4] * In_channel->List[channel_ptr][(y_ptr) * (In_channel->WNL+2) + x_ptr];
//                    cout << tmp << endl;
                    tmp += fliter[5] * In_channel->List[channel_ptr][(y_ptr) * (In_channel->WNL+2) + x_ptr + 1];
//                    cout << tmp << endl;
                    tmp += fliter[6] * In_channel->List[channel_ptr][(y_ptr+1) * (In_channel->WNL+2) + x_ptr - 1];
//                    cout << tmp << endl;
                    tmp += fliter[7] * In_channel->List[channel_ptr][(y_ptr+1) * (In_channel->WNL+2) + x_ptr];
//                    cout << tmp << endl;
                    tmp += fliter[8] * In_channel->List[channel_ptr][(y_ptr+1) * (In_channel->WNL+2) + x_ptr + 1];
//                    cout << tmp << endl;
//                }
                fliter = fliter + 9;
//                cout << tmp << endl;
//                if(x_ptr == 2)
//                    return nullptr;
//                Storage[x_ptr + y_ptr*(In_channel->WNL + 2)] += tmp;
            }
            if(tmp + conv_params[layer].p_bias[fliter_num] > 0) // RELU
                Storage[x_ptr + y_ptr*(In_channel->WNL + 2)] += tmp + conv_params[layer].p_bias[fliter_num];
        }
    }
    return Storage;
}
