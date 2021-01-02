#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "face_binary_cls.cpp"
#include <algorithm>
#include <cmath>

using namespace std;

struct T{
    int size;
    int WNL;
    float **List;
};

float *ConvBNReLU(int layer, int fliter_num, T *In_channel);
float *MaxPooling(T *In_channel, int num, int layer);
float *Flatten(T *In_channel);
float *dot_product(float *Large);

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
    first->WNL = 64;
    for(int i = 0; i < 16; i++){
        first->List[i] = ConvBNReLU(0, i, start);
    }



    for(int i = 0; i < 16; i++){
        float *tmp = MaxPooling(first, i, 1);
        delete first->List[i];
        first->List[i] = tmp;
    }
    first->WNL = 32;

//        for(int i = 0; i < 34; i++){
//            for(int j = 0; j < 34; j++){
//                    cout << first->List[0][i * 34 + j] << " ";
//            }
//            cout << endl;
//        }

    second->WNL = 32;
    second->size = 32;
    second->List = new float*[32]();
    cout << "wa" << endl;
    for(int i = 0; i < 32; i++){
        second->List[i] = ConvBNReLU(1, i, first);
    }

//    for(int i = 0; i < 34; i++){
//        for(int j = 0; j < 34; j++){
//                cout << second->List[0][i * 34 + j] << " ";
//        }
//        cout << endl;
//    }

    for(int i = 0; i < 32; i++){
        float *tmp = MaxPooling(second, i, 2);
        delete second->List[i];
        second->List[i] = tmp;
    }
    second->WNL = 16;

//        for(int i = 0; i < 18; i++){
//            for(int j = 0; j < 18; j++){
//                    cout << second->List[1][i * 18 + j] << " ";
//            }
//            cout << endl;
//        }

    third->size = 32;
    third->List = new float*[32]();
    third->WNL = 16;
    for(int i = 0; i < 32; i++){
        third->List[i] = ConvBNReLU(2, i, second);
    }
    third->WNL = 8;

//    for(int i = 0; i < 10; i++){
//        for(int j = 0; j < 10; j++){
//                cout << third->List[30][i * 10 + j] << " ";
//        }
//        cout << endl;
//    }

    float *end = Flatten(third);

//    for(int i = 0; i < 2048; i++){
//        cout << end[i] << " ";
//    }

    float *to_out = dot_product(end);
    cout << (exp(to_out[0])/(exp(to_out[0]) + exp(to_out[1]))) << endl;


}

float *ConvBNReLU(int layer, int fliter_num, T *In_channel){
    int strike = conv_params[layer].stride;

    if(layer == 2){
        int x_ptr = 2;
        int y_ptr = 2;
        float * Storage = new float[(In_channel->WNL/strike + 2) * (In_channel->WNL/strike + 2)]();
        float* fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
        for(;y_ptr <= In_channel->WNL; y_ptr += strike){
            for(x_ptr = 2; x_ptr <= In_channel->WNL; x_ptr += strike){
                fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
                float tmp = 0;
//            cout << x_ptr << " " << y_ptr << endl;
                for(int channel_ptr = 0; channel_ptr < In_channel->size; channel_ptr++){
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
                    Storage[(strike - 2 + x_ptr/strike) + (strike - 2 + y_ptr/strike)*(In_channel->WNL/strike + 2)] += tmp + conv_params[layer].p_bias[fliter_num];
            }
        }
        return Storage;
    }

    if(strike == 2 && layer != 2){
        int x_ptr = 1;
        int y_ptr = 1;
        float * Storage = new float[(In_channel->WNL/strike + 2) * (In_channel->WNL/strike + 2)]();
        float* fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
        for(;y_ptr <= In_channel->WNL; y_ptr += strike){
            for(x_ptr = 1; x_ptr <= In_channel->WNL; x_ptr += strike){
                fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
                float tmp = 0;
//            cout << x_ptr << " " << y_ptr << endl;
                for(int channel_ptr = 0; channel_ptr < In_channel->size; channel_ptr++){
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
                    Storage[(strike - 1 + x_ptr/strike) + (strike - 1 + y_ptr/strike)*(In_channel->WNL/strike + 2)] += tmp + conv_params[layer].p_bias[fliter_num];
            }
        }
        return Storage;
    }else{
        int x_ptr = 2;
        int y_ptr = 2;
        float * Storage = new float[(In_channel->WNL/strike + 2) * (In_channel->WNL/strike + 2)]();
        float* fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
        for(;y_ptr < In_channel->WNL; y_ptr += strike){
            for(x_ptr = 2; x_ptr < In_channel->WNL; x_ptr += strike){
                fliter = conv_params[layer].p_weight + fliter_num * In_channel->size * 9;
                float tmp = 0;
//            cout << x_ptr << " " << y_ptr << endl;
                for(int channel_ptr = 0; channel_ptr < In_channel->size; channel_ptr++){
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
                    Storage[(strike - 1 + x_ptr/strike) + (strike - 1 + y_ptr/strike)*(In_channel->WNL/strike + 2)] += tmp + conv_params[layer].p_bias[fliter_num];
            }
        }
        return Storage;
    }
}



float *MaxPooling(T *In_channel, int num, int layer){
    if(layer == 1){
        float *Storage = new float[(In_channel->WNL/2 + 2) * (In_channel->WNL/2 + 2)]();
        int x_ptr = 1;
        int y_ptr = 1;
        for(;y_ptr <= In_channel->WNL; y_ptr += 2){
            for(x_ptr = 1; x_ptr <= In_channel->WNL; x_ptr += 2){
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(In_channel->List[num][x_ptr + y_ptr * (In_channel->WNL + 2)],In_channel->List[num][x_ptr + 1 + y_ptr * (In_channel->WNL + 2)]);
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)],In_channel->List[num][x_ptr + 1 + (y_ptr + 1) * (In_channel->WNL + 2)]);
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)],In_channel->List[num][x_ptr + (y_ptr + 1) * (In_channel->WNL + 2)]);
            }
        }
        return Storage;
    }
    else{
        float *Storage = new float[(In_channel->WNL/2 + 2) * (In_channel->WNL/2 + 2)]();
        int x_ptr = 2;
        int y_ptr = 2;
        for(;y_ptr < In_channel->WNL; y_ptr += 2){
            for(x_ptr = 2; x_ptr < In_channel->WNL; x_ptr += 2){
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(In_channel->List[num][x_ptr + y_ptr * (In_channel->WNL + 2)],In_channel->List[num][x_ptr + 1 + y_ptr * (In_channel->WNL + 2)]);
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)],In_channel->List[num][x_ptr + 1 + (y_ptr + 1) * (In_channel->WNL + 2)]);
                Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)] = max(Storage[1 + x_ptr/2 + (1 + y_ptr/2) * (In_channel->WNL/2 + 2)],In_channel->List[num][x_ptr + (y_ptr + 1) * (In_channel->WNL + 2)]);
            }
        }
        return Storage;
    }
}

float *Flatten(T *In_channel){
    float *Large = new float[In_channel->size * In_channel->WNL * In_channel->WNL]();
    int Large_ptr = 0;
    for(int m = 0; m < In_channel->size; m++){
        for(int i = 1; i <= In_channel->WNL; i++){
            for(int j = 1; j <= In_channel->WNL; j++){
                Large[Large_ptr] = In_channel->List[m][i * (In_channel->WNL + 2) + j];
                Large_ptr++;
            }
        }
    }
    return Large;
}


float *dot_product(float *Large){
    float *out = new float[2]();
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2048; j++){
            out[i] += Large[j] * fc_params->p_weight[i * 2048 + j];
//            cout << out[i] << " ";
        }
        out[i] += fc_params->p_bias[i];
    }
    return out;
}































