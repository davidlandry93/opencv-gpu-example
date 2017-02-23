#include <iostream>
#include <chrono>

using namespace std;
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>

int main (int argc, char* argv[])
{
    try
        {
            cv::cuda::printCudaDeviceInfo(0);

            cv::Mat src_host = cv::imread("../example.png", CV_LOAD_IMAGE_GRAYSCALE);

            cv::cuda::GpuMat dst, src;
            src.upload(src_host);

            auto start = chrono::steady_clock::now();
            cv::cuda::threshold(src, dst, 100.0, 255.0, CV_THRESH_BINARY);
            auto accelerated_duration = chrono::steady_clock::now() - start;

            cv::Mat result_host;
            dst.download(result_host);
            // cv::imshow("Input", src_host);
            // cv::imshow("Result", result_host);
            // cv::waitKey();

            cv::Mat dst_host;

            start = chrono::steady_clock::now();
            cv::threshold(src_host, dst_host, 100.0, 255.0, CV_THRESH_BINARY);
            auto host_duration = chrono::steady_clock::now() - start;


            cout << "=== Thresholding ===" << endl;
            cout << "Host time: " << chrono::duration<float,milli>(host_duration).count() << " ms." << endl;
            cout << "Device time: " << chrono::duration<float,milli>(accelerated_duration).count() << " ms." << endl;
            cout << "Speedup: " << chrono::duration<float,milli>(host_duration).count() / chrono::duration<float, milli>(accelerated_duration).count()  << "x." << endl;

        }
    catch(const cv::Exception& ex)
        {
            std::cout << "Error: " << ex.what() << std::endl;
        }
    return 0;
}
