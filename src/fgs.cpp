#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using std::vector;
using std::chrono::system_clock;
using std::chrono::duration;

#define CVPLOT_HEADER_ONLY
#include "CvPlot/cvplot.h"

#ifndef uchar
typedef unsigned char uchar;
#endif

#define SQ(x) ((x)*(x))
#define DATA_TYPE float
#define DEPTH_SCALE 10.F
#define DEPTH_OFFSET -5.F

// Graph attrbs
#define GRAPH_WIDTH 1400
#define GRAPH_HEIGHT 600
#define SPECIFIC_ROW 100

class Timer {
private:
	system_clock::time_point start;
public:
	Timer() { tic(); }
	void tic() {
		start = system_clock::now();
	}

	double toc() {
		duration<double> elapsed = system_clock::now() - start;
		return elapsed.count();
	}
};

template <typename T>
void printParams(const double sigma, const T lambda, const int iter, const T attn)
{
	fprintf(stdout, "Parameters:\n");
	fprintf(stdout, "    Sigma = %.4f\n", sigma);
	fprintf(stdout, "    Lambda = %.4f\n", lambda);
	fprintf(stdout, "    Iteration = %d\n", iter);
	fprintf(stdout, "    Attenuation = %.4f\n", attn);
}

// Build LUT for bilateral kernel weight
template <typename T>
void prepareBLFKernel(vector<T>& BLFKernel, const double sigma)
{
	const int max_size_of_filter = 195075;
	BLFKernel.resize(max_size_of_filter);

	for (int m = 0; m < max_size_of_filter; m++) {
		BLFKernel[m] = (T)exp(-sqrt((double)m) / (sigma)); // Kernel LUT
	}
}

template <typename T>
void solve_tridiagonal_in_place_destructive(
	T x[],
	const size_t N,
	const T a[],
	const T b[],
	T c[])
{
	int n;

	c[0] = c[0] / b[0];
	x[0] = x[0] / b[0];

	// loop from 1 to N - 1 inclusive 
	for (n = 1; n < N; n++) {
		T m = (T)(1.0 / (b[n] - a[n] * c[n - 1]));
		c[n] = c[n] * m;
		x[n] = (x[n] - a[n] * x[n - 1]) * m;
	}

	// loop from N - 2 to 0 inclusive 
	for (n = N - 2; n >= 0; n--) {
		x[n] = x[n] - c[n] * x[n + 1];
	}
}

template <typename T>
void copyUcharDataToArr(T* dst, const uchar* src, const int width, const int height, const int channels)
{
	for (int row = 0; row < height; row++)
	{
		const uchar* row_ptr = &src[channels * (row * width)];
		for (int col = 0; col < width; col++)
		{
			int idx = channels * (row * width + col);
			for (int ch = 0; ch < channels; ch++) {
				dst[idx + ch] = (T)row_ptr[channels * col + ch];
			}
		}
	}
}

template <typename T>
void copyUcharDataToVec(vector<T>& dst, const uchar* src, const int width, const int height, const int channels)
{
	dst.resize(width * height * channels);
	for (int row = 0; row < height; row++)
	{
		const uchar* row_ptr = &src[channels * (row * width)];
		for (int col = 0; col < width; col++)
		{
			int idx = channels * (row * width + col);
			for (int ch = 0; ch < channels; ch++) {
				dst[idx + ch] = (T)row_ptr[channels * col + ch];
			}
		}
	}
}

template <typename T>
void copyArrToUcharData(uchar* dst, const T* src, const int width, const int height, const int channels)
{
	for (int row = 0; row < height; row++)
	{
		const T* row_ptr = &src[channels * (row * width)];
		for (int col = 0; col < width; col++)
		{
			int idx = channels * (row * width + col);
			for (int ch = 0; ch < channels; ch++) {
				dst[idx + ch] = static_cast<uchar>(row_ptr[channels * col + ch]);
			}
		}
	}
}

template <typename T>
void copyVecToUcharData(uchar* dst, const vector<T>& src, const int width, const int height, const int channels)
{
	for (int row = 0; row < height; row++)
	{
		const T* row_ptr = &src.data()[channels * (row * width)];
		for (int col = 0; col < width; col++)
		{
			int idx = channels * (row * width + col);
			for (int ch = 0; ch < channels; ch++) {
				dst[idx + ch] = static_cast<uchar>(row_ptr[channels * col + ch]);
			}
		}
	}
}

template <typename T>
void FGS_simple(
	T* image,
	T* joint_image,
	const int width,
	const int height,
	const int nChannels,
	const int nChannels_guide,
	const T sigma,
	const T lambda,
	const int solver_iteration,
	const T solver_attenuation)
{
	int color_diff = 0;

	if (joint_image == nullptr) joint_image = image;

	T *a_vec = new T[width];
	T *b_vec = new T[width];
	T *c_vec = new T[width];
	T *x_vec = new T[width];
	T *c_ori_vec = new T[width];

	T *a2_vec = new T[width];
	T *b2_vec = new T[width];
	T *c2_vec = new T[width];
	T *x2_vec = new T[width];
	T *c2_ori_vec = new T[width];

	std::vector<T> BLFKernelI;
	prepareBLFKernel(BLFKernelI, sigma);

	//Variation of lambda (NEW)
	T lambda_in = (T)1.5*lambda*pow(4.0, solver_iteration - 1) / (pow(4.0, solver_iteration) - 1.0);
	for (int iter = 0; iter < solver_iteration; iter++)
	{
		//for each row
		for (int i = 0; i < height; i++)
		{
			memset(a_vec, 0, sizeof(T)*width);
			memset(b_vec, 0, sizeof(T)*width);
			memset(c_vec, 0, sizeof(T)*width);
			memset(c_ori_vec, 0, sizeof(T)*width);
			memset(x_vec, 0, sizeof(T)*width);
			for (int j = 1; j < width; j++)
			{
				int color_diff = 0;
				// compute bilateral weight for all channels
				for (int c = 0; c < nChannels_guide; c++)
					color_diff += SQ(joint_image[nChannels_guide * (i * width + j) + c] - joint_image[nChannels_guide * (i * width + j - 1) + c]);

				a_vec[j] = -lambda_in * BLFKernelI[color_diff];		//WLS
			}
			for (int j = 0; j < width - 1; j++) {
				c_ori_vec[j] = a_vec[j + 1];
			}
			for (int j = 0; j < width; j++) {
				b_vec[j] = 1.f - a_vec[j] - c_ori_vec[j];		//WLS
			}
			for (int c = 0; c < nChannels; c++)
			{
				memcpy(c_vec, c_ori_vec, sizeof(T)*width);
				for (int j = 0; j < width; j++) {
					x_vec[j] = image[nChannels * (i * width + j) + c];
				}
				solve_tridiagonal_in_place_destructive(x_vec, width, a_vec, b_vec, c_vec);
				for (int j = 0; j < width; j++) {
					image[nChannels * (i * width + j) + c] = x_vec[j];
				}
			}
		}

		//for each column
		for (int j = 0; j < width; j++)
		{
			memset(a2_vec, 0, sizeof(T)*height);
			memset(b2_vec, 0, sizeof(T)*height);
			memset(c2_vec, 0, sizeof(T)*height);
			memset(c2_ori_vec, 0, sizeof(T)*height);
			memset(x2_vec, 0, sizeof(T)*height);
			for (int i = 1; i < height; i++)
			{
				int color_diff = 0;
				// compute bilateral weight for all channels
				for (int c = 0; c < nChannels_guide; c++)
					color_diff += SQ(joint_image[nChannels_guide * (i * width + j) + c] - joint_image[nChannels_guide * ((i - 1) * width + j) + c]);

				a2_vec[i] = -lambda_in * BLFKernelI[color_diff];		//WLS
			}
			for (int i = 0; i < height - 1; i++) {
				c2_ori_vec[i] = a2_vec[i + 1];
			}
			for (int i = 0; i < height; i++) {
				b2_vec[i] = 1.f - a2_vec[i] - c2_ori_vec[i];		//WLS
			}
			for (int c = 0; c < nChannels; c++)
			{
				memcpy(c2_vec, c2_ori_vec, sizeof(T)*height);
				for (int i = 0; i < height; i++) {
					x2_vec[i] = image[nChannels * (i * width + j) + c];
				}
				solve_tridiagonal_in_place_destructive(x2_vec, height, a2_vec, b2_vec, c2_vec);
				for (int i = 0; i < height; i++) {
					image[nChannels * (i * width + j) + c] = x2_vec[i];
				}
			}
		}

		//Variation of lambda (NEW)
		lambda_in /= solver_attenuation;
	}	//iter	
}

void getBinaryData(std::vector<float>& vec, const char* data_path)
{
	FILE* fp = fopen(data_path, "rb");
	int32_t count = vec.size();
	if (data_path != nullptr) {
		fread(&vec[0], sizeof(float), count, fp);
	}
	fclose(fp);
}

void convertDisparityToDepth(Mat &disparityMat, Mat &depthMat)
{
	if (depthMat.empty())
	{
		depthMat = Mat(disparityMat.size(), disparityMat.type());
	}
	depthMat = (1.F / disparityMat) * DEPTH_SCALE + DEPTH_OFFSET;
}

void getGraphAtRow(Mat &input, Mat &output, int row)
{
	std::vector<double> row_vec;
	row_vec.resize(input.cols);

	for (int i = 0; i < input.cols; i++)
	{
		row_vec[i] = static_cast<double>(input.ptr<float>(row)[i]);
	}
	auto axes = CvPlot::plot(row_vec, "-");
	output = axes.render(GRAPH_HEIGHT, GRAPH_WIDTH);
}

int main(int argc, char* argv[])
{
	const int width = 960;
	const int height = 480;
	const char* usage = { "fgs.exe [input data] [guidance data]\n" };
	std::string input_img_path("0.dat");
	std::string guidance_img_path;
	if (argc > 3)
	{
		fprintf(stderr, "Invalid arguments, the number of arguments must be under 3\n");
		fprintf(stdout, "%s", usage);
	}
	else if (argc > 2)
	{
		input_img_path = argv[1];
		guidance_img_path = argv[2];
	}
	else if (argc > 1)
	{
		input_img_path = argv[1];
	}

	// input data
	std::vector<float> input_data;
	input_data.resize(width * height);
	getBinaryData(input_data, input_img_path.data());

	// input image(disparity)
	Mat img = Mat(cv::Size(width, height), CV_32F, input_data.data());
	Mat img_depth, img_depth_graph;
	convertDisparityToDepth(img, img_depth);

	// guide data
	std::vector<float> guidance_data;
	if (guidance_img_path.empty() == false)
	{
		getBinaryData(guidance_data, guidance_img_path.data());
	}

	// guidance image
	Mat guidance_img, guidance_img_depth;
	if (guidance_img_path.empty() == false)
	{
		guidance_img = Mat(cv::Size(width, height), CV_32F, guidance_data.data());
	}

	// output image
	Mat output = Mat(cv::Size(width, height), CV_32F);

	// common attrbs
	int channels = 1;

	// show input image
	cv::imshow("input_disparity", img); // no need to scale, since tensor has value bet 0 ~ 1.0
	cv::imshow("input_depth", img_depth / 150.0); // since our depth max value is 150m
	getGraphAtRow(img_depth, img_depth_graph, SPECIFIC_ROW);
	cv::imshow("input_depth_graph", img_depth_graph);
	cv::waitKey(10);

	// data copy: input data to array
	DATA_TYPE* image_filtered = new DATA_TYPE[height * width * channels]();
	if (img_depth.type() == CV_32F || img_depth.type() == CV_32FC2 || img_depth.type() == CV_32FC3 || img_depth.type() == CV_32FC4) {
		memcpy(image_filtered, img_depth.data, sizeof(float) * width * height * channels);
	}
	else {
		copyUcharDataToArr(image_filtered, img_depth.data, width, height, channels);
	}

	// data copy: guidance data to array
	DATA_TYPE* image_guidance = nullptr;
	if (guidance_img_path.empty() == false)
	{
		convertDisparityToDepth(guidance_img, guidance_img_depth);
		image_guidance = new DATA_TYPE[height * width * channels]();
		if (guidance_img_depth.type() == CV_32F || guidance_img_depth.type() == CV_32FC2 || guidance_img_depth.type() == CV_32FC3 || guidance_img_depth.type() == CV_32FC4) {
			memcpy(image_guidance, guidance_img_depth.data, sizeof(float) * width * height * channels);
		}
		else {
			copyUcharDataToArr(image_guidance, guidance_img_depth.data, width, height, channels);
		}
	}
	
	// params
	DATA_TYPE sigma = 0.015f;
	DATA_TYPE lambda = SQ(5.f);
	int solver_iteration = 2;
	DATA_TYPE solver_attenuation = 10.f; // lambda = lambda / solver_attenuation in every iteration.
	printParams(sigma, lambda, solver_iteration, solver_attenuation);

	// Timer instance
	Timer t;
	t.tic();
	// run FGS
	FGS_simple(
		image_filtered,
		image_guidance,
		width,
		height,
		channels, /* input image channels */
		channels, /* guidance image channels */
		sigma,
		lambda,
		solver_iteration,
		solver_attenuation);
	double elapsed = t.toc();

	// print elapsed time
	fprintf(stdout, "Elapsed time: %.4f ms\n", elapsed * 1000.0);

	// result mat for saving output
	Mat result_depth = Mat(img_depth.size(), img_depth.type());
	Mat result_depth_graph;

	// filtered array to Mat
	if (img_depth.type() == CV_32F || img_depth.type() == CV_32FC2 || img_depth.type() == CV_32FC3 || img_depth.type() == CV_32FC4) {
		memcpy(result_depth.data, image_filtered, sizeof(float) * width * height * channels);
	}
	else {
		copyArrToUcharData(result_depth.data, image_filtered, width, height, channels);
	}

	// show result
	cv::imshow("result_depth", result_depth / 150.0); // since our depth max value is 150m
	getGraphAtRow(result_depth, result_depth_graph, SPECIFIC_ROW);
	cv::imshow("result_depth_graph", result_depth_graph);
	cv::waitKey(0);

	// release memory
	delete[] image_filtered;
	if (guidance_img_path.empty() == false)
	{
		delete[] image_guidance;
	}
	return 0;
}