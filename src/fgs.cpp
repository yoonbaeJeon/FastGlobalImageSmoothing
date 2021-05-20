#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using std::vector;
using std::chrono::system_clock;
using std::chrono::duration;

#ifndef uchar
typedef unsigned char uchar;
#endif

#define SQ(x) ((x)*(x))
#define DATA_TYPE float

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

int main(int argc, char* argv[])
{
	const char* usage = { "fgs.exe [input image path] [guidance image path]\n" };
	std::string input_img_path("input.exr");
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

	// input image
	Mat img = imread(input_img_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

	// guide image
	Mat guidance_img;
	if (guidance_img_path.empty() == false)
	{
		guidance_img = imread(guidance_img_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
		if (img.cols != guidance_img.cols || img.rows != guidance_img.rows || img.channels() != guidance_img.channels())
		{
			fprintf(stderr, "Invalid guidance image, the size of guidance image must match with that of the input image\n");
			return -1;
		}
	}

	// output image
	Mat output = Mat(img.size(), img.type());

	// common attrbs
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	// show input image
	cv::imshow("input", img / 150.0); // since our depth max val is 150m
	cv::waitKey(10);

	// data copy: input Mat to array
	DATA_TYPE* image_filtered = new DATA_TYPE[height * width * channels]();
	if (img.type() == CV_32F || img.type() == CV_32FC2 || img.type() == CV_32FC3 || img.type() == CV_32FC4) {
		memcpy(image_filtered, img.data, sizeof(float) * width * height * channels);
	}
	else {
		copyUcharDataToArr(image_filtered, img.data, width, height, channels);
	}

	// data copy: guidance Mat to array
	DATA_TYPE* image_guidance = nullptr;
	if (guidance_img_path.empty() == false)
	{
		image_guidance = new DATA_TYPE[height * width * channels]();
		if (guidance_img.type() == CV_32F || guidance_img.type() == CV_32FC2 || guidance_img.type() == CV_32FC3 || guidance_img.type() == CV_32FC4) {
			memcpy(image_guidance, guidance_img.data, sizeof(float) * width * height * channels);
		}
		else {
			copyUcharDataToArr(image_guidance, guidance_img.data, width, height, channels);
		}
	}
	
	// params
	DATA_TYPE sigma = 0.035f;
	DATA_TYPE lambda = SQ(10.f);
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

	// filtered array to Mat
	if (img.type() == CV_32F || img.type() == CV_32FC2 || img.type() == CV_32FC3 || img.type() == CV_32FC4) {
		memcpy(img.data, image_filtered, sizeof(float) * width * height * channels);
	}
	else {
		copyArrToUcharData(img.data, image_filtered, width, height, channels);
	}

	// show result
	cv::imshow("result", img / 150.0); // since our depth max val is 150m
	cv::waitKey(0);

	// release memory
	delete[] image_filtered;
	if (guidance_img_path.empty() == false)
	{
		delete[] image_guidance;
	}
	return 0;
}