#include "Plotters.h"

void PlotSet_CPU(std::string FileName, uint64_t rez, double RE_Start, double RE_End, double IM_Start, double IM_End) {
	int n = 0;
	std::complex<float> z = 0,
		c = 0;

	std::cout << "Creating Image object." << std::endl;
	cv::Mat image(rez, rez, CV_8UC3);

	std::cout << "Image object created." << std::endl;

	uint8_t* pixelptr = (uint8_t*)image.data;

	std::cout << "Looping through the mandelbrot set." << std::endl;
	for (uint64_t i = 0; i < image.rows; i++) {
		for (uint64_t j = 0; j < image.cols; j++) {
			// complex number things
			z = 0;
			c = std::complex(RE_Start + ((float)i / image.cols) * (RE_End - RE_Start),
				IM_Start + ((float)j / image.rows) * (IM_End - IM_Start));
			n = 0;


			while (abs(z) <= 2 && n < MAX_ITER) {
				z = z * z + c;
				n++;
			}

			pixelptr[i * image.cols * 3 + j * 3 + 0] = 255 * ((double)n / MAX_ITER) * 2;
			pixelptr[i * image.cols * 3 + j * 3 + 1] = 255 * ((double)n / MAX_ITER);
			pixelptr[i * image.cols * 3 + j * 3 + 2] = 255 * ((double)n / MAX_ITER);
		}
	}
	std::cout << "Loop through the set complete." << std::endl;

	std::string FileWriteName = "./" + FileName + ".png";
	bool check = cv::imwrite(FileWriteName, image);

	if (!check) {
		std::cout << "Shit, something went wrong." << std::endl;
	}

	std::cout << "Image written?" << std::endl;
};