#include "Plotters.h"

int main(int argc, char* argv[]) {
	std::string FileName = "Mandelbrot_Plot";
	uint64_t rez = 1920;
	double RE_Start = -2,
		RE_End = 1,
		IM_Start = -1,
		IM_End = 1;

	if (argc <= 1) {
		std::cout << "No args passed, defaulting to defaults..." << std::endl;
		PlotSet_DPCPP(FileName, rez, RE_Start, RE_End, IM_Start, IM_End);
	}
	if (argc == 2) {
		if (argv[1] == (std::string)"DPCPP") PlotSet_DPCPP(FileName, rez, RE_Start, RE_End, IM_Start, IM_End);

		else if (argv[1] == (std::string)"CPU") PlotSet_CPU(FileName, rez, RE_Start, RE_End, IM_Start, IM_End);

		else std::cout << "Incorrect argument passed." << std::endl; return 0;
	}
	return 0;
}