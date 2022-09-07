#pragma once

#include<vector>
#include "Neuron.h"


class Net
{
public:
	Net(const std::vector<unsigned>& topology);

	void feedForward(const std::vector<double>& data);
	void backPropagation(const std::vector<double>& expectedValues);
	void getResults(std::vector<double>& result);

	double getRecentAverageError()
	{
		return recentAvarageError;
	}
private:
	std::vector<Layer> layers;

	double error = 0.0;

	double recentAvarageError = 0.0;
	double recentAvarageSmoothingFactor = 0.0;
};

