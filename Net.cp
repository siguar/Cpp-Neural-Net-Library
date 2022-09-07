#include "Net.h"


#include <iostream>
#include <cstdlib>
#include <cassert>
#include <math.h>


Net::Net(const std::vector<unsigned>& topology)
{
	for (int i = 0; i < topology.size(); ++i)
	{
		layers.push_back(Layer());

		unsigned numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

		for (unsigned j = 0; j <= topology[i]; ++j)
		{
			layers.back().push_back(Neuron(numOutputs, j));
			std::cout << "Made A Neuron!" << std::endl;
		}

		layers.back().back().setOutputValue(1.0);
	}
}

void Net::feedForward(const std::vector<double>& data)
{
	assert(data.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < data.size(); ++i)
	{
		layers[0][i].setOutputValue(data[i]);
	}

	for (unsigned layersNum = 1; layersNum < layers.size(); ++layersNum)
	{
		Layer& prevLayer = layers[layersNum - 1];
		for (unsigned j = 0; j < layers[layersNum].size() - 1; ++j)
		{
			layers[layersNum][j].feedForward(prevLayer);
		}
	}
}

void Net::backPropagation(const std::vector<double>& expectedValues)
{
	Layer& outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = expectedValues[n] - outputLayer[n].getOutputValue();
		error += delta * delta;
	}
	///
	error /= outputLayer.size() - 1;
	error = sqrt(error);

	recentAvarageError = (recentAvarageError * recentAvarageSmoothingFactor + error) / (recentAvarageSmoothingFactor + 1.0);

	///
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calculateOutputGradients(expectedValues[n]);
	}

	for (unsigned i = layers.size() - 2; i > 0; --i)
	{
		Layer& hiddenLayer = layers[i];
		Layer& nextLayer = layers[i + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	for (unsigned i = layers.size() - 1; i > 0; --i)
	{
		Layer& layer = layers[i];
		Layer& prevLayer = layers[i - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(std::vector<double>& result)
{
	result.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		result.push_back(layers.back()[n].getOutputValue());
	}
}
