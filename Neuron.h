#pragma once

#include <vector>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned selfIndex);

	void setOutputValue(const double& value);
	void feedForward(Layer &prevLayer);
	double getOutputValue();
	void calculateOutputGradients(const double& expectedValue);
	void calculateHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	static double randomWeight();

	double transferFunction(const double& value);
	double transferFunctionDerivative(const double& value);
	double sumDOW(const Layer& nextLayer) const;

	double outputValue;
	std::vector<Connection> outputWeights;
	unsigned selfIndex = 0;

	double eta = 0.10;
	double alpha = 0.5;

	double gradient = 0.0;
};

