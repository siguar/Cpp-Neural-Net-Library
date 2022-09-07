#include "Neuron.h"
#include "Net.h"
#include <cmath>
#include <iostream>

Neuron::Neuron(unsigned numOutputs, unsigned selfIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	this->selfIndex = selfIndex;
}

void Neuron::setOutputValue(const double& value)
{
	outputValue = value;
}

void Neuron::feedForward(Layer& prevLayer) 
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[selfIndex].weight;
	}

	outputValue = transferFunction(sum);
}

double Neuron::getOutputValue()
{
	return outputValue;
}

void Neuron::calculateOutputGradients(const double& expectedValue)
{
	double delta = expectedValue - outputValue;
	gradient = delta * transferFunctionDerivative(outputValue);
}

void Neuron::calculateHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * transferFunctionDerivative(outputValue);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];

		double oldDeltaWeight = neuron.outputWeights[selfIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputValue() * gradient + alpha * oldDeltaWeight;

		neuron.outputWeights[selfIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[selfIndex].weight += newDeltaWeight;
	}
}

double Neuron::randomWeight()
{
	return rand() / double(RAND_MAX);
}

double Neuron::transferFunction(const double& value)
{
	return tanh(value);
}

double Neuron::transferFunctionDerivative(const double& value)
{
	return 1 - tanh(value) * tanh(value);
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() -1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}
	return sum;
}
