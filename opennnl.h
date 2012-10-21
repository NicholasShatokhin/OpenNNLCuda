#ifndef _OPENNNL_H_
#define _OPENNNL_H_

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "utils.h"
#include "cuda_helper.h"

using namespace std;

typedef enum {LIN, SIG} TActivationKind;

class OpenNNL
{

    private:
        int _layersCount; // num of layers
        int * _neuronsPerLayerCount; // num of neurons in each layer
        int * _deviceNeuronsPerLayerCount; // device // num of neurons in each layer
        int _inputsCount;    // num of network inputs
        int _weightsCount;   // num of weights of all neurons in network
        int _neuronsCount;   // num of all neurons in network (and also num of biases count)
        int _outputsCount;   // num of network outputs

        int * _neuronsInPreviousLayers; // device // the sum of the number of neurons in previous layers
        int * _inputsInPreviousLayers; // the sum of the inputs of each neuron in previous layers
        int * _deviceInputsInPreviousLayers; // device // the sum of the inputs of each neuron in previous layers
        int * _inputsInCurrentLayer; // device // the inputs of each neuron in current layer (not sum)

        float * _neuronsInputsWeights; // device // weights of neurons inputs
        float * _neuronsBiases; // device // biases of neurons

        float * _inputs;    // inputs of network
        float * _outputs;   // outputs of network
        //float * _derivatives; // device // derivatives of output of each neuron

        float * _deviceInputs;  // device
        float * _deviceOutputs; // device

        float * _Bs;    // device // B for IDBD training
        float * _BsForBias;    // device // B for IDBD training
        float * _Hs;    // device // H for IDBD training
        float * _HsForBias;    // device // H for IDBD training

        /*float activation(float output, int type); // activation function
        float derivative(float x, int type);  // derivative of activation function

        inline float sigmoid(float output, float a);
        inline float sigmoid_simple(float output);*/

        void calculateNeuronsOutputsAndDerivatives(float * inputs=NULL, float * outputs=NULL, float * derivatives=NULL); // calculates neurons outputs and derivatives for training functions

        inline void setB(int layer, int neuron, int input, float value);  // set B for current neuron's input
        inline float getB(int layer, int neuron, int input);  // get B of current neuron's input

        inline void setBForBias(int layer, int neuron, float value);  // set B for current neuron's bias
        inline float getBForBias(int layer, int neuron);  // get B of current neuron's bias

        inline void setH(int layer, int neuron, int input, float value); // set H for current neuron's input
        inline float getH(int layer, int neuron, int input);  // get H of current neuron's input

        inline void setHForBias(int layer, int neuron, float value); // set H for current neuron's input
        inline float getHForBias(int layer, int neuron);  // get H of current neuron's input

        inline void setWeight(int layer, int neuron, int input, float value); // set weight for current neuron's input
        inline float getWeight(int layer, int neuron, int input); // get weight current neuron's input

        inline void setBias(int layer, int neuron, float value);  // set bias for current neuron
        inline float getBias(int layer, int neuron);  // get bias of current neuron

        //inline void setDerivative(int layer, int neuron, float value); // sets neuron's derivative value
        //inline float getDerivative(int layer, int neuron); // gets neuron's derivative value

        void resetHs();
        void resetHsForBias();
        void resetHsAndHsForBias();

        void randomizeBs();
        void randomizeBsForBias();
        void randomizeBsAndBsForBias();

        inline int indexByLayerAndNeuron(int layer, int neuron);
        inline int indexByLayerNeuronAndInput(int layer, int neuron, int input);

        inline float activation(float x, TActivationKind kind=SIG);
        inline float activation_derivative(float x, TActivationKind kind=SIG);

        float * _calculateWorker(float * inputs = NULL); // worker for calculation network outputs
        float _changeWeightsByBP(float * trainingInputs, float * trainingOutputs, float speed, float sample_weight=1.0);
        float _changeWeightsByIDBD(float * trainingInputs, float *trainingOutputs, float speed, float sample_weight=1.0);

        bool _doEpochBP(int samplesCount, float * trainingInputs, float * trainingOutputs, int numEpoch, float speed, float error);
        bool _doEpochIDBD(int samplesCount, float * trainingInputs, float * trainingOutputs, int numEpoch, float speed, float error);
        void _trainingBP(int samplesCount, float * trainingInputs, float * trainingOutputs, int maxEpochsCount, float speed, float error);
        void _trainingIDBD(int samplesCount, float * trainingInputs, float * trainingOutputs, int maxEpochsCount, float speed, float error);

        //inline void cudaCall(cudaError_t error, char *file=__FILE__, int line=__LINE__);

    public:
        OpenNNL(const int inptCount, const int lrCount, const int * neuronsInLayer);
        OpenNNL(const char * filename); // creates object and loads network and its parameters from file
        ~OpenNNL();

        void printDebugInfo();
        void randomizeWeights();    // randomizes neurons weights
        void randomizeBiases(); // randomizes neurons biases
        void randomizeWeightsAndBiases();   // randomizes weights and biases

        inline void setInput(int index, float value);  // sets value to input by index
        inline float getOutput(int index); // gets output by index

        void setWeights(float * weights);  // sets neurons weights from argument
        void setWeightsRef(float * weights);  // sets neurons weights by ref in argument (data must be alive while OpenNNL's object lives)
        void setBiases(float * biases);    // sets neurons biases from argument
        void setBiasesRef(float * biases);    // sets neurons biases by ref in argument (data must be alive while OpenNNL's object lives)
        void setWeightsAndBiases(float * weights, float * biases);    // sets neurons weights and biases from argument
        void setWeightsAndBiasesRef(float * weights, float * biases);    // sets neurons weights and biases by refs in arguments (data must be alive while OpenNNL's object lives)

        bool loadWeights(const char * filename);

        void loadNetwork(const char * filename);    // this function loads network and its parameters from file
        void saveNetwork(const char * filename);    // this function stores network and its parameters to file

        float * calculate(float * inputs=NULL);   // calculates network outputs and returns pointer to outputs array (copy 'inputs' data )
        float * calculateRef(float * inputs=NULL);    // calculates network outputs and returns pointer to outputs array (sets internal inputs array by 'inputs' ref - data must be alive while OpenNNL's object lives)

        /*void training(int trainingSetSize, float ** trainingInputs, float **trainingOutputs, float speed, float error, int maxEpochs);
        void trainingByFile(const char * filename, float speed, float error, int maxEpochs);
        void trainingByFileBatch(const char * filename, float speed, float error, int maxEpochs, int batchSize=0, int offset=0);*/

        void trainingBP(int samplesCount, float * trainingInputs, float *trainingOutputs, int maxEpochsCount, float speed, float error);
        void trainingIDBD(int samplesCount, float * trainingInputs, float *trainingOutputs, int maxEpochsCount, float speed, float error);

        void setInputs(float * in);    // copies in to inputs
        void setInputsRef(float * in);    // sets inputs by ref in argument (data must be alive while OpenNNL's object lives)

        void getOutputs(float * out);  // copies network outputs to out
        float * getOutputs();  // returns pointer to outputs array

};

#endif // _OPENNNL_H_
