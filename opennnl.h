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

#include "LittleBigEndian.h"

using namespace std;

#define BLOCK_SIZE 256

#define REAL float

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

        int _maxLayerSize;
        int _maxInputsCount;

        int * _neuronsInPreviousLayers; // the sum of the number of neurons in previous layers
        int * _deviceNeuronsInPreviousLayers; // device
        int * _inputsInPreviousLayers; // the sum of the inputs of each neuron in previous layers
        int * _deviceInputsInPreviousLayers; // device // the sum of the inputs of each neuron in previous layers
        int * _inputsInCurrentLayer; // the inputs of each neuron in current layer (not sum)
        int * _deviceInputsInCurrentLayer; // device

        REAL * _neuronsInputsWeights; // device // weights of neurons inputs
        REAL * _neuronsBiases; // device // biases of neurons

        REAL * _inputs;    // inputs of network
        REAL * _outputs;   // outputs of network
        //REAL * _derivatives; // device // derivatives of output of each neuron

        REAL * _deviceInputs;  // device
        REAL * _deviceOutputs; // device

        REAL * _layerInputs;    // temp variable for output calculation
        REAL * _weightedLayerInputs;    // temp variable for output calculation

        REAL * _Bs;    // device // B for IDBD training
        REAL * _BsForBias;    // device // B for IDBD training
        REAL * _Hs;    // device // H for IDBD training
        REAL * _HsForBias;    // device // H for IDBD training

        /*REAL activation(REAL output, int type); // activation function
        REAL derivative(REAL x, int type);  // derivative of activation function

        inline REAL sigmoid(REAL output, REAL a);
        inline REAL sigmoid_simple(REAL output);*/

        void createNetwork(const int inputsCount, const int layersCount, const int * neuronsPerLayerCount);
        void destroyNetwork();

        void load(const char * filename);
        void save(const char * filename);

        void calculateNeuronsOutputsAndDerivatives(REAL * trainingInputs, REAL * deviceOutputs, REAL * deviceDerivatives); // calculates neurons outputs and derivatives for training functions

        inline void setB(int layer, int neuron, int input, REAL value);  // set B for current neuron's input
        inline REAL getB(int layer, int neuron, int input);  // get B of current neuron's input

        inline void setBForBias(int layer, int neuron, REAL value);  // set B for current neuron's bias
        inline REAL getBForBias(int layer, int neuron);  // get B of current neuron's bias

        inline void setH(int layer, int neuron, int input, REAL value); // set H for current neuron's input
        inline REAL getH(int layer, int neuron, int input);  // get H of current neuron's input

        inline void setHForBias(int layer, int neuron, REAL value); // set H for current neuron's input
        inline REAL getHForBias(int layer, int neuron);  // get H of current neuron's input

        inline void setWeight(int layer, int neuron, int input, REAL value); // set weight for current neuron's input
        inline REAL getWeight(int layer, int neuron, int input); // get weight current neuron's input

        inline void setBias(int layer, int neuron, REAL value);  // set bias for current neuron
        inline REAL getBias(int layer, int neuron);  // get bias of current neuron

        //inline void setDerivative(int layer, int neuron, REAL value); // sets neuron's derivative value
        //inline REAL getDerivative(int layer, int neuron); // gets neuron's derivative value

        void resetHs();
        void resetHsForBias();
        void resetHsAndHsForBias();

        void randomizeBs();
        void randomizeBsForBias();
        void randomizeBsAndBsForBias();

        inline int indexByLayerAndNeuron(int layer, int neuron);
        inline int indexByLayerNeuronAndInput(int layer, int neuron, int input);

        //inline REAL activation(REAL x, TActivationKind kind=SIG);
        //inline REAL activation_derivative(REAL x, TActivationKind kind=SIG);

        REAL * _calculateSingle(REAL * inputs); // worker for calculation network outputs
        void _doCalculation(REAL * inputs, REAL * outputs);
        REAL _changeWeightsByBP(REAL * deviceTrainingInputs, REAL * deviceTrainingOutputs, REAL * deviceOutputs, REAL * deviceDerivatives, REAL * deviceLocalGradients, REAL * deviceErrors[], REAL speed, REAL sample_weight=1.0);
        REAL _changeWeightsByIDBD(REAL * trainingInputs, REAL *trainingOutputs, REAL speed, REAL sample_weight=1.0);

        bool _doEpochBP(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, REAL * deviceOutputs, REAL * deviceDerivatives, REAL * deviceLocalGradients, REAL * deviceErrors[], int numEpoch, REAL speed, REAL minError);
        bool _doEpochIDBD(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int numEpoch, REAL speed, REAL error);
        void _trainingBP(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int maxEpochsCount, REAL speed, REAL error);
        void _trainingIDBD(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int maxEpochsCount, REAL speed, REAL error);

        //inline void cudaCall(cudaError_t error, char *file=__FILE__, int line=__LINE__);

    public:
        OpenNNL(const int inptCount, const int lrCount, const int * neuronsInLayer);
        OpenNNL(const char * filename); // creates object and loads network and its parameters from file
        ~OpenNNL();

        void printDebugInfo();
        void randomizeWeights();    // randomizes neurons weights
        void randomizeBiases(); // randomizes neurons biases
        void randomizeWeightsAndBiases();   // randomizes weights and biases

        inline void setInput(int index, REAL value);  // sets value to input by index
        inline REAL getOutput(int index); // gets output by index

        void setWeights(REAL * weights);  // sets neurons weights from argument
        void setWeightsRef(REAL * weights);  // sets neurons weights by ref in argument (data must be alive while OpenNNL's object lives)
        void setBiases(REAL * biases);    // sets neurons biases from argument
        void setBiasesRef(REAL * biases);    // sets neurons biases by ref in argument (data must be alive while OpenNNL's object lives)
        void setWeightsAndBiases(REAL * weights, REAL * biases);    // sets neurons weights and biases from argument
        void setWeightsAndBiasesRef(REAL * weights, REAL * biases);    // sets neurons weights and biases by refs in arguments (data must be alive while OpenNNL's object lives)

        bool loadWeights(const char * filename);

        void loadNetwork(const char * filename);    // this function loads network and its parameters from file
        void saveNetwork(const char * filename);    // this function stores network and its parameters to file

        REAL * calculate(REAL * inputs=NULL);   // calculates network outputs and returns pointer to outputs array (copy 'inputs' data )
        REAL * calculateRef(REAL * inputs=NULL);    // calculates network outputs and returns pointer to outputs array (sets internal inputs array by 'inputs' ref - data must be alive while OpenNNL's object lives)

        void calculate(REAL * inputs, REAL * outputs, int samplesCount);

        /*void training(int trainingSetSize, REAL ** trainingInputs, REAL **trainingOutputs, REAL speed, REAL error, int maxEpochs);
        void trainingByFile(const char * filename, REAL speed, REAL error, int maxEpochs);
        void trainingByFileBatch(const char * filename, REAL speed, REAL error, int maxEpochs, int batchSize=0, int offset=0);*/

        void trainingBP(int samplesCount, REAL * trainingInputs, REAL *trainingOutputs, int maxEpochsCount, REAL speed, REAL error);
        void trainingIDBD(int samplesCount, REAL * trainingInputs, REAL *trainingOutputs, int maxEpochsCount, REAL speed, REAL error);

        void setInputs(REAL * in);    // copies in to inputs
        void setInputsRef(REAL * in);    // sets inputs by ref in argument (data must be alive while OpenNNL's object lives)

        void getOutputs(REAL * out);  // copies network outputs to out
        REAL * getOutputs();  // returns pointer to outputs array

};

#endif // _OPENNNL_H_
