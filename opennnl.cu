#include "opennnl.h"

__device__ float activation(float x)
{
    return 1 / (1 + exp((-1)*x));
}

__global__ void weighting(float * outputs, float * inputs, float * weights, int start, int inputsCount, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < inputsCount * neuronsCount)
    {
        outputs[idx] = inputs[idx%inputsCount] * weights[start+idx];
    }
}

__global__ void calculateOut(float * outputs, float * inputs, int inputsCount, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        float temp = 0;

        for(int k=0;k<inputsCount;k++)
        {
            temp += inputs[idx*inputsCount+k];
        }

        outputs[idx] = activation(temp);
    }
}

#define cudaCall(x) { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE);}}

OpenNNL::OpenNNL(const int inputsCount, const int layersCount, const int * neuronsPerLayerCount)
{
    _inputsCount = inputsCount;
    _layersCount = layersCount;
    _weightsCount = 0;
    _neuronsCount = 0;

    int * _hostNeuronsPerLayerCount = new int[_layersCount];
    int * _hostNeuronsInPreviousLayers = new int[_layersCount];
    int * _hostInputsInPreviousLayers = new int[_layersCount];
    int * _hostInputsInCurrentLayer = new int[_layersCount];

    cudaCall(cudaMalloc(&_neuronsPerLayerCount, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_neuronsInPreviousLayers, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_inputsInPreviousLayers, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_inputsInCurrentLayer, _layersCount*sizeof(int)));

    _inputs = new float[_inputsCount];

    cudaCall(cudaMalloc(&_deviceInputs, _inputsCount*sizeof(float)));

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        _hostNeuronsInPreviousLayers[i] = _neuronsCount;
        _hostInputsInPreviousLayers[i] = _weightsCount;

        _hostInputsInCurrentLayer[i] = inputs;

        _weightsCount += neuronsPerLayerCount[i] * inputs;
        _neuronsCount += neuronsPerLayerCount[i];

        inputs = _hostNeuronsPerLayerCount[i] = neuronsPerLayerCount[i];
    }

    _outputsCount = inputs;
    _outputs = new float[_outputsCount];

    cudaCall(cudaMalloc(&_deviceOutputs, _outputsCount*sizeof(float)));

    cudaCall(cudaMemcpy(_neuronsPerLayerCount, _hostNeuronsPerLayerCount, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_neuronsInPreviousLayers, _hostNeuronsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_inputsInPreviousLayers, _hostInputsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_inputsInCurrentLayer, _hostInputsInCurrentLayer, _layersCount*sizeof(int), cudaMemcpyHostToDevice));

    delete _hostNeuronsPerLayerCount;
    delete _hostNeuronsInPreviousLayers;
    delete _hostInputsInPreviousLayers;
    delete _hostInputsInCurrentLayer;
    //_derivatives = new float[_neuronsCount];

    //_neuronsInputsWeights = new float[_weightsCount];
    //_neuronsBiases = new float[_neuronsCount];
    cudaCall(cudaMalloc(&_neuronsInputsWeights, _weightsCount*sizeof(float)));
    cudaCall(cudaMalloc(&_neuronsBiases, _neuronsCount*sizeof(float)));
}

OpenNNL::~OpenNNL()
{
    //delete[] _neuronsPerLayerCount;
    //delete[] _neuronsInPreviousLayers;
    //delete[] _inputsInPreviousLayers;
    //delete[] _inputsInCurrentLayer;
    cudaCall(cudaFree(_neuronsPerLayerCount));
    cudaCall(cudaFree(_neuronsInPreviousLayers));
    cudaCall(cudaFree(_inputsInPreviousLayers));
    cudaCall(cudaFree(_inputsInCurrentLayer));
    delete[] _inputs;
    delete[] _outputs;
    cudaCall(cudaFree(_deviceInputs));
    cudaCall(cudaFree(_deviceOutputs));
    //delete[] _derivatives;
    //delete[] _neuronsInputsWeights;
    //delete[] _neuronsBiases;
    cudaCall(cudaFree(_neuronsInputsWeights));
    cudaCall(cudaFree(_neuronsBiases));
}

void OpenNNL::printDebugInfo()
{
    /*printf("inputsCount=%d\n", _inputsCount);
    printf("outputsCount=%d\n", _outputsCount);
    printf("layersCount=%d\n", _layersCount);
    printf("neuronsCount=%d\n", _neuronsCount);
    printf("weightsCount=%d\n", _weightsCount);

    for(int i=0;i<_layersCount;i++)
    {
        printf("neurons in layer %d: %d\n", i, _neuronsPerLayerCount[i]);
        printf("neurons in all layers before %d: %d\n", i, _neuronsInPreviousLayers[i]);
        printf("inputs in all layers before %d: %d\n", i, _inputsInPreviousLayers[i]);
        printf("inputs of each neuron in layer %d: %d\n", i, _inputsInCurrentLayer[i]);
    }*/
}

inline int OpenNNL::indexByLayerAndNeuron(int layer, int neuron)
{
    return _neuronsInPreviousLayers[layer] + neuron;
}

inline int OpenNNL::indexByLayerNeuronAndInput(int layer, int neuron, int input)
{
    return _inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input;
}

inline void OpenNNL::setB(int layer, int neuron, int input, float value)
{
    _Bs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline float OpenNNL::getB(int layer, int neuron, int input)
{
    return _Bs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBForBias(int layer, int neuron, float value)
{
    _BsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline float OpenNNL::getBForBias(int layer, int neuron)
{
    return _BsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setH(int layer, int neuron, int input, float value)
{
    _Hs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline float OpenNNL::getH(int layer, int neuron, int input)
{
    return _Hs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setHForBias(int layer, int neuron, float value)
{
    _HsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline float OpenNNL::getHForBias(int layer, int neuron)
{
    return _HsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setWeight(int layer, int neuron, int input, float value)
{
    _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline float OpenNNL::getWeight(int layer, int neuron, int input)
{
    return _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBias(int layer, int neuron, float value)
{
    _neuronsBiases[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline float OpenNNL::getBias(int layer, int neuron)
{
    return _neuronsBiases[indexByLayerAndNeuron(layer, neuron)];
}

/*inline void OpenNNL::setDerivative(int layer, int neuron, float value)
{
    _derivatives[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline float OpenNNL::getDerivative(int layer, int neuron)
{
    return _derivatives[indexByLayerAndNeuron(layer, neuron)];
}*/

inline void OpenNNL::setInput(int index, float value)
{
    _inputs[index] = value;
}

inline float OpenNNL::getOutput(int index)
{
    return _outputs[index];
}

__global__ void _cudaRandomizeWeights(float * neuronsInputsWeights, int weightsCount, int inputsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < weightsCount)
    {
        neuronsInputsWeights[idx] = /*unified_random() /*/ sqrtf(inputsCount);
    }
}

void OpenNNL::randomizeWeights()
{
    /*initialize_random_generator();

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<inputs*_neuronsPerLayerCount[i];j++)
        {
            _neuronsInputsWeights[_inputsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _neuronsPerLayerCount[i];
    }*/

}

void OpenNNL::randomizeBiases()
{
    /*initialize_random_generator();

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            _neuronsBiases[_neuronsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _neuronsPerLayerCount[i];
    }*/
}

void OpenNNL::randomizeWeightsAndBiases()
{
    this->randomizeWeights();
    this->randomizeBiases();
}

/*****************************************************************************/
/* Вычислить активационную функцию y(x) = 2x / (1 + abs(x)). */
/*****************************************************************************/
inline float OpenNNL::activation(float x, TActivationKind kind)
{
    return ((kind == SIG) ? (2.0 * x / (1 + fabs(x))):x);
}

/*****************************************************************************/
/* Вычислить производную активационной функции y(x) по формуле:
   dy(x)         2.0
   ----- = ---------------.
    dx     (1 + abs(x))^2
*/
/*****************************************************************************/
inline float OpenNNL::activation_derivative(float x, TActivationKind kind)
{
    float temp = 1.0 + fabs(x);
    return ((kind == SIG) ? (2.0 / (temp * temp)):1.0);
}

float * OpenNNL::_calculateWorker(float *inpt)
{
    int inputsCount;
    float * temp;
    float * inputs = new float[_inputsCount];

    memcpy(inputs, inpt, sizeof(float)*_inputsCount);

    inputsCount = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        temp = new float[_neuronsPerLayerCount[i]*inputsCount];
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<inputsCount;k++)
            {
            temp[j*inputsCount+k] = inputs[k] * _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)];
            }
        }

        delete[] inputs;

        inputs = new float[_neuronsPerLayerCount[i]];

        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            inputs[j] = 0;

            for(int k=0;k<inputsCount;k++)
            {
                inputs[j] += temp[j*inputsCount+k];
            }

            inputs[j]  -= _neuronsBiases[indexByLayerAndNeuron(i, j)];

            inputs[j] = activation(inputs[j]);

        }

        inputsCount = _neuronsPerLayerCount[i];
        delete[] temp;
    }

    memcpy(_outputs, inputs, sizeof(float)*inputsCount);

    delete[] inputs;

    return _outputs;
}

float * OpenNNL::calculate(float *inputs)
{
    if(inputs)
    {
        memcpy(_inputs, inputs, _inputsCount*sizeof(float));
    }

    return _calculateWorker(_inputs);
}

float * OpenNNL::calculateRef(float *inputs)
{
    if(!inputs)
        inputs = _inputs;

    return _calculateWorker(inputs);
}

void OpenNNL::calculateNeuronsOutputsAndDerivatives(float *inpt, float *outputs, float *derivatives)
{
    int inputsCount, neuronIndex = 0;
    float * temp;
    float * inputs = new float[_inputsCount];

    memcpy(inputs, inpt, sizeof(float)*_inputsCount);

    inputsCount = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        temp = new float[_neuronsPerLayerCount[i]*inputsCount];
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<inputsCount;k++)
            {
                temp[j*inputsCount+k] = inputs[k] * _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)];
            }
        }

        delete[] inputs;

        inputs = new float[_neuronsPerLayerCount[i]];

        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            inputs[j] = 0;

            for(int k=0;k<inputsCount;k++)
            {
                inputs[j] += temp[j*inputsCount+k];
            }

            inputs[j] -= _neuronsBiases[indexByLayerAndNeuron(i, j)];

            outputs[neuronIndex] = inputs[j] = activation(inputs[j]);
            derivatives[neuronIndex] = activation_derivative(inputs[j]);

            neuronIndex++;
        }

        inputsCount = _neuronsPerLayerCount[i];

        delete[] temp;
    }

    delete[] inputs;
}

float OpenNNL::_changeWeightsByBP(float * trainingInputs, float *trainingOutputs, float speed, float sample_weight)
{
    float error = 0, current_error;
    float * localGradients = new float[_neuronsCount];
    float * outputs = new float[_neuronsCount];
    float * derivatives = new float[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_neuronsPerLayerCount[_layersCount-1];j++) // cuda kernel
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            for(int j=0;j<_neuronsPerLayerCount[i];j++) // cuda kernel
            {
                localGradients[indexByLayerAndNeuron(i, j)] = 0;

                for(int k=0;k<_neuronsPerLayerCount[i+1];k++)
                {
                    localGradients[indexByLayerAndNeuron(i, j)] += _neuronsInputsWeights[indexByLayerNeuronAndInput(i+1, k, j)]
                                                                    * localGradients[indexByLayerAndNeuron(i+1, k)];
                }

                localGradients[indexByLayerAndNeuron(i, j)] *= derivatives[indexByLayerAndNeuron(i, j)];
            }
        }
    }

    for(int j=0;j<_neuronsPerLayerCount[0];j++) // this and next cicle for cuda kernel (j*k threads)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += speed * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k];
        }

        _neuronsBiases[indexByLayerAndNeuron(0, j)] -= speed * localGradients[indexByLayerAndNeuron(0, j)];
    }

    for(int i=1;i<_layersCount;i++) // try to parallelize all three cicles in one kernel. If it's impossible, only two inner
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_neuronsPerLayerCount[i-1];k++)
            {
                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += speed * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];
            }

            _neuronsBiases[indexByLayerAndNeuron(i, j)] -= speed * localGradients[indexByLayerAndNeuron(i, j)];
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;

    error /= 2;
    return error;
}

float OpenNNL::_changeWeightsByIDBD(float * trainingInputs, float *trainingOutputs, float speed, float sample_weight)
{
    float error = 0, current_error;
    float cur_rate, delta, deltaB, deltaH;
    float * localGradients = new float[_neuronsCount];
    float * outputs = new float[_neuronsCount];
    float * derivatives = new float[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_neuronsPerLayerCount[_layersCount-1];j++)
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            for(int j=0;j<_neuronsPerLayerCount[i];j++)
            {
                localGradients[indexByLayerAndNeuron(i, j)] = 0;

                for(int k=0;k<_neuronsPerLayerCount[i+1];k++)
                {
                    localGradients[indexByLayerAndNeuron(i, j)] += _neuronsInputsWeights[indexByLayerNeuronAndInput(i+1, k, j)]
                                                                    * localGradients[indexByLayerAndNeuron(i+1, k)];
                }

                localGradients[indexByLayerAndNeuron(i, j)] *= derivatives[indexByLayerAndNeuron(i, j)];
            }
        }
    }

    for(int j=0;j<_neuronsPerLayerCount[0];j++)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            deltaB = speed * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k] * getH(0, j, k);

            if (deltaB > 2.0)
            {
                deltaB = 2.0;
            }
            else
            {
                if (deltaB < -2.0)
                {
                    deltaB = -2.0;
                }
            }

            setB(0, j, k, getB(0, j, k) + deltaB);
            cur_rate = exp(getB(0, j, k));
            delta = cur_rate * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k];

            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += delta;

            deltaH = 1 - cur_rate * trainingInputs[k] * trainingInputs[k];
            if(deltaH <= 0)
                setH(0, j, k, delta);
            else
                setH(0, j, k, getH(0, j, k) * deltaH + delta);
        }

        deltaB = speed * localGradients[indexByLayerAndNeuron(0, j)] * getHForBias(0, j);

        if (deltaB > 2.0)
        {
            deltaB = 2.0;
        }
        else
        {
            if (deltaB < -2.0)
            {
                deltaB = -2.0;
            }
        }

        setBForBias(0, j, getBForBias(0, j) - deltaB);
        cur_rate = exp(getBForBias(0, j));
        delta = cur_rate * localGradients[indexByLayerAndNeuron(0, j)];

        _neuronsBiases[indexByLayerAndNeuron(0, j)] -= delta;

        deltaH = 1 - cur_rate;
        if(deltaH <= 0)
            setHForBias(0, j, -delta);
        else
            setHForBias(0, j, getHForBias(0, j) * deltaH - delta);
    }

    for(int i=1;i<_layersCount;i++)
    {
        for(int j=0;j<_neuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_neuronsPerLayerCount[i-1];k++)
            {
                deltaB = speed * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)] * getH(i, j, k);

                if (deltaB > 2.0)
                {
                    deltaB = 2.0;
                }
                else
                {
                    if (deltaB < -2.0)
                    {
                        deltaB = -2.0;
                    }
                }

                setB(i, j, k, getB(i, j, k) + deltaB);
                cur_rate = exp(getB(i, j, k));
                delta = cur_rate * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];

                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += delta;

                deltaH = 1 - cur_rate * outputs[indexByLayerAndNeuron(i-1, k)] * outputs[indexByLayerAndNeuron(i-1, k)];
                if(deltaH <= 0)
                    setH(i, j, k, delta);
                else
                    setH(i, j, k, getH(i, j, k) * deltaH + delta);
            }

            deltaB = speed * localGradients[indexByLayerAndNeuron(i, j)] * getHForBias(i, j);

            if (deltaB > 2.0)
            {
                deltaB = 2.0;
            }
            else
            {
                if (deltaB < -2.0)
                {
                    deltaB = -2.0;
                }
            }

            setBForBias(i, j, getBForBias(i, j) - deltaB);
            cur_rate = exp(getBForBias(i, j));
            delta = cur_rate * localGradients[indexByLayerAndNeuron(i, j)];

            _neuronsBiases[indexByLayerAndNeuron(i, j)] -= delta;

            deltaH = 1 - cur_rate;
            if(deltaH <= 0)
                setHForBias(i, j, -delta);
            else
                setHForBias(i, j, getHForBias(i, j) * deltaH - delta);
        }
    }

    delete[] localGradients;
    delete[] outputs;
    delete[] derivatives;

    error /= 2;
    return error;
}

bool OpenNNL::_doEpochBP(int samplesCount, float * trainingInputs, float * trainingOutputs, int numEpoch, float speed, float minError)
{
    float error = 0;
    float * currentSampleInputs = new float[_inputsCount];
    float * currentSampleOutputs = new float[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Epoch: " << numEpoch << ", Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(float));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(float));

        error = _changeWeightsByBP(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

bool OpenNNL::_doEpochIDBD(int samplesCount, float * trainingInputs, float * trainingOutputs, int numEpoch, float speed, float minError)
{
    float error = 0;
    float * currentSampleInputs = new float[_inputsCount];
    float * currentSampleOutputs = new float[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(float));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(float));

        error = _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

void OpenNNL::_trainingBP(int samplesCount, float * trainingInputs, float * trainingOutputs, int maxEpochsCount, float speed, float error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochBP(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}


void OpenNNL::_trainingIDBD(int samplesCount, float * trainingInputs, float * trainingOutputs, int maxEpochsCount, float speed, float error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochIDBD(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}

void OpenNNL::trainingBP(int samplesCount, float * trainingInputs, float *trainingOutputs, int maxEpochsCount, float speed, float error)
{
    _trainingBP(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);
}

void OpenNNL::trainingIDBD(int samplesCount, float * trainingInputs, float *trainingOutputs, int maxEpochsCount, float speed, float error)
{
    _Bs = new float[_weightsCount];
    _Hs = new float[_weightsCount];

    _BsForBias = new float[_neuronsCount];
    _HsForBias = new float[_neuronsCount];

    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingIDBD(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);

    delete[] _Bs;
    delete[] _Hs;
    delete[] _BsForBias;
    delete[] _HsForBias;
}

void OpenNNL::getOutputs(float * out)
{
    memcpy(out, _outputs, sizeof(float)*_outputsCount);
}

void OpenNNL::resetHs()
{
    for(int i=0;i<_weightsCount;i++)
        _Hs[i] = 0;
}

void OpenNNL::resetHsForBias()
{
    for(int i=0;i<_neuronsCount;i++)
        _HsForBias[i] = 0;
}

void OpenNNL::resetHsAndHsForBias()
{
    resetHs();
    resetHsForBias();
}

void OpenNNL::randomizeBs()
{
    /*initialize_random_generator();
    for(int i=0;i<_weightsCount;i++)
        _Bs[i] = unified_random();*/
}

void OpenNNL::randomizeBsForBias()
{
    /*initialize_random_generator();
    for(int i=0;i<_neuronsCount;i++)
        _BsForBias[i] = unified_random();*/
}

void OpenNNL::randomizeBsAndBsForBias()
{
    randomizeBs();
    randomizeBsForBias();
}
