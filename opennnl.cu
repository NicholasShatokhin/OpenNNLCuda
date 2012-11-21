#include "opennnl.h"

/*****************************************************************************/
/* Вычислить активационную функцию y(x) = 2x / (1 + abs(x)). */
/*****************************************************************************/
//inline REAL OpenNNL::activation(REAL x, TActivationKind kind)
__device__ REAL activation(REAL x)
{
    return (2.0 * x / (1 + fabs(x)));
}

/*****************************************************************************/
/* Вычислить производную активационной функции y(x) по формуле:
   dy(x)         2.0
   ----- = ---------------.
    dx     (1 + abs(x))^2
*/
/*****************************************************************************/
//inline REAL OpenNNL::activation_derivative(REAL x, TActivationKind kind)
__device__ REAL activation_derivative(REAL x)
{
    REAL temp = 1.0 + fabs(x);
    return (2.0 / (temp * temp));
}

__global__ void testKernel(REAL * output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
    {
        output[idx] = 5;
    }
}

__global__ void weighting(REAL * outputs, REAL * inputs, REAL * weights, int start, int inputsCount, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < inputsCount * neuronsCount)
    {
        outputs[idx] = inputs[idx%inputsCount] * weights[start+idx];
    }
}

__global__ void calculateOut(REAL * outputs, REAL * inputs, REAL * biases, int inputsCount, int neuronsCount, int neuronsInPreviousLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        REAL temp = 0;

        for(int k=0;k<inputsCount;k++)
        {
            temp += inputs[idx*inputsCount+k];
        }

        temp -= biases[neuronsInPreviousLayers+idx];

        outputs[idx] = activation(temp);
    }
}

__global__ void copyData(REAL * output, REAL * input, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
    {
        output[idx] = input[idx];
    }
}

__global__ void initializeRandomGenerator ( curandState * state, unsigned long seed, int count )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
        curand_init ( seed, idx, 0, &state[idx] );
}

__global__ void generateRandomArray( curandState* globalState, REAL * randomArray, int count )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < count)
    {
        curandState localState = globalState[idx];
        REAL RANDOM = curand_uniform( &localState )*2-1;
        randomArray[idx] = RANDOM;
        globalState[idx] = localState;
    }
}

OpenNNL::OpenNNL(const int inputsCount, const int layersCount, const int * neuronsPerLayerCount)
{
    _maxLayerSize = 0;
    _maxInputsCount = 0;

    _inputsCount = inputsCount;
    _layersCount = layersCount;
    _weightsCount = 0;
    _neuronsCount = 0;

    _neuronsPerLayerCount = new int[_layersCount];
    _neuronsInPreviousLayers = new int[_layersCount];
    _inputsInPreviousLayers = new int[_layersCount];
    _inputsInCurrentLayer = new int[_layersCount];


    cudaCall(cudaMalloc(&_deviceNeuronsPerLayerCount, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_deviceNeuronsInPreviousLayers, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_deviceInputsInPreviousLayers, _layersCount*sizeof(int)));
    cudaCall(cudaMalloc(&_deviceInputsInCurrentLayer, _layersCount*sizeof(int)));

    _inputs = new REAL[_inputsCount];

    //cudaCall(cudaMalloc(&_deviceInputs, _inputsCount*sizeof(REAL)));

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        if(inputs > _maxInputsCount)
            _maxInputsCount = inputs;

        _neuronsInPreviousLayers[i] = _neuronsCount;
        _inputsInPreviousLayers[i] = _weightsCount;

        _inputsInCurrentLayer[i] = inputs;

        _weightsCount += neuronsPerLayerCount[i] * inputs;
        _neuronsCount += neuronsPerLayerCount[i];

        inputs = _neuronsPerLayerCount[i] = neuronsPerLayerCount[i];

        if(_neuronsPerLayerCount[i] > _maxLayerSize)
        {
            _maxLayerSize = _neuronsPerLayerCount[i];
        }
    }

    _outputsCount = inputs;
    _outputs = new REAL[_outputsCount];

    cudaCall(cudaMalloc(&_deviceOutputs, _outputsCount*sizeof(REAL)));

    cudaCall(cudaMemcpy(_deviceNeuronsPerLayerCount, _neuronsPerLayerCount, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceNeuronsInPreviousLayers, _neuronsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceInputsInPreviousLayers, _inputsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceInputsInCurrentLayer, _inputsInCurrentLayer, _layersCount*sizeof(int), cudaMemcpyHostToDevice));

    //delete _hostNeuronsInPreviousLayers;
    //delete _hostInputsInCurrentLayer;
    //_derivatives = new REAL[_neuronsCount];

    //_neuronsInputsWeights = new REAL[_weightsCount];
    //_neuronsBiases = new REAL[_neuronsCount];
    cudaCall(cudaMalloc(&_neuronsInputsWeights, _weightsCount*sizeof(REAL)));
    cudaCall(cudaMalloc(&_neuronsBiases, _neuronsCount*sizeof(REAL)));

    cudaCall(cudaMalloc ( (void**)&_layerInputs, _maxInputsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc((void**)&_weightedLayerInputs, _maxInputsCount*_maxLayerSize*sizeof(REAL)));
}

OpenNNL::~OpenNNL()
{
    delete[] _neuronsPerLayerCount;
    delete[] _inputsInPreviousLayers;
    //delete[] _deviceNeuronsPerLayerCount;
    //delete[] _neuronsInPreviousLayers;
    //delete[] _inputsInPreviousLayers;
    delete[] _inputsInCurrentLayer;
    cudaCall(cudaFree(_deviceNeuronsPerLayerCount));
    cudaCall(cudaFree(_deviceNeuronsInPreviousLayers));
    delete[] _neuronsInPreviousLayers;
    cudaCall(cudaFree(_deviceInputsInPreviousLayers));
    cudaCall(cudaFree(_deviceInputsInCurrentLayer));
    delete[] _inputs;
    delete[] _outputs;
    //cudaCall(cudaFree(_deviceInputs));
    cudaCall(cudaFree(_deviceOutputs));
    //delete[] _derivatives;
    //delete[] _neuronsInputsWeights;
    //delete[] _neuronsBiases;
    cudaCall(cudaFree(_neuronsInputsWeights));
    cudaCall(cudaFree(_neuronsBiases));

    cudaCall(cudaFree(_layerInputs));
    cudaCall(cudaFree(_weightedLayerInputs));
}

void OpenNNL::printDebugInfo()
{
    printf("inputsCount=%d\n", _inputsCount);
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
    }
}

inline int OpenNNL::indexByLayerAndNeuron(int layer, int neuron)
{
    return _neuronsInPreviousLayers[layer] + neuron;
}

inline int OpenNNL::indexByLayerNeuronAndInput(int layer, int neuron, int input)
{
    return _inputsInPreviousLayers[layer] + neuron*_inputsInCurrentLayer[layer] + input;
}

inline void OpenNNL::setB(int layer, int neuron, int input, REAL value)
{
    _Bs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline REAL OpenNNL::getB(int layer, int neuron, int input)
{
    return _Bs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBForBias(int layer, int neuron, REAL value)
{
    _BsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline REAL OpenNNL::getBForBias(int layer, int neuron)
{
    return _BsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setH(int layer, int neuron, int input, REAL value)
{
    _Hs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline REAL OpenNNL::getH(int layer, int neuron, int input)
{
    return _Hs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setHForBias(int layer, int neuron, REAL value)
{
    _HsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline REAL OpenNNL::getHForBias(int layer, int neuron)
{
    return _HsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setWeight(int layer, int neuron, int input, REAL value)
{
    _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline REAL OpenNNL::getWeight(int layer, int neuron, int input)
{
    return _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBias(int layer, int neuron, REAL value)
{
    _neuronsBiases[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline REAL OpenNNL::getBias(int layer, int neuron)
{
    return _neuronsBiases[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setInput(int index, REAL value)
{
    _inputs[index] = value;
}

inline REAL OpenNNL::getOutput(int index)
{
    return _outputs[index];
}

__global__ void weightsMultiplicationBySqrtFromInputs(REAL * neuronsInputsWeights, int * inputsInCurrentLayer, int * inputsInPreviousLayers, int weightsCount, int layersCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < weightsCount)
    {
        // searching layer's number. ugly method -- TODO: write better
        int layer = layersCount-1;
        for(int i=0;i<layersCount-1;i++)
        {
            if(idx >= inputsInPreviousLayers[i] && idx < inputsInPreviousLayers[i+1])
            {
                layer = i;
                break;
            }
        }

        neuronsInputsWeights[idx] /= sqrtf(inputsInCurrentLayer[layer]);
    }
}

__global__ void biasesMultiplicationBySqrtFromInputs(REAL * neuronsBiases, int * inputsInCurrentLayer, int * neuronsInPreviousLayers, int neuronsCount, int layersCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        // searching layer's number. ugly method -- TODO: write better
        int layer = layersCount-1;
        for(int i=0;i<layersCount-1;i++)
        {
            if(idx >= neuronsInPreviousLayers[i] && idx < neuronsInPreviousLayers[i+1])
            {
                layer = i;
                break;
            }
        }

        neuronsBiases[idx] /= sqrtf(inputsInCurrentLayer[layer]);
    }
}

void OpenNNL::randomizeWeights()
{
    dim3 threads = dim3(BLOCK_SIZE, 1, 1);
    int blocksCount = floor((REAL)_weightsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1, 1);
    curandState* devStates;

    cudaCall(cudaMalloc ( &devStates, _weightsCount*sizeof( curandState ) ));

    initializeRandomGenerator <<<blocks, threads>>> ( devStates, time(NULL), _weightsCount );

    generateRandomArray <<<blocks, threads>>> ( devStates, _neuronsInputsWeights, _weightsCount );

    weightsMultiplicationBySqrtFromInputs <<< blocks, threads >>> (_neuronsInputsWeights, _deviceInputsInCurrentLayer, _deviceInputsInPreviousLayers, _weightsCount, _layersCount);

    cudaCall(cudaFree(devStates));
}

void OpenNNL::randomizeBiases()
{
    dim3 threads = dim3(256, 1);
    int blocksCount = floor((REAL) _neuronsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);
    curandState* devStates;

    cudaCall(cudaMalloc ( &devStates, _neuronsCount*sizeof( curandState ) ));

    initializeRandomGenerator <<<blocks, threads>>> ( devStates, time(NULL), _neuronsCount );

    generateRandomArray <<<blocks, threads>>> ( devStates, _neuronsBiases, _neuronsCount );

    biasesMultiplicationBySqrtFromInputs <<< blocks, threads >>> (_neuronsBiases, _deviceInputsInCurrentLayer, _deviceNeuronsInPreviousLayers, _neuronsCount, _layersCount);

    cudaCall(cudaFree(devStates));
}

void OpenNNL::randomizeWeightsAndBiases()
{
    this->randomizeWeights();
    this->randomizeBiases();
}

REAL * OpenNNL::_calculateSingle(REAL *inputs)
{
    REAL *deviceInputs;
    REAL *deviceOutputs;

    cudaCall(cudaMalloc ( (void**)&deviceInputs, _inputsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _outputsCount*sizeof(REAL) ));

    cudaCall(cudaMemcpy      ( deviceInputs, inputs, _inputsCount*sizeof(REAL), cudaMemcpyHostToDevice ));

    _doCalculation(deviceInputs, deviceOutputs);

    cudaCall(cudaMemcpy      ( _outputs, deviceOutputs, _outputsCount*sizeof(REAL), cudaMemcpyDeviceToHost ));

    cudaCall(cudaFree(deviceInputs));
    cudaCall(cudaFree(deviceOutputs));

    return _outputs;
}

void OpenNNL::_doCalculation(REAL *inputs, REAL * outputs)
{   
    int inputsCount = _inputsCount;

    //REAL * deviceTemp;
    //REAL *deviceInputs;

    //cudaCall(cudaMalloc ( (void**)&deviceInputs, inputsCount*sizeof(REAL) ));

    //cudaCall(cudaMemcpy      ( _layerInputs, inputs, inputsCount*sizeof(REAL), cudaMemcpyHostToDevice ));

    dim3 threads = dim3(512, 1);
    int blocksCount = floor((REAL) _inputsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);

    copyData<<<blocks, threads>>>(_layerInputs, inputs, _inputsCount);

    for(int i=0;i<_layersCount;i++)
    {
        //cudaCall(cudaMemoryAllocate((void***)&deviceTemp, sizeof(REAL), _neuronsPerLayerCount[i]*inputsCount));
        //cudaCall(cudaMalloc((void**)&deviceTemp, _neuronsPerLayerCount[i]*inputsCount*sizeof(REAL)));

        blocksCount = floor((REAL) _neuronsPerLayerCount[i]*inputsCount / threads.x) + 1;
        blocks = dim3(blocksCount, 1);

        weighting<<<blocks, threads>>>(_weightedLayerInputs, _layerInputs, _neuronsInputsWeights, _inputsInPreviousLayers[i], inputsCount, _neuronsPerLayerCount[i]);

        //cudaCall(cudaFree(deviceInputs));

        //cudaCall(cudaMemoryAllocate((void ***)&deviceInputs, sizeof(REAL), _neuronsPerLayerCount[i]));
        //cudaCall(cudaMalloc((void**)&deviceInputs, _neuronsPerLayerCount[i]*sizeof(REAL)));

        blocksCount = floor((REAL) _neuronsPerLayerCount[i] / threads.x) + 1;
        blocks = dim3(blocksCount, 1);

        calculateOut <<<blocks, threads>>> (_layerInputs, _weightedLayerInputs, _neuronsBiases, inputsCount, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i]);

        inputsCount = _neuronsPerLayerCount[i];

        //cudaCall(cudaFree(deviceTemp));
    }

    //cudaCall(cudaMemcpy      ( _outputs, _layerInputs, inputsCount*sizeof(REAL), cudaMemcpyDeviceToHost ));

    blocksCount = floor((REAL) _outputsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    copyData<<<blocks, threads>>>(outputs, _layerInputs, _outputsCount);

    //cudaCall(cudaFree(deviceInputs));

    //return _outputs;
}

REAL * OpenNNL::calculate(REAL *inputs)
{
    if(inputs)
    {
        memcpy(_inputs, inputs, _inputsCount*sizeof(REAL));
    }

    return _calculateSingle(_inputs);
}

REAL * OpenNNL::calculateRef(REAL *inputs)
{
    if(!inputs)
        inputs = _inputs;

    return _calculateSingle(inputs);
}

void OpenNNL::calculate(REAL * inputs, REAL * outputs, int samplesCount)
{
    REAL *deviceInputs;
    REAL *deviceOutputs;

    cudaCall(cudaMalloc ( (void**)&deviceInputs, _inputsCount*samplesCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _outputsCount*samplesCount*sizeof(REAL) ));

    cudaCall(cudaMemcpy      ( deviceInputs, inputs, _inputsCount*samplesCount*sizeof(REAL), cudaMemcpyHostToDevice ));

    for(int i=0;i<samplesCount;i++)
    {
        _doCalculation(deviceInputs+i*_inputsCount, deviceOutputs+i*_outputsCount);
    }

    cudaCall(cudaMemcpy      ( outputs, deviceOutputs, _outputsCount*samplesCount*sizeof(REAL), cudaMemcpyDeviceToHost ));

    cudaCall(cudaFree(deviceInputs));
    cudaCall(cudaFree(deviceOutputs));
}

__global__ void calculateOutputsAndDerivatives(REAL * outputs, REAL * derivatives, REAL * layerOutputs, REAL * inputs, REAL * biases, int inputsCount, int neuronsCount, int neuronsInPreviousLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        REAL temp = 0;

        for(int k=0;k<inputsCount;k++)
        {
            temp += inputs[idx*inputsCount+k];
        }

        int neuron = neuronsInPreviousLayers + idx;

        temp -= biases[neuron];

        outputs[neuron] = layerOutputs[idx] = activation(temp);
        derivatives[neuron] = activation_derivative(temp);
    }
}

void OpenNNL::calculateNeuronsOutputsAndDerivatives(REAL * inputs, REAL * deviceOutputs, REAL * deviceDerivatives)
{
    dim3 threads = dim3(BLOCK_SIZE, 1);

    int inputsCount = _inputsCount;

    //REAL * deviceTemp;
    //REAL * deviceInputs;

    // BE CAREFUL WITH SIZE OF THIS!
    //cudaCall(cudaMalloc ( (void**)&deviceInputs, inputsCount*sizeof(REAL) ));

    //cudaCall(cudaMemcpy      ( deviceInputs, inputs, inputsCount*sizeof(REAL), cudaMemcpyDeviceToDevice ));
    int blocksCount = floor((REAL) inputsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);

    copyData<<<blocks, threads>>>(_layerInputs, inputs, inputsCount);

    for(int i=0;i<_layersCount;i++)
    {
        //cudaCall(cudaMalloc((void**)&deviceTemp, _neuronsPerLayerCount[i]*inputsCount*sizeof(REAL)));

        int blocksCount = floor((REAL) _neuronsPerLayerCount[i]*inputsCount / threads.x) + 1;
        dim3 blocksMul  = dim3(blocksCount, 1);

        weighting<<<blocksMul, threads>>>(_weightedLayerInputs, _layerInputs, _neuronsInputsWeights, _inputsInPreviousLayers[i], inputsCount, _neuronsPerLayerCount[i]);

        //cudaCall(cudaFree(deviceInputs));

        //cudaCall(cudaMalloc((void**)&deviceInputs, _neuronsPerLayerCount[i]*sizeof(REAL)));

        blocksCount = floor((REAL) _neuronsPerLayerCount[i] / threads.x) + 1;
        dim3 blocksSum  = dim3(blocksCount, 1);

        calculateOutputsAndDerivatives<<<blocksSum, threads>>>(deviceOutputs, deviceDerivatives, _layerInputs, _weightedLayerInputs, _neuronsBiases, inputsCount, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i]);

        inputsCount = _neuronsPerLayerCount[i];

        //cudaCall(cudaFree(deviceTemp));
    }

    //cudaCall(cudaFree(deviceInputs));
}

__global__ void calculateLocalGradientsForLastLayer(REAL * localGradients, REAL * error, REAL * outputs, REAL * derivatives, REAL * trainingOutputs, REAL sample_weight, int neuronsCount, int neuronsInLastLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        REAL current_error = trainingOutputs[idx] - outputs[neuronsInLastLayers + idx];
        localGradients[neuronsInLastLayers + idx] = current_error * sample_weight * derivatives[neuronsInLastLayers + idx];

        error[idx] = current_error * current_error;
    }
}

__global__ void calculateLocalGradientsForAnotherLayers(REAL * localGradients, REAL * neuronsInputsWeights, REAL * derivatives, int neuronsCount, int neuronsInPreviousLayers, int neuronsInPreviousLayersWithCurrent, int neuronsInNextLayer, int inputsInPreviousLayersWithCurrent, int inputsInNextLayer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        int neuron = neuronsInPreviousLayers + idx;

        //localGradients[neuron] = 0;
        REAL temp = 0;

        // this to Kernel, then reduce localGradients.
        for(int k=0;k<neuronsInNextLayer;k++)
        {
            /*localGradients[neuron]*/ temp += neuronsInputsWeights[inputsInPreviousLayersWithCurrent + k*inputsInNextLayer + idx]
                                                            * localGradients[neuronsInPreviousLayersWithCurrent + k];
        }

        localGradients[neuron] = temp * derivatives[neuron];
    }
}

__global__ void changeWeightsForFirstLayer(REAL * neuronsInputsWeights, REAL * trainingInputs, REAL * localGradients, REAL speed, int inputsInLayer, int inputsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < inputsInLayer)
    {
        REAL deltaWeight = speed * localGradients[(int) (idx / inputsCount)] * trainingInputs[idx % inputsCount];
        REAL temp = neuronsInputsWeights[idx] + deltaWeight;
        neuronsInputsWeights[idx] = temp;
    }
}

__global__ void changeBiasesForFirstLayer(REAL * neuronsBiases, REAL * localGradients, REAL speed, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        REAL deltaBias = speed * localGradients[idx];
        REAL temp = neuronsBiases[idx] - deltaBias;
        neuronsBiases[idx] = temp;
    }
}

__global__ void changeWeightsForAnotherLayers(REAL * neuronsInputsWeights, REAL * localGradients, REAL * outputs, int * neuronsInPreviousLayers, int * inputsInPreviousLayers, int * inputsInCurrentLayer, REAL speed, int inputsCount, int layersCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= inputsInPreviousLayers[1] && idx < inputsCount)
    {
        // searching layer's number. ugly method -- TODO: write better
        int layer = layersCount-1;
        for(int i=0;i<layersCount-1;i++)
        {
            if(idx >= inputsInPreviousLayers[i] && idx < inputsInPreviousLayers[i+1])
            {
                layer = i;
                break;
            }
        }

        int index1 = neuronsInPreviousLayers[layer] + ((int) ((idx - inputsInPreviousLayers[layer]) / inputsInCurrentLayer[layer]));
        int index2 = neuronsInPreviousLayers[layer-1] + ((idx - inputsInPreviousLayers[layer]) % inputsInCurrentLayer[layer]);
        neuronsInputsWeights[idx] += speed * localGradients[index1] * outputs[index2];
    }
}

__global__ void changeBiasesForAnotherLayers(REAL * neuronsBiases, REAL * localGradients, REAL speed, int neuronsCount, int neuronsInFirstLayer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= neuronsInFirstLayer && idx < neuronsCount)
    {
        neuronsBiases[idx] -= speed * localGradients[idx];
    }
}

__global__ void reduceSum ( REAL * inData, REAL * outData )
{
    __shared__ REAL data [BLOCK_SIZE];
    int   tid = threadIdx.x;
    int   i   = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    data [tid] = inData [i] + inData [i+blockDim.x];     // load into shared memeory

    __syncthreads ();

    for ( int s = blockDim.x / 2; s > 32; s >>= 1 )
    {
        if ( tid < s )
            data [tid] += data [tid + s];

        __syncthreads ();
    }

    if ( tid < 32 )                                   // unroll last iterations
    {
        data [tid] += data [tid + 32];
        data [tid] += data [tid + 16];
        data [tid] += data [tid +  8];
        data [tid] += data [tid +  4];
        data [tid] += data [tid +  2];
        data [tid] += data [tid +  1];
    }

    if ( tid == 0 )                                      // write result of block reduction
        outData [blockIdx.x] = data [0];
}

REAL OpenNNL::_changeWeightsByBP(REAL * deviceTrainingInputs, REAL * deviceTrainingOutputs, REAL * deviceOutputs, REAL * deviceDerivatives, REAL * deviceLocalGradients, REAL * deviceErrors[], REAL speed, REAL sample_weight)
{
    dim3 threads = dim3(BLOCK_SIZE, 1);
    int blocksCount;
    dim3 blocks;

    REAL error = 0, current_error;
    int neuronsInLastLayer = _neuronsPerLayerCount[_layersCount-1];
    int index = 0, n = neuronsInLastLayer;
    REAL * errors = new REAL[neuronsInLastLayer];

    /*REAL * deviceOutputs;
    REAL * deviceDerivatives;

    REAL * deviceLocalGradients;
    REAL * deviceErrors[2] = {NULL, NULL};

    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceDerivatives, _neuronsCount*sizeof(REAL) ));

    cudaCall(cudaMalloc ( (void**)&deviceLocalGradients, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[0], neuronsInLastLayer*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[1], neuronsInLastLayer*sizeof(REAL) ));*/

    calculateNeuronsOutputsAndDerivatives(deviceTrainingInputs, deviceOutputs, deviceDerivatives);

    blocksCount = floor((REAL) neuronsInLastLayer / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    calculateLocalGradientsForLastLayer <<<blocks, threads>>>(deviceLocalGradients, deviceErrors[0], deviceOutputs, deviceDerivatives, deviceTrainingOutputs, sample_weight, neuronsInLastLayer, _neuronsInPreviousLayers[_layersCount-1]);

    /*for (index = 0; n >= BLOCK_SIZE; n /= (2*BLOCK_SIZE), index ^= 1 )
    {
                               // set kernel launch configuration
        dim3 dimBlock ( BLOCK_SIZE, 1, 1 );
        dim3 dimGrid  ( n / (2*dimBlock.x), 1, 1 );

        reduceSum <<<dimGrid, dimBlock>>> ( deviceErrors [index], deviceErrors [index^1] );
    }

    cudaMemcpy      ( errors, deviceErrors [index], neuronsInLastLayer*sizeof(REAL), cudaMemcpyDeviceToHost );

    for ( index = 1; index < n; index++ )
            errors [0] += errors [index];*/
    errors[0] = 10;

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            blocksCount = floor((REAL) _neuronsPerLayerCount[i] / threads.x) + 1;
            blocks  = dim3(blocksCount, 1);

            calculateLocalGradientsForAnotherLayers <<<blocks, threads>>> (deviceLocalGradients, _neuronsInputsWeights, deviceDerivatives, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i], _neuronsInPreviousLayers[i+1], _neuronsPerLayerCount[i+1], _inputsInPreviousLayers[i+1], _inputsInCurrentLayer[i+1]);

            cudaThreadSynchronize();
        }
    }

    blocksCount = floor((REAL) _neuronsPerLayerCount[0] * _inputsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeWeightsForFirstLayer <<<blocks, threads>>> (_neuronsInputsWeights, deviceTrainingInputs, deviceLocalGradients, speed, _neuronsPerLayerCount[0] * _inputsCount, _inputsCount);

    blocksCount = floor((REAL) _neuronsPerLayerCount[0] / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeBiasesForFirstLayer <<<blocks, threads>>> (_neuronsBiases, deviceLocalGradients, speed, _neuronsPerLayerCount[0]);

    blocksCount = floor((REAL) _weightsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeWeightsForAnotherLayers <<<blocks, threads>>> (_neuronsInputsWeights, deviceLocalGradients, deviceOutputs, _deviceNeuronsInPreviousLayers, _deviceInputsInPreviousLayers, _deviceInputsInCurrentLayer, speed, _weightsCount, _layersCount);

    blocksCount = floor((REAL) _neuronsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeBiasesForAnotherLayers <<<blocks, threads>>> (_neuronsBiases, deviceLocalGradients, speed, _neuronsCount, _neuronsInPreviousLayers[1]);

    error = errors[0]/2;
    return error;
}

REAL OpenNNL::_changeWeightsByIDBD(REAL * trainingInputs, REAL *trainingOutputs, REAL speed, REAL sample_weight)
{
    /*REAL error = 0, current_error;
    REAL cur_rate, delta, deltaB, deltaH;
    REAL * localGradients = new REAL[_neuronsCount];
    REAL * outputs = new REAL[_neuronsCount];
    REAL * derivatives = new REAL[_neuronsCount];

    calculateNeuronsOutputsAndDerivatives(trainingInputs, outputs, derivatives);

    for(int j=0;j<_deviceNeuronsPerLayerCount[_layersCount-1];j++)
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
            {
                localGradients[indexByLayerAndNeuron(i, j)] = 0;

                for(int k=0;k<_deviceNeuronsPerLayerCount[i+1];k++)
                {
                    localGradients[indexByLayerAndNeuron(i, j)] += _neuronsInputsWeights[indexByLayerNeuronAndInput(i+1, k, j)]
                                                                    * localGradients[indexByLayerAndNeuron(i+1, k)];
                }

                localGradients[indexByLayerAndNeuron(i, j)] *= derivatives[indexByLayerAndNeuron(i, j)];
            }
        }
    }

    for(int j=0;j<_deviceNeuronsPerLayerCount[0];j++)
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
        for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_deviceNeuronsPerLayerCount[i-1];k++)
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
    return error;*/
    return 0;
}

bool OpenNNL::_doEpochBP(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, REAL * deviceOutputs, REAL * deviceDerivatives, REAL * deviceLocalGradients, REAL * deviceErrors[], int numEpoch, REAL speed, REAL minError)
{
    REAL error = 0;

    /*REAL * deviceOutputs;
    REAL * deviceDerivatives;

    REAL * deviceLocalGradients;
    REAL * deviceErrors[2] = {NULL, NULL};

    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceDerivatives, _neuronsCount*sizeof(REAL) ));

    cudaCall(cudaMalloc ( (void**)&deviceLocalGradients, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[0], _neuronsPerLayerCount[_layersCount-1]*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[1], _neuronsPerLayerCount[_layersCount-1]*sizeof(REAL) ));*/

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Epoch: " << numEpoch << ", Sample: " << sample << endl;

        error = _changeWeightsByBP(trainingInputs+sample*_inputsCount, trainingOutputs+sample*_outputsCount, deviceOutputs, deviceDerivatives, deviceLocalGradients, deviceErrors, speed, 1);

        cout << "Error: " << error << endl;
    }

    /*cudaCall(cudaFree(deviceLocalGradients));
    cudaCall(cudaFree(deviceOutputs));
    cudaCall(cudaFree(deviceDerivatives));

    cudaCall(cudaFree(deviceErrors[0]));
    cudaCall(cudaFree(deviceErrors[1]));*/

    return (error <= minError);
}

bool OpenNNL::_doEpochIDBD(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int numEpoch, REAL speed, REAL minError)
{
    REAL error = 0;
    REAL * currentSampleInputs = new REAL[_inputsCount];
    REAL * currentSampleOutputs = new REAL[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(REAL));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(REAL));

        error = _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

void OpenNNL::_trainingBP(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int maxEpochsCount, REAL speed, REAL error)
{
    REAL * deviceTrainingInputs, * deviceTrainingOutputs;

    REAL * deviceOutputs;
    REAL * deviceDerivatives;

    REAL * deviceLocalGradients;
    REAL * deviceErrors[2] = {NULL, NULL};

    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceDerivatives, _neuronsCount*sizeof(REAL) ));

    cudaCall(cudaMalloc ( (void**)&deviceLocalGradients, _neuronsCount*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[0], _neuronsPerLayerCount[_layersCount-1]*sizeof(REAL) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[1], _neuronsPerLayerCount[_layersCount-1]*sizeof(REAL) ));

    cudaCall(cudaMalloc ( (void**)&deviceTrainingInputs, _inputsCount*samplesCount*sizeof(REAL) ));
    cudaCall(cudaMemcpy      ( deviceTrainingInputs, trainingInputs, _inputsCount*samplesCount*sizeof(REAL), cudaMemcpyHostToDevice ));

    cudaCall(cudaMalloc ( (void**)&deviceTrainingOutputs, _outputsCount*samplesCount*sizeof(REAL) ));
    cudaCall(cudaMemcpy      ( deviceTrainingOutputs, trainingOutputs, _outputsCount*samplesCount*sizeof(REAL), cudaMemcpyHostToDevice ));

    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochBP(samplesCount, deviceTrainingInputs, deviceTrainingOutputs, deviceOutputs, deviceDerivatives, deviceLocalGradients, deviceErrors, i, speed, error))
        {
            break;
        }
    }

    cudaCall(cudaFree(deviceTrainingInputs));
    cudaCall(cudaFree(deviceTrainingOutputs));

    cudaCall(cudaFree(deviceLocalGradients));
    cudaCall(cudaFree(deviceOutputs));
    cudaCall(cudaFree(deviceDerivatives));

    cudaCall(cudaFree(deviceErrors[0]));
    cudaCall(cudaFree(deviceErrors[1]));
}


void OpenNNL::_trainingIDBD(int samplesCount, REAL * trainingInputs, REAL * trainingOutputs, int maxEpochsCount, REAL speed, REAL error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochIDBD(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}

void OpenNNL::trainingBP(int samplesCount, REAL * trainingInputs, REAL *trainingOutputs, int maxEpochsCount, REAL speed, REAL error)
{
    _trainingBP(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);
}

void OpenNNL::trainingIDBD(int samplesCount, REAL * trainingInputs, REAL *trainingOutputs, int maxEpochsCount, REAL speed, REAL error)
{
    _Bs = new REAL[_weightsCount];
    _Hs = new REAL[_weightsCount];

    _BsForBias = new REAL[_neuronsCount];
    _HsForBias = new REAL[_neuronsCount];

    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingIDBD(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);

    delete[] _Bs;
    delete[] _Hs;
    delete[] _BsForBias;
    delete[] _HsForBias;
}

void OpenNNL::getOutputs(REAL * out)
{
    memcpy(out, _outputs, sizeof(REAL)*_outputsCount);
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

void OpenNNL::setWeights(REAL *weights)
{
    cudaCall(cudaMemcpy(_neuronsInputsWeights, weights, _weightsCount*sizeof(REAL), cudaMemcpyHostToDevice));
}

void OpenNNL::setBiases(REAL *biases)
{
    cudaCall(cudaMemcpy(_neuronsBiases, biases, _neuronsCount*sizeof(REAL), cudaMemcpyHostToDevice));
}

void OpenNNL::setWeightsAndBiases(REAL *weights, REAL *biases)
{
    setWeights(weights);
    setBiases(biases);
}
