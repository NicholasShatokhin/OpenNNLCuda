#include "opennnl.h"

/*****************************************************************************/
/* Вычислить активационную функцию y(x) = 2x / (1 + abs(x)). */
/*****************************************************************************/
//inline double OpenNNL::activation(double x, TActivationKind kind)
__device__ double activation(double x)
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
//inline double OpenNNL::activation_derivative(double x, TActivationKind kind)
__device__ double activation_derivative(double x)
{
    double temp = 1.0 + fabs(x);
    return (2.0 / (temp * temp));
}

__global__ void testKernel(double * output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
    {
        output[idx] = 5;
    }
}

__global__ void weighting(double * outputs, double * inputs, double * weights, int start, int inputsCount, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < inputsCount * neuronsCount)
    {
        outputs[idx] = inputs[idx%inputsCount] * weights[start+idx];
    }
}

__global__ void calculateOut(double * outputs, double * inputs, double * biases, int inputsCount, int neuronsCount, int neuronsInPreviousLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        double temp = 0;

        for(int k=0;k<inputsCount;k++)
        {
            temp += inputs[idx*inputsCount+k];
        }

        temp -= biases[neuronsInPreviousLayers+idx];

        outputs[idx] = activation(temp);
    }
}

__global__ void initializeRandomGenerator ( curandState * state, unsigned long seed, int count )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < count)
        curand_init ( seed, idx, 0, &state[idx] );
}

__global__ void generateRandomArray( curandState* globalState, double * randomArray, int count )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < count)
    {
        curandState localState = globalState[idx];
        double RANDOM = curand_uniform( &localState )*2-1;
        randomArray[idx] = RANDOM;
        globalState[idx] = localState;
    }
}

OpenNNL::OpenNNL(const int inputsCount, const int layersCount, const int * neuronsPerLayerCount)
{
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

    _inputs = new double[_inputsCount];

    //cudaCall(cudaMalloc(&_deviceInputs, _inputsCount*sizeof(double)));

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        _neuronsInPreviousLayers[i] = _neuronsCount;
        _inputsInPreviousLayers[i] = _weightsCount;

        _inputsInCurrentLayer[i] = inputs;

        _weightsCount += neuronsPerLayerCount[i] * inputs;
        _neuronsCount += neuronsPerLayerCount[i];

        inputs = _neuronsPerLayerCount[i] = neuronsPerLayerCount[i];
    }

    _outputsCount = inputs;
    _outputs = new double[_outputsCount];

    cudaCall(cudaMalloc(&_deviceOutputs, _outputsCount*sizeof(double)));

    cudaCall(cudaMemcpy(_deviceNeuronsPerLayerCount, _neuronsPerLayerCount, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceNeuronsInPreviousLayers, _neuronsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceInputsInPreviousLayers, _inputsInPreviousLayers, _layersCount*sizeof(int), cudaMemcpyHostToDevice));
    cudaCall(cudaMemcpy(_deviceInputsInCurrentLayer, _inputsInCurrentLayer, _layersCount*sizeof(int), cudaMemcpyHostToDevice));

    //delete _hostNeuronsInPreviousLayers;
    //delete _hostInputsInCurrentLayer;
    //_derivatives = new double[_neuronsCount];

    //_neuronsInputsWeights = new double[_weightsCount];
    //_neuronsBiases = new double[_neuronsCount];
    cudaCall(cudaMalloc(&_neuronsInputsWeights, _weightsCount*sizeof(double)));
    cudaCall(cudaMalloc(&_neuronsBiases, _neuronsCount*sizeof(double)));
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

inline void OpenNNL::setB(int layer, int neuron, int input, double value)
{
    _Bs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getB(int layer, int neuron, int input)
{
    return _Bs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBForBias(int layer, int neuron, double value)
{
    _BsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getBForBias(int layer, int neuron)
{
    return _BsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setH(int layer, int neuron, int input, double value)
{
    _Hs[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getH(int layer, int neuron, int input)
{
    return _Hs[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setHForBias(int layer, int neuron, double value)
{
    _HsForBias[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getHForBias(int layer, int neuron)
{
    return _HsForBias[indexByLayerAndNeuron(layer, neuron)];
}

inline void OpenNNL::setWeight(int layer, int neuron, int input, double value)
{
    _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)] = value;
}

inline double OpenNNL::getWeight(int layer, int neuron, int input)
{
    return _neuronsInputsWeights[indexByLayerNeuronAndInput(layer, neuron, input)];
}

inline void OpenNNL::setBias(int layer, int neuron, double value)
{
    _neuronsBiases[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getBias(int layer, int neuron)
{
    return _neuronsBiases[indexByLayerAndNeuron(layer, neuron)];
}

/*inline void OpenNNL::setDerivative(int layer, int neuron, double value)
{
    _derivatives[indexByLayerAndNeuron(layer, neuron)] = value;
}

inline double OpenNNL::getDerivative(int layer, int neuron)
{
    return _derivatives[indexByLayerAndNeuron(layer, neuron)];
}*/

inline void OpenNNL::setInput(int index, double value)
{
    _inputs[index] = value;
}

inline double OpenNNL::getOutput(int index)
{
    return _outputs[index];
}

__global__ void weightsMultiplicationBySqrtFromInputs(double * neuronsInputsWeights, int * inputsInCurrentLayer, int * inputsInPreviousLayers, int weightsCount, int layersCount)
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

__global__ void biasesMultiplicationBySqrtFromInputs(double * neuronsBiases, int * inputsInCurrentLayer, int * neuronsInPreviousLayers, int neuronsCount, int layersCount)
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
    dim3 threads = dim3(BLOCK_SIZE, 1);
    int blocksCount = floor((double)_weightsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);
    curandState* devStates;

    cudaCall(cudaMalloc ( &devStates, _weightsCount*sizeof( curandState ) ));

    initializeRandomGenerator <<<blocks, threads>>> ( devStates, time(NULL), _weightsCount );

    generateRandomArray <<<blocks, threads>>> ( devStates, _neuronsInputsWeights, _weightsCount );

    weightsMultiplicationBySqrtFromInputs <<< blocks, threads >>> (_neuronsInputsWeights, _deviceInputsInCurrentLayer, _deviceInputsInPreviousLayers, _weightsCount, _layersCount);

    cudaCall(cudaFree(devStates));
    /*initialize_random_generator();

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<inputs*_deviceNeuronsPerLayerCount[i];j++)
        {
            _neuronsInputsWeights[_inputsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _deviceNeuronsPerLayerCount[i];
    }*/

}

void OpenNNL::randomizeBiases()
{
    dim3 threads = dim3(256, 1);
    int blocksCount = floor((double) _neuronsCount / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);
    curandState* devStates;

    cudaCall(cudaMalloc ( &devStates, _neuronsCount*sizeof( curandState ) ));

    initializeRandomGenerator <<<blocks, threads>>> ( devStates, time(NULL), _neuronsCount );

    generateRandomArray <<<blocks, threads>>> ( devStates, _neuronsBiases, _neuronsCount );

    biasesMultiplicationBySqrtFromInputs <<< blocks, threads >>> (_neuronsBiases, _deviceInputsInCurrentLayer, _deviceNeuronsInPreviousLayers, _neuronsCount, _layersCount);

    cudaCall(cudaFree(devStates));
    /*initialize_random_generator();

    int inputs = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
        {
            _neuronsBiases[_neuronsInPreviousLayers[i]+j] = unified_random() / sqrt(inputs);
        }

        inputs = _deviceNeuronsPerLayerCount[i];
    }*/
}

void OpenNNL::randomizeWeightsAndBiases()
{
    this->randomizeWeights();
    this->randomizeBiases();
}

double * OpenNNL::_calculateWorker(double *inpt)
{   
    int inputsCount = _inputsCount;

    double * deviceTemp;
    double *deviceInputs;

    cudaCall(cudaMalloc ( (void**)&deviceInputs, inputsCount*sizeof(double) ));

    cudaCall(cudaMemcpy      ( deviceInputs, inpt, inputsCount*sizeof(double), cudaMemcpyHostToDevice ));

    for(int i=0;i<_layersCount;i++)
    {
        //cudaCall(cudaMemoryAllocate((void***)&deviceTemp, sizeof(double), _neuronsPerLayerCount[i]*inputsCount));
        cudaCall(cudaMalloc((void**)&deviceTemp, _neuronsPerLayerCount[i]*inputsCount*sizeof(double)));

        dim3 threadsMul = dim3(512, 1);
        int blocksCount = floor((double) _neuronsPerLayerCount[i]*inputsCount / threadsMul.x) + 1;
        dim3 blocksMul  = dim3(blocksCount, 1);

        weighting<<<blocksMul, threadsMul>>>(deviceTemp, deviceInputs, _neuronsInputsWeights, _inputsInPreviousLayers[i], inputsCount, _neuronsPerLayerCount[i]);

        cudaCall(cudaFree(deviceInputs));

        //cudaCall(cudaMemoryAllocate((void ***)&deviceInputs, sizeof(double), _neuronsPerLayerCount[i]));
        cudaCall(cudaMalloc((void**)&deviceInputs, _neuronsPerLayerCount[i]*sizeof(double)));

        dim3 threadsSum = dim3(512, 1);
        blocksCount = floor((double) _neuronsPerLayerCount[i] / threadsSum.x) + 1;
        dim3 blocksSum  = dim3(blocksCount, 1);

        calculateOut <<<blocksSum, threadsSum>>> (deviceInputs, deviceTemp, _neuronsBiases, inputsCount, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i]);

        inputsCount = _neuronsPerLayerCount[i];

        cudaCall(cudaFree(deviceTemp));
    }

    cudaCall(cudaMemcpy      ( _outputs, deviceInputs, inputsCount*sizeof(double), cudaMemcpyDeviceToHost ));
    cudaCall(cudaFree(deviceInputs));

    return _outputs;
    /*int inputsCount;
    double * temp;
    double * inputs = new double[_inputsCount];

    memcpy(inputs, inpt, sizeof(double)*_inputsCount);

    inputsCount = _inputsCount;

    for(int i=0;i<_layersCount;i++)
    {
        temp = new double[_deviceNeuronsPerLayerCount[i]*inputsCount];
        for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<inputsCount;k++)
            {
            temp[j*inputsCount+k] = inputs[k] * _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)];
            }
        }

        delete[] inputs;

        inputs = new double[_deviceNeuronsPerLayerCount[i]];

        for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
        {
            inputs[j] = 0;

            for(int k=0;k<inputsCount;k++)
            {
                inputs[j] += temp[j*inputsCount+k];
            }

            inputs[j]  -= _neuronsBiases[indexByLayerAndNeuron(i, j)];

            inputs[j] = activation(inputs[j]);

        }

        inputsCount = _deviceNeuronsPerLayerCount[i];
        delete[] temp;
    }

    memcpy(_outputs, inputs, sizeof(double)*inputsCount);

    delete[] inputs;

    return _outputs;*/
}

double * OpenNNL::calculate(double *inputs)
{
    if(inputs)
    {
        memcpy(_inputs, inputs, _inputsCount*sizeof(double));
    }

    return _calculateWorker(_inputs);
}

double * OpenNNL::calculateRef(double *inputs)
{
    if(!inputs)
        inputs = _inputs;

    return _calculateWorker(inputs);
}

__global__ void calculateOutputsAndDerivatives(double * outputs, double * derivatives, double * layerOutputs, double * inputs, double * biases, int inputsCount, int neuronsCount, int neuronsInPreviousLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        double temp = 0;

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

void OpenNNL::calculateNeuronsOutputsAndDerivatives(double * inputs, double * deviceOutputs, double * deviceDerivatives)
{
    int inputsCount = _inputsCount;
    //double * temp;
    //double * inputs = new double[_inputsCount];

    //memcpy(inputs, inpt, sizeof(double)*_inputsCount);

    double * deviceTemp;
    double * deviceInputs;
    //double * deviceOutputs;
    //double * deviceDerivatives;

    cudaCall(cudaMalloc ( (void**)&deviceInputs, inputsCount*sizeof(double) ));
    //cudaCall(cudaMalloc ( (void**)&deviceOutputs, _neuronsCount*sizeof(double) ));
    //cudaCall(cudaMalloc ( (void**)&deviceDerivatives, _neuronsCount*sizeof(double) ));

    cudaCall(cudaMemcpy      ( deviceInputs, inputs, inputsCount*sizeof(double), cudaMemcpyDeviceToDevice ));

    for(int i=0;i<_layersCount;i++)
    {
        cudaCall(cudaMalloc((void**)&deviceTemp, _neuronsPerLayerCount[i]*inputsCount*sizeof(double)));

        dim3 threadsMul = dim3(512, 1);
        int blocksCount = floor((double) _neuronsPerLayerCount[i]*inputsCount / threadsMul.x) + 1;
        dim3 blocksMul  = dim3(blocksCount, 1);

        weighting<<<blocksMul, threadsMul>>>(deviceTemp, deviceInputs, _neuronsInputsWeights, _inputsInPreviousLayers[i], inputsCount, _neuronsPerLayerCount[i]);

        cudaCall(cudaFree(deviceInputs));

        cudaCall(cudaMalloc((void**)&deviceInputs, _neuronsPerLayerCount[i]*sizeof(double)));

        dim3 threadsSum = dim3(512, 1);
        blocksCount = floor((double) _neuronsPerLayerCount[i] / threadsSum.x) + 1;
        dim3 blocksSum  = dim3(blocksCount, 1);

        calculateOutputsAndDerivatives<<<blocksSum, threadsSum>>>(deviceOutputs, deviceDerivatives, deviceInputs, deviceTemp, _neuronsBiases, inputsCount, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i]);

        inputsCount = _neuronsPerLayerCount[i];

        cudaCall(cudaFree(deviceTemp));
    }

    //cudaCall(cudaMemcpy      ( outputs, deviceOutputs, inputsCount*sizeof(double), cudaMemcpyDeviceToHost ));
    //cudaCall(cudaMemcpy      ( derivatives, deviceDerivatives, inputsCount*sizeof(double), cudaMemcpyDeviceToHost ));
    cudaCall(cudaFree(deviceInputs));
}

__global__ void calculateLocalGradientsForLastLayer(double * localGradients, double * error, double * outputs, double * derivatives, double * trainingOutputs, double sample_weight, int neuronsCount, int neuronsInLastLayers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        double current_error = trainingOutputs[idx] - outputs[neuronsInLastLayers + idx];
        localGradients[neuronsInLastLayers + idx] = current_error * sample_weight * derivatives[neuronsInLastLayers + idx];

        error[idx] = current_error * current_error;
    }
}

__global__ void calculateLocalGradientsForAnotherLayers(double * localGradients, double * neuronsInputsWeights, double * derivatives, int neuronsCount, int neuronsInPreviousLayers, int neuronsInPreviousLayersWithCurrent, int neuronsInNextLayer, int inputsInPreviousLayers, int inputsInCurrentLayer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        int neuron = neuronsInPreviousLayers + idx;

        localGradients[neuron] = 0;

        // this to Kernel, then reduce localGradients.
        for(int k=0;k<neuronsInNextLayer;k++)
        {
            localGradients[neuron] += neuronsInputsWeights[inputsInPreviousLayers + k*inputsInCurrentLayer + idx]
                                                            * localGradients[neuronsInPreviousLayersWithCurrent + k];
        }

        localGradients[neuron] *= derivatives[neuron];
    }
}

__global__ void changeWeightsForFirstLayer(double * neuronsInputsWeights, double * trainingInputs, double * localGradients, double speed, int inputsInLayer, int inputsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < inputsInLayer)
    {
        double deltaWeight = speed * localGradients[(int) (idx / inputsCount)] * trainingInputs[idx % inputsCount];
        double temp = neuronsInputsWeights[idx] + deltaWeight;
        neuronsInputsWeights[idx] = temp;
    }
}

__global__ void changeBiasesForFirstLayer(double * neuronsBiases, double * localGradients, double speed, int neuronsCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < neuronsCount)
    {
        double deltaBias = speed * localGradients[idx];
        double temp = neuronsBiases[idx] - deltaBias;
        neuronsBiases[idx] = temp;
    }
}

__global__ void changeWeightsForAnotherLayers(double * neuronsInputsWeights, double * localGradients, double * outputs, int * neuronsInPreviousLayers, int * inputsInPreviousLayers, int * inputsInCurrentLayer, double speed, int inputsCount, int layersCount)
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

__global__ void changeBiasesForAnotherLayers(double * neuronsBiases, double * localGradients, double speed, int neuronsCount, int neuronsInFirstLayer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= neuronsInFirstLayer && idx < neuronsCount)
    {
        neuronsBiases[idx] -= speed * localGradients[idx];
    }
}

__global__ void reduceSum ( double * inData, double * outData )
{
    __shared__ double data [BLOCK_SIZE];
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

double OpenNNL::_changeWeightsByBP(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
{
    dim3 threads = dim3(BLOCK_SIZE, 1);
    int blocksCount;
    dim3 blocks;

    double error = 0, current_error;
    int neuronsInLastLayer = _neuronsPerLayerCount[_layersCount-1];
    int index = 0, n = neuronsInLastLayer;
    double * errors = new double[neuronsInLastLayer];
    //double * localGradients = new double[_neuronsCount];
    //double * outputs = new double[_neuronsCount];
    //double * derivatives = new double[_neuronsCount];

    // calculateNeuronsOutputsAndDerivatives
    double * deviceOutputs;
    double * deviceDerivatives;

    double * deviceTrainingInputs;
    double * deviceTrainingOutputs;

    double * deviceLocalGradients;
    double * deviceErrors[2] = {NULL, NULL};

    cudaCall(cudaMalloc ( (void**)&deviceOutputs, _neuronsCount*sizeof(double) ));
    cudaCall(cudaMalloc ( (void**)&deviceDerivatives, _neuronsCount*sizeof(double) ));

    cudaCall(cudaMalloc ( (void**)&deviceTrainingInputs, _inputsCount*sizeof(double) ));
    cudaCall(cudaMemcpy      ( deviceTrainingInputs, trainingInputs, _inputsCount*sizeof(double), cudaMemcpyHostToDevice ));

    cudaCall(cudaMalloc ( (void**)&deviceTrainingOutputs, _outputsCount*sizeof(double) ));
    cudaCall(cudaMemcpy      ( deviceTrainingOutputs, trainingOutputs, _outputsCount*sizeof(double), cudaMemcpyHostToDevice ));

    cudaCall(cudaMalloc ( (void**)&deviceLocalGradients, _neuronsCount*sizeof(double) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[0], neuronsInLastLayer*sizeof(double) ));
    cudaCall(cudaMalloc ( (void**)&deviceErrors[1], neuronsInLastLayer*sizeof(double) ));

    calculateNeuronsOutputsAndDerivatives(deviceTrainingInputs, deviceOutputs, deviceDerivatives);

    /*double * outputs = new double[_neuronsCount];
    cudaCall(cudaMemcpy      ( outputs, deviceOutputs, _neuronsCount*sizeof(double), cudaMemcpyDeviceToHost ));

    cout << "outputs:" << endl;
    for(int i=0;i<_neuronsCount;i++)
        cout << outputs[i] << endl;
    cout << "------------" << endl;

    delete[] outputs;

    double * derivatives = new double[_neuronsCount];
    cudaCall(cudaMemcpy      ( derivatives, deviceDerivatives, _neuronsCount*sizeof(double), cudaMemcpyDeviceToHost ));

    cout << "derivatives:" << endl;
    for(int i=0;i<_neuronsCount;i++)
        cout << derivatives[i] << endl;
    cout << "------------" << endl;

    delete[] derivatives;*/

    // calculateLocalGradientsForLastLayer
    /*for(int j=0;j<_deviceNeuronsPerLayerCount[_layersCount-1];j++) // cuda kernel
    {
        current_error = trainingOutputs[j] - outputs[indexByLayerAndNeuron(_layersCount-1, j)];
        localGradients[indexByLayerAndNeuron(_layersCount-1, j)] = current_error * sample_weight * derivatives[indexByLayerAndNeuron(_layersCount-1, j)];

        error += current_error * current_error;
    }*/

    blocksCount = floor((double) neuronsInLastLayer / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    calculateLocalGradientsForLastLayer <<<blocks, threads>>>(deviceLocalGradients, deviceErrors[0], deviceOutputs, deviceDerivatives, deviceTrainingOutputs, sample_weight, neuronsInLastLayer, _neuronsInPreviousLayers[_layersCount-1]);

    /*cudaMemcpy      ( errors, deviceErrors [0], neuronsInLastLayer*sizeof(double), cudaMemcpyDeviceToHost );

    cout << "errors: " << endl;
    for(int i=0;i<n;i++)
        cout << errors[i] << endl;
    cout << "-------------" << endl;*/

    for (index = 0; n >= BLOCK_SIZE; n /= (2*BLOCK_SIZE), index ^= 1 )
    {
                               // set kernel launch configuration
        dim3 dimBlock ( BLOCK_SIZE, 1, 1 );
        dim3 dimGrid  ( n / (2*dimBlock.x), 1, 1 );

        reduceSum <<<dimGrid, dimBlock>>> ( deviceErrors [index], deviceErrors [index^1] );
    }

    cudaMemcpy      ( errors, deviceErrors [index], neuronsInLastLayer*sizeof(double), cudaMemcpyDeviceToHost );

    /*cout << "errors: " << endl;
    for(int i=0;i<n;i++)
        cout << errors[i] << endl;
    cout << "-------------" << endl;*/

    for ( index = 1; index < n; index++ )
            errors [0] += errors [index];

    if(_layersCount > 1)
    {
        for(int i=_layersCount-2;i>=0;i--)
        {
            // calculateLocalGradientsForAnotherLayers
            blocksCount = floor((double) _neuronsPerLayerCount[i] / threads.x) + 1;
            blocks  = dim3(blocksCount, 1);

            calculateLocalGradientsForAnotherLayers <<<blocks, threads>>> (deviceLocalGradients, _neuronsInputsWeights, deviceDerivatives, _neuronsPerLayerCount[i], _neuronsInPreviousLayers[i], _neuronsInPreviousLayers[i+1], _neuronsPerLayerCount[i+1], _inputsInPreviousLayers[i], _inputsInCurrentLayer[i]);
        }
    }

    /*double * localGradients = new double[_neuronsCount];
    cudaCall(cudaMemcpy      ( localGradients, deviceLocalGradients, _neuronsCount*sizeof(double), cudaMemcpyDeviceToHost ));

    cout << "localGradients:" << endl;
    for(int i=0;i<_neuronsCount;i++)
        cout << localGradients[i] << endl;
    cout << "------------" << endl;

    delete[] localGradients;*/


    // changeWeightsForFirstLayer
    /*for(int j=0;j<_neuronsPerLayerCount[0];j++) // this and next cicle for cuda kernel (j*k threads)
    {
        for(int k=0;k<_inputsCount;k++)
        {
            _neuronsInputsWeights[indexByLayerNeuronAndInput(0, j, k)] += speed * localGradients[indexByLayerAndNeuron(0, j)] * trainingInputs[k];
        }

        _neuronsBiases[indexByLayerAndNeuron(0, j)] -= speed * localGradients[indexByLayerAndNeuron(0, j)];
    }*/
    blocksCount = floor((double) _neuronsPerLayerCount[0] * _inputsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeWeightsForFirstLayer <<<blocks, threads>>> (_neuronsInputsWeights, deviceTrainingInputs, deviceLocalGradients, speed, _neuronsPerLayerCount[0] * _inputsCount, _inputsCount);

    blocksCount = floor((double) _neuronsPerLayerCount[0] / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeBiasesForFirstLayer <<<blocks, threads>>> (_neuronsBiases, deviceLocalGradients, speed, _neuronsPerLayerCount[0]);

    // changeWeightsForAnotherLayers
    /*for(int i=1;i<_layersCount;i++) // try to parallelize all three cicles in one kernel. If it's impossible, only two inner
    {
        for(int j=0;j<_deviceNeuronsPerLayerCount[i];j++)
        {
            for(int k=0;k<_deviceNeuronsPerLayerCount[i-1];k++)
            {
                _neuronsInputsWeights[indexByLayerNeuronAndInput(i, j, k)] += speed * localGradients[indexByLayerAndNeuron(i, j)] * outputs[indexByLayerAndNeuron(i-1, k)];
            }

            _neuronsBiases[indexByLayerAndNeuron(i, j)] -= speed * localGradients[indexByLayerAndNeuron(i, j)];
        }
    }*/

    blocksCount = floor((double) _weightsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeWeightsForAnotherLayers <<<blocks, threads>>> (_neuronsInputsWeights, deviceLocalGradients, deviceOutputs, _deviceNeuronsInPreviousLayers, _deviceInputsInPreviousLayers, _deviceInputsInCurrentLayer, speed, _weightsCount, _layersCount);

    /*double * weights = new double[_weightsCount];
    cudaCall(cudaMemcpy      ( weights, _neuronsInputsWeights, _weightsCount*sizeof(double), cudaMemcpyDeviceToHost ));

    cout << "weights:" << endl;
    for(int i=0;i<_weightsCount;i++)
        cout << weights[i] << endl;
    cout << "------------" << endl;

    delete[] weights;*/

    blocksCount = floor((double) _neuronsCount / threads.x) + 1;
    blocks  = dim3(blocksCount, 1);

    changeBiasesForAnotherLayers <<<blocks, threads>>> (_neuronsBiases, deviceLocalGradients, speed, _neuronsCount, _neuronsInPreviousLayers[1]);

    /*double * biases = new double[_neuronsCount];
    cudaCall(cudaMemcpy      ( biases, _neuronsBiases, _neuronsCount*sizeof(double), cudaMemcpyDeviceToHost ));

    cout << "biases:" << endl;
    for(int i=0;i<_neuronsCount;i++)
        cout << biases[i] << endl;
    cout << "------------" << endl;

    delete[] biases;*/

    //delete[] localGradients;
    //delete[] outputs;
    //delete[] derivatives;
    cudaCall(cudaFree(deviceLocalGradients));
    cudaCall(cudaFree(deviceOutputs));
    cudaCall(cudaFree(deviceDerivatives));

    cudaCall(cudaFree(deviceErrors[0]));
    cudaCall(cudaFree(deviceErrors[1]));

    error = errors[0]/2;
    return error;
}

double OpenNNL::_changeWeightsByIDBD(double * trainingInputs, double *trainingOutputs, double speed, double sample_weight)
{
    double error = 0, current_error;
    double cur_rate, delta, deltaB, deltaH;
    double * localGradients = new double[_neuronsCount];
    double * outputs = new double[_neuronsCount];
    double * derivatives = new double[_neuronsCount];

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
    return error;
}

bool OpenNNL::_doEpochBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed, double minError)
{
    double error = 0;
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Epoch: " << numEpoch << ", Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(double));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(double));

        error = _changeWeightsByBP(currentSampleInputs, currentSampleOutputs, speed, 1);

        cout << "Error: " << error << endl;
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

bool OpenNNL::_doEpochIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int numEpoch, double speed, double minError)
{
    double error = 0;
    double * currentSampleInputs = new double[_inputsCount];
    double * currentSampleOutputs = new double[_outputsCount];

    for(int sample=0;sample<samplesCount;sample++)
    {
        cout << "Sample: " << sample << endl;
        memcpy(currentSampleInputs, trainingInputs+sample*_inputsCount, _inputsCount*sizeof(double));
        memcpy(currentSampleOutputs, trainingOutputs+sample*_outputsCount, _outputsCount*sizeof(double));

        error = _changeWeightsByIDBD(currentSampleInputs, currentSampleOutputs, speed, 1);
    }

    delete[] currentSampleInputs;
    delete[] currentSampleOutputs;

    return (error <= minError);
}

void OpenNNL::_trainingBP(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed, double error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochBP(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}


void OpenNNL::_trainingIDBD(int samplesCount, double * trainingInputs, double * trainingOutputs, int maxEpochsCount, double speed, double error)
{
    for(int i=0;i<maxEpochsCount;i++)
    {
        if(_doEpochIDBD(samplesCount, trainingInputs, trainingOutputs, i, speed, error))
        {
            break;
        }
    }
}

void OpenNNL::trainingBP(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    _trainingBP(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);
}

void OpenNNL::trainingIDBD(int samplesCount, double * trainingInputs, double *trainingOutputs, int maxEpochsCount, double speed, double error)
{
    _Bs = new double[_weightsCount];
    _Hs = new double[_weightsCount];

    _BsForBias = new double[_neuronsCount];
    _HsForBias = new double[_neuronsCount];

    resetHsAndHsForBias();
    randomizeBsAndBsForBias();

    _trainingIDBD(samplesCount, trainingInputs, trainingOutputs, maxEpochsCount, speed, error);

    delete[] _Bs;
    delete[] _Hs;
    delete[] _BsForBias;
    delete[] _HsForBias;
}

void OpenNNL::getOutputs(double * out)
{
    memcpy(out, _outputs, sizeof(double)*_outputsCount);
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

void OpenNNL::setWeights(double *weights)
{
    cudaCall(cudaMemcpy(_neuronsInputsWeights, weights, _weightsCount*sizeof(double), cudaMemcpyHostToDevice));
}

void OpenNNL::setBiases(double *biases)
{
    cudaCall(cudaMemcpy(_neuronsBiases, biases, _neuronsCount*sizeof(double), cudaMemcpyHostToDevice));
}

void OpenNNL::setWeightsAndBiases(double *weights, double *biases)
{
    setWeights(weights);
    setBiases(biases);
}
