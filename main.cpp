#include <iostream>
#include <iomanip>

#include "opennnl.h"
#include "mnistfile.h"

using namespace std;

int maxArrayElementsIndex(double array[], int count);

void testNetwork1();
void testNetwork2();

void startTimer(struct timespec * tp)
{
    clock_gettime(CLOCK_MONOTONIC, tp);

    cout << "Time: 0 ns" << endl;
}

void printTimerValue(struct timespec * tp)
{
    long int startTime = tp->tv_nsec;

    clock_gettime(CLOCK_MONOTONIC, tp);

    cout << "Time: " << tp->tv_nsec - startTime << " ns" << endl;
}

int main()
{
    testNetwork2();
    return 0;
}

void testNetwork1()
{
    struct timespec tp;

    const int INPUTS_COUNT = 3;
    const int LAYERS_COUNT = 4;
    const int OUTPUTS_COUNT = 2;

    const int TRAINING_SAMPLES_COUNT = 21;
    const double SPEED = 0.015;
    const double ERROR = 0.005;

    const int TEST_INPUTS_COUNT = 9;

    int neuronsInLayers[LAYERS_COUNT] = {3, 10, 10, 2};
    double trainingInputs[TRAINING_SAMPLES_COUNT*INPUTS_COUNT] = {1.0, 	1.0, 	1.0,
                                                                  0.5, 	1.0, 	1.0,
                                                                  1.0, 	1.0, 	0.5,
                                                                  1.0, 	0.5, 	1.0,
                                                                  0.5, 	1.0, 	0.5,
                                                                  0.0, 	0.0, 	0.0,
                                                                  0.5, 	0.5, 	0.5,
                                                                  0.0, 	1.0, 	1.0,
                                                                  1.0, 	1.0, 	0.0,
                                                                  1.0, 	0.0, 	1.0,
                                                                  0.0, 	1.0, 	0.0,
                                                                  0.0, 	0.0, 	1.0,
                                                                  1.0, 	0.0, 	0.0,
                                                                  0.3, 	0.4, 	0.1,
                                                                  0.1, 	0.4, 	0.3,
                                                                  0.0, 	0.1, 	0.2,
                                                                  0.2, 	0.1, 	0.0,
                                                                  0.0, 	0.3, 	0.6,
                                                                  0.6, 	0.3, 	0.0,
                                                                  0.2, 	0.3, 	0.4,
                                                                  0.4, 	0.3, 	0.2 };
    double trainingOutputs[TRAINING_SAMPLES_COUNT*OUTPUTS_COUNT] = {
                                                                           1.0, 	0.5,
                                                                           0.6, 	0.7,
                                                                           0.6, 	0.3,
                                                                           0.3, 	0.4,
                                                                           0.7, 	0.5,
                                                                           0.2, 	0.2,
                                                                           0.5, 	0.4,
                                                                           0.4, 	0.9,
                                                                           0.4, 	0.1,
                                                                           0.2, 	0.2,
                                                                           1.0, 	0.5,
                                                                           0.3, 	0.8,
                                                                           0.3, 	0.2,
                                                                           0.5, 	0.3,
                                                                           0.5, 	0.7,
                                                                           0.3, 	0.9,
                                                                           0.3, 	0.1,
                                                                           0.5, 	0.8,
                                                                           0.5, 	0.2,
                                                                           0.5, 	0.9,
                                                                           0.4, 	0.1
                                                                   };

    double testInputs[TEST_INPUTS_COUNT*INPUTS_COUNT] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.5,
        0.0, 0.5, 1.0,
        0.2, 0.1, 0.3,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 0.5, 0.0,
        1.0, 1.0, 1.0
    };

    double weights[159];
    double biases[25];

    for(int i=0;i<159;i++)
        weights[i] = 0.1;

    for(int i=0;i<25;i++)
        biases[i] = 0.1;


    startTimer(&tp);

    cout << "Creating object..." << endl;
    OpenNNL * opennnl = new OpenNNL(INPUTS_COUNT, LAYERS_COUNT, neuronsInLayers);

    printTimerValue(&tp);

    cout << "Randomizing weights..." << endl;
    //opennnl->randomizeWeights();
    opennnl->setWeights(weights);

    cout << "Randomizing biases..." << endl;
    //opennnl->randomizeBiases();
    opennnl->setBiases(biases);

    cout << "Training..." << endl;

    printTimerValue(&tp);

    opennnl->trainingBP(TRAINING_SAMPLES_COUNT, trainingInputs, trainingOutputs, 1, SPEED, ERROR);

    printTimerValue(&tp);

    opennnl->printDebugInfo();

    cout << "Calculations..." << endl;

    double inputs[INPUTS_COUNT];
    double outputs[OUTPUTS_COUNT];

    printTimerValue(&tp);

    for(int i=0;i<TEST_INPUTS_COUNT;i++)
    {
        memcpy(inputs, testInputs+i*INPUTS_COUNT, INPUTS_COUNT*sizeof(double));

        opennnl->calculate(inputs);
        opennnl->getOutputs(outputs);

        cout << "test sample #" << i+1 << ":" << endl;
        for(int j=0;j<INPUTS_COUNT;j++)
            cout << inputs[j] << " ";
        cout << " --> ";
        for(int j=0;j<OUTPUTS_COUNT;j++)
            cout << outputs[j] << " ";
        cout << endl;
    }

    printTimerValue(&tp);

    cout << "Deleting object..." << endl;
    delete opennnl;

    printTimerValue(&tp);

    cout << "Done!" << endl;
}

void testNetwork2()
{
    cout << "Creating network..." << endl;
    const int layers_count = 3;
    const int inputs_count = 784;
    const int outputs_count = 10;
    int neuronsInLayers[layers_count] = {300, 100, outputs_count};
    const double error = 0.005;

    OpenNNL * opennnl = new OpenNNL(inputs_count, layers_count, neuronsInLayers);
    opennnl->randomizeWeightsAndBiases();

    cout << "Preparing train data..." << endl;

    MnistFile images;
    MnistFile labels;
    if(!images.openFile("../OpenNNLCuda/data/mnist/train-images.idx3-ubyte"))
    {
        cout << "Couldn't find train images file" << endl;
        return;
    }

    if(!labels.openFile("../OpenNNLCuda/data/mnist/train-labels.idx1-ubyte"))
    {
        cout << "Couldn't find train labels file" << endl;
        return;
    }

    cout << "Files opened. Reading..." << endl;

    unsigned char * image = new unsigned char[images.getRows()*images.getCols()];
    unsigned char label;

    const int trainingSamplesCount = 2000;//images.getLength();
    const double speed = 1 / (double) trainingSamplesCount;

    double * trainingInputs = new double[trainingSamplesCount*inputs_count];
    double * trainingOutputs = new double[trainingSamplesCount*outputs_count];

    for(int i=0;i<trainingSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&label);

        for(int j=0;j<inputs_count;j++)
        {
            trainingInputs[i*inputs_count+j] = ((double) image[j] - 127.5) / 127.5;
        }

        for(int k=0;k<label;k++)
            trainingOutputs[i*outputs_count+k] = -1;
        trainingOutputs[i*outputs_count+label] = 1;
        for(int k=label+1;k<outputs_count;k++)
            trainingOutputs[i*outputs_count+k] = -1;
    }

    images.closeFile();
    labels.closeFile();

    cout << "Training..." << endl;

    opennnl->trainingIDBD(trainingSamplesCount, trainingInputs, trainingOutputs, 1, speed, error);

    delete trainingInputs;
    delete trainingOutputs;

    if(!images.openFile("../OpenNNLCuda/data/mnist/t10k-images.idx3-ubyte"))
    {
        cout << "Couldn't find test images file" << endl;
        return;
    }

    if(!labels.openFile("../OpenNNLCuda/data/mnist/t10k-labels.idx1-ubyte"))
    {
        cout << "Couldn't find test labels file" << endl;
        return;
    }

    const int testSamplesCount = 500;//images.getLength();

    double * testInputs = new double[inputs_count];
    double * testOutputs = new double[outputs_count];
    int outputLabel, correctAnswers=0;

    cout << "Testing..." << endl;

    for(int i=0;i<testSamplesCount;i++)
    {
        images.readRecord(image);
        labels.readRecord(&label);

        for(int j=0;j<inputs_count;j++)
        {
            testInputs[j] = ((double) image[j] - 127.5) / 127.5;
        }

        opennnl->calculate(testInputs);
        opennnl->getOutputs(testOutputs);

        outputLabel = maxArrayElementsIndex(testOutputs, outputs_count);
        if(outputLabel == (int) label)
            correctAnswers++;
        else
        {
            cout << i << ": Incorrect answer: " << outputLabel << " instead: " << (int) label << endl;

            cout << "Outputs: " << setprecision(30);
            for(int k=0;k<outputs_count;k++)
            {
                cout << testOutputs[i] << " ";
            }
            cout << endl;
        }
    }

    images.closeFile();
    labels.closeFile();

    delete image;

    delete testInputs;
    delete testOutputs;

    cout << endl;
    cout << "Correct answers " << correctAnswers << " from " << testSamplesCount << " labels" << endl;
    cout << "Error rate: " << 100.0 - ((double) correctAnswers) / ((double) testSamplesCount) * 100.0 << "%" << endl;

    delete opennnl;
}

int maxArrayElementsIndex(double array[], int count)
{
    double maxElement = -10;
    int index = 0;

    for(int i=0;i<count;i++)
    {
        if(array[i] > maxElement)
        {
            maxElement = array[i];
            index = i;
        }
    }

    return index;
}
