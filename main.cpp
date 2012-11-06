#include <iostream>

#include "opennnl.h"

using namespace std;

int main()
{
    const int INPUTS_COUNT = 3;
    const int LAYERS_COUNT = 4;
    const int OUTPUTS_COUNT = 2;

    const int TRAINING_SAMPLES_COUNT = 21;
    const float SPEED = 0.015;
    const float ERROR = 0.005;

    const int TEST_INPUTS_COUNT = 9;

    int neuronsInLayers[LAYERS_COUNT] = {3, 10, 10, 2};
    float trainingInputs[TRAINING_SAMPLES_COUNT*INPUTS_COUNT] = {1.0, 	1.0, 	1.0,
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
    float trainingOutputs[TRAINING_SAMPLES_COUNT*OUTPUTS_COUNT] = {
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

    float testInputs[TEST_INPUTS_COUNT*INPUTS_COUNT] = {
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

    float weights[159];
    float biases[25];

    for(int i=0;i<159;i++)
        weights[i] = 0;

    for(int i=0;i<25;i++)
        biases[i] = 0;

    cout << "Creating object..." << endl;
    OpenNNL * opennnl = new OpenNNL(INPUTS_COUNT, LAYERS_COUNT, neuronsInLayers);

    cout << "Randomizing weights..." << endl;
    //opennnl->randomizeWeights();
    opennnl->setWeights(weights);

    cout << "Randomizing biases..." << endl;
    //opennnl->randomizeBiases();
    opennnl->setBiases(biases);

    cout << "Training..." << endl;

    opennnl->trainingBP(TRAINING_SAMPLES_COUNT, trainingInputs, trainingOutputs, 1, SPEED, ERROR);

    opennnl->printDebugInfo();

    cout << "Calculations..." << endl;

    float inputs[INPUTS_COUNT];
    float outputs[OUTPUTS_COUNT];

    for(int i=0;i<TEST_INPUTS_COUNT;i++)
    {
        memcpy(inputs, testInputs+i*INPUTS_COUNT, INPUTS_COUNT*sizeof(float));

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

    cout << "Deleting object..." << endl;
    delete opennnl;

    cout << "Done!" << endl;

    return 0;
}

