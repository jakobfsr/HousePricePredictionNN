#include "nn.h"
#include <stdlib.h>
#include <math.h>

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void initialize_network(NeuralNetwork *nn, int input_size, int hidden_size, int output_size) {
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->weights_input_hidden = malloc(input_size * hidden_size * sizeof(double));
    nn->weights_hidden_output = malloc(hidden_size * output_size * sizeof(double));
    nn->hidden_bias = malloc(hidden_size * sizeof(double));
    nn->output_bias = malloc(output_size * sizeof(double));

    // Zufällige Initialisierung der Gewichte in [-1,1] 0.25515163375770289
    //                                                  0.25515163375660355
    for (int i = 0; i < input_size * hidden_size; i++)
        nn->weights_input_hidden[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    for (int i = 0; i < hidden_size * output_size; i++)
        nn->weights_hidden_output[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    for (int i = 0; i < hidden_size; i++) {
        nn->hidden_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    for (int i = 0; i < output_size; i++) {
        nn->output_bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void forward(NeuralNetwork *nn, double *inputs, double *hidden, double *outputs) {
    // Hidden-Layer berechnen
    for (int i = 0; i < nn->hidden_size; i++) {
        double sum = nn->hidden_bias[i];
        for (int j = 0; j < nn->input_size; j++) {
            sum += inputs[j] * nn->weights_input_hidden[i * nn->input_size + j];
        }
        hidden[i] = sum;//sigmoid(sum);
    }

    // Output-Layer berechnen
    for (int i = 0; i < nn->output_size; i++) {
        double sum = nn->output_bias[i];
        for (int j = 0; j < nn->hidden_size; j++) {
            sum += hidden[j] * nn->weights_hidden_output[i * nn->hidden_size + j];
        }
        outputs[i] = sum;//sigmoid(sum);
    }
}

void free_network(NeuralNetwork *nn) {
    free(nn->weights_input_hidden);
    free(nn->weights_hidden_output);
    free(nn->hidden_bias);
    free(nn->output_bias);
}
