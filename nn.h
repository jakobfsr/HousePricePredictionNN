#ifndef NN_H
#define NN_H

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weights_input_hidden;
    double *weights_hidden_output;
    double *hidden_bias;
    double *output_bias;
} NeuralNetwork;

/** Initialisiert das neuronale Netz mit zufälligen Gewichten. **/
void initialize_network(NeuralNetwork *nn, int input_size, int hidden_size, int output_size);

/** Führt einen Forward-Pass durch. inputs -> hidden -> outputs **/
void forward(NeuralNetwork *nn, double *inputs, double *hidden, double *outputs);

/** Gibt den zugewiesenen Speicher frei **/
void free_network(NeuralNetwork *nn);

#endif
