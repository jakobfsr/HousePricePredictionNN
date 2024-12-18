#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/** Generiert Trainingsdaten:
    inputs: [num_samples][5]
    targets: [num_samples][1]

   Modell: Preis = 50*rooms + 2*size + (2020-year)*10 - distance*3 + 30*bathrooms
   und dann Normalisierung durch division durch 1000.0.
**/
void generate_training_data(double inputs[][5], double targets[][1], int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        double rooms = (rand() % 5) + 1;       // 1-6
        double size = (rand() % 200) + 50;     // 50-249
        double year = (rand() % 41) + 1980;    // 1980-2020
        double distance = (rand() % 30) + 1;   // 1-30
        double bathrooms = (rand() % 3) + 1;   // 1-3

        double price = 50.0 * rooms + 2.0 * size + (2020 - year)*10.0 - distance*3.0 + 30.0 * bathrooms;
        // Optional: leichtes Rauschen hinzufügen:
        // double noise = (double)(rand()%201 -100) * 0.1; // z.B. -10 bis +10
        // price += noise;

        inputs[i][0] = rooms;
        inputs[i][1] = size;
        inputs[i][2] = year;
        inputs[i][3] = distance;
        inputs[i][4] = bathrooms;

        targets[i][0] = price / 1000.0; // Normalisierung
    }
}

/** Führt einen Forward/Backward-Pass für ein Batch durch und akkumuliert die Gewichtsänderungen. **/
void train_parallel(NeuralNetwork *nn, double inputs[][5], double targets[][1], int start_idx, int end_idx, int batch_size, double lr) {
    int num_samples = end_idx - start_idx;
    int num_batches = num_samples / batch_size;

    for (int batch = 0; batch < num_batches; batch++) {
        double *weight_deltas_input_hidden = calloc(nn->input_size * nn->hidden_size, sizeof(double));
        double *weight_deltas_hidden_output = calloc(nn->hidden_size * nn->output_size, sizeof(double));
        double *bias_deltas_hidden = calloc(nn->hidden_size, sizeof(double));
        double *bias_deltas_output = calloc(nn->output_size, sizeof(double));

        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            int sample_idx = start_idx + batch * batch_size + i;
            double hidden[nn->hidden_size];
            double output[nn->output_size];
            double hidden_errors[nn->hidden_size];
            double output_errors[nn->output_size];

            // Forward-Pass
            forward(nn, inputs[sample_idx], hidden, output);

            // Output-Fehler
            for (int j = 0; j < nn->output_size; j++) {
                double error = targets[sample_idx][j] - output[j];
                output_errors[j] = error * output[j] * (1.0 - output[j]);
            }

            // Hidden-Fehler
            for (int j = 0; j < nn->hidden_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < nn->output_size; k++) {
                    sum += output_errors[k] * nn->weights_hidden_output[k * nn->hidden_size + j];
                }
                hidden_errors[j] = sum * hidden[j] * (1.0 - hidden[j]);
            }

            // Gradienten-Update Output-Layer
            for (int j = 0; j < nn->output_size; j++) {
                for (int k = 0; k < nn->hidden_size; k++) {
                    #pragma omp atomic
                    weight_deltas_hidden_output[j * nn->hidden_size + k] += output_errors[j] * hidden[k];
                }
                #pragma omp atomic
                bias_deltas_output[j] += output_errors[j];
            }

            // Gradienten-Update Hidden-Layer
            for (int j = 0; j < nn->hidden_size; j++) {
                for (int k = 0; k < nn->input_size; k++) {
                    #pragma omp atomic
                    weight_deltas_input_hidden[j * nn->input_size + k] += hidden_errors[j] * inputs[sample_idx][k];
                }
                #pragma omp atomic
                bias_deltas_hidden[j] += hidden_errors[j];
            }
        }

        // Anwenden der Gradienten auf Gewichte und Biases
        for (int j = 0; j < nn->hidden_size * nn->output_size; j++) {
            nn->weights_hidden_output[j] += lr * weight_deltas_hidden_output[j];
        }
        for (int j = 0; j < nn->input_size * nn->hidden_size; j++) {
            nn->weights_input_hidden[j] += lr * weight_deltas_input_hidden[j];
        }
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->hidden_bias[j] += lr * bias_deltas_hidden[j];
        }
        for (int j = 0; j < nn->output_size; j++) {
            nn->output_bias[j] += lr * bias_deltas_output[j];
        }

        free(weight_deltas_input_hidden);
        free(weight_deltas_hidden_output);
        free(bias_deltas_hidden);
        free(bias_deltas_output);
    }
}

/** Bewertet das Netz auf einem Test-Set und berechnet den mittleren quadratischen Fehler (MSE). **/
double evaluate(NeuralNetwork *nn, double inputs[][5], double targets[][1], int start_idx, int end_idx) {
    double mse = 0.0;
    int count = end_idx - start_idx;
    for (int i = start_idx; i < end_idx; i++) {
        double hidden[nn->hidden_size];
        double output[nn->output_size];
        forward(nn, inputs[i], hidden, output);
        double error = (targets[i][0] - output[0]);
        double sq_error = error*error;
        mse += sq_error;
        if(i % 250 == 0){
            printf("Rooms: %f, Size %f, Year: %f, Distance: %f, Bathrooms: %f\n", inputs[i][0],inputs[i][1], inputs[i][2], inputs[i][3], inputs[i][4]);
            printf("Predicted Value: %.3f, Actual Value: %.3f, Squared Error: %.3f\n", output[0], targets[i][0], sq_error);
        }
    }
    return mse / (double)count;
}

/** Mischt die Daten (inputs und targets) zufällig durch. **/
void shuffle_data(double inputs[][5], double targets[][1], int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // swap inputs
        for (int k = 0; k < 5; k++) {
            double tmp = inputs[i][k];
            inputs[i][k] = inputs[j][k];
            inputs[j][k] = tmp;
        }
        // swap targets
        double tmp_t = targets[i][0];
        targets[i][0] = targets[j][0];
        targets[j][0] = tmp_t;
    }
}

/** Führt eine k-fache Cross Validation durch.
    Wir nehmen k=5. Der Datensatz wird in 5 gleiche Teile aufgeteilt.
    Jeder Teil dient einmal als Testset, die restlichen 4 als Trainingsset.
**/
void k_fold_cross_validation(double inputs[][5], double targets[][1], int num_samples, int k, int epochs, int batch_size, double lr) {
    int fold_size = num_samples / k;
    double total_test_mse = 0.0;

    for (int fold = 0; fold < k; fold++) {
        // Testset: von fold*fold_size bis (fold+1)*fold_size
        int test_start = fold * fold_size;
        int test_end = (fold + 1) * fold_size;

        // Trainingsset: alles außer test_start...test_end
        // Wir initialisieren ein neues Netz für jeden Fold.
        NeuralNetwork nn;
        initialize_network(&nn, 5, 10, 1);

        // Training
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Erster Teil des Trainingssets: von 0 bis test_start
            if (test_start > 0) {
                train_parallel(&nn, inputs, targets, 0, test_start, batch_size, lr);
            }
            // Zweiter Teil des Trainingssets: von test_end bis num_samples
            if (test_end < num_samples) {
                train_parallel(&nn, inputs, targets, test_end, num_samples, batch_size, lr);
            }
        }

        // Evaluieren auf dem Test-Fold
        double mse = evaluate(&nn, inputs, targets, test_start, test_end);
        total_test_mse += mse;

        printf("Fold %d: Test MSE = %.6f\n", fold, mse);

        free_network(&nn);
    }

    double avg_mse = total_test_mse / (double)k;
    printf("Durchschnittlicher Test-MSE über alle Folds: %.6f\n", avg_mse);
}

int main() {
    srand(time(NULL));

    const int num_samples = 10000;
    double inputs[num_samples][5];
    double targets[num_samples][1];

    // Daten generieren
    generate_training_data(inputs, targets, num_samples);

    // Daten mischen für eine faire Cross-Validation
    shuffle_data(inputs, targets, num_samples);

    // Parameter
    int k = 5;          // 5-fache Cross-Validation
    int epochs = 100;   // weniger Epochenzahl für jede Cross-Validation Runde
    int batch_size = 10;
    double lr = 0.0001;

    // Cross Validation durchführen
    k_fold_cross_validation(inputs, targets, num_samples, k, epochs, batch_size, lr);

    return 0;
}

// {0.906488464170363, -0.89876206028217542, -0.34491357549322466, 0.76549372904258495, -0.60205667074865499, -0.8982673566314705, 0.66124219245335181, -0.32701107735140766, 0.60414842404618319, -0.46801895716601005}
// {0.906488464170363, -0.89876206028217542, -0.34491357549322466, 0.76549372904258495, -0.60205667074865499, -0.8982673566314705, 0.66124219245335181, -0.32701107735140766, 0.60414842404618319, -0.46801895716601005}