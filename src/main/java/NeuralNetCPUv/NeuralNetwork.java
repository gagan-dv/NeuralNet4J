package NeuralNetCPUv;

import java.util.Random;

/**
 * NeuralNetwork class
 *
 * Implements a feedforward neural network with:
 * - Multiple hidden layers
 * - ReLU activation for hidden layers
 * - Linear + Softmax for output layer
 * - Cross-entropy loss
 * - Backpropagation with gradient descent
 *
 * This class ties together the Layer objects into a full model.
 */
public class NeuralNetwork {

    /** Array of layers in the network */
    public Layer[] layers;

    /** Learning rate for weight updates */
    private float learningRate;

    // ============================
    // Constructor
    // ============================

    /**
     * Create a new feedforward neural network.
     *
     * @param inputSize    number of input features
     * @param hiddenSizes  array specifying the size of each hidden layer
     * @param outputSize   number of output neurons (classes)
     * @param learningRate learning rate for training
     */
    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, float learningRate) {
        this.learningRate = learningRate;

        // Total number of layers = hidden layers + 1 output layer
        int totalLayers = hiddenSizes.length + 1;
        layers = new Layer[totalLayers];

        // Keep track of previous layer size
        int prevSize = inputSize;

        // Create hidden layers (with ReLU activation)
        for (int i = 0; i < hiddenSizes.length; i++) {
            layers[i] = new Layer(
                    prevSize,
                    hiddenSizes[i],
                    (Float x) -> Math.max(0f, x),       // ReLU activation
                    (Float z) -> z > 0f ? 1f : 0f       // ReLU derivative
            );
            prevSize = hiddenSizes[i];
        }

        // Create output layer (linear -> softmax outside)
        layers[totalLayers - 1] = new Layer(
                prevSize,
                outputSize,
                (Float x) -> x,   // linear activation
                null              // derivative handled via softmax + cross entropy
        );
    }

    // ============================
    // Forward propagation
    // ============================

    /**
     * Performs forward propagation through all layers.
     * Applies softmax to the final output for probabilities.
     *
     * @param input input vector
     * @return probability distribution over output classes
     */
    public float[] forward(float[] input) {
        float[] output = input;

        // Pass through each layer
        for (Layer layer : layers) {
            output = layer.forward(output);
        }

        // Apply softmax at the end (for classification)
        return Activations.softmax(output);
    }

    // ============================
    // Training
    // ============================

    /**
     * Trains the network for multiple epochs on a dataset.
     *
     * @param dataset training dataset
     * @param epochs  number of epochs
     */
    public void train(TrainDataset dataset, int epochs) {
        for (int e = 0; e < epochs; e++) {

            // Shuffle dataset each epoch
            RandomUtil.shuffle(dataset.features.data, dataset.labels.data);

            float totalLoss = 0f;

            // Train on each example
            for (int i = 0; i < dataset.numExamples; i++) {
                float[] input = dataset.features.data[i];
                float[] target = dataset.labels.data[i];

                // Forward pass
                float[] output = forward(input);

                // Compute loss
                totalLoss += crossEntropyLoss(output, target);

                // Backward pass (update weights)
                trainSample(input, target);
            }

            // Print progress every 10 epochs
            if ((e + 1) % 10 == 0) {
                System.out.printf("Epoch %d: Loss = %.4f%n",
                        e + 1,
                        totalLoss / dataset.numExamples);
            }
        }
    }

    /**
     * Trains the network on a single training example using backpropagation.
     *
     * @param input  feature vector
     * @param target one-hot encoded target vector
     */
    public void trainSample(float[] input, float[] target) {
        // Forward pass with softmax at the end
        float[] softmaxOut = forward(input);

        // Compute output error (delta for softmax + cross-entropy)
        float[] delta = new float[softmaxOut.length];
        for (int i = 0; i < delta.length; i++) {
            delta[i] = softmaxOut[i] - target[i];
        }

        // Backward pass through all layers
        for (int l = layers.length - 1; l >= 0; l--) {
            Layer currentLayer = layers[l];

            // Backprop through this layer (update weights & biases)
            float[] dAprev = currentLayer.backwardFromDZ(delta, learningRate);

            // Prepare delta for previous layer (if not input layer)
            if (l > 0) {
                Layer prevLayer = layers[l - 1];
                float[] dZprev = new float[dAprev.length];

                // Multiply by derivative of activation
                for (int i = 0; i < dZprev.length; i++) {
                    float deriv = (prevLayer.activationDerivative != null)
                            ? prevLayer.activationDerivative.apply(prevLayer.lastZ[i])
                            : 1f;
                    dZprev[i] = dAprev[i] * deriv;
                }

                // Pass delta backward
                delta = dZprev;
            }
        }
    }

    // ============================
    // Utilities
    // ============================

    /**
     * Prints the structure of the network (layers and sizes).
     */
    public void printStructure() {
        System.out.println("Neural Network Structure:");
        for (int i = 0; i < layers.length; i++) {
            Layer l = layers[i];
            System.out.printf("Layer %d: inputs=%d, outputs=%d%n",
                    i + 1, l.inputSize, l.outputSize);
        }
    }

    /**
     * Computes cross-entropy loss between predicted and target.
     *
     * @param predicted predicted probabilities
     * @param target    one-hot encoded target
     * @return loss value
     */
    public static float crossEntropyLoss(float[] predicted, float[] target) {
        float loss = 0f;
        for (int i = 0; i < predicted.length; i++) {
            // Add small epsilon to avoid log(0)
            loss -= target[i] * Math.log(predicted[i] + 1e-10f);
        }
        return loss;
    }

    /**
     * Evaluates the network accuracy on a test dataset.
     *
     * @param test test dataset
     * @param nn   trained neural network
     * @return accuracy (0.0 – 1.0)
     */
    public static float evaluate(TestDataset test, NeuralNetwork nn) {
        int correct = 0;

        for (int i = 0; i < test.numExamples; i++) {
            float[] prediction = nn.forward(test.features.data[i]);

            // Compare predicted class with true class
            if (argMax(prediction) == argMax(test.labels.data[i])) {
                correct++;
            }
        }

        return (float) correct / test.numExamples;
    }

    /**
     * Returns the index of the maximum element in an array.
     */
    public static int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }

        return idx;
    }
}
