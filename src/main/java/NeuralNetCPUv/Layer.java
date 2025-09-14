package NeuralNetCPUv;

import java.util.Random;
import java.util.function.Function;

/**
 * Represents a single fully-connected (dense) layer in a neural network.
 * Handles forward propagation (computing outputs) and backward propagation (updating weights).
 */
public class Layer {

    // === Layer structure ===
    public int inputSize;   // number of input neurons
    public int outputSize;  // number of output neurons

    // === Parameters ===
    public Matrix weights;  // weight matrix of shape (inputSize x outputSize)
    public float[] biases;  // bias vector of length outputSize

    // === Cached values for backpropagation ===
    public float[] lastInput;   // input passed to this layer
    public float[] lastZ;       // linear combination (Wx + b) before activation
    public float[] lastOutput;  // output after applying activation

    // === Activation function and its derivative ===
    public Function<Float, Float> activation;
    public Function<Float, Float> activationDerivative;

    // Random number generator for initialization
    private Random rand = new Random();

    /**
     * Constructor for the Layer.
     *
     * @param inputSize              number of inputs to this layer
     * @param outputSize             number of neurons (outputs) in this layer
     * @param activation             activation function (e.g., ReLU, sigmoid, etc.)
     * @param activationDerivative   derivative of activation function
     */
    public Layer(int inputSize,
                 int outputSize,
                 Function<Float, Float> activation,
                 Function<Float, Float> activationDerivative) {

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        // Save activation functions
        this.activation = activation;
        this.activationDerivative = activationDerivative;

        // Initialize weights as a matrix (inputSize x outputSize)
        this.weights = new Matrix(inputSize, outputSize);

        // Randomize weights using uniform distribution
        this.weights.randomizeUniform();

        // Initialize biases as zero
        this.biases = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            this.biases[i] = 0.0f;
        }
    }

    /**
     * Forward propagation step.
     * Computes: z = W*x + b, then applies activation function.
     *
     * @param input input vector of length inputSize
     * @return output vector of length outputSize
     */
    public float[] forward(float[] input) {
        // Store the input for backpropagation
        this.lastInput = input.clone();

        // Step 1: Multiply weights and input
        float[] z = Matrix.multiplyVector(this.weights.data, input);

        // Step 2: Add biases to each neuron
        for (int j = 0; j < outputSize; j++) {
            z[j] = z[j] + this.biases[j];
        }

        // Store z for backpropagation
        this.lastZ = z.clone();

        // Step 3: Apply activation function element-wise
        this.lastOutput = new float[outputSize];
        for (int j = 0; j < outputSize; j++) {
            this.lastOutput[j] = this.activation.apply(z[j]);
        }

        return this.lastOutput;
    }

    /**
     * Backward propagation step.
     * Updates weights and biases using gradients and computes error to pass back.
     *
     * @param dZ           gradient of loss w.r.t. this layer’s pre-activation (z)
     * @param learningRate step size for weight updates
     * @return gradient of loss w.r.t. previous layer’s activations (dA_prev)
     */
    public float[] backwardFromDZ(float[] dZ, float learningRate) {
        // Initialize gradient to pass back to previous layer
        float[] dAprev = new float[inputSize];

        // Step 1: Compute dAprev = W * dZ
        for (int i = 0; i < inputSize; i++) {
            float sum = 0.0f;
            for (int j = 0; j < outputSize; j++) {
                sum = sum + (this.weights.data[i][j] * dZ[j]);
            }
            dAprev[i] = sum;
        }

        // Step 2: Update weights using gradient descent
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float gradient = this.lastInput[i] * dZ[j];
                this.weights.data[i][j] = this.weights.data[i][j] - (learningRate * gradient);
            }
        }

        // Step 3: Update biases
        for (int j = 0; j < outputSize; j++) {
            this.biases[j] = this.biases[j] - (learningRate * dZ[j]);
        }

        // Step 4: Return gradient for previous layer
        return dAprev;
    }
}


