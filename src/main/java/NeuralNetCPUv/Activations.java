package NeuralNetCPUv;

/**
 * The Activations class provides implementations of common
 * activation functions and their derivatives, which are
 * essential in building and training neural networks.
 */
public class Activations {

    /**
     * Sigmoid activation function.
     * Formula: σ(x) = 1 / (1 + e^(-x))
     *
     * @param x the input value
     * @return the sigmoid of x
     */
    public static float sigmoid(float x) {
        double exponent = Math.exp(-1.0 * x);   // compute e^(-x)
        double denominator = 1.0 + exponent;    // compute 1 + e^(-x)
        double result = 1.0 / denominator;      // compute 1 / (1 + e^(-x))
        return (float) result;
    }

    /**
     * Derivative of the sigmoid function.
     * Formula: σ'(x) = σ(x) * (1 - σ(x))
     *
     * @param x the input value
     * @return the derivative of sigmoid at x
     */
    public static float sigmoidDerivative(float x) {
        float sigmoidValue = sigmoid(x);                  // compute σ(x)
        float result = sigmoidValue * (1.0f - sigmoidValue);  // compute σ(x) * (1 - σ(x))
        return result;
    }

    /**
     * ReLU (Rectified Linear Unit) activation function.
     * Formula: ReLU(x) = max(0, x)
     *
     * @param x the input value
     * @return the ReLU of x
     */
    public static float relu(float x) {
        if (x > 0.0f) {
            return x;   // positive values remain unchanged
        } else {
            return 0.0f; // negative values become 0
        }
    }

    /**
     * Derivative of the ReLU function.
     * Formula: ReLU'(x) = 1 if x > 0, else 0
     *
     * @param x the input value
     * @return the derivative of ReLU at x
     */
    public static float reluDerivative(float x) {
        if (x > 0.0f) {
            return 1.0f; // derivative is 1 for positive inputs
        } else {
            return 0.0f; // derivative is 0 for zero or negative inputs
        }
    }

    /**
     * Hyperbolic tangent (tanh) activation function.
     * Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     *
     * @param x the input value
     * @return the tanh of x
     */
    public static float tanh(float x) {
        double numerator = Math.exp(x) - Math.exp(-1.0 * x);    // e^x - e^(-x)
        double denominator = Math.exp(x) + Math.exp(-1.0 * x);  // e^x + e^(-x)
        double result = numerator / denominator;                // fraction
        return (float) result;
    }

    /**
     * Derivative of the tanh function.
     * Formula: tanh'(x) = 1 - (tanh(x))^2
     *
     * @param x the input value
     * @return the derivative of tanh at x
     */
    public static float tanhDerivative(float x) {
        float tanhValue = tanh(x);                 // compute tanh(x)
        float result = 1.0f - (tanhValue * tanhValue); // compute 1 - (tanh(x))^2
        return result;
    }

    /**
     * Softmax activation function.
     * Converts a vector of raw scores into probabilities.
     * Formula: softmax(z_i) = e^(z_i - max(z)) / Σ e^(z_j - max(z))
     *
     * Subtracting max(z) improves numerical stability.
     *
     * @param z the input vector of values
     * @return the softmax output vector
     */
    public static float[] softmax(float[] z) {
        // Step 1: find the maximum value in the array for numerical stability
        float maxValue = z[0];
        for (int i = 1; i < z.length; i++) {
            if (z[i] > maxValue) {
                maxValue = z[i];
            }
        }

        // Step 2: exponentiate each element (after subtracting maxValue)
        float[] exponentials = new float[z.length];
        float sumOfExponentials = 0.0f;
        for (int i = 0; i < z.length; i++) {
            double exponent = Math.exp(z[i] - maxValue);
            exponentials[i] = (float) exponent;
            sumOfExponentials += exponentials[i];
        }

        // Step 3: divide each exponentiated value by the sum to normalize
        float[] softmaxOutput = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            softmaxOutput[i] = exponentials[i] / sumOfExponentials;
        }

        return softmaxOutput;
    }

    /**
     * Linear activation function.
     * Formula: f(x) = x
     *
     * @param x the input value
     * @return the same value as input
     */
    public static float linear(float x) {
        return x;
    }

    /**
     * Derivative of the linear activation function.
     * Formula: f'(x) = 1
     *
     * @param x the input value
     * @return 1.0 (since derivative is constant)
     */
    public static float linearDerivative(float x) {
        return 1.0f;
    }
}


