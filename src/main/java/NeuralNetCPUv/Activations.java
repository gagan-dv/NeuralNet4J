package NeuralNetCPUv;

public class Activations {

    public static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    public static float sigmoidDerivative(float x) {
        float s = sigmoid(x);
        return s * (1 - s);
    }

    public static float relu(float x) {
        return Math.max(0, x);
    }

    public static float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    public static float tanh(float x) {
        return (float) Math.tanh(x);
    }

    public static float tanhDerivative(float x) {
        float t = tanh(x);
        return 1 - t * t;
    }

    public static float[] softmax(float[] z) {
        float max = z[0];
        for (float val : z) max = Math.max(max, val);

        float sum = 0;
        float[] out = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            out[i] = (float) Math.exp(z[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < z.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    public static float linear(float x) {
        return x;
    }

    public static float linearDerivative(float x) {
        return 1.0f;
    }
}
