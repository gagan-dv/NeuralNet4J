package NeuralNetCPUv;

import java.util.Random;

/**
 * Utility class for vector operations (1D arrays of floats).
 */
public class Vector {



    private static Random rand = new Random();

    /**
     * Allocate a vector of given size, optionally zero-initialized.
     */
    public static float[] alloc(int size, boolean zeroInit) {
        float[] vec = new float[size];
        if (zeroInit == true) {
            for (int i = 0; i < size; i++) {
                vec[i] = 0.0f;
            }
        } else {
            // Leave uninitialized (default float = 0.0 anyway, but explicit branch kept)
            for (int i = 0; i < size; i++) {
                vec[i] = vec[i]; // no-op, explicit else
            }
        }
        return vec;
    }

    /**
     * Copy a vector into a new array.
     */
    public static float[] copy(float[] vec) {
        float[] newVec = new float[vec.length];
        System.arraycopy(vec, 0, newVec, 0, vec.length);
        return newVec;
    }

    /**
     * Compute dot product of two vectors.
     */
    public static float dot(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Add two vectors elementwise.
     */
    public static float[] add(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /**
     * Subtract two vectors elementwise.
     */
    public static float[] subtract(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    /**
     * Multiply two vectors elementwise.
     */
    public static float[] multiplyElem(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    /**
     * Divide two vectors elementwise.
     */
    public static float[] divideElem(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] / b[i];
        }
        return result;
    }

    /**
     * Multiply vector by a scalar.
     */
    public static float[] scale(float[] a, float scalar) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * scalar;
        }
        return result;
    }

    /**
     * Compute Euclidean norm of a vector.
     */
    public static float norm(float[] a) {
        float sum = 0.0f;
        for (float v : a) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    /**
     * Normalize a vector to unit length.
     */
    public static float[] normalize(float[] a) {
        float mag = norm(a);
        if (mag == 0.0f) {
            return copy(a);
        } else {
            return scale(a, 1.0f / mag);
        }
    }

    /**
     * Fill a vector with zeros.
     */
    public static void zeros(float[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = 0.0f;
        }
    }

    /**
     * Fill a vector with ones.
     */
    public static void ones(float[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = 1.0f;
        }
    }

    /**
     * Fill a vector with random values from uniform distribution [-1, 1].
     */
    public static void randomUniform(float[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = rand.nextFloat() * 2f - 1f;
        }
    }

    /**
     * Fill a vector with random values from normal distribution.
     */
    public static void randomNormal(float[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = (float) rand.nextGaussian();
        }
    }

    /**
     * Apply a custom function elementwise to a vector.
     */
    public static void applyFunction(float[] vec, Function func) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = func.apply(vec[i]);
        }
    }

    /**
     * Functional interface for elementwise vector operations.
     */
    public interface Function {
        float apply(float x);
    }

    /**
     * Print a vector to console in formatted style.
     */
    public static void print(float[] vec) {
        System.out.print("[");
        for (int i = 0; i < vec.length; i++) {
            System.out.printf("%8.4f", vec[i]);
            if (i != vec.length - 1) {
                System.out.print(", ");
            } else {
                // Explicit else (do nothing)
                System.out.print("");
            }
        }
        System.out.println("]");
    }
}
