package NeuralNetCPUv;

import java.util.Random;

/**
 * Matrix class
 *
 * Provides core linear algebra operations used for neural network training.
 * Includes constructors, initialization methods, arithmetic operations,
 * elementwise operations, transposition, and printing.
 *
 * Designed for educational clarity (not optimized for speed).
 */
public class Matrix {

    /** Number of rows in the matrix */
    public int rows;

    /** Number of columns in the matrix */
    public int cols;

    /** 2D array holding matrix data */
    public float[][] data;

    /** Random number generator (used for initialization) */
    private static Random rand = new Random();

    // ============================
    // Constructors
    // ============================

    /**
     * Creates a new matrix of size rows x cols filled with zeros by default.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols]; // automatically filled with 0.0f
    }

    /**
     * Creates an empty matrix (0x0) with no data.
     * Useful for placeholders.
     */
    public Matrix() {
        this.rows = 0;
        this.cols = 0;
        this.data = null;
    }

    // ============================
    // Static Matrix Operations
    // ============================

    /**
     * Performs standard matrix multiplication: C = A * B
     *
     * @param A left-hand side matrix
     * @param B right-hand side matrix
     * @return result matrix (rows(A) x cols(B))
     */
    public static float[][] multiply(float[][] A, float[][] B) {
        int rows = A.length;        // rows of A
        int cols = B[0].length;     // cols of B
        int inner = B.length;       // shared dimension

        float[][] result = new float[rows][cols];

        // Triple nested loop for standard matrix multiplication
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < inner; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    /**
     * Multiplies a matrix with a vector: out = mat^T * vec
     *
     * @param mat 2D array (matrix)
     * @param vec 1D array (vector)
     * @return resulting vector
     */
    public static float[] multiplyVector(float[][] mat, float[] vec) {
        int rows = mat.length;
        int cols = mat[0].length;

        float[] out = new float[cols];

        // Compute weighted sum for each column
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                out[j] += mat[i][j] * vec[i];
            }
        }
        return out;
    }

    /**
     * Transposes a matrix.
     *
     * @param mat input matrix
     * @return transposed matrix
     */
    public static float[][] transpose(float[][] mat) {
        int rows = mat.length;
        int cols = mat[0].length;

        float[][] transposed = new float[cols][rows];

        // Swap rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = mat[i][j];
            }
        }

        return transposed;
    }

    /**
     * Element-wise addition of two matrices.
     */
    public static float[][] add(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A[i][j] + B[i][j];
            }
        }
        return out;
    }

    /**
     * Element-wise subtraction of two matrices.
     */
    public static float[][] subtract(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A[i][j] - B[i][j];
            }
        }
        return out;
    }

    /**
     * Element-wise multiplication of two matrices (Hadamard product).
     */
    public static float[][] multiplyElem(float[][] A, float[][] B) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A[i][j] * B[i][j];
            }
        }
        return out;
    }

    /**
     * Multiplies all elements of a matrix by a scalar.
     */
    public static float[][] multiplyScalar(float[][] A, float scalar) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = A[i][j] * scalar;
            }
        }
        return out;
    }

    // ============================
    // Initialization methods
    // ============================

    /** Fill with zeros */
    public void zeros() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 0.0f;
            }
        }
    }

    /** Fill with ones */
    public void ones() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 1.0f;
            }
        }
    }

    /** Fill with random values between -1 and +1 */
    public void randomizeUniform() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextFloat() * 2f - 1f; // range [-1, 1]
            }
        }
    }

    /** Fill with normally distributed random values (mean=0, std=1) */
    public void randomizeNormal() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) rand.nextGaussian();
            }
        }
    }

    // ============================
    // Utility methods
    // ============================

    /** Deep copy of a matrix */
    public static float[][] copy(float[][] A) {
        int rows = A.length;
        int cols = A[0].length;
        float[][] out = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(A[i], 0, out[i], 0, cols);
        }
        return out;
    }

    /** Apply a function to each element of the matrix */
    public void applyFunction(Function func) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = func.apply(data[i][j]);
            }
        }
    }

    /** Simple functional interface for elementwise operations */
    public interface Function {
        float apply(float x);
    }

    /** Print the matrix in a readable format */
    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%8.4f ", data[i][j]);
            }
            System.out.println();
        }
    }

    /**
     * Multiply this matrix with a vector: out = this^T * vec
     */
    public float[] multiplyVec(float[] vec) {
        float[] out = new float[cols];
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                out[j] += data[i][j] * vec[i];
            }
        }
        return out;
    }
}
