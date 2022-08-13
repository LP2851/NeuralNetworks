package data;

import static mnist.MnistReader.MNIST_IMAGE_TOTAL_SIZE;
import static mnist.MnistReader.MNIST_IMAGE_WIDTH_HEIGHT;

/**
 * @version 0.1
 * @author lucas
 */
public class DataPoint {

    public double[] inputs, expectedOutputs;
    public int label;

    /**
     *
     * @param inputs Values that are for the input of the neural network
     * @param expectedOutputs Expected outputs for the neural network
     */
    public DataPoint(double[] inputs, double[] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    /**
     *
     * @param inputs Values that are for the input of the neural network
     * @param expectedOutputs Expected outputs for the neural network
     * @param label Label that has been given to the data point
     */
    public DataPoint(double[] inputs, double[] expectedOutputs, int label) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.label = label;
    }

    /**
     * Creates a data point from a Mnist input
     * @param inputImage Input read from Mnist
     * @param expectedOutput Label given to the data point
     * @return The newly created data point from the Mnist data
     */
    public static DataPoint createMnistDataPoint(double[][] inputImage, int expectedOutput) {
        double[] inputs = new double[MNIST_IMAGE_TOTAL_SIZE];

        for (int x = 0; x < MNIST_IMAGE_WIDTH_HEIGHT; x++) {
            for (int y = 0; y < MNIST_IMAGE_WIDTH_HEIGHT; y++) {
                inputs[x * MNIST_IMAGE_WIDTH_HEIGHT + y] = inputImage[x][y] / 255d;
            }
        }

        double[] out = new double[10];
        out[expectedOutput] = 1d;

        return new DataPoint(inputs, out, expectedOutput);
    }
}
