package mnist;

import data.DataPoint;

import static mnist.MnistReader.MNIST_IMAGE_TOTAL_SIZE;
import static mnist.MnistReader.MNIST_IMAGE_WIDTH_HEIGHT;

/**
 * An Mnist Data Point from the Mnist dataset4
 */
public class MnistDataPoint extends DataPoint<Integer> {

    /**
     * @param inputs          Values that are for the input of the neural network
     * @param expectedOutputs Expected outputs for the neural network
     * @param label           Label that has been given to the data point
     */
    public MnistDataPoint(double[] inputs, double[] expectedOutputs, int label) {
        super(inputs, expectedOutputs, label);
    }

    /**
     * Creates a data point from a Mnist input
     *
     * @param inputImage     Input read from Mnist
     * @param expectedOutput Label given to the data point
     * @return The newly created data point from the Mnist data
     */
    public static MnistDataPoint createMnistDataPoint(double[][] inputImage, int expectedOutput) {
        double[] inputs = new double[MNIST_IMAGE_TOTAL_SIZE];

        for (int x = 0; x < MNIST_IMAGE_WIDTH_HEIGHT; x++) {
            for (int y = 0; y < MNIST_IMAGE_WIDTH_HEIGHT; y++) {
                inputs[x * MNIST_IMAGE_WIDTH_HEIGHT + y] = inputImage[x][y] / 255d;
            }
        }

        double[] out = new double[10];
        out[expectedOutput] = 1d;

        return new MnistDataPoint(inputs, out, expectedOutput);
    }
}
