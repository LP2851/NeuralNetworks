package data;


/**
 * @author lucas
 * @version 0.2
 */
public abstract class DataPoint<E> {

    public double[] inputs, expectedOutputs;
    public final E label;

    /**
     *
     * @param inputs Values that are for the input of the neural network
     * @param expectedOutputs Expected outputs for the neural network
     */
    public DataPoint(double[] inputs, double[] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        label = null;
    }

    /**
     * @param inputs          Values that are for the input of the neural network
     * @param expectedOutputs Expected outputs for the neural network
     * @param label           Label that has been given to the data point
     */
    public DataPoint(double[] inputs, double[] expectedOutputs, E label) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.label = label;
    }

}
