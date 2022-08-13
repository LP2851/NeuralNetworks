package network;

/**
 * Represents a layer in the neural network
 * @version 0.1
 * @author lucas
 */
public class Layer {

    private final int numNodesIn, numNodesOut;

    // Weights for each of the connections
    private final double[][] weights;
    private final double[] biases;

    /**
     *
     * @param numNodesIn Number of nodes coming into the layer (previous layer)
     * @param numNodesOut Number of nodes going to (next layer)
     */
    public Layer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        weights = new double[numNodesIn][numNodesOut];
        biases = new double[numNodesOut];
    }

    /**
     * Calculates the layer's output activations
     * @param inputs The inputs to this layer
     * @return The layer's output activations.
     */
    public double[] calculateOutputs(double[] inputs) {
        double[] activations = new double[numNodesOut];

        // For each of the output nodes
        for (int outNode = 0; outNode < numNodesOut; outNode++) {
            // Starts at the bias amount
            double weightedInput = biases[outNode];
            for (int inNode = 0; inNode < numNodesIn; inNode++) {
                weightedInput += inputs[inNode] * weights[inNode][outNode];
            }
            // Applies and activation function on the weighted input calculated
            activations[outNode] = sigmoidFunction(weightedInput);
        }

        return activations;
    }

    /**
     * Calculates the cost of the node (how far off it is from expected).
     * Error squared cost
     * @param outputActivation Layer's calculated activation
     * @param expectedOutput Predicted/expected activation
     * @return Error squared cost
     */
    public double nodeCost(double outputActivation, double expectedOutput) {
        double error = outputActivation - expectedOutput;
        return error * error;
    }

    /**
     * @return Number of in nodes
     */
    public int getNumNodesIn() {
        return numNodesIn;
    }


    // ACTIVATION FUNCTIONS

    /**
     * Step function (either 0 or 1)
     * 1 if greater than 0 else 0
     * @param x Function input
     * @return Either 0 or 1
     */
    private double stepFunction(double x) {
        return (x > 0) ? 1 : 0;
    }

    /**
     * Sigmoid function (between 0 and 1 in an 'S' shape)
     * @param x Function input
     * @return Value between 0 and 1
     */
    private double sigmoidFunction(double x) {
        return 1d / (1d + Math.exp(-x));
    }

    /**
     * Hyperbolic Tangent function (between -1 and 1 in an 'S' shape)
     * @param x Function input
     * @return Value between -1 and 1
     */
    private double hyperbolicTangentFunction(double x) {
        double e2w = Math.exp(2 * x);
        return (e2w - 1) / (e2w + 1);
    }

    /**
     * SiLU (Sigmoid Linear Unit) function
     * @param x Function input
     * @return Value on SiLU curve
     */
    private double siLU(double x) {
        return x / (1 + Math.exp(-x));
    }
    /**
     * ReLU (Rectified Linear Unit) function
     * 0 if less than or equal to 0 else positive
     * @param x Function input
     * @return Value greater than or equal to 0.
     */
    private double reLU(double x) {
        return Math.max(0, x);
    }
}
