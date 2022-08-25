package network;

import java.util.Random;

/**
 * Represents a layer in the neural network
 * @version 0.1
 * @author lucas
 */
public class Layer {

    private final int numNodesIn, numNodesOut;

    // Weights for each of the connections
    public double[][] weights;
    public double[] biases;

    // Gradient Decent- Cost gradients
    public double[][] costGradientWeights;
    public double[] costGradientBias;

    private double[] activations, weightedInputs, inputs;


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

        clearGradients();
        initRandomWeights();
    }

    public void initRandomWeights() {
        Random rand = new Random();
        for (int i = 0; i < numNodesIn; i++) {
            for (int j = 0; j < numNodesOut; j++) {
                double randVal = rand.nextDouble() * 2 - 1;
                weights[i][j] = randVal / Math.sqrt(numNodesIn);
            }
        }
    }

    public void applyGradients(double learnRate) {
        for (int i = 0; i < numNodesOut; i++) {
            biases[i] -= costGradientBias[i] * learnRate;
            for (int j = 0; j < numNodesIn; j++) {
                weights[j][i] -= costGradientWeights[j][i] * learnRate;
            }
        }
    }

    /**
     * Calculates the layer's output activations
     * @param inputs The inputs to this layer
     * @return The layer's output activations.
     */
    public double[] calculateOutputs(double[] inputs) {
        activations = new double[numNodesOut];
        weightedInputs = new double[numNodesOut];
        this.inputs = inputs;

        // For each of the output nodes
        for (int outNode = 0; outNode < numNodesOut; outNode++) {
            // Starts at the bias amount
            double weightedInput = biases[outNode];
            for (int inNode = 0; inNode < numNodesIn; inNode++) {
                weightedInput += inputs[inNode] * weights[inNode][outNode];
            }
            // Applies and activation function on the weighted input calculated
            weightedInputs[outNode] = weightedInput;
            activations[outNode] = sigmoidFunction(weightedInput);
        }

        return activations;
    }

    public double[] calculateOutputsLayerNodeValues(double[] expectedOutputs) {
        double[] nodeValues = new double[expectedOutputs.length];
        for (int i = 0; i < nodeValues.length; i++) {
            double costDerivative = nodeCostDerivative(activations[i],  expectedOutputs[i]);
            double activationDerivative = sigmoidDerivativeFunction(weightedInputs[i]);
            nodeValues[i] = activationDerivative * costDerivative;
        }

        return nodeValues;
    }

    public double[] calculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues) {
        double[] newNodeValues = new double[numNodesOut];
        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < newNodeValues.length; oldNodeIndex++) {

                // Partial derivative of the weighted input with respect to index
                double weightedInputDerivative = oldLayer.weights[newNodeIndex][oldNodeIndex];
            }

            newNodeValue *= sigmoidDerivativeFunction(weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }
        return newNodeValues;
    }

    public void updateGradients(double[] nodeValues) {
        for (int out = 0; out < numNodesOut; out++) {
            for (int in = 0; in < numNodesIn; in++) {
                double derivativeCostWrtWeight = inputs[in] * nodeValues[out];
                costGradientWeights[in][out] += derivativeCostWrtWeight;
            }

            double derivativeCostWrtBias =  1 * nodeValues[out];
            costGradientBias[out] += derivativeCostWrtBias;
        }
    }

    public void clearGradients() {
        costGradientWeights = new double[numNodesIn][numNodesOut];
        costGradientBias = new double[numNodesOut];
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

    public double nodeCostDerivative(double outputActivation, double expectedOutput) {
        return 2 * (outputActivation - expectedOutput);
    }

    /**
     * @return Number of in nodes
     */
    public int getNumNodesIn() {
        return numNodesIn;
    }

    /**
     * @return Number of out nodes
     */
    public int getNumNodesOut() {
        return numNodesOut;
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

    private double sigmoidDerivativeFunction(double x) {
        double activation = sigmoidFunction(x);
        return activation * (1 - activation);
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
