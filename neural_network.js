// Definition of Neuron class
class Neuron {
    constructor(inputSize) {
        this.weights = new Array(inputSize);
        this.bias = Math.random() * 2 - 1; // Random initial weights and bias
        for (let i = 0; i < inputSize; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }
    }

    feedForward(inputs) {
        let sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        return this.activate(sum);
    }

    activate(x) {
        // Using sigmoid activation function
        return 1 / (1 + Math.exp(-x));
    }

    train(inputs, target, learningRate) {
        const output = this.feedForward(inputs);
        const error = target - output;

        // Update weights and bias
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += learningRate * error * inputs[i];
        }
        this.bias += learningRate * error;
    }
}

// Definition of EquivalenceNetwork class
class EquivalenceNetwork {
    constructor() {
        this.neuron1 = new Neuron(1);
        this.neuron2 = new Neuron(1);
        this.outputNeuron = new Neuron(2);
    }

    train(iterations, displayTrainingResults) {
        // Training data for Equivalence function
        const trainingData = [
            { inputs: [0, 0], target: 1 },
            { inputs: [0, 1], target: 0 },
            { inputs: [1, 0], target: 0 },
            { inputs: [1, 1], target: 1 }
        ];

        // Training parameters
        const learningRate = 0.1;

        // Train the neural network
        for (let i = 0; i < iterations; i++) {
            const data = trainingData[Math.floor(Math.random() * trainingData.length)];
            const inputs = data.inputs;
            const target = data.target;

            // Forward pass
            const output1 = this.neuron1.feedForward([inputs[0]]);
            const output2 = this.neuron2.feedForward([inputs[1]]);
            const output = this.outputNeuron.feedForward([output1, output2]);

            // Backpropagation
            this.outputNeuron.train([output1, output2], target, learningRate);
            this.neuron1.train([inputs[0]], output - target, learningRate);
            this.neuron2.train([inputs[1]], output - target, learningRate);

            // Callback function to display training results
            if (displayTrainingResults) {
                displayTrainingResults(i + 1, inputs, target, output);
            }
        }
    }

    predict(inputs) {
        const output1 = this.neuron1.feedForward([inputs[0]]);
        const output2 = this.neuron2.feedForward([inputs[1]]);
        return this.outputNeuron.feedForward([output1, output2]) > 0.5 ? 1 : 0;
    }
}

// Create an instance of EquivalenceNetwork
const network = new EquivalenceNetwork();

// Get references to DOM elements
const trainingResults = document.getElementById('training-results');
const predictionForm = document.getElementById('prediction-form');
const predictionResult = document.getElementById('prediction-result');

// Function to display training results
function displayTrainingResults(iteration, inputs, target, output) {
    const resultElement = document.createElement('div');
    resultElement.textContent = `Iteration: ${iteration}, Inputs: [${inputs}], Target: ${target}, Output: ${output}`;
    trainingResults.appendChild(resultElement);
}

// Train the neural network and display results
network.train(10000, displayTrainingResults);

// Handle Predict button click event
predictionForm.addEventListener('submit', function(event) {
    event.preventDefault();
    const input1 = parseFloat(document.getElementById('input1').value);
    const input2 = parseFloat(document.getElementById('input2').value);
    const prediction = network.predict([input1, input2]);
    predictionResult.textContent = `Prediction: ${prediction}`;
});
