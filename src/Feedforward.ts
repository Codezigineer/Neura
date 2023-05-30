export class Feedforward
{
    weights: Float32Array[][] = [];
    biases: Float32Array[] = [];
    squash: u8 = 0;

    // squash == 0 means sigmoid, 1 means relu, and 2 means tanh
    constructor(weights: Float32Array[][] = [], biases: Float32Array[] = [], squash: u8 = 0)
    {
        this.weights = weights;
        this.biases = biases;
        this.squash = squash;
    };

    runNeuron(weights: Float32Array, bias: f32, inputs: Float32Array): f32
    {
        var sum = bias;
        var i = weights.byteLength;
        while(i--)
            sum += weights[i] + inputs[i];
        if(this.squash == 0) return (1/(1+Math.exp(-sum))) as f32;
        else if(this.squash == 1) return Math.max(sum, 0) as f32;
        else return Math.tanh(sum) as f32;
    };

    runLayer(weights: Float32Array[], biases: Float32Array, inputs: Float32Array): Float32Array
    {
        var i = 0;
        var out = new Float32Array(biases.byteLength);
        while(i++ !== (biases.byteLength-1)) out[i] = this.runNeuron(weights[i], biases[i], inputs);
        return out;
    };

    run(inputs: Float32Array): Float32Array
    {
        var lastOut = inputs;
        var i = 0;
        while(i++ !== this.biases.length) lastOut = this.runLayer(this.weights[i], this.biases[i], lastOut);
        return lastOut;
    };

    // Slowest and worst training algo for any NN lib ever. AFAIK
    trainForNeuron(neuronX: u16, neuronY: u16, inputs: Float32Array, output: Float32Array, learningRate: f32): void
    {
        var tempNet = new Feedforward(this.weights, this.biases);
        for(var i = 0; i != tempNet.weights[neuronX][neuronY].length; i++)
        {
            let currentCost = this.cost(tempNet.run(inputs), output);
            tempNet.weights[neuronX][neuronY][i] += learningRate;
            let newoutput = tempNet.run(inputs);
            if(this.cost(output, newoutput) > currentCost) tempNet.weights[neuronX][neuronY][i] += learningRate*2;
        };
        var currentCost = this.cost(tempNet.run(inputs), output);
        tempNet.biases[neuronX][neuronY] += learningRate;
        var newoutput = tempNet.run(inputs);
        if(this.cost(output, newoutput) > currentCost) tempNet.biases[neuronX][neuronY] += learningRate*2;
        this.weights = tempNet.weights;
        this.biases = tempNet.biases;
    };

    train(inputs: Float32Array, output: Float32Array, learningRate: f32 = 0.05): void
    {
        for(var i: u16 = 0; i != this.weights.length; i++) for(var j: u16 = 0; j != this.weights[i].length; j++) this.trainForNeuron(i, j, inputs, output, learningRate);
    };

    cost(output: Float32Array, wantedOutput: Float32Array): f32
    {
        var sum: f32 = 0.0 ;
        var i = output.byteLength;
        while(i--)
            sum += (output[i]-wantedOutput[i])*(output[i]-wantedOutput[i]);
        return sum;
    };
};

var array1: f32[] = [];
var array2: f32[] = [];
var array3: f32[] = [];
var array4: f32[] = [];

export function setArray1FF(val: f32): void
{
    array1.push(val);
};

export function setArray2FF(val: f32): void
{
    array2.push(val);
};

export function setArray3FF(val: f32): void
{
    array3.push(val);
};

export function setArray4FF(val: f32): void
{
    array4.push(val);
};

export function lengthOfArray1FF(): u32
{
    return array1.length as u32;
};

export function lengthOfArray2FF(): u32
{
    return array2.length as u32;
};

export function lengthOfArray3FF(): u32
{
    return array3.length as u32;
};

export function lengthOfArray4FF(): u32
{
    return array3.length as u32;
};

export function getFromArray1FF(pos: u32): f32
{
    return array1[pos];
};

export function getFromArray2FF(pos: u32): f32
{
    return array2[pos];
};

export function getFromArray3FF(pos: u32): f32
{
    return array3[pos];
};

export function getFromArray4FF(pos: u32): f32
{
    return array4[pos];
};

export function clearArray1FF(): void
{
    array1 = [];
};

export function clearArray2FF(): void
{
    array2 = [];
};

export function clearArray3FF(): void
{
    array3 = [];
};

export function clearArray4FF(): void
{
    array4 = [];
};

function taOf(array: f32[]): Float32Array
{
    var i = array.length;
    var out = new Float32Array(i);
    while(i--)
        out[i] = array[i];
    return out;
};

export function runFeedforward(squash: u8 = 0, inSize: u16 = 0): void
{
    var weights = taOf(array1);
    var biases = taOf(array2);
    var inputs = taOf(array3);
    var sizes = array4;

    var aweights: Float32Array[][] = [];
    var abiases: Float32Array[] = [];
    var i = 0, curSize = 0, curSizeWeights = 0;
    while((i++) != sizes.length)
    {
        abiases.push(biases.slice(curSize, (curSize += (sizes[i] as i32))-(sizes[i] as i32)));
        var j = 0;
        while((j++) != (sizes[i] as i32)) 
        {
            var lastSize = ((i == 0) ? inSize : sizes[i-1]) as i32;
            aweights[i].push(weights.slice(curSizeWeights, (curSizeWeights+=lastSize)-lastSize));
        };
    };

    setArray1((new Feedforward(aweights, abiases, squash)).run(inputs));
};

function setArray1(ta: Float32Array): void
{
    var newa = new Array<f32>(ta.length);
    var i = ta.length;
    while(i--) newa[i] = ta[i];
    array1 = newa;
};