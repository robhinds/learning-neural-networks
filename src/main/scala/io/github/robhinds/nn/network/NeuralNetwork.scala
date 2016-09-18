package io.github.robhinds.nn.network

import io.github.robhinds.nn.neurons.Neuron
import scala.util.Random

/**
  * Based on https://github.com/perborgen/NeuralNetworkNoob/blob/master/net.py
  *
  * Params should be provided using the NNBuilder to create an instance of this class.
  *
  * @param inputNeurons
  * @param hiddenNeurons
  * @param outputNeurons
  * @param learningRate
  * @param iterations
  */
case class NeuralNetwork(
  inputNeurons: List[Neuron],
  hiddenNeurons: List[Neuron],
  outputNeurons: List[Neuron],
  learningRate: Double,
  iterations: Integer) {

  def train(inputs: List[List[Double]], outputs: List[List[Double]], testSetSize: Integer) = {
    assert (testSetSize > inputs.size/2,
      "Specified test set size is too large - use less than half the data for testing")
    assert (inputs.nonEmpty && inputs.head.size == inputNeurons.size,
      s"inputs provided do not match the network set up (network has ${inputNeurons.size} input nodes)")
    assert (outputs.nonEmpty && outputs.head.size == outputNeurons.size,
      s"outputs provided do not match the network set up (network has ${outputNeurons.size} input nodes)")

    //split training vs test data
    val trainingInputs = inputs.dropRight(testSetSize)
    val trainingOutputs = outputs.dropRight(testSetSize)
    val testInputs = inputs.takeRight(testSetSize)
    val testOutputs = outputs.takeRight(testSetSize)

    1 to iterations foreach { i =>
      //randomly select an input/output row to train on
      val trainingIndex = new Random().nextInt(trainingInputs.size)
      val trainingInput = inputs(trainingIndex)
      val trainingOutput = outputs(trainingIndex)

      //run the feed forward pass
      feedForward(trainingInput)

      // run the back prop calcs
      backPropogation(trainingOutput)

    }

    testPerformance(testInputs, testOutputs)
  }

  def feedForward(inputs: List[Double]): List[Double] ={
    assert(inputs.size == inputNeurons.size)
    val hiddenLayerOutputs = hiddenNeurons map ( _.calculateOutput(inputs) )
    outputNeurons map ( _.calculateOutput(hiddenLayerOutputs) )
  }

  def backPropogation(targetOutputs: List[Double]): Unit = {
    val outputNeuronsWithTargets = outputNeurons zip targetOutputs
    val squaredErrorsDerivativePerOutput = outputNeuronsWithTargets map { n =>
      //Output of each neuron - the target output
      n._1.activationOutput - n._2
    }

    //Derivative errors for hidden layer
    val sqErrorDerivativeWithOutputNeurons = outputNeurons zip squaredErrorsDerivativePerOutput
    val derivativeErrorsPerHidden = 0 to hiddenNeurons.size-1 map { hiddenNeuronIndex =>
      val derivativeErrs = sqErrorDerivativeWithOutputNeurons map {
        case (neuron, sqErrDerivative) =>
          neuron.sigmoidDerivativeByWeight(hiddenNeuronIndex) * sqErrDerivative
      }
      derivativeErrs.foldLeft(0.0)(_ + _)
    }

    //iterate over all hidden layer neurons and calculate the weight error,
    //then we multiply by learning rate and apply new weight
    val derivativeErrorsWithHidden = derivativeErrorsPerHidden zip hiddenNeurons
    derivativeErrorsWithHidden foreach { case (derivativeError, neuron) =>
      //iterate over all weights pointing to this hidden neuron to calc error
      neuron.weights = neuron.inputs.zipWithIndex.map{ case (input, index) =>
        val derivedWeightError = derivativeError * neuron.sigmoidDerivative * input
        neuron.weights(index) - (derivedWeightError * learningRate)
      }
    }

    //now do similar for the output layer, but used squared errors
    val squaredErrorsWithOutputs = squaredErrorsDerivativePerOutput zip outputNeurons
    squaredErrorsWithOutputs foreach { case (squaredError, neuron) =>
      //iterate over all weights pointing to this output neuron to calc error
      neuron.weights = neuron.inputs.zipWithIndex.map{ case (input, index) =>
        val derivedWeightError = squaredError * neuron.sigmoidDerivative * input
        neuron.weights(index) - (derivedWeightError * learningRate)
      }
    }
  }

  def testPerformance(inputs: List[List[Double]], outputs: List[List[Double]]) {
    var score = 0.0
    val inputAndOutput = inputs zip outputs
    inputAndOutput map { test =>
      val predictedOutputs = feedForward(test._1) map ( BigDecimal(_).setScale(0, BigDecimal.RoundingMode.HALF_UP).toDouble )
      if (predictedOutputs == test._2)
        score = score + 1.0
    }
    println(s"SCORE : ${(score/inputs.size) * 100}%")

  }

}
