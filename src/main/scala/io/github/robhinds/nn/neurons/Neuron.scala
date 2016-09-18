package io.github.robhinds.nn.neurons

import io.github.robhinds.nn.activation.ActivationFunction

class Neuron(var weights: List[Double], bias: Double) {
  self: ActivationFunction =>

  var biasedOutput = 0.0
  var activationOutput = 0.0
  var sigmoidDerivative = 0.0
  var sigmoidDerivativeByWeight: List[Double] = Nil
  var inputs: List[Double] = Nil

  def calculateOutput(in: List[Double]): Double = {
    inputs = in
    val weightedInputs = (weights zip inputs) map ( a => a._1 * a._2 )
    val summedWeightedInput = weightedInputs.foldLeft(0.0)(_ + _)
    biasedOutput = summedWeightedInput + bias
    activationOutput = activation(biasedOutput)
    sigmoidDerivative = activationOutput * (1.0 - activationOutput)
    sigmoidDerivativeByWeight = weights map ( _ * sigmoidDerivative )
    activationOutput
  }

}
