package io.github.robhinds.nn.builder

import io.github.robhinds.nn.activation.{NoOpActivation, SigmoidActivation}
import io.github.robhinds.nn.network.NeuralNetwork
import io.github.robhinds.nn.neurons.Neuron

import scala.util.Random


object NNBuilder {
  var inNeurons: Integer = _
  var hidNeurons: Integer = _
  var outNeurons: Integer = _
  var iters: Integer = _
  var alpha: Double = _
  var randomSeed: Integer = 42

  def inputNeurons(numNeurons: Integer) = {
    inNeurons = numNeurons
    this
  }

  def hiddenNeurons(numNeurons: Integer) = {
    hidNeurons = numNeurons
    this
  }

  def outputNeurons(numNeurons: Integer) = {
    outNeurons = numNeurons
    this
  }

  def iterations(n: Integer) = {
    iters = n
    this
  }

  def learningRate(a: Double) = {
    alpha = a
    this
  }

  def seed(x: Integer) = {
    randomSeed = x
    this
  }

  def build() = {
    val random = new Random(randomSeed)

    val inputs = 1 to inNeurons map { i =>
      new Neuron(List(1.0), 0.0) with NoOpActivation
    }

    val hidden = 1 to hidNeurons map { i =>
      val inputToHiddenWeights = inputs map { input =>
        random.nextDouble
      }
      new Neuron(inputToHiddenWeights.to[List], 1.0) with SigmoidActivation
    }

    val outputs = 1 to outNeurons map { i =>
      val hiddenToOutputWeights = hidden map { input =>
        random.nextDouble
      }
      new Neuron(hiddenToOutputWeights.to[List], 1.0) with SigmoidActivation
    }

    NeuralNetwork(inputs.to[List], hidden.to[List], outputs.to[List], alpha, iters)
  }
}
