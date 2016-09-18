package io.github.robhinds.nn.activation

import org.scalatest._
import io.github.robhinds.nn.neurons.Neuron

class NeuronSpec extends FunSpec with Matchers {

  describe("Using a basic neuron setup") {
    val neuron = new Neuron(List(4.0, 5.0), 1.0) with NoOpActivation
    val inputs = List(2.0, 3.0)
    val expected = (4.0 * 2.0) + (5.0 * 3.0) + 1.0
    val expectedSigDer = expected * (1 - expected)

    neuron.calculateOutput(inputs)

    it("should calculate the biased output correctly"){
      neuron.biasedOutput should equal (expected)
    }

    it("should calculate the sigmoid derivative for the neuron correctly") {
      neuron.sigmoidDerivative should equal (expectedSigDer)
    }

    it("should calculate the weighted sigmoid derivatives"){
      neuron.sigmoidDerivativeByWeight should equal (List(4.0*expectedSigDer, 5.0*expectedSigDer))
    }
  }

}
