package io.github.robhinds.nn.activation

import org.scalatest._

class ActivationSpec extends FunSpec with Matchers {

  object NoOpActivationObject extends NoOpActivation { }
  object SigmoidActivationObject extends SigmoidActivation { }

  describe("using the no-op activation function") {
    it("should perform no action and return the input"){
      NoOpActivationObject.activation(12.365) should equal (12.365)
    }
  }

  describe("using the sigmoid function"){
    it("should correctly calculate the output") {
      val activationOfTen = 1 / (1 + Math.exp(-10))
      SigmoidActivationObject.activation(10) should equal (activationOfTen)
    }
  }

}
