package io.github.robhinds.nn.activation

import org.scalatest._
import io.github.robhinds.nn.builder.NNBuilder
import scala.io.Source

class NeuralNetworkSpec extends FunSpec with Matchers {

  describe("using a basic neural network") {

    val network = NNBuilder
      .inputNeurons(2)
      .hiddenNeurons(2)
      .outputNeurons(2)
      .learningRate(0.1)
      .iterations(10000)
      .build()

    val data = Source.fromURL(getClass.getResource("/testdata.csv"))
    val iter = data.getLines().map(_.split(","))
    val inputData = iter map ( columns => List(columns(1).toDouble,columns(2).toDouble) )

    val dataOut = Source.fromURL(getClass.getResource("/testdata.csv"))
    val iterOut = dataOut.getLines().map(_.split(","))
    val outputData = iterOut map { columns =>
      if (columns(3) == "1.0")
        List(1.0, 0.0)
      else
        List(0.0,1.0)
    }

    it("should try and train ok.."){
      network.train(inputData.toList, outputData.toList, 70)
    }
  }

}
