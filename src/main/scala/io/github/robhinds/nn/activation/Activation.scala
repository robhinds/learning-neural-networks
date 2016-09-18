package io.github.robhinds.nn.activation

trait ActivationFunction {
  def activation(x: Double): Double
}
trait SigmoidActivation extends ActivationFunction {
  override def activation(x: Double) = 1 / (1 + Math.exp(-x))
}
trait NoOpActivation extends ActivationFunction {
  override def activation(x: Double) = x
}
