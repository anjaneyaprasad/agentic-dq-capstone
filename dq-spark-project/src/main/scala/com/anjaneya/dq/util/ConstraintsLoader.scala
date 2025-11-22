package com.anjaneya.dq.util

import com.anjaneya.dq.config._
import org.yaml.snakeyaml.Yaml

import java.io.InputStream
import scala.jdk.CollectionConverters._

object ConstraintsLoader {

  // reading constraints YAML for a specific dataset. Returns DatasetConstraintsConfig.
  // constraints YAML path pattern: configs/constraints/<datasetName>.yaml (path hardcoded here for simplicity)
  def loadConstraintsFor(datasetName: String): DatasetConstraintsConfig = {
    val path = s"configs/constraints/$datasetName.yaml"

    val is: InputStream = getClass.getClassLoader.getResourceAsStream(path)
    if (is == null) {
      throw new IllegalArgumentException(s"Constraint config not found on classpath: $path")
    }

    val yaml = new Yaml()
    val root  = yaml.load(is).asInstanceOf[java.util.Map[String, Object]]
    val m     = root.asScala // Map[String, Object]

    val dataset = m.getOrElse("dataset",
      throw new IllegalArgumentException("Missing 'dataset' key in constraints YAML")
    ).toString

    val constraintsAny = m.getOrElse("constraints",
      throw new IllegalArgumentException("Missing 'constraints' key in constraints YAML")
    )

    val constraintsJava =
      constraintsAny.asInstanceOf[java.util.List[java.util.Map[String, Object]]]

    val rules: Seq[ConstraintRule] = constraintsJava.asScala.toSeq.map { cJava =>
      val c = cJava.asScala // Map[String, Object]

      def str(key: String, default: String = ""): String =
        c.get(key).map(_.toString).getOrElse(default)

      def strList(key: String): Seq[String] =
        c.get(key) match {
          case Some(list: java.util.List[_]) =>
            list.asScala.map(_.toString).toSeq
          case Some(single) =>
            Seq(single.toString)
          case None =>
            Seq.empty
        }

      def doubleOpt(key: String): Option[Double] =
        c.get(key).map(v => v.toString.toDouble)

      ConstraintRule(
        `type`         = str("type"),
        column         = str("column"),
        allowed_values = strList("allowed_values"), // ex: currency allowed values "INR", "USD", "EUR", etc.
        level          = str("level", "ERROR"), // if not provided, default to "ERROR"
        min            = doubleOpt("min"),
        max            = doubleOpt("max"),
        threshold      = doubleOpt("threshold")
      )
    }

    DatasetConstraintsConfig(dataset = dataset, constraints = rules)
  }
}