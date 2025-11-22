ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "dq-spark-project",
    version := "0.1.0",
    logLevel := Level.Info,

    libraryDependencies ++= Seq(
      // Spark Libraries
      "org.apache.spark" %% "spark-core" % "3.5.1" % "provided",
      "org.apache.spark" %% "spark-sql"  % "3.5.1" % "provided",

      // Deequ compatible with Spark 3.5.x
      "com.amazon.deequ" % "deequ" % "2.0.7-spark-3.5",

      // Config and YAML parsing
      "com.typesafe" % "config" % "1.4.3",
      "org.yaml" % "snakeyaml" % "2.2",

      // Logging
      "org.slf4j" % "slf4j-api" % "2.0.9",
      "ch.qos.logback" % "logback-classic" % "1.5.6" % Runtime,

      // Test Framework
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    ),

    // Avoid dependency conflicts with old Hadoop versions
    dependencyOverrides ++= Seq(
      "com.google.protobuf" % "protobuf-java" % "3.25.5"
    )
  )
