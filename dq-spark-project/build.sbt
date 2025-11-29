import sbtassembly.AssemblyPlugin.autoImport._
import sbt.Keys._

ThisBuild / scalaVersion := "2.12.18" // Spark 3.4.x -> Scala 2.12
ThisBuild / organization := "com.anjaneya.dq"

lazy val sparkVersion = "3.4.3"
lazy val deequVersion = "2.0.7-spark-3.4" // Deequ build for Spark 3.4
lazy val snowflakeConnector = "2.16.0-spark_3.4" // Spark 3.4, Scala 2.12
lazy val snowflakeJdbc = "3.24.2"
lazy val slf4jVersion = "1.7.36"
lazy val scalaTestVersion = "3.2.18"

lazy val root = (project in file("."))
  .settings(
    name := "dq-spark-project",
    version := "0.1.0-SNAPSHOT",
    libraryDependencies ++= Seq(
      // Spark (provided by your local Spark install)
      "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
      "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",

      // Deequ (Spark 3.4, Scala 2.12)
      "com.amazon.deequ" % "deequ" % deequVersion,

      // Snowflake connector (Spark 3.4, Scala 2.12) + JDBC
      "net.snowflake" % "spark-snowflake_2.12" % snowflakeConnector,
      "net.snowflake" % "snowflake-jdbc" % snowflakeJdbc,

      // Logging
      "org.slf4j" % "slf4j-api" % slf4jVersion,

      // Tests
      "org.scalatest" %% "scalatest" % scalaTestVersion % Test
    ),

    // Force the newer JDBC version in case the connector pulls an older one
    dependencyOverrides += "net.snowflake" % "snowflake-jdbc" % snowflakeJdbc,
    testFrameworks += new TestFramework("org.scalatest.tools.Framework"),

    // Assembly settings
    assembly / test := {},
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", _ @_*) => MergeStrategy.discard
      case "reference.conf"            => MergeStrategy.concat
      case _                           => MergeStrategy.first
    }
  )
